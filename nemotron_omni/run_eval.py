"""Open ASR Leaderboard runner for Nemotron-3-Nano-Omni using HF transformers.
"""

import argparse
import os
import time

import evaluate
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from normalizer import data_utils


wer_metric = evaluate.load("wer")
torch.set_float32_matmul_precision("medium")


def main(args):
    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    model_source = args.model_path or args.model_id
    torch_dtype = torch.bfloat16 if args.device >= 0 else torch.float32

    load_kwargs = {"trust_remote_code": True}
    if args.revision:
        load_kwargs["revision"] = args.revision

    processor = AutoProcessor.from_pretrained(model_source, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_source, **load_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        device_map=device,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        **load_kwargs,
    ).eval()

    audio_token = getattr(tokenizer, "audio_token", "<so_embedding>")
    sample_rate = getattr(processor, "audio_sampling_rate", 16_000)

    # One chat-prompt template per sample; inline a single `<so_embedding>`
    # placeholder which the (patched) processor will expand to the encoder's
    # actual token count for that clip.
    user_message = f"{audio_token}\n{args.user_prompt}"
    chat_messages = [
        {"role": "system", "content": "/no_think"},
        {"role": "user", "content": user_message},
    ]
    prompt_template = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )

    def transcribe_batch(audios: list) -> list:
        prompts = [prompt_template] * len(audios)
        inputs = processor(
            text=prompts,
            audio=audios,
            padding=True,
            return_tensors="pt",
        )
        sound_clips = inputs.pop("sound_clips", None)
        inputs = inputs.to(device)

        with torch.inference_mode():
            generated_ids = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                sound_clips=sound_clips,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs.input_ids.shape[1]
        return [
            tokenizer.decode(
                gid[prompt_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for gid in generated_ids
        ]

    def benchmark(batch):
        audios = [np.asarray(audio["array"], dtype=np.float32) for audio in batch["audio"]]
        minibatch_size = len(audios)
        batch["audio_length_s"] = [len(audio) / sample_rate for audio in audios]

        start_time = time.time()
        raw_texts = transcribe_batch(audios)
        runtime = time.time() - start_time
        per_sample_runtime = runtime / max(1, minibatch_size)

        batch["transcription_time_s"] = [per_sample_runtime] * minibatch_size
        batch["predictions"] = [data_utils.normalizer(t) for t in raw_texts]
        batch["references"] = batch["norm_text"]
        return batch

    if args.warmup_steps is not None:
        warmup_dataset = data_utils.load_data(args)
        warmup_dataset = data_utils.prepare_data(warmup_dataset)

        num_warmup_samples = args.warmup_steps * args.batch_size
        if args.streaming:
            warmup_dataset = warmup_dataset.take(num_warmup_samples)
        else:
            warmup_dataset = warmup_dataset.select(
                range(min(num_warmup_samples, len(warmup_dataset)))
            )
        warmup_dataset = iter(
            warmup_dataset.map(benchmark, batch_size=args.batch_size, batched=True)
        )

        for _ in tqdm(warmup_dataset, desc="Warming up..."):
            continue

    dataset = data_utils.load_data(args)
    dataset = data_utils.prepare_data(dataset)

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    dataset = dataset.map(
        benchmark,
        batch_size=args.batch_size,
        batched=True,
        remove_columns=["audio"],
    )

    all_results = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": [],
    }
    for result in tqdm(iter(dataset), desc="Samples..."):
        for key in all_results:
            all_results[key].append(result[key])

    manifest_path = data_utils.write_manifest(
        all_results["references"],
        all_results["predictions"],
        args.model_id,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=all_results["audio_length_s"],
        transcription_time=all_results["transcription_time_s"],
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    wer = wer_metric.compute(
        references=all_results["references"],
        predictions=all_results["predictions"],
    )
    wer = round(100 * wer, 2)
    rtfx = round(
        sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2
    )
    print("WER:", wer, "%", "RTFx:", rtfx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default="esb/datasets")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--user_prompt", type=str,
                        default="Transcribe the audio clip into text. Return only the transcription.")
    args = parser.parse_args()
    parser.set_defaults(streaming=False)
    main(args)
