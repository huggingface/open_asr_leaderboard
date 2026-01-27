
import argparse
import os
import torch
import os 
import sys
from transformers import AutoModelForCausalLM, AutoConfig, GenerationConfig, AutoTokenizer
from normalizer import data_utils
import evaluate
import time
from tqdm import tqdm
import random
import numpy as np


wer_metric = evaluate.load("wer")
torch.set_float32_matmul_precision("medium")

def main(args,min_new_tokens=None):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True
    ).to(args.device)


    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    generation_config = GenerationConfig.from_pretrained(args.model_id)

    TASK_TOKEN = "<|ASR|>"
    AUDIO_TOKEN = "<|audio_bos|><|AUDIO|><|audio_eos|>\n"

    prompt = [
    {"role": "user", "content": f"{TASK_TOKEN}{AUDIO_TOKEN}{args.user_prompt}"},
    ]
    prompt = tokenizer.apply_chat_template(prompt,add_generation_prompt=True,tokenize=False)
    gen_kwargs = {"max_new_tokens": args.max_new_tokens, "num_beams": args.num_beams}

    def benchmark(batch):
        # Load audio inputs
        audios = [torch.tensor(audio["array"],dtype=torch.float32).cuda() for audio in batch["audio"]]
        minibatch_size = len(audios)
        
        # START TIMING
        start_time = time.time()

        inputs = tokenizer(text=[prompt] * minibatch_size, return_tensors="pt").to('cuda')

        # Model Inference
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            pred_ids = model.generate(
                input_ids=inputs.input_ids,
                audio=audios,
                generation_config=generation_config,
                **gen_kwargs
            )

        # Convert token ids to text transcription
        pred_text = tokenizer.batch_decode(
            pred_ids, 
            skip_special_tokens=True
        )

        # END TIMING
        runtime = time.time() - start_time

        # normalize by minibatch size since we want the per-sample time
        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]

        # normalize transcriptions with English normalizer
        batch["predictions"] = [data_utils.normalizer(pred) for pred in pred_text]
        batch["references"] = batch["norm_text"]
        return batch

    if args.warmup_steps is not None:
        dataset = data_utils.load_data(args)
        dataset = data_utils.prepare_data(dataset)

        num_warmup_samples = args.warmup_steps * args.batch_size
        if args.streaming:
            warmup_dataset = dataset.take(num_warmup_samples)
        else:
            warmup_dataset = dataset.select(range(min(num_warmup_samples, len(dataset))))
        warmup_dataset = iter(warmup_dataset.map(benchmark, batch_size=args.batch_size // 2, batched=True))

        for _ in tqdm(warmup_dataset, desc="Warming up..."):
            continue

    dataset = data_utils.load_data(args)
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))
    dataset = data_utils.prepare_data(dataset)

    dataset = dataset.map(
        benchmark, batch_size=args.batch_size, batched=True, remove_columns=["audio"],
    )

    all_results = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": [],
    }
    result_iter = iter(dataset)
    for result in tqdm(result_iter, desc="Samples..."):
        for key in all_results:
            all_results[key].append(result[key])

    # Write manifest results (WER and RTFX)
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
        references=all_results["references"], predictions=all_results["predictions"]
    )
    wer = round(100 * wer, 2)
    rtfx = round(sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2)
    print("WER:", wer, "%", "RTFx:", rtfx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with 🤗 Transformers",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="esb/datasets",
        help="Dataset path. By default, it is `esb/datasets`",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name. *E.g.* `'librispeech_asr` for the LibriSpeech ASR dataset, or `'common_voice'` for Common Voice. The full list of dataset names "
        "can be found at `https://huggingface.co/datasets/esb/datasets`",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset. *E.g.* `'validation`' for the dev split, or `'test'` for the test split.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Number of beams for beam search.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Number of samples to be evaluated. Put a lower number e.g. 64 for testing this script.",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Choose whether you'd like to download the entire dataset or stream it during the evaluation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate (for auto-regressive models).",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5,
        help="Number of warm-up steps to run before launching the timed runs.",
    )
    parser.add_argument(
        "--user_prompt",
        type=str,
        default="Transcribe the audio clip into text.",
        help="User prompt string.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="ASR",
        help='Selct Speech Text-to-text Task'
    )
    args = parser.parse_args()
    parser.set_defaults(streaming=False)

    main(args)
