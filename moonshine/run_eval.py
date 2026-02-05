import argparse
import os
import torch
from transformers import MoonshineForConditionalGeneration, AutoProcessor

import evaluate
from normalizer import data_utils, cuda_sync
import time
from tqdm import tqdm
import numpy as np

wer_metric = evaluate.load("wer")
torch.set_float32_matmul_precision('high')

def main(args):
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = MoonshineForConditionalGeneration.from_pretrained(args.model_id).to(args.device).to(torch_dtype)
    model.eval()  # Set model to evaluation mode
    processor = AutoProcessor.from_pretrained(args.model_id)

    if args.torch_compile:
        model.forward = torch.compile(model.forward, mode=args.compile_mode, fullgraph=True)
        if model.can_generate():
            # enable static k/v cache for autoregressive models
            model.generation_config.cache_implementation = "static"

    def benchmark(batch, min_new_tokens=None):
        # Load audio inputs
        audios = [audio["array"] for audio in batch["audio"]]
        minibatch_size = len(audios)

        # 1. Pre-Processing (outside timed block)
        # 1.1 Pad audios to max batch size if using torch compile to prevent re-compilations
        padding_size = 0
        if minibatch_size != args.batch_size and args.torch_compile:
            padding_size = args.batch_size - minibatch_size
            padding_audios = [audios[-1] for _ in range(padding_size)]
            audios.extend(padding_audios)

        inputs = processor(audios, return_tensors="pt", padding=True, sampling_rate=16000).to(args.device).to(torch_dtype)

        # Create a mask for output tokens to limit length based on input audio clip length.
        # Add 2 to token limits to account for <sot> and <eot>.
        token_generation_limits = [len(clip) * 6.5 // 16000 + 2 for clip in audios]
        max_new_tokens = torch.tensor(token_generation_limits).reshape((-1, 1)).to(args.device)

        # START TIMING - CUDA sync for accurate GPU timing
        cuda_sync(args.device)
        start_time = time.time()

        # Model inference only in timed block
        with torch.inference_mode(): 
            pred_ids = model.generate(**inputs, max_new_tokens=max_new_tokens.max())

        # END TIMING - CUDA sync before measuring
        cuda_sync(args.device)
        runtime = time.time() - start_time

        # Post-processing outside timed block
        output_mask = torch.arange(pred_ids.shape[-1]).repeat((pred_ids.shape[0], 1)).to(args.device)
        output_mask = output_mask > max_new_tokens

        eot_token = model.config.eos_token_id
        pred_ids.masked_fill(output_mask, eot_token)

        # 3.2 Convert token ids to text transcription
        pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True)

        # normalize by minibatch size since we want the per-sample time
        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]

        # normalize transcriptions with English normalizer
        pred_text = pred_text if padding_size == 0 else pred_text[:-padding_size]
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
        warmup_dataset = iter(warmup_dataset.map(benchmark, batch_size=args.batch_size, batched=True, fn_kwargs={"min_new_tokens": args.max_new_tokens}))

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
        "--torch_compile",
        action="store_true",
        help="Whether to JIT compile the forward pass of the model.",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="max-autotune",
        help="Mode for torch compiling model forward pass. Can be either 'default', 'reduce-overhead', 'max-autotune' or 'max-autotune-no-cudagraphs'.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10,
        help="Number of warm-up steps to run before launching the timed runs.",
    )
    args = parser.parse_args()
    parser.set_defaults(streaming=False)

    main(args)
