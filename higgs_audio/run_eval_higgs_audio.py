"""Evaluation script for Higgs Audio v3 models on ESB benchmark datasets.

No external dependencies beyond transformers + torch. The model bundles
its own audio preprocessing via trust_remote_code=True.

Usage:
    python run_eval_higgs_audio.py \
        --model_id bosonai/higgs-audio-v3-8b-stt \
        --dataset_path hf-audio/esb-datasets-test-only-sorted \
        --dataset ami --split test --device 0 --batch_size 4
"""

import argparse
import os
import sys
import time
import runpy

import torch
import evaluate
from normalizer import data_utils
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

wer_metric = evaluate.load("wer")


def load_transcribe_fn(model_id):
    """Load the bundled transcribe_batch function from the model repo.

    Downloads all Python files needed by transcribe.py, then loads it via
    runpy with the download directory on sys.path so plain (non-relative)
    imports resolve to sibling files.
    """
    from transformers.utils import cached_file

    for filename in [
        "transcribe.py",
        "higgs_audio_collator.py",
        "modeling_higgs_audio_xcodec.py",
        "utils.py",
        "common.py",
        "configuration_higgs_audio.py",
    ]:
        cached_file(model_id, filename)

    path = cached_file(model_id, "transcribe.py")
    module_dir = os.path.dirname(path)

    sys.path.insert(0, module_dir)
    try:
        module_globals = runpy.run_path(path)
    finally:
        sys.path.pop(0)

    return module_globals["transcribe_batch"]


def main(args):
    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"

    model = AutoModel.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model.eval()
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")

    # Required for generation stop conditions
    model.audio_out_bos_token_id = tokenizer.convert_tokens_to_ids("<|audio_out_bos|>")
    model.audio_eos_token_id = tokenizer.convert_tokens_to_ids("<|audio_eos|>")

    transcribe_batch = load_transcribe_fn(args.model_id)

    def benchmark(batch):
        # Load audio inputs
        audios = [audio["array"] for audio in batch["audio"]]
        batch["audio_length_s"] = [
            len(audio) / batch["audio"][0]["sampling_rate"] for audio in audios
        ]
        minibatch_size = len(audios)

        # START TIMING
        start_time = time.time()

        # INFERENCE
        pred_text = transcribe_batch(
            model, tokenizer, audios, sample_rates=16000,
            max_new_tokens=args.max_new_tokens,
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
            dataset = dataset.select(
                range(min(args.max_eval_samples, len(dataset)))
            )

    dataset = dataset.map(
        benchmark, batch_size=args.batch_size, batched=True,
        remove_columns=["audio"],
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
    rtfx = round(
        sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2
    )
    print("WER:", wer, "%", "RTFx:", rtfx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be a HiggsAudio3 checkpoint on the HF Hub.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="esb/datasets",
        help="Dataset path. By default, it is `esb/datasets`.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset.",
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
        default=4,
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
        default=1024,
        help="Maximum number of tokens to generate (includes the chain-of-thought block).",
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
