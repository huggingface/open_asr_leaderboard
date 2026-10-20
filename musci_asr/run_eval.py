"""Evaluate Musci-ASR-qwen-1019 on the Open ASR Leaderboard datasets.

This evaluator is intentionally self-contained for the Musci Space.  It uses the
qwen-asr inference API, keeps raw text in the manifest, and lets the leaderboard
normalizer score every dataset with one decoding configuration.
"""

import argparse
import os
import time
from typing import Any

import evaluate
import numpy as np
import torch
from qwen_asr import Qwen3ASRModel
from tqdm import tqdm

from normalizer import data_utils

wer_metric = evaluate.load("wer")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_model(args: argparse.Namespace) -> Qwen3ASRModel:
    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16 if args.device >= 0 else torch.float32,
        "device_map": device,
        "max_inference_batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
    }
    # revision is optional for local smoke tests, but submit_jobs.sh pins it for
    # reproducible HF Jobs runs.
    if args.model_revision:
        kwargs["revision"] = args.model_revision
    model = Qwen3ASRModel.from_pretrained(args.model_id, **kwargs)
    parameter_model = getattr(model, "model", model)
    print(f"Model size: {sum(p.numel() for p in parameter_model.parameters()) / 1e9:.2f}B parameters")
    return model


def _benchmark(batch: dict[str, list[Any]], model: Qwen3ASRModel, args: argparse.Namespace) -> dict[str, list[Any]]:
    audio_inputs = []
    audio_lengths = []
    for item in batch["audio"]:
        waveform = np.asarray(item["array"], dtype=np.float32)
        sample_rate = int(item["sampling_rate"])  # dict or datasets>=4 AudioDecoder (no .get)
        audio_inputs.append((waveform, sample_rate))
        audio_lengths.append(len(waveform) / sample_rate)

    batch_size = len(audio_inputs)
    batch["audio_length_s"] = audio_lengths
    batch["audio_filepath"] = data_utils.extract_audio_filepaths_from_batch(batch, batch_size)

    device = torch.device(f"cuda:{args.device}" if args.device >= 0 else "cpu")
    _sync(device)
    started = time.perf_counter()
    outputs = model.transcribe(
        audio=audio_inputs,
        language=args.language,
    )
    _sync(device)
    elapsed = time.perf_counter() - started

    batch["transcription_time_s"] = [elapsed / batch_size] * batch_size
    batch["predictions"] = [output.text.strip() for output in outputs]
    batch["references"] = batch["original_text"]
    return batch


def _run(args: argparse.Namespace) -> None:
    model = _load_model(args)

    def run_dataset(dataset):
        return dataset.map(
            lambda batch: _benchmark(batch, model, args),
            batch_size=args.batch_size,
            batched=True,
            remove_columns=["audio"],
        )

    if args.warmup_steps > 0:
        warmup = data_utils.prepare_data(data_utils.load_data(args))
        count = args.warmup_steps * args.batch_size
        warmup = warmup.take(count) if args.streaming else warmup.select(range(min(count, len(warmup))))
        for _ in tqdm(run_dataset(warmup), desc="Warming up"):
            pass

    dataset = data_utils.prepare_data(data_utils.load_data(args))
    if args.max_eval_samples > 0:
        dataset = dataset.take(args.max_eval_samples) if args.streaming else dataset.select(
            range(min(args.max_eval_samples, len(dataset)))
        )
    dataset = run_dataset(dataset)

    result_keys = [
        "audio_length_s",
        "transcription_time_s",
        "predictions",
        "references",
        "audio_filepath",
    ]
    if data_utils.is_chunked_dataset(args.dataset_path):
        result_keys += data_utils.CHUNK_METADATA_KEYS
    results = {key: [] for key in result_keys}
    for row in tqdm(dataset, desc="Samples"):
        for key in result_keys:
            results[key].append(row[key])

    manifest = data_utils.write_manifest(
        results["references"],
        results["predictions"],
        args.model_id,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=results["audio_length_s"],
        transcription_time=results["transcription_time_s"],
        audio_filepaths=results["audio_filepath"],
        extra_fields=(
            {key: results[key] for key in data_utils.CHUNK_METADATA_KEYS}
            if data_utils.is_chunked_dataset(args.dataset_path)
            else None
        ),
    )
    print(f"Manifest: {os.path.abspath(manifest)}")

    if data_utils.is_chunked_dataset(args.dataset_path):
        scored_rows = data_utils.merge_chunked_manifest(data_utils.read_manifest(manifest))
        references = [row["text"] for row in scored_rows]
        predictions = [row["pred_text"] for row in scored_rows]
    else:
        references = results["references"]
        predictions = results["predictions"]

    norm_refs = [data_utils.normalizer(text) for text in references]
    norm_preds = [data_utils.normalizer(text) for text in predictions]
    wer = round(100 * wer_metric.compute(references=norm_refs, predictions=norm_preds), 2)
    total_audio = sum(results["audio_length_s"])
    total_time = sum(results["transcription_time_s"])
    rtfx = round(total_audio / total_time, 2) if total_time else 0.0
    print(f"WER: {wer}%  RTFx: {rtfx}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--model_revision", default="", help="Optional immutable Hub commit or tag.")
    parser.add_argument("--dataset_path", default="hf-audio/open-asr-leaderboard")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--language", default="English")
    return parser.parse_args()


if __name__ == "__main__":
    _run(parse_args())
