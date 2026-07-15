"""Evaluate local ONNX ASR exports on one multilingual leaderboard split."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from importlib.metadata import version
from itertools import chain
from pathlib import Path
from typing import Protocol

os.environ.setdefault("DATASETS_USE_TORCHCODEC", "0")

import numpy as np
import onnxruntime as ort
from datasets import Audio, load_dataset
from huggingface_hub import snapshot_download
from jiwer import wer as compute_wer
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from normalizer import data_utils  # noqa: E402
from normalizer.eval_utils import normalize_compound_pairs  # noqa: E402


SAMPLE_RATE = 16_000
DATASET_ID = "nithinraok/asr-leaderboard-datasets"


@dataclass
class Sample:
    waveform: np.ndarray
    reference: str
    duration: float
    audio_filepath: str | None


class Backend(Protocol):
    def transcribe(self, waveforms: list[np.ndarray]) -> list[str]: ...


class OnnxAsrBackend:
    def __init__(self, model_path: str, num_threads: int) -> None:
        import onnx_asr

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = num_threads
        session_options.inter_op_num_threads = 1
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.model = onnx_asr.load_model(
            "nemo-conformer-tdt",
            path=model_path,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

    def transcribe(self, waveforms: list[np.ndarray]) -> list[str]:
        result = self.model.recognize(waveforms, sample_rate=SAMPLE_RATE)
        if not isinstance(result, list):
            result = [result]
        return [str(text) for text in result]


class SherpaOnnxBackend:
    def __init__(self, model_path: str, num_threads: int) -> None:
        import sherpa_onnx

        root = Path(model_path)
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(root / "encoder.int8.onnx"),
            decoder=str(root / "decoder.int8.onnx"),
            joiner=str(root / "joiner.int8.onnx"),
            tokens=str(root / "tokens.txt"),
            model_type="nemo_transducer",
            num_threads=num_threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=128,
            decoding_method="greedy_search",
            debug=False,
        )

    def transcribe(self, waveforms: list[np.ndarray]) -> list[str]:
        streams = []
        for waveform in waveforms:
            stream = self.recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, waveform)
            streams.append(stream)
        self.recognizer.decode_streams(streams)
        return [stream.result.text for stream in streams]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("onnx-asr", "sherpa-onnx"), required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def download_model(args: argparse.Namespace) -> str:
    kwargs: dict[str, object] = {
        "repo_id": args.repo_id,
        "revision": args.revision,
    }
    if args.backend == "onnx-asr":
        # The repository also contains a separate int8 export. The fp32 encoder
        # uses external data files, so retain everything except the int8 pair.
        kwargs["ignore_patterns"] = ["*-int8.onnx", "scriber-quantizations.json"]
    return snapshot_download(**kwargs)


def load_backend(args: argparse.Namespace, model_path: str) -> Backend:
    if args.backend == "onnx-asr":
        return OnnxAsrBackend(model_path, args.num_threads)
    return SherpaOnnxBackend(model_path, args.num_threads)


def decode_audio(audio: object) -> tuple[np.ndarray, int]:
    if not isinstance(audio, dict) or "array" not in audio:
        raise TypeError(f"Expected a decoded datasets.Audio dictionary, got {type(audio)!r}")
    waveform = np.asarray(audio["array"], dtype=np.float32)
    sample_rate = int(audio["sampling_rate"])
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=0, dtype=np.float32)
    if waveform.ndim != 1:
        raise ValueError(f"Expected mono audio, got shape {waveform.shape}")
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz audio after casting, got {sample_rate} Hz")
    return np.ascontiguousarray(waveform), sample_rate


def iter_batches(dataset, batch_size: int):
    batch: list[Sample] = []
    for row in tqdm(dataset, total=len(dataset), desc="Preparing and transcribing"):
        reference = str(row.get("text", "")).strip()
        if not data_utils.is_target_text_in_range(reference):
            continue
        waveform, sample_rate = decode_audio(row["audio"])
        duration = len(waveform) / sample_rate
        batch.append(
            Sample(
                waveform=waveform,
                reference=reference,
                duration=duration,
                audio_filepath=data_utils.extract_audio_filepath_from_sample(row),
            )
        )
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def write_manifest(path: Path, samples: list[Sample], predictions: list[str], times: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, (sample, prediction, inference_time) in enumerate(zip(samples, predictions, times)):
            record = {
                "audio_filepath": sample.audio_filepath or f"sample_{index}",
                "duration": sample.duration,
                "time": inference_time,
                "text": sample.reference,
                "pred_text": prediction,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_threads < 1 or args.max_samples < 0:
        raise ValueError("batch-size and num-threads must be positive; max-samples must be non-negative")

    print(f"Downloading {args.repo_id}@{args.revision} for {args.variant}")
    model_path = download_model(args)
    print(f"Loading {args.backend} model from immutable snapshot {model_path}")
    backend = load_backend(args, model_path)

    print(f"Loading {args.dataset_id}/{args.config_name}:{args.split}")
    dataset = load_dataset(
        args.dataset_id,
        args.config_name,
        split=args.split,
        streaming=False,
    )
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    if "duration" in dataset.column_names:
        dataset = dataset.sort("duration", reverse=True)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE, decode=True))
    selected_samples = len(dataset)

    batches = iter(iter_batches(dataset, args.batch_size))
    try:
        first_batch = next(batches)
    except StopIteration as exc:
        raise RuntimeError("Dataset contains no scorable samples") from exc

    print(f"Warming up with {len(first_batch)} samples")
    backend.transcribe([sample.waveform for sample in first_batch])

    all_samples: list[Sample] = []
    predictions: list[str] = []
    inference_times: list[float] = []
    batch_count = 0

    for batch in chain((first_batch,), batches):
        started = time.perf_counter()
        batch_predictions = backend.transcribe([sample.waveform for sample in batch])
        elapsed = time.perf_counter() - started
        if len(batch_predictions) != len(batch):
            raise RuntimeError(f"Backend returned {len(batch_predictions)} predictions for {len(batch)} samples")
        batch_count += 1
        all_samples.extend(batch)
        predictions.extend(batch_predictions)
        inference_times.extend([elapsed / len(batch)] * len(batch))

    if not any(text.strip() for text in predictions):
        raise RuntimeError("The model returned only empty predictions")

    normalized_references = [data_utils.ml_normalizer(sample.reference, lang=args.language) for sample in all_samples]
    normalized_predictions = [data_utils.ml_normalizer(text, lang=args.language) for text in predictions]
    normalized_references, normalized_predictions = normalize_compound_pairs(
        normalized_references, normalized_predictions
    )
    wer_percent = round(100 * compute_wer(normalized_references, normalized_predictions), 2)
    total_audio_seconds = sum(sample.duration for sample in all_samples)
    total_inference_seconds = sum(inference_times)
    rtfx = round(total_audio_seconds / total_inference_seconds, 4)

    safe_model_id = args.model_id.replace("/", "-")
    safe_dataset_id = args.dataset_id.replace("/", "-")
    manifest_path = args.output_dir / (
        f"MODEL_{safe_model_id}_DATASET_{safe_dataset_id}_{args.config_name}_{args.split}.jsonl"
    )
    write_manifest(manifest_path, all_samples, predictions, inference_times)

    metrics = {
        "schema_version": 1,
        "variant": args.variant,
        "backend": args.backend,
        "model_id": args.model_id,
        "repo_id": args.repo_id,
        "revision": args.revision,
        "dataset_id": args.dataset_id,
        "config_name": args.config_name,
        "language": args.language,
        "split": args.split,
        "selected_samples": selected_samples,
        "processed_samples": len(all_samples),
        "empty_predictions": sum(not text.strip() for text in predictions),
        "batch_size": args.batch_size,
        "batch_count": batch_count,
        "num_threads": args.num_threads,
        "total_audio_seconds": total_audio_seconds,
        "total_inference_seconds": total_inference_seconds,
        "rtfx": rtfx,
        "wer": wer_percent,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "onnx_asr": package_version("onnx-asr"),
            "onnxruntime": package_version("onnxruntime"),
            "sherpa_onnx": package_version("sherpa-onnx"),
            "runner_name": os.environ.get("RUNNER_NAME"),
            "runner_os": os.environ.get("RUNNER_OS"),
            "runner_arch": os.environ.get("RUNNER_ARCH"),
        },
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({"WER": wer_percent, "RTFx": rtfx, "samples": len(all_samples)}, indent=2))
    print(f"Manifest: {manifest_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
