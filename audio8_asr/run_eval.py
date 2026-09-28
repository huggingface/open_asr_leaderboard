#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.nn.functional as F
from datasets import IterableDataset
from kaldialign import batch_error_rate
from transformers import AutoModelForCausalLM, AutoProcessor, CompileConfig

from normalizer import data_utils


DEFAULT_MODEL_ID = "AutoArk-AI/Audio8-ASR-0.1B"
DEFAULT_MODEL_REVISION = "b812eff124893ecd76a1dcde74ee58db5adab59c"
PROMPT = "Please transcribe this audio."


@dataclass
class LocalRow:
    sample_id: str
    reference: str
    waveform: np.ndarray
    sampling_rate: int
    duration: float


@dataclass
class BatchOutput:
    predictions: list[str]
    generated_ids: list[list[int]]
    runtime_seconds: float
    stop_hits: list[bool]
    max_new_hits: list[bool]
    stop_token_ids: list[int | None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Audio8-ASR with the Open ASR Leaderboard contract.")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--manifest_model_id")
    parser.add_argument("--model_revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--dataset_path", default="hf-audio/open-asr-leaderboard")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--skip_samples", type=int, default=0)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_audio_seconds", type=float, default=30.0)
    parser.add_argument("--sampling_rate", type=int, default=16000)
    parser.add_argument("--model_max_length", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--feature_workers", type=int, default=int(os.environ.get("AUDIO8_FEATURE_WORKERS", "16")))
    parser.add_argument("--torch_cpu_threads", type=int, default=int(os.environ.get("AUDIO8_TORCH_CPU_THREADS", "1")))
    parser.add_argument("--torch_compile", choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"))
    parser.add_argument("--compile_fullgraph", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--local_parquet_dir", type=Path)
    parser.add_argument("--local_parquet_file", type=Path, action="append", default=[])
    parser.add_argument("--output_root", type=Path, default=Path("."))
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    value = str(value)
    if value in {"-1", "cpu"}:
        return torch.device("cpu")
    if value.startswith("cuda"):
        return torch.device(value)
    return torch.device(f"cuda:{value}")


def resolve_dtype(value: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[value]


def unique_token_ids(values: Iterable[Any]) -> list[int]:
    result: list[int] = []
    for value in values:
        children = value if isinstance(value, (list, tuple)) else [value]
        for child in children:
            if isinstance(child, int) and child >= 0 and child not in result:
                result.append(child)
    return result


def build_conversation(waveform: np.ndarray) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "array": waveform},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def waveform_from_audio_item(audio: Any, target_sampling_rate: int) -> tuple[np.ndarray, int]:
    if isinstance(audio, dict):
        waveform = audio.get("array")
        sampling_rate = audio.get("sampling_rate", target_sampling_rate)
    elif hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        waveform = getattr(samples, "data", samples)
        sampling_rate = getattr(samples, "sample_rate", target_sampling_rate)
    else:
        waveform = audio
        sampling_rate = target_sampling_rate

    if torch.is_tensor(waveform):
        waveform = waveform.detach().cpu().float().numpy()
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        channel_axis = 0 if waveform.shape[0] <= waveform.shape[1] else 1
        waveform = waveform.mean(axis=channel_axis)
    waveform = waveform.reshape(-1)
    sampling_rate = int(sampling_rate)
    if sampling_rate != int(target_sampling_rate):
        waveform = librosa.resample(
            waveform,
            orig_sr=sampling_rate,
            target_sr=int(target_sampling_rate),
        )
        sampling_rate = int(target_sampling_rate)
    return np.asarray(waveform, dtype=np.float32), sampling_rate


def parquet_paths(args: argparse.Namespace) -> list[Path]:
    paths = [path.resolve() for path in args.local_parquet_file]
    if args.local_parquet_dir is not None:
        paths.extend(path.resolve() for path in sorted(args.local_parquet_dir.glob("*.parquet")))
    unique: list[Path] = []
    for path in paths:
        if path not in unique:
            unique.append(path)
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing local Parquet files: {missing}")
    return unique


def row_sample_id(row: dict[str, Any], fallback_index: int) -> str:
    for key in ("id", "wav_filename", "file", "path"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"sample_{fallback_index}"


def decode_embedded_audio(row: dict[str, Any], sampling_rate: int) -> tuple[np.ndarray, float]:
    audio = row.get("audio")
    if isinstance(audio, dict):
        audio_bytes = audio.get("bytes")
    else:
        audio_bytes = row.get("bytes")
    if audio_bytes is None:
        raise ValueError(f"Local Parquet row has no embedded audio bytes; keys={sorted(row)}")
    decoded, original_sr = sf.read(io.BytesIO(bytes(audio_bytes)), dtype="float32", always_2d=True)
    duration = float(decoded.shape[0]) / float(original_sr)
    waveform = decoded.mean(axis=1)
    if int(original_sr) != int(sampling_rate):
        waveform = librosa.resample(waveform, orig_sr=int(original_sr), target_sr=int(sampling_rate))
    return np.asarray(waveform, dtype=np.float32), duration


def iter_local_rows(args: argparse.Namespace) -> Iterator[LocalRow]:
    paths = parquet_paths(args)
    if not paths:
        raise ValueError("Local mode requires --local_parquet_dir or --local_parquet_file")
    skipped = 0
    emitted = 0
    source_index = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        for row_group_index in range(parquet.num_row_groups):
            rows = parquet.read_row_group(row_group_index).to_pylist()
            for row in rows:
                reference = str(data_utils.get_text(row) or "")
                normalized = data_utils.normalizer(reference)
                if not data_utils.is_target_text_in_range(normalized):
                    source_index += 1
                    continue
                if skipped < max(0, int(args.skip_samples)):
                    skipped += 1
                    source_index += 1
                    continue
                if args.max_eval_samples > 0 and emitted >= int(args.max_eval_samples):
                    return
                waveform, duration = decode_embedded_audio(row, args.sampling_rate)
                yield LocalRow(
                    sample_id=row_sample_id(row, source_index),
                    reference=reference,
                    waveform=waveform,
                    sampling_rate=int(args.sampling_rate),
                    duration=duration,
                )
                emitted += 1
                source_index += 1


def local_batch_from_rows(rows: Sequence[LocalRow]) -> dict[str, list[Any]]:
    return {
        "audio": [{"array": row.waveform, "sampling_rate": row.sampling_rate} for row in rows],
        "original_text": [row.reference for row in rows],
        "id": [row.sample_id for row in rows],
    }


def rows_to_batch(rows: Sequence[dict[str, Any]]) -> dict[str, list[Any]]:
    keys: set[str] = set()
    for row in rows:
        keys.update(row.keys())
    return {key: [row.get(key) for row in rows] for key in keys}


def chunked_rows(rows: Iterable[Any], batch_size: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def iter_official_batches(dataset: Any, batch_size: int) -> Iterator[dict[str, list[Any]]]:
    if isinstance(dataset, IterableDataset):
        for rows in chunked_rows(iter(dataset), batch_size):
            yield rows_to_batch(rows)
        return
    for start in range(0, len(dataset), batch_size):
        yield dict(dataset[start : start + batch_size])


def load_official_dataset(args: argparse.Namespace) -> Any:
    dataset = data_utils.load_data(args)
    dataset = data_utils.prepare_data(dataset, sampling_rate=args.sampling_rate)
    if args.skip_samples > 0:
        if isinstance(dataset, IterableDataset):
            dataset = dataset.skip(int(args.skip_samples))
        else:
            dataset = dataset.select(range(min(int(args.skip_samples), len(dataset)), len(dataset)))
    if args.max_eval_samples > 0:
        if isinstance(dataset, IterableDataset):
            dataset = dataset.take(int(args.max_eval_samples))
        else:
            dataset = dataset.select(range(min(int(args.max_eval_samples), len(dataset))))
    return dataset


class Audio8Evaluator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = resolve_device(args.device)
        self.dtype = resolve_dtype(args.dtype, self.device)
        model_path = Path(args.model_id)
        revision = None if model_path.exists() else args.model_revision
        self.processor = AutoProcessor.from_pretrained(
            args.model_id,
            revision=revision,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
        self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            revision=revision,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
            dtype=self.dtype,
            attn_implementation=args.attn_implementation,
        ).to(self.device)
        self.model.eval()
        self.model_size = sum(parameter.numel() for parameter in self.model.parameters())
        self.hop_length = int(getattr(self.processor.feature_extractor, "hop_length", 160))
        self.eos_token_ids = unique_token_ids(
            [
                self.processor.tokenizer.eos_token_id,
                self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                getattr(self.model.config, "eos_token_id", None),
                getattr(self.model.generation_config, "eos_token_id", None),
            ]
        )
        self.generate_kwargs: dict[str, Any] = {
            "max_new_tokens": int(args.max_new_tokens),
            "do_sample": False,
            "use_cache": True,
        }
        if args.torch_compile is not None:
            self.model.generation_config.cache_implementation = "static"
            self.generate_kwargs["compile_config"] = CompileConfig(
                mode=args.torch_compile,
                fullgraph=args.compile_fullgraph,
            )

    def _prepare_single_input(self, waveform: np.ndarray) -> dict[str, torch.Tensor]:
        max_samples = int(round(float(self.args.max_audio_seconds) * int(self.args.sampling_rate)))
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if max_samples > 0:
            waveform = waveform[:max_samples]
        return dict(
            self.processor.apply_chat_template(
                build_conversation(waveform),
                return_tensors="pt",
                sampling_rate=int(self.args.sampling_rate),
                audio_padding="longest",
                add_generation_prompt=True,
                audio_max_length=max_samples,
                audio_torch_dtype=self.dtype,
                text_kwargs={
                    "padding": "longest",
                    "truncation": True,
                    "max_length": int(self.args.model_max_length),
                },
            )
        )

    def prepare_batch_inputs(
        self,
        waveforms: Sequence[np.ndarray],
    ) -> tuple[dict[str, torch.Tensor], int]:
        workers = max(1, min(int(self.args.feature_workers), len(waveforms)))
        if workers == 1:
            per_sample_inputs = [self._prepare_single_input(waveform) for waveform in waveforms]
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="audio8-features") as executor:
                per_sample_inputs = list(executor.map(self._prepare_single_input, waveforms))

        prompt_length = max(int(item["input_ids"].shape[1]) for item in per_sample_inputs)
        feature_width = max(int(item["input_features"].shape[-1]) for item in per_sample_inputs)
        pad_token_id = int(self.processor.tokenizer.pad_token_id)
        input_ids: list[torch.Tensor] = []
        attention_masks: list[torch.Tensor] = []
        input_features: list[torch.Tensor] = []
        feature_lengths: list[int] = []
        for item in per_sample_inputs:
            ids = item["input_ids"][0]
            attention_mask = item["attention_mask"][0]
            text_padding = prompt_length - int(ids.shape[0])
            if text_padding > 0:
                ids = F.pad(ids, (text_padding, 0), value=pad_token_id)
                attention_mask = F.pad(attention_mask, (text_padding, 0), value=0)
            features = item["input_features"][0]
            valid_feature_width = int(features.shape[-1])
            feature_padding = feature_width - valid_feature_width
            if feature_padding > 0:
                features = F.pad(features, (0, feature_padding), value=0.0)
            input_ids.append(ids)
            attention_masks.append(attention_mask)
            input_features.append(features)
            feature_lengths.append(valid_feature_width)

        return {
            "input_ids": torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attention_masks, dim=0),
            "input_features": torch.stack(input_features, dim=0),
            "feature_lens": torch.tensor(feature_lengths, dtype=torch.long),
        }, prompt_length

    def transcribe_batch(self, waveforms: Sequence[np.ndarray]) -> BatchOutput:
        if not waveforms:
            return BatchOutput([], [], 0.0, [], [], [])
        clipped_waveforms = []
        max_samples = int(round(float(self.args.max_audio_seconds) * int(self.args.sampling_rate)))
        for waveform in waveforms:
            waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
            clipped_waveforms.append(waveform[:max_samples] if max_samples > 0 else waveform)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.perf_counter()

        inputs, prompt_length = self.prepare_batch_inputs(clipped_waveforms)
        inputs = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in dict(inputs).items()
        }

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **self.generate_kwargs)

        if self.device.type == "cuda":
            end_event.record()
            torch.cuda.synchronize(self.device)
            runtime = start_event.elapsed_time(end_event) / 1000.0
        else:
            runtime = time.perf_counter() - start_time

        generated_batch = output_ids[:, prompt_length:].detach().cpu().tolist()
        eos_set = set(self.eos_token_ids)
        pad_token_id = self.processor.tokenizer.pad_token_id
        predictions: list[str] = []
        visible_batch: list[list[int]] = []
        stop_hits: list[bool] = []
        max_new_hits: list[bool] = []
        stop_token_ids: list[int | None] = []
        for generated in generated_batch:
            visible_ids: list[int] = []
            stop_token_id: int | None = None
            for token_id in generated:
                token_id = int(token_id)
                if token_id in eos_set:
                    stop_token_id = token_id
                    break
                if pad_token_id is not None and token_id == int(pad_token_id):
                    continue
                visible_ids.append(token_id)
            prediction = self.processor.decode(visible_ids, skip_special_tokens=True)
            predictions.append(normalize_whitespace(prediction))
            visible_batch.append(visible_ids)
            stop_hits.append(stop_token_id is not None)
            max_new_hits.append(stop_token_id is None and len(visible_ids) >= int(self.args.max_new_tokens))
            stop_token_ids.append(stop_token_id)
        return BatchOutput(
            predictions=predictions,
            generated_ids=visible_batch,
            runtime_seconds=runtime,
            stop_hits=stop_hits,
            max_new_hits=max_new_hits,
            stop_token_ids=stop_token_ids,
        )


def process_batch(
    batch: dict[str, list[Any]],
    evaluator: Audio8Evaluator,
    args: argparse.Namespace,
) -> dict[str, list[Any]]:
    audio_items = batch["audio"]
    waveforms: list[np.ndarray] = []
    durations: list[float] = []
    for audio in audio_items:
        waveform, sampling_rate = waveform_from_audio_item(audio, args.sampling_rate)
        waveforms.append(waveform)
        durations.append(float(len(waveform)) / float(sampling_rate))
    output = evaluator.transcribe_batch(waveforms)
    per_sample_time = output.runtime_seconds / len(waveforms)
    references = list(batch.get("original_text") or [data_utils.get_text(row) for row in batch])
    audio_filepaths = data_utils.extract_audio_filepaths_from_batch(batch, len(waveforms))
    result = {
        "audio_length_s": durations,
        "transcription_time_s": [per_sample_time] * len(waveforms),
        "predictions": output.predictions,
        "references": references,
        "audio_filepath": audio_filepaths,
        "generated_tokens": [len(ids) for ids in output.generated_ids],
        "generation_hit_stop": output.stop_hits,
        "generation_hit_max_new_tokens": output.max_new_hits,
        "generation_stop_token_id": output.stop_token_ids,
    }
    if data_utils.is_chunked_dataset(args.dataset_path):
        result.update({key: list(batch[key]) for key in data_utils.CHUNK_METADATA_KEYS})
    return result


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def write_outputs(
    all_results: dict[str, list[Any]],
    evaluator: Audio8Evaluator,
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any]]:
    is_chunked = data_utils.is_chunked_dataset(args.dataset_path)
    output_root = args.output_root.resolve()
    manifest_model_id = args.manifest_model_id or args.model_id
    with working_directory(output_root):
        manifest_relative = data_utils.write_manifest(
            all_results["references"],
            all_results["predictions"],
            manifest_model_id,
            args.dataset_path,
            args.dataset,
            args.split,
            audio_length=all_results["audio_length_s"],
            transcription_time=all_results["transcription_time_s"],
            audio_filepaths=all_results["audio_filepath"],
            extra_fields={key: all_results[key] for key in data_utils.CHUNK_METADATA_KEYS}
            if is_chunked
            else None,
        )
    manifest_path = output_root / manifest_relative
    if is_chunked:
        sessions = data_utils.merge_chunked_manifest(data_utils.read_manifest(manifest_path))
        raw_references = [session["text"] for session in sessions]
        raw_predictions = [session["pred_text"] for session in sessions]
    else:
        raw_references = all_results["references"]
        raw_predictions = all_results["predictions"]
    references = [data_utils.normalizer(text) for text in raw_references]
    predictions = [data_utils.normalizer(text) for text in raw_predictions]
    refs_split = [tuple(text.split()) for text in references]
    preds_split = [tuple(text.split()) for text in predictions]
    errors = batch_error_rate(refs_split, preds_split, merge_compounds=True)
    total_audio = sum(float(value) for value in all_results["audio_length_s"])
    total_time = sum(float(value) for value in all_results["transcription_time_s"])
    max_hit_indices = [
        index
        for index, hit in enumerate(all_results["generation_hit_max_new_tokens"])
        if hit
    ]
    summary = {
        "model_id": args.model_id,
        "manifest_model_id": manifest_model_id,
        "model_revision": args.model_revision,
        "dataset_path": args.dataset_path,
        "dataset": args.dataset,
        "split": args.split,
        "manifest": str(manifest_path),
        "samples": len(all_results["predictions"]),
        # chunked datasets are scored per parent session, not per sample
        "scored_units": len(references),
        "wer": round(100.0 * float(errors["err_rate"]), 4),
        "ins": int(errors["ins"]),
        "del": int(errors["del"]),
        "sub": int(errors["sub"]),
        "audio_seconds": total_audio,
        "inference_seconds": total_time,
        "rtfx": total_audio / total_time if total_time > 0 else None,
        "stop_hits": sum(bool(value) for value in all_results["generation_hit_stop"]),
        "max_new_hits": sum(bool(value) for value in all_results["generation_hit_max_new_tokens"]),
        "empty_predictions": sum(not str(value).strip() for value in all_results["predictions"]),
        "mean_generated_tokens": (
            sum(int(value) for value in all_results["generated_tokens"]) / len(all_results["generated_tokens"])
            if all_results["generated_tokens"]
            else 0.0
        ),
        "max_new_audio_filepaths": [all_results["audio_filepath"][index] for index in max_hit_indices],
        "model_parameters": evaluator.model_size,
        "settings": {
            "device": str(evaluator.device),
            "dtype": str(evaluator.dtype),
            "attn_implementation": args.attn_implementation,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "max_audio_seconds": args.max_audio_seconds,
            "sampling_rate": args.sampling_rate,
            "model_max_length": args.model_max_length,
            "padding_side": evaluator.processor.tokenizer.padding_side,
            "feature_lengths_supplied": True,
            "torch_compile": args.torch_compile,
            "feature_workers": args.feature_workers,
            "torch_cpu_threads": args.torch_cpu_threads,
        },
    }
    summary_path = manifest_path.with_suffix(manifest_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path, summary


def source_batches(args: argparse.Namespace) -> Iterator[dict[str, list[Any]]]:
    if parquet_paths(args):
        for rows in chunked_rows(iter_local_rows(args), max(1, int(args.batch_size))):
            yield local_batch_from_rows(rows)
        return
    dataset = load_official_dataset(args)
    yield from iter_official_batches(dataset, max(1, int(args.batch_size)))


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be positive")
    if data_utils.is_chunked_dataset(args.dataset_path) and parquet_paths(args):
        raise ValueError(
            "local Parquet input cannot be used with a chunked dataset: the local rows "
            "carry no chunk metadata to reassemble sessions with."
        )
    if args.feature_workers < 1 or args.torch_cpu_threads < 1:
        raise ValueError("--feature_workers and --torch_cpu_threads must be positive")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.set_num_threads(int(args.torch_cpu_threads))
    torch.set_float32_matmul_precision("high")

    evaluator = Audio8Evaluator(args)
    print(f"Model size: {evaluator.model_size / 1e9:.6f}B parameters")
    print(
        f"model={args.model_id}@{args.model_revision} dataset={args.dataset}/{args.split} "
        f"device={evaluator.device} dtype={evaluator.dtype} batch_size={args.batch_size} "
        f"padding_side={evaluator.processor.tokenizer.padding_side} "
        f"feature_workers={args.feature_workers} torch_cpu_threads={args.torch_cpu_threads}",
        flush=True,
    )

    if args.warmup_steps > 0:
        try:
            warmup_batch = next(source_batches(args))
        except StopIteration:
            warmup_batch = None
        if warmup_batch is not None:
            warmup_waveforms = [
                waveform_from_audio_item(audio, args.sampling_rate)[0]
                for audio in warmup_batch["audio"]
            ]
            for step in range(int(args.warmup_steps)):
                evaluator.transcribe_batch(warmup_waveforms)
                print(f"warmup_step={step + 1}/{args.warmup_steps}", flush=True)

    all_results: dict[str, list[Any]] = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": [],
        "audio_filepath": [],
        "generated_tokens": [],
        "generation_hit_stop": [],
        "generation_hit_max_new_tokens": [],
        "generation_stop_token_id": [],
    }
    if data_utils.is_chunked_dataset(args.dataset_path):
        all_results.update({key: [] for key in data_utils.CHUNK_METADATA_KEYS})
    processed = 0
    for batch in source_batches(args):
        result = process_batch(batch, evaluator, args)
        for key in all_results:
            all_results[key].extend(result[key])
        processed += len(result["predictions"])
        if processed == len(result["predictions"]) or processed % 100 == 0:
            print(f"processed={processed}", flush=True)

    manifest_path, summary = write_outputs(all_results, evaluator, args)
    print(f"Results saved at path: {manifest_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()