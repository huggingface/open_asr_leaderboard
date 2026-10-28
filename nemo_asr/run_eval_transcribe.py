#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import copy
import io
import itertools
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable, Iterator, TypeVar

os.environ.setdefault("HF_AUDIO_DECODER_BACKEND", "soundfile")

import numpy as np
import soundfile as sf
from datasets import Audio, IterableDataset, load_dataset
import httpx
from openai import OpenAI
from tqdm import tqdm

from transcribe_eval_helpers import (
    aggregate_transcripts,
    decoded_audio_fields,
    longform_chunk_plan,
)
from transcribe_model_card_contract import (
    MODEL_ID,
    MODEL_REVISION,
    PROMPTS,
    build_chat_completion_request,
    cleanup_generation_loops,
)
from normalizer import data_utils


SAMPLE_RATE = 16_000
WARMUP_DURATIONS_S = (1, 5, 15, 30)
T = TypeVar("T")


@dataclass
class SampleResult:
    reference: str
    prediction: str
    raw_model_text: str | list[str]
    duration_s: float
    audio_filepath: str
    parent_id: str | None = None
    chunk_index: int | None = None


@dataclass
class PreparedSample:
    request: dict | None
    longform_chunks: list[np.ndarray] | None
    aggregate_chunk_outputs: bool
    sampling_rate: int
    reference: str
    duration_s: float
    audio_filepath: str
    parent_id: str | None = None
    chunk_index: int | None = None


def _dataset_config(args: argparse.Namespace) -> str | None:
    return args.dataset or None


def load_eval_dataset(args: argparse.Namespace):
    if args.language == "hi":
        dataset = load_dataset(
            args.dataset_path,
            _dataset_config(args),
            split=args.split,
            streaming=args.streaming,
            token=True,
            revision=args.dataset_revision,
        )
        dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    else:
        load_args = copy.copy(args)
        load_args.dataset = _dataset_config(args)
        dataset = data_utils.load_data(load_args)
        dataset = data_utils.prepare_data(dataset, sampling_rate=SAMPLE_RATE)

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        if isinstance(dataset, IterableDataset) or args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))
    return dataset


def extract_reference(row: dict) -> str:
    lattice = row.get("lattice")
    if lattice is not None:
        return json.dumps(lattice, ensure_ascii=False)
    if "original_text" in row:
        return row["original_text"]
    return data_utils.get_text(row)


def extract_audio(row: dict) -> tuple[np.ndarray, int]:
    decoded_samples, sampling_rate = decoded_audio_fields(row["audio"])
    samples = np.asarray(decoded_samples, dtype=np.float32)
    if samples.ndim == 2:
        channel_axis = 0 if samples.shape[0] <= samples.shape[1] else 1
        samples = samples.mean(axis=channel_axis, dtype=np.float32)
    if samples.ndim != 1:
        raise ValueError(
            f"Expected mono audio after downmix, got shape {samples.shape}"
        )
    if sampling_rate != SAMPLE_RATE:
        raise ValueError(
            f"Expected {SAMPLE_RATE} Hz after resampling, got {sampling_rate} Hz"
        )
    return np.ascontiguousarray(samples, dtype=np.float32), sampling_rate


def wav_data_url(
    samples: np.ndarray,
    sampling_rate: int,
    subtype: str = "FLOAT",
) -> str:
    buffer = io.BytesIO()
    sf.write(buffer, samples, sampling_rate, format="WAV", subtype=subtype)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


def split_audio(
    samples: np.ndarray,
    sampling_rate: int,
    chunk_seconds: float,
    chunk_overlap_seconds: float,
) -> tuple[list[np.ndarray], bool]:
    if chunk_seconds <= 0:
        return [samples], False
    ranges, aggregate_outputs = longform_chunk_plan(
        len(samples), sampling_rate, chunk_seconds, chunk_overlap_seconds
    )
    return [samples[start:end] for start, end in ranges], aggregate_outputs


def batched(iterable: Iterable[T], batch_size: int) -> Iterator[list[T]]:
    iterator = iter(iterable)
    while batch := list(itertools.islice(iterator, batch_size)):
        yield batch


def prepare_sample(
    row: dict,
    args: argparse.Namespace,
    sample_index: int,
) -> PreparedSample:
    samples, sampling_rate = extract_audio(row)
    duration_s = len(samples) / sampling_rate
    chunks, aggregate_chunk_outputs = split_audio(
        samples,
        sampling_rate,
        args.longform_chunk_seconds,
        args.longform_chunk_overlap_seconds,
    )

    if args.longform_chunk_seconds > 0:
        request = None
        longform_chunks = chunks
    else:
        request = build_chat_completion_request(
            wav_data_url(chunks[0], sampling_rate),
            args.prompt_profile,
            model=args.served_model_name,
        )
        longform_chunks = None

    audio_filepath = data_utils.extract_audio_filepath_from_sample(row)
    if audio_filepath is None:
        audio_filepath = f"sample_{sample_index}"

    return PreparedSample(
        request=request,
        longform_chunks=longform_chunks,
        aggregate_chunk_outputs=aggregate_chunk_outputs,
        sampling_rate=sampling_rate,
        reference=extract_reference(row),
        duration_s=duration_s,
        audio_filepath=audio_filepath,
        parent_id=row.get("parent_id"),
        chunk_index=row.get("chunk_index"),
    )


def transcribe_request(client: OpenAI, request: dict) -> str:
    response = client.chat.completions.create(**request)
    content = response.choices[0].message.content
    if not isinstance(content, str):
        raise TypeError(f"Expected a text completion, got {type(content).__name__}")
    return content


def finalize_sample(prepared: PreparedSample, raw_parts: list[str]) -> SampleResult:
    combined_raw = aggregate_transcripts(raw_parts, prepared.aggregate_chunk_outputs)
    prediction = cleanup_generation_loops(combined_raw)
    return SampleResult(
        reference=prepared.reference,
        prediction=prediction,
        raw_model_text=raw_parts[0] if len(raw_parts) == 1 else raw_parts,
        duration_s=prepared.duration_s,
        audio_filepath=prepared.audio_filepath,
        parent_id=prepared.parent_id,
        chunk_index=prepared.chunk_index,
    )


def run_shortform_batch(
    executor: ThreadPoolExecutor,
    client: OpenAI,
    prepared: list[PreparedSample],
) -> tuple[list[SampleResult], list[float]]:
    requests = [sample.request for sample in prepared]
    if any(request is None for request in requests):
        raise RuntimeError("short-form batch contains a long-form sample")

    start = time.perf_counter()
    futures = [
        executor.submit(transcribe_request, client, request) for request in requests
    ]
    raw_outputs = [[future.result()] for future in futures]
    batch_wall_time = time.perf_counter() - start
    results = [
        finalize_sample(sample, raw_parts)
        for sample, raw_parts in zip(prepared, raw_outputs)
    ]
    return results, [batch_wall_time / len(results)] * len(results)


def run_longform_batch(
    executor: ThreadPoolExecutor,
    client: OpenAI,
    prepared: list[PreparedSample],
    args: argparse.Namespace,
) -> tuple[list[SampleResult], list[float]]:
    chunks = [sample.longform_chunks for sample in prepared]
    if any(sample_chunks is None for sample_chunks in chunks):
        raise RuntimeError("long-form batch contains a short-form sample")

    raw_outputs: list[list[str]] = [
        [""] * len(sample_chunks) for sample_chunks in chunks
    ]
    batch_wall_time = 0.0
    tasks = (
        (sample_index, chunk_index, chunk)
        for sample_index, sample_chunks in enumerate(chunks)
        for chunk_index, chunk in enumerate(sample_chunks)
    )
    for task_batch in batched(tasks, args.batch_size):
        requests: list[tuple[int, int, dict]] = []
        for sample_index, chunk_index, chunk in task_batch:
            sample = prepared[sample_index]
            requests.append(
                (
                    sample_index,
                    chunk_index,
                    build_chat_completion_request(
                        wav_data_url(chunk, sample.sampling_rate, subtype="PCM_16"),
                        args.prompt_profile,
                        model=args.served_model_name,
                    ),
                )
            )

        start = time.perf_counter()
        futures = [
            (
                sample_index,
                chunk_index,
                executor.submit(transcribe_request, client, request),
            )
            for sample_index, chunk_index, request in requests
        ]
        for sample_index, chunk_index, future in futures:
            raw_outputs[sample_index][chunk_index] = future.result()
        batch_wall_time += time.perf_counter() - start

    results = [
        finalize_sample(sample, raw_parts)
        for sample, raw_parts in zip(prepared, raw_outputs)
    ]
    total_duration = sum(result.duration_s for result in results)
    sample_times = [
        batch_wall_time * result.duration_s / total_duration for result in results
    ]
    return results, sample_times


def run_batch(
    executor: ThreadPoolExecutor,
    client: OpenAI,
    rows: list[dict],
    args: argparse.Namespace,
    first_index: int,
) -> tuple[list[SampleResult], list[float]]:
    prepared = [
        prepare_sample(row, args, first_index + offset)
        for offset, row in enumerate(rows)
    ]
    if args.longform_chunk_seconds > 0:
        return run_longform_batch(executor, client, prepared, args)
    return run_shortform_batch(executor, client, prepared)


def warm_up(
    client: OpenAI, executor: ThreadPoolExecutor, args: argparse.Namespace
) -> None:
    if args.warmup_samples <= 0:
        return

    requests = []
    for index in range(args.warmup_samples):
        duration_s = WARMUP_DURATIONS_S[index % len(WARMUP_DURATIONS_S)]
        samples = np.zeros(duration_s * SAMPLE_RATE, dtype=np.float32)
        requests.append(
            build_chat_completion_request(
                wav_data_url(samples, SAMPLE_RATE),
                args.prompt_profile,
                model=args.served_model_name,
            )
        )

    print(f"Warming up with {len(requests)} synthetic request(s)...")
    for request_batch in batched(requests, args.batch_size):
        futures = [
            executor.submit(transcribe_request, client, request)
            for request in request_batch
        ]
        for future in futures:
            future.result()


def evaluate(
    client: OpenAI, executor: ThreadPoolExecutor, args: argparse.Namespace
) -> str:
    dataset = load_eval_dataset(args)
    row_batch_size = (
        args.longform_record_batch_size
        if args.longform_chunk_seconds > 0
        else args.batch_size
    )
    all_results: list[SampleResult] = []
    transcription_times: list[float] = []
    processed = 0

    progress = tqdm(desc="Transcribing", unit="sample")
    try:
        for rows in batched(dataset, row_batch_size):
            results, sample_times = run_batch(
                executor, client, rows, args, first_index=processed
            )
            all_results.extend(results)
            transcription_times.extend(sample_times)
            processed += len(results)
            progress.update(len(results))
    finally:
        progress.close()

    if not all_results:
        raise RuntimeError("No evaluation samples remained after dataset filtering")

    extra_fields: dict[str, list] = {
        "raw_model_text": [result.raw_model_text for result in all_results],
        "prompt_profile": [args.prompt_profile] * len(all_results),
        "model_revision": [args.model_revision] * len(all_results),
        "nemo_speech_revision": [os.environ["NEMO_SPEECH_REVISION"]] * len(all_results),
        "speculative_mode": [os.environ["SPECULATIVE"]] * len(all_results),
        "vllm_version": [os.environ["VLLM_VERSION"]] * len(all_results),
        "longform_chunk_seconds": [args.longform_chunk_seconds] * len(all_results),
        "longform_chunk_overlap_seconds": [args.longform_chunk_overlap_seconds]
        * len(all_results),
        "generation_loop_cleanup_applied": [True] * len(all_results),
    }
    if data_utils.is_chunked_dataset(args.dataset_path):
        extra_fields.update(
            parent_id=[result.parent_id for result in all_results],
            chunk_index=[result.chunk_index for result in all_results],
        )

    manifest_path = data_utils.write_manifest(
        [result.reference for result in all_results],
        [result.prediction for result in all_results],
        args.model_id,
        args.dataset_path,
        args.dataset or "",
        args.split,
        audio_length=[result.duration_s for result in all_results],
        transcription_time=transcription_times,
        audio_filepaths=[result.audio_filepath for result in all_results],
        extra_fields=extra_fields,
    )
    inference_time = sum(transcription_times)
    audio_duration = sum(result.duration_s for result in all_results)
    rtfx = audio_duration / inference_time if inference_time else float("inf")
    print(f"Results saved at path: {os.path.abspath(manifest_path)}")
    print(f"Samples: {len(all_results)}")
    print(f"Timed inference: {inference_time:.3f} s")
    print(f"RTFx: {rtfx:.2f}")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--model_revision", default=MODEL_REVISION)
    parser.add_argument("--served_model_name", default=MODEL_ID)
    parser.add_argument("--server_url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--dataset_revision", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--language", default="en")
    parser.add_argument("--prompt_profile", required=True, choices=sorted(PROMPTS))
    parser.add_argument(
        "--batch_size",
        "--max_concurrency",
        dest="batch_size",
        type=int,
        default=192,
    )
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--warmup_samples", type=int, default=4)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--longform_chunk_seconds", type=float, default=0)
    parser.add_argument("--longform_chunk_overlap_seconds", type=float, default=0)
    parser.add_argument("--longform_record_batch_size", type=int, default=2)
    parser.add_argument("--request_timeout_s", type=float, default=3600)
    parser.add_argument("--max_retries", type=int, default=2)
    args = parser.parse_args()

    if args.model_id != MODEL_ID:
        parser.error(f"--model_id must remain pinned to {MODEL_ID}")
    if args.model_revision != MODEL_REVISION:
        parser.error(f"--model_revision must remain pinned to {MODEL_REVISION}")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if args.longform_record_batch_size <= 0:
        parser.error("--longform_record_batch_size must be positive")
    if args.warmup_samples < 0:
        parser.error("--warmup_samples must be non-negative")
    if args.longform_chunk_seconds < 0:
        parser.error("--longform_chunk_seconds must be non-negative")
    if args.longform_chunk_overlap_seconds < 0:
        parser.error("--longform_chunk_overlap_seconds must be non-negative")
    if args.longform_chunk_seconds == 0 and args.longform_chunk_overlap_seconds != 0:
        parser.error("--longform_chunk_overlap_seconds requires long-form chunking")
    if (
        args.longform_chunk_seconds > 0
        and args.longform_chunk_overlap_seconds >= args.longform_chunk_seconds
    ):
        parser.error("--longform_chunk_overlap_seconds must be shorter than a chunk")
    return args


def main() -> None:
    args = parse_args()
    total_parameters = int(os.environ["MODEL_TOTAL_PARAMETERS"])
    print(f"Model size: {total_parameters / 1e9:.4f}B parameters")
    limits = httpx.Limits(
        max_connections=args.batch_size,
        max_keepalive_connections=args.batch_size,
    )
    with httpx.Client(limits=limits) as http_client:
        client = OpenAI(
            base_url=args.server_url,
            api_key="EMPTY",
            timeout=args.request_timeout_s,
            max_retries=args.max_retries,
            http_client=http_client,
        )
        with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
            warm_up(client, executor, args)
            evaluate(client, executor, args)


if __name__ == "__main__":
    main()
