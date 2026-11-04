"""Bounded async requests referencing local encoded audio files."""

import asyncio
import errno
import json
import math
import os
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path

import aiohttp

AUDIO_ROOT = Path("/tmp/rumble-audio")


@dataclass
class Sample:
    index: int
    audio_path: Path
    reference: str
    duration: float
    filepath: str | None = None
    metadata: dict | None = None


def request_body(sample, model, language, max_tokens):
    return {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [{
                "type": "audio_url",
                "audio_url": {"url": sample.audio_path.as_uri()},
            }],
        }],
        "chat_template_kwargs": {
            "digits": True, "filler": False, "verbatim": True, "language": language,
        },
        "temperature": 0,
        "seed": 0,
        "max_tokens": max_tokens,
    }


def parse_response(data):
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Expected exactly one completion")
    choice = choices[0]
    raw = choice["message"]["content"]
    finish = choice["finish_reason"]
    if not isinstance(raw, str) or finish not in {"stop", "length"}:
        raise ValueError("Invalid completion")
    text = raw.replace("<|im_end|>", "").strip()
    if "<asr_text>" in text:
        header, transcript = text.split("<asr_text>", 1)
        # Preserve malformed output so WER exposes it instead of hiding an error.
        if header.startswith("language "):
            text = transcript.strip()
    return {
        "prediction": text,
        "raw_completion": raw,
        "finish_reason": finish,
        "completion_tokens": data.get("usage", {}).get("completion_tokens"),
    }


async def run_requests(
    samples, *, base_urls, model, concurrency, language, max_tokens, timeout=360,
):
    """Time all requests through the final response; exclude setup and persistence."""
    if not base_urls or concurrency < 1:
        raise ValueError("Expected server endpoints and positive concurrency")
    iterator = iter(samples)
    rows = []
    workers = min(concurrency, len(samples))
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(
        connector=connector, timeout=aiohttp.ClientTimeout(total=timeout), trust_env=False,
    ) as session:
        pending_audio = [0.0] * len(base_urls)
        pending_requests = [0] * len(base_urls)
        next_engine = 0

        async def worker():
            nonlocal next_engine
            while True:
                sample = next(iterator, None)
                if sample is None:
                    return
                # Selection and reservation happen before yielding to another worker.
                engine = min(
                    range(len(base_urls)),
                    key=lambda i: (
                        pending_audio[i], pending_requests[i],
                        (i - next_engine) % len(base_urls),
                    ),
                )
                next_engine = (engine + 1) % len(base_urls)
                pending_audio[engine] += sample.duration
                pending_requests[engine] += 1
                try:
                    async with session.post(
                        base_urls[engine].rstrip("/") + "/chat/completions",
                        json=request_body(sample, model, language, max_tokens),
                    ) as response:
                        if response.status != 200:
                            raise RuntimeError(f"vLLM HTTP status {response.status}")
                        prediction = parse_response(await response.json())
                finally:
                    pending_requests[engine] -= 1
                    pending_audio[engine] = (
                        pending_audio[engine] - sample.duration
                        if pending_requests[engine] else 0.0
                    )
                rows.append({
                    "index": sample.index,
                    "reference": sample.reference,
                    "filepath": sample.filepath,
                    "duration": sample.duration,
                    "metadata": sample.metadata or {},
                    **prediction,
                })

        started = time.perf_counter()
        async with asyncio.TaskGroup() as group:
            for _ in range(workers):
                group.create_task(worker())
        elapsed = time.perf_counter() - started
    rows.sort(key=lambda row: row["index"])
    return rows, elapsed


def prepare_inputs(args, audio_dir):
    """Stage original encoded files; leave decoding and resampling to vLLM."""
    from normalizer import data_utils

    dataset = data_utils.load_data(args)
    dataset = dataset.cast_column("audio", replace(dataset.features["audio"], decode=False))
    chunked = data_utils.is_chunked_dataset(args.dataset_path)
    multilingual = getattr(args, "multilingual", False)
    audio_dir = Path(audio_dir).resolve()
    audio_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for row in dataset:
        if row.get("lattice") is not None:
            reference = json.dumps(row["lattice"], ensure_ascii=False)
        else:
            reference = data_utils.get_text(row)
        text = reference if multilingual else data_utils.normalizer(reference)
        if not data_utils.is_target_text_in_range(text):
            continue
        duration = row.get("audio_length_s", row.get("duration"))
        if duration is None or not math.isfinite(float(duration)) or float(duration) <= 0:
            raise ValueError("Expected positive audio duration metadata")
        encoded = row["audio"]
        audio_path = audio_dir / str(len(samples))
        if encoded.get("bytes") is not None:
            audio_path.write_bytes(encoded["bytes"])
        else:
            source = Path(encoded["path"]).resolve(strict=True)
            try:
                os.link(source, audio_path)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                shutil.copyfile(source, audio_path)
        if not audio_path.stat().st_size:
            raise ValueError("Empty audio file")
        samples.append(Sample(
            len(samples), audio_path, reference, float(duration),
            data_utils.extract_audio_filepath_from_sample(row),
            {key: row[key] for key in data_utils.CHUNK_METADATA_KEYS} if chunked else {},
        ))
    return samples


def results_from_rows(rows, elapsed, dataset_path):
    """Distribute dataset wall time across rows for the upstream scorer to sum."""
    from normalizer import data_utils

    results = {
        "audio_length_s": [row["duration"] for row in rows],
        "transcription_time_s": [elapsed / len(rows)] * len(rows),
        "predictions": [row["prediction"] for row in rows],
        "references": [row["reference"] for row in rows],
        "audio_filepath": [row["filepath"] for row in rows],
    }
    for key in ("raw_completion", "finish_reason", "completion_tokens"):
        results[key] = [row[key] for row in rows]
    if data_utils.is_chunked_dataset(dataset_path):
        for key in data_utils.CHUNK_METADATA_KEYS:
            results[key] = [row["metadata"][key] for row in rows]
    return results
