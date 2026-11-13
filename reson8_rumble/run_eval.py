"""Download selected datasets or evaluate one dataset against running vLLM engines."""

import argparse
import asyncio
import errno
import json
import math
import os
import shlex
import shutil
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

import aiohttp

BASE_PORT = 8000

DEFAULT_DATASET_PATH = "hf-audio/open-asr-leaderboard"

LANGUAGE_NAMES = {
    # Note: it supports more languages
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "nl": "Dutch",
    "pt": "Portuguese",
}


def parse_datasets(spec, default_path=DEFAULT_DATASET_PATH):
    selections = []
    seen = set()
    for line in spec.splitlines():
        if not line.strip():
            continue
        fields = shlex.split(line)
        if not 2 <= len(fields) <= 5:
            raise ValueError(
                f"Expected 'name split [dataset_path [config [language]]]': {line}"
            )
        name, split = fields[:2]
        path = fields[2] if len(fields) >= 3 else default_path
        config = fields[3] if len(fields) >= 4 else (name if len(fields) == 2 else None)
        config = None if config == "-" else config
        language = fields[4] if len(fields) == 5 else "en"
        if language not in LANGUAGE_NAMES:
            raise ValueError(f"Unsupported language: {language}")
        identity = (path, config, split)
        if identity in seen:
            raise ValueError(f"Duplicate dataset selection: {line}")
        seen.add(identity)
        selections.append((*identity, language))
    if not selections:
        raise ValueError("Select at least one dataset")
    return selections


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


DURATION_COLUMNS = ("audio_length_s", "duration", "duration_s")


def audio_duration(row, audio_path):
    """Duration in seconds from the row's metadata, else from the file header.

    The header read does not decode the audio, so staging stays cheap; librosa
    is the fallback for formats libsndfile cannot open.
    """
    duration = next((row[k] for k in DURATION_COLUMNS if row.get(k) is not None), None)
    if duration is None:
        try:
            import soundfile

            duration = soundfile.info(str(audio_path)).duration
        except Exception:
            import librosa

            duration = librosa.get_duration(path=str(audio_path))
    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Expected a positive audio duration, got {duration} for {audio_path}")
    return duration


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
        duration = audio_duration(row, audio_path)
        samples.append(Sample(
            len(samples), audio_path, reference, duration,
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


def write_results(args, all_results):
    from normalizer import data_utils

    is_chunked = data_utils.is_chunked_dataset(args.dataset_path)
    extra_fields = (
        {key: all_results[key] for key in data_utils.CHUNK_METADATA_KEYS}
        if is_chunked
        else {}
    )
    extra_fields.update(
        {
            key: all_results[key]
            for key in (
                "raw_completion",
                "finish_reason",
                "completion_tokens",
            )
        }
    )
    manifest_path = data_utils.write_manifest(
        all_results["references"],
        all_results["predictions"],
        args.model_id,
        args.dataset_path,
        args.dataset or "",
        args.split,
        audio_length=all_results["audio_length_s"],
        transcription_time=all_results["transcription_time_s"],
        audio_filepaths=all_results["audio_filepath"],
        extra_fields=extra_fields or None,
    )
    return manifest_path


def evaluate(args, samples):
    if not samples:
        raise ValueError("No evaluation samples")
    print(f"Evaluating {args.dataset_path} / {args.dataset} / {args.split}", flush=True)
    rows, elapsed = asyncio.run(run_requests(
        samples, base_urls=args.base_urls, model=args.model_id,
        concurrency=args.concurrency, language=LANGUAGE_NAMES[args.language],
        max_tokens=args.max_new_tokens,
    ))
    results = results_from_rows(rows, elapsed, args.dataset_path)
    manifest = Path(write_results(args, results)).resolve()
    print("Results saved at path:", manifest, flush=True)
    duration = sum(row["duration"] for row in rows)
    print(
        f"{args.dataset_path} / {args.dataset} / {args.split}: "
        f"RTFx: {duration / elapsed:.2f} "
        f"({duration:.3f}s audio / {elapsed:.3f}s wall)",
        flush=True,
    )


def main(args):
    selections = parse_datasets(args.datasets, args.default_dataset_path)
    if not args.download and len(selections) != 1:
        raise ValueError("Evaluate one dataset at a time; use run_rumble.sh for a full run")
    if args.num_engines < 1:
        raise ValueError("Engine count must be positive")

    from normalizer import data_utils

    for path, config, split, language in selections:
        selection = argparse.Namespace(**vars(args))
        selection.dataset_path, selection.dataset, selection.split = path, config, split
        selection.language = language
        selection.multilingual = language != "en"
        selection.streaming = False
        if args.download:
            dataset = data_utils.load_data(selection)
            print(f"Cached {path} / {config} / {split}: {len(dataset)} samples", flush=True)
            continue

        selection.base_urls = [
            f"http://127.0.0.1:{BASE_PORT + engine}/v1"
            for engine in range(args.num_engines)
        ]
        AUDIO_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="dataset-", dir=AUDIO_ROOT) as audio_dir:
            evaluate(selection, prepare_inputs(selection, audio_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", required=True,
        help="One 'name split [dataset_path [config [language]]]' entry; newline-separated for --download",
    )
    parser.add_argument("--download", action="store_true", help="Download to the shared cache and exit")
    parser.add_argument("--num_engines", type=int, default=4)
    args = parser.parse_args()
    args.model_id = "Reson8/Rumble-2-Flash"
    args.default_dataset_path = DEFAULT_DATASET_PATH
    args.concurrency = 2048
    args.max_new_tokens = 512
    main(args)
