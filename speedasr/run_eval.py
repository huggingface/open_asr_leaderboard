#!/usr/bin/env python3
"""Batched SpeedASR evaluation using the official Open ASR data/manifest code.

Run from the model package root. The Dockerfile installs the official scorer at
the pinned commit; for a local run set --open-asr-repo to that checkout. Raw
references/predictions are retained, including Earnings22 session metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


OFFICIAL_COMMIT = "4a113dd90c9ffd011d7d3d105da76980d511d029"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_id", "--model-id", default="SpeedAudio261119/SpeedASR-v1-65M")
    p.add_argument("--revision", help="Pinned model commit/tag; required for a Hub download.")
    p.add_argument("--model-dir", type=Path, help="Use an already downloaded package instead of Hub weights.")
    p.add_argument("--dataset_path", "--dataset-path", default="hf-audio/open-asr-leaderboard")
    p.add_argument("--dataset", default="", help="Configuration name, or empty for a dedicated dataset repo.")
    p.add_argument("--dataset-revision", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--device", default="0", help="CUDA index, cuda:N, or cpu.")
    p.add_argument("--batch_size", "--batch-size", type=int, default=2048)
    p.add_argument("--batch-audio-seconds", type=float, default=5000.0)
    p.add_argument("--torch-threads", type=int, default=16)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--warmup-seconds", type=float, default=10.0,
                   help="Fixed synthetic audio duration, independent of the test set.")
    p.add_argument("--max_eval_samples", "--max-eval-samples", type=int, default=-1)
    p.add_argument("--open-asr-repo", type=Path,
                   default=Path(os.getenv("OPEN_ASR_REPO", "/opt/open_asr_leaderboard")))
    p.add_argument("--skip-score", action="store_true", help="Write raw manifests without local scoring.")
    a = p.parse_args()
    a.streaming = False
    if (a.batch_size < 1 or not math.isfinite(a.batch_audio_seconds)
            or a.batch_audio_seconds <= 0 or a.torch_threads < 1):
        p.error("Batch size, batch audio seconds and torch threads must be positive.")
    if a.warmup_steps < 0 or not math.isfinite(a.warmup_seconds) or a.warmup_seconds <= 0:
        p.error("Warmup steps must be nonnegative and warmup duration positive.")
    if a.max_eval_samples == 0 or a.max_eval_samples < -1:
        p.error("max_eval_samples must be -1 or positive.")
    if a.model_dir is None and not a.revision:
        p.error("Set --revision to the pinned Hub model commit/tag, or use --model-dir.")
    return a


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepared_batches(dataset, batch_size, batch_audio_seconds, np):
    """Decode bounded staging windows, then batch by count and padded duration.

    Audio decoding/resampling and sorting occur before the timed inference call.
    No example is truncated or dropped. Long single examples form their own
    batch, even when they exceed the nominal batch audio budget.
    """
    def stage(samples):
        rows = []
        for sample in samples:
            audio = sample["audio"]
            waveform = np.asarray(audio["array"], dtype=np.float32)
            if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
                raise ValueError("Expected nonempty finite mono audio from official prepare_data.")
            if int(audio["sampling_rate"]) != 16000:
                raise ValueError("Official prepare_data must resample audio to 16000 Hz.")
            rows.append((sample, waveform))
        rows.sort(key=lambda r: len(r[1]), reverse=True)
        batch = []
        max_samples = 0
        for row in rows:
            proposed_max = max(max_samples, len(row[1]))
            if batch and (len(batch) >= batch_size or
                          proposed_max * (len(batch) + 1) / 16000 > batch_audio_seconds):
                yield batch
                batch, max_samples = [], 0
            batch.append(row)
            max_samples = max(max_samples, len(row[1]))
        if batch:
            yield batch

    window = []
    for sample in dataset:
        window.append(sample)
        if len(window) >= batch_size:
            yield from stage(window)
            window = []
    if window:
        yield from stage(window)


def main(a):
    sys.path.insert(0, str(PACKAGE_ROOT))
    sys.path.insert(0, str(a.open_asr_repo.resolve()))
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(a.open_asr_repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as e:
        raise RuntimeError("--open-asr-repo must point to the pinned official Git checkout.") from e
    if commit != OFFICIAL_COMMIT:
        raise RuntimeError(f"Expected official scorer {OFFICIAL_COMMIT}, found {commit}.")

    import numpy as np
    import torch
    from huggingface_hub import HfApi
    from normalizer import data_utils
    from normalizer.eval_utils import score_manifest
    from inference import SpeedASR

    is_chunked = data_utils.is_chunked_dataset(a.dataset_path)
    if is_chunked and a.max_eval_samples > 0:
        raise ValueError("Do not truncate Earnings22 chunks: complete parent sessions are required.")
    if is_chunked and a.dataset_revision:
        raise ValueError("This pinned official chunk loader does not support --dataset-revision.")
    if a.device in {"cpu", "-1"}:
        device = "cpu"
    else:
        device = a.device if a.device.startswith("cuda") else f"cuda:{int(a.device)}"
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
    torch.set_num_threads(a.torch_threads)

    api = HfApi()
    model_revision = a.revision
    if a.model_dir:
        model_dir = a.model_dir.resolve()
        model = SpeedASR(checkpoint=model_dir / "pretrained.pt", tokens=model_dir / "tokens.txt",
                         config=model_dir / "model_config.json", device=device,
                         batch_size=a.batch_size, batch_audio_seconds=a.batch_audio_seconds,
                         torch_threads=a.torch_threads)
    else:
        model_revision = api.model_info(a.model_id, revision=a.revision).sha
        model = SpeedASR.from_pretrained(
            a.model_id, revision=model_revision, device=device, batch_size=a.batch_size,
            batch_audio_seconds=a.batch_audio_seconds, torch_threads=a.torch_threads,
        )
    count = sum(p.numel() for p in model.model.parameters())
    if model.beam_size != 4:
        raise ValueError("This submission uses the same fixed beam size 4 on every dataset.")
    print(f"Model size: {count} parameters ({count / 1e9:.9f}B)", flush=True)

    dataset_revision = api.dataset_info(a.dataset_path, revision=a.dataset_revision).sha
    parent_revision = None
    if is_chunked:
        parent_revision = api.dataset_info(
            data_utils.CHUNKED_DATASETS[a.dataset_path.lower()]["parent_dataset_path"]
        ).sha
    # Dedicated dataset repos use no config. Keep the manifest config as an empty
    # string while passing None to load_dataset through the official helper.
    load_args = argparse.Namespace(**vars(a))
    load_args.dataset = a.dataset or None
    if not is_chunked:
        load_args.dataset_revision = dataset_revision
    dataset = data_utils.prepare_data(data_utils.load_data(load_args))
    if a.max_eval_samples > 0:
        dataset = dataset.select(range(min(a.max_eval_samples, len(dataset))))

    def sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)

    dummy = np.zeros(round(16000 * a.warmup_seconds), dtype=np.float32)
    for _ in range(a.warmup_steps):
        model.transcribe_batch([dummy], sample_rate=16000)
    sync()
    cuda_batch_retries = getattr(model, "cuda_batch_retries", None)
    if cuda_batch_retries is not None:
        cuda_batch_retries.clear()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
        model.max_gpu_allocated_bytes = 0

    refs, preds, durations, times, paths = [], [], [], [], []
    extra = {key: [] for key in data_utils.CHUNK_METADATA_KEYS} if is_chunked else {}
    batches = 0
    for batch in prepared_batches(dataset, a.batch_size, a.batch_audio_seconds, np):
        waveforms = [waveform for _, waveform in batch]
        sync()
        started = time.perf_counter()
        output = model.transcribe_batch(waveforms, sample_rate=16000)
        sync()
        runtime = time.perf_counter() - started
        if len(output) != len(batch) or any(not isinstance(text, str) for text in output):
            raise RuntimeError("Inference returned an invalid number/type of predictions.")
        for (sample, waveform), prediction in zip(batch, output):
            refs.append(sample["original_text"])
            preds.append(prediction)
            durations.append(len(waveform) / 16000)
            times.append(runtime / len(batch))
            paths.append(data_utils.extract_audio_filepath_from_sample(sample))
            for key in extra:
                extra[key].append(sample[key])
        batches += 1
        print(f"Finished batch {batches}: {len(refs)} examples", flush=True)
    if not refs:
        raise RuntimeError("No evaluation examples remain after official reference filtering.")

    manifest = Path(data_utils.write_manifest(
        refs, preds, a.model_id, a.dataset_path, a.dataset or "", a.split,
        audio_length=durations, transcription_time=times,
        audio_filepaths=paths, extra_fields=extra or None,
    ))
    metadata = {
        "model_id": a.model_id, "model_revision": model_revision,
        "model_parameters": count, "dataset_path": a.dataset_path,
        "dataset": a.dataset, "dataset_revision": dataset_revision,
        "parent_dataset_revision_observed": parent_revision,
        "chunk_revision_note": "Official chunk loader resolves main; observed revisions recorded, not a snapshot lock." if is_chunked else None,
        "split": a.split, "official_commit": commit, "examples": len(refs),
        "batches": batches, "precision": "fp32", "causal": False,
        "decoding_method": "modified_beam_search", "beam_size": 4,
        "context_size": 3, "batch_size": a.batch_size,
        "long_chunk_seconds": model.long_chunk_seconds,
        "long_chunk_batch_size": model.long_chunk_batch_size,
        "min_input_seconds": model.min_input_seconds,
        "batch_audio_seconds": a.batch_audio_seconds,
        "warmup_steps": a.warmup_steps, "warmup_seconds": a.warmup_seconds,
        "warmup_input": "synthetic zeros, independent of evaluation data",
        "audio_seconds": sum(durations), "inference_seconds": sum(times),
        "rtfx": sum(durations) / sum(times),
        "timing_scope": "Synchronized wall time of complete transcribe_batch calls; excludes audio loading/resampling, staging/sorting, model loading, warmup and scoring.",
        "device": device,
        "gpu_name": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
        "peak_gpu_allocated_bytes": model.max_gpu_allocated_bytes if device.startswith("cuda") else None,
        "cuda_batch_retries": cuda_batch_retries,
        "python": platform.python_version(), "torch": torch.__version__,
        "package_versions": {n: importlib.metadata.version(n) for n in ("k2", "kaldifeat", "numpy", "datasets", "huggingface_hub")},
        "manifest_sha256": sha256(manifest),
        "run_eval_sha256": sha256(Path(__file__)),
        "inference_code_sha256": sha256(PACKAGE_ROOT / "inference.py"),
        "vendor_code_sha256": {p.name: sha256(p) for p in sorted((PACKAGE_ROOT / "speedasr_core").glob("*.py"))},
        "model_configuration": model.config,
        "partial_run": a.max_eval_samples > 0,
    }
    if not a.skip_score:
        rows = data_utils.read_manifest(manifest)
        if is_chunked:
            rows = data_utils.merge_chunked_manifest(rows)
        score = score_manifest(rows, language="en", multilingual=False, num_workers=1)
        metadata["official_score"] = score
        print("Official score:", json.dumps(score, ensure_ascii=False), flush=True)
    manifest.with_suffix(".meta.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Manifest: {manifest.resolve()}\nRTFx: {metadata['rtfx']:.4f}", flush=True)


if __name__ == "__main__":
    main(parse_args())
