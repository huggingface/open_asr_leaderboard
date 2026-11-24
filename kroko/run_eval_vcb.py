#!/usr/bin/env python3
"""Evaluate one English dataset with Fast GPU ASR's TensorRT runtime.

ONNX release bundles are resolved from the Hub and compiled on the selected GPU. Audio
is resampled and sorted by decreasing length before fixed-profile synthetic warm-ups per worker
and one measured pass. One ASR instance is used by default. Timings cover
the wall clock of the queue of synchronized full ASR calls,
including text and timestamps, but exclude loading, export, audio preparation,
and scoring. RTFx divides real audio duration by total measured inference time.

Raw transcripts and per-utterance timings are written under ``./results`` with
an environment metadata sidecar. The upstream English scorer normalizes both
references and hypotheses, merges Earnings22 chunks, and scores compounds.
Use a separate working directory for each configuration: manifest filenames
identify the model and dataset, not the decoder, precision, or batch size.
"""

import json
import os
import subprocess
from argparse import ArgumentParser, Namespace
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from hashlib import sha256
from io import BytesIO
from math import gcd
from pathlib import Path
from platform import platform
from sys import executable
from tempfile import TemporaryDirectory
from time import perf_counter

import cupy as cp
import numpy as np
import soundfile as sf
from datasets import (
    Audio,
    Dataset,
    DownloadConfig,
    Features,
    Value,
    concatenate_datasets,
)
from datasets.utils.file_utils import xopen
from fast_gpu_asr import ASR
from filelock import FileLock
from kaldialign import batch_error_rate
from librosa import resample
from scipy.signal import resample_poly
from tqdm.auto import tqdm

from normalizer import data_utils, voice_code_bench
from preflight_model import download_release
from revision_pins import install as install_dataset_pins

install_dataset_pins(data_utils)

SAMPLE_RATE = 16000


def parse_args() -> Namespace:
    """Parse and validate arguments without downloading data or selecting a GPU.

    Model revision and decoder are explicit inputs; model selection belongs
    to the shell launchers. The release supplies inference-only ONNX assets.
    Streaming is disabled so the full dataset can be sorted before inference.

    Returns
    -------
    Namespace
        Validated model, dataset, export, and evaluation settings, with
        ``streaming=False`` and ``beam=1`` for greedy decoder modes.
    """

    parser = ArgumentParser(
        description=(
            "Evaluate a Zipformer ONNX release with Fast GPU ASR and report "
            "English word error rate (WER) and end-to-end throughput (RTFx)."
        )
    )

    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Hugging Face repository containing the ONNX release and config.json.",
    )
    parser.add_argument("--model-revision", required=True, help="Immutable Hub model commit SHA")
    parser.add_argument("--dataset-revision", required=True, help="Pinned dataset commit SHA")
    parser.add_argument(
        "--report-name",
        type=str,
        default=None,
        help=(
            "Model name recorded in result manifests and scoring, such as "
            "'nvidia/parakeet-tdt-0.6b-v3 (fast-gpu-asr)'. Defaults to --model-id."
        ),
    )
    parser.add_argument(
        "--model-family",
        type=str,
        choices=("zipformer",),
        required=True,
        help="Model family for the ONNX bundle adapter.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="hf-audio/open-asr-leaderboard",
        help="Hugging Face dataset repository or local dataset path.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset configuration name, such as librispeech; use '' if none is needed.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate, such as test, test.clean, or test.other.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help=(
            "CUDA device index for export and inference; "
            "use without remapped CUDA_VISIBLE_DEVICES."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Rows per call, at most the ONNX export's fixed capacity; partial batches are included.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent ASR instances on the selected GPU.",
    )
    parser.add_argument(
        "--beam",
        type=int,
        required=True,
        help="Transducer beam width; greedy modes override this with one.",
    )
    parser.add_argument(
        "--blank-penalty",
        type=float,
        default=0.0,
        help="Penalty subtracted from blank log probabilities; positive discourages blanks.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=("fp16", "bf16", "fp32"),
        required=True,
        help="Precision of the published encoder and decoder ONNX graphs.",
    )
    parser.add_argument(
        "--decoder-type",
        type=str,
        choices=("transducer_modified_beam_search",),
        required=True,
        help="RNN-T decoder matching the ONNX release manifest.",
    )
    parser.add_argument(
        "--min-audio-seconds",
        type=float,
        required=True,
        help="Minimum TensorRT profile duration; shorter clips are padded internally.",
    )
    parser.add_argument(
        "--opt-audio-seconds",
        type=float,
        required=True,
        help="Typical duration TensorRT optimizes for.",
    )
    parser.add_argument(
        "--max-audio-seconds",
        type=float,
        required=True,
        help="Maximum supported clip duration; longer clips fail without truncation.",
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        choices=range(6),
        default=5,
        help="TensorRT builder optimization level, from 0 to 5.",
    )
    parser.add_argument(
        "--engine-cache",
        type=Path,
        default=Path("engines"),
        help="Directory for reusable, target-GPU TensorRT bundles.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=-1,
        help="Smoke-test limit after reference filtering; -1 evaluates the full split.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Calls per fixed synthetic profile and worker, excluded from reported timings.",
    )
    parser.set_defaults(streaming=False)

    args = parser.parse_args()

    if not args.report_name:
        args.report_name = args.model_id
    if args.batch_size < 1 or args.beam < 1 or args.workers < 1:
        parser.error("batch_size, beam, and workers must be positive.")
    if args.warmup_steps < 0 or args.max_eval_samples == 0 or args.max_eval_samples < -1:
        parser.error(
            "warmup_steps must be nonnegative; max_eval_samples must be -1 or positive."
        )
    if not 0 < args.min_audio_seconds <= args.opt_audio_seconds <= args.max_audio_seconds:
        parser.error(
            "Expected finite 0 < min_audio_seconds <= opt_audio_seconds <= max_audio_seconds."
        )
    if args.max_eval_samples > 0 and data_utils.is_chunked_dataset(args.dataset_path):
        parser.error(
            "Do not truncate a chunked dataset: WER requires complete parent sessions."
        )

    if args.decoder_type != "transducer_modified_beam_search":
        args.beam = 1

    return args


def inspect_audio(
    audio: dict[str, str | bytes | None],
    utterance_id: str,
    max_samples: int,
    model_family: str,
) -> dict[str, int | dict[str, int | np.typing.NDArray[np.float32]]]:
    """Decode, downmix, validate, and resample a clip before batching.

    Parameters
    ----------
    audio : dict[str, str | bytes | None]
        Encoded ``bytes`` or a local, remote, or archive ``path``. Bytes take
        precedence over the path. SoundFile decodes without TorchCodec;
        multiple channels are averaged to mono before float32 conversion.
    utterance_id : str
        Identifier resolved during dataset preparation, used in error messages.
    max_samples : int
        Maximum clip length at ``SAMPLE_RATE``, checked after resampling.
    model_family : str
        ``zipformer`` resamples mono float32 audio with SciPy's
        ``resample_poly`` using a reduced integer ratio. ``parakeet`` uses
        librosa's ``soxr_hq`` resampling, matching the earlier NumPy-based
        upstream loader. Clips already at ``SAMPLE_RATE`` bypass resampling.

    Returns
    -------
    dict[str, int | dict[str, int | np.typing.NDArray[np.float32]]]
        Mono float32 ``audio`` at ``SAMPLE_RATE`` and ``num_samples``.
        Other dataset columns, including ``utterance_id``, are preserved by
        ``Dataset.map``. All rows use the same float32 storage.

    Raises
    ------
    ValueError
        Audio has no path or bytes, is empty, has an invalid sample rate,
        exceeds [-1, 1] before resampling, or exceeds the duration profile afterward.
    """

    if audio["bytes"] is None and audio["path"] is None:
        raise ValueError(f"{utterance_id}: audio must contain a path or bytes.")

    with (
        BytesIO(audio["bytes"])
        if audio["bytes"] is not None
        else xopen(audio["path"], "rb", download_config=DownloadConfig(token=True))
    ) as source:
        waveform, sampling_rate = sf.read(source)

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if waveform.ndim != 1 or waveform.size == 0 or sampling_rate <= 0:
        raise ValueError(
            f"{utterance_id}: expected nonempty mono audio with a positive sample rate."
        )
    if np.max(np.abs(waveform)) > 1.0:
        raise ValueError(f"{utterance_id}: audio must be normalized to [-1, 1].")

    waveform = waveform.astype(np.float32)
    if sampling_rate != SAMPLE_RATE:
        if model_family == "parakeet":
            waveform = resample(
                waveform, orig_sr=sampling_rate, target_sr=SAMPLE_RATE, res_type="soxr_hq"
            )
        else:
            divisor = gcd(sampling_rate, SAMPLE_RATE)
            waveform = resample_poly(
                waveform, SAMPLE_RATE // divisor, sampling_rate // divisor
            ).astype(np.float32)

    if waveform.size > max_samples:
        raise ValueError(
            f"{utterance_id}: {waveform.size / SAMPLE_RATE:.3f}s exceeds the export profile; "
            "increase --max-audio-seconds for the whole run."
        )

    return {
        "audio": {"array": waveform, "sampling_rate": SAMPLE_RATE},
        "num_samples": waveform.size,
    }


def export_bundle(
    args: Namespace, hardware: dict[str, str | int | list[int]]
) -> tuple[Path, dict[str, str | dict[str, str | int | float | list[int]]]]:
    """Build a target-GPU TensorRT bundle from the pinned ONNX release.

    Download and validate inference assets only. Cache by release revision,
    runtime versions, settings and hardware. The child builder releases its GPU
    resources before ASR loads the engines. No training checkpoint is required.
    """
    from importlib.metadata import version

    _, release, _ = download_release(args.model_id, args.model_revision)
    if release["language"] != "en":
        raise ValueError("This suite uses English datasets and normalization; select an English release.")
    settings = {
        "batch_size": args.batch_size,
        "decoder_type": args.decoder_type,
        "beam": args.beam,
        "blank_penalty": args.blank_penalty,
        "encoder_precision": args.precision,
        "decoder_precision": args.precision,
        "min_audio_seconds": args.min_audio_seconds,
        "opt_audio_seconds": args.opt_audio_seconds,
        "max_audio_seconds": args.max_audio_seconds,
        "optimization_level": args.optimization_level,
    }
    metadata = {
        "model_id": args.model_id,
        "model_type": args.model_family,
        "release_format": release["format"],
        "revision": args.model_revision,
        "export_settings": settings,
        "onnx_export_settings": release["export"],
        "runtime_versions": {name: version(name) for name in
            ("fast-gpu-asr", "tensorrt-cu13", "tensorrt-cu13-bindings")},
        "hardware": hardware,
    }
    key = sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:20]
    args.engine_cache.mkdir(parents=True, exist_ok=True)
    bundle = args.engine_cache.resolve() / key
    with FileLock(f"{bundle}.lock"):
        if bundle.exists():
            with open(bundle / "export.json", encoding="utf-8") as source:
                if json.load(source) != metadata:
                    raise ValueError(f"Engine-cache metadata mismatch: {bundle}")
            return bundle, metadata
        with TemporaryDirectory(dir=args.engine_cache, prefix="build-") as tmp:
            output = Path(tmp).resolve() / "bundle"
            command = [executable, str(Path(__file__).with_name("build_onnx_bundle.py")),
                "--model-id", args.model_id, "--model-revision", args.model_revision,
                "--output-dir", str(output), "--batch-size", str(args.batch_size),
                "--beam", str(args.beam), "--blank-penalty", str(args.blank_penalty),
                "--precision", args.precision,
                "--min-audio-seconds", str(args.min_audio_seconds),
                "--opt-audio-seconds", str(args.opt_audio_seconds),
                "--max-audio-seconds", str(args.max_audio_seconds),
                "--optimization-level", str(args.optimization_level)]
            subprocess.run(command, check=True, env={**os.environ,
                "CUDA_VISIBLE_DEVICES": str(args.device)})
            with open(output / "export.json", "w", encoding="utf-8") as destination:
                json.dump(metadata, destination, indent=2, sort_keys=True)
            output.rename(bundle)
    return bundle, metadata


VCB_MAX_AUDIO_SECONDS = 130.0  # VoiceCodeBench recordings last 35-123 s
VCB_CHUNK_SECONDS = 30.0


class ChunkedASR:
    """Wrap an ASR instance so clips beyond its export profile are split into equal parts.

    Each clip longer than ``VCB_CHUNK_SECONDS`` is cut into the fewest equal parts of at most that
    length, the parts are transcribed in engine-sized batches, and their texts are joined per clip.
    """

    def __init__(self, backend: ASR, batch_size: int) -> None:
        self.backend = backend
        self.batch_size = batch_size
        self.stream = backend.stream

    def __call__(self, audios):
        limit = round(VCB_CHUNK_SECONDS * SAMPLE_RATE)
        pieces, owners = [], []
        for i, audio in enumerate(audios):
            parts = max(1, -(-len(audio) // limit))
            for part in np.array_split(audio, parts):
                pieces.append(np.ascontiguousarray(part, dtype=np.float32))
                owners.append(i)
        texts = []
        for start in range(0, len(pieces), self.batch_size):
            predictions, _ = self.backend(pieces[start:start + self.batch_size])
            self.backend.stream.synchronize()
            texts.extend(predictions)
        joined = [[] for _ in audios]
        for owner, text in zip(owners, texts):
            if text:
                joined[owner].append(text)
        return [" ".join(parts) for parts in joined], [[] for _ in audios]


def benchmark(
    dataset: Dataset, backends: list[ASR], batch_size: int, warmup_steps: int,
    warmup_seconds: tuple[float, ...] = (0.1, 10.0, 40.0),
) -> dict[str, list[str | int | float | None]]:
    """Transcribe a dataset with independent ASR instances and measure throughput.

    Materialize contiguous float32 audio in host memory before timing. Keep one
    call in flight per instance and synchronize its stream before assigning the
    next batch. Every batch, including the partial tail, is evaluated once after
    warmup; completion order does not change the returned row order.

    Queue wall time starts before the first submission and includes scheduling,
    full ASR calls, stream synchronization, progress updates, and bookkeeping.
    Audio preparation, warmups, and final result aggregation are excluded. Text
    and word timestamps are both computed, but timestamps are not returned.
    Garbage collection is left unchanged.

    Parameters
    ----------
    dataset : Dataset
        Nonempty prepared dataset with mono 16 kHz waveforms in ``audio.array``
        and reference transcripts in ``original_text``. Rows are batched in
        their existing order; this function does not sort or resample them.
    backends : list[ASR]
        Nonempty list of distinct instances with the same model and decoding
        settings on the selected GPU. Instances must not be used concurrently
        elsewhere. At most one instance per batch is activated.
    batch_size : int
        Positive number of rows per call, within each engine's batch capacity.
        The final batch may contain fewer rows.
    warmup_steps : int
        Nonnegative number of untimed calls per synthetic min/opt/max profile
        for each active instance. Warmup predictions are discarded.

    Returns
    -------
    dict[str, list[str | int | float | None]]
        Lists in dataset order: ``references``, ``predictions``,
        ``audio_length_s``, ``transcription_time_s``, ``audio_filepaths``, and
        available chunk metadata. Audio durations exclude padding. Each row's
        transcription time is an equal share of total queue wall time, not a
        per-clip latency. Pooled RTFx is the sum of audio durations divided by
        the sum of these time shares.

    Raises
    ------
    ValueError
        The dataset or backend list is empty.
    RuntimeError
        A call returns a different number of transcripts or timestamp lists
        than input clips. Inference failures also propagate; submitted worker
        calls return or raise before the executor exits.
    """

    batches = list(dataset.iter(batch_size=batch_size))
    if not batches or not backends:
        raise ValueError("Evaluation requires audio batches and at least one ASR instance.")

    audios = [
        [np.ascontiguousarray(clip["array"], dtype=np.float32) for clip in batch["audio"]]
        for batch in batches
    ]

    backends = backends[: len(batches)]
    for backend in backends:
        backend.stream.synchronize()
        # Fixed profile shapes and synthetic audio, independent of evaluation data.
        for seconds in sorted(set(warmup_seconds)):
            dummy = [np.zeros(round(seconds * SAMPLE_RATE), dtype=np.float32)] * batch_size
            for _ in range(warmup_steps):
                predictions, timestamps = backend(dummy)
                backend.stream.synchronize()
                if len(predictions) != len(dummy) or len(timestamps) != len(dummy):
                    raise RuntimeError("ASR must return one transcript and timestamp list per input.")

    outputs: dict[int, list[str]] = {}
    with (
        ThreadPoolExecutor(max_workers=len(backends)) as pool,
        tqdm(total=len(batches), desc="Batches", mininterval=1.0) as progress,
    ):
        start = perf_counter()

        pending = {
            pool.submit(backend, audios[index]): (index, backend)
            for index, backend in enumerate(backends)
        }
        next_index = len(pending)
        while pending:
            ready, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in ready:
                index, backend = pending.pop(future)
                predictions, timestamps = future.result()
                backend.stream.synchronize()

                if len(predictions) != len(audios[index]) or len(timestamps) != len(audios[index]):
                    raise RuntimeError(
                        "ASR must return one transcript and timestamp list per input."
                    )

                outputs[index] = predictions
                if next_index < len(batches):
                    pending[pool.submit(backend, audios[next_index])] = (next_index, backend)
                    next_index += 1
                progress.update(1)

        elapsed = perf_counter() - start

    results: dict[str, list[str | int | float | None]] = {}
    for index, batch in enumerate(batches):
        rows = {
            "references": batch["original_text"],
            "predictions": outputs[index],
            "audio_length_s": [len(audio) / SAMPLE_RATE for audio in audios[index]],
            "transcription_time_s": [elapsed / len(dataset)] * len(batch["audio"]),
            "audio_filepaths": data_utils.extract_audio_filepaths_from_batch(batch),
            **{key: batch[key] for key in (*data_utils.CHUNK_METADATA_KEYS, *voice_code_bench.MANIFEST_KEYS) if key in batch},
        }
        for key, values in rows.items():
            results.setdefault(key, []).extend(values)

    return results


def main() -> None:
    """Prepare audio, run one measured pass, and save scored raw transcripts.

    Warm-ups are discarded; final partial batches contribute to pooled RTFx.
    WER uses upstream normalization and compound merging after reconstructing
    Earnings22 parent transcripts. Results are written under ``./results``.

    The metadata sidecar records arguments, resolved ONNX release revision, GPU
    properties, and dataset totals. ``driver_cuda_version`` is the driver's
    supported CUDA API version, not the NVIDIA driver release number.
    """

    args = parse_args()
    is_vcb = voice_code_bench.is_voice_code_bench(args.dataset_path)
    with cp.cuda.Device(args.device):
        dataset = (voice_code_bench.load_data(args) if is_vcb else data_utils.load_data(args)).cast_column("audio", Audio(decode=False))
        # Normalize metadata without copying or decoding the encoded audio column.
        metadata = dataset.remove_columns("audio").map(data_utils.normalize)
        indices = [
            index
            for index, text in enumerate(metadata["norm_text"])
            if data_utils.is_target_text_in_range(text)
        ]
        if args.max_eval_samples > 0:
            indices = indices[: args.max_eval_samples]

        if not indices:
            raise ValueError("No evaluable samples remain after upstream filtering.")

        ids = list(metadata["id"]) if "id" in metadata.column_names else [None] * len(metadata)
        file_names = (
            list(metadata["file_name"])
            if "file_name" in metadata.column_names
            else [None] * len(metadata)
        )
        utterance_ids = [""] * len(metadata)
        for position, index in enumerate(indices):
            utterance_ids[index] = str(
                ids[index] or file_names[index] or f"{position:012d}"
            )
        if "utterance_id" in metadata.column_names:
            metadata = metadata.remove_columns("utterance_id")
        metadata = metadata.add_column("utterance_id", utterance_ids)
        # Join before selecting rows so adding IDs cannot flatten/copy the audio.
        dataset = concatenate_datasets(
            [dataset.select_columns("audio"), metadata], axis=1
        ).select(indices)

        dataset = dataset.map(
            inspect_audio,
            input_columns=["audio", "utterance_id"],
            # Store samples directly; Audio would re-encode them as PCM16 WAV.
            features=Features(
                {
                    **dataset.features,
                    "audio": {"array": [Value("float32")], "sampling_rate": Value("int32")},
                    "num_samples": Value("int64"),
                }
            ),
            fn_kwargs={
                "max_samples": round((VCB_MAX_AUDIO_SECONDS if is_vcb else args.max_audio_seconds) * SAMPLE_RATE),
                "model_family": args.model_family,
            },
        )

        lengths, ids = list(dataset["num_samples"]), list(dataset["utterance_id"])
        order = sorted(range(len(dataset)), key=lambda i: (-lengths[i], ids[i]))
        # Python audio lists make garbage collection scan every sample during ASR.
        dataset = dataset.select(order).with_format(
            "numpy", columns=["audio"], output_all_columns=True
        )

        properties = cp.cuda.runtime.getDeviceProperties(args.device)
        hardware: dict[str, str | int | list[int]] = {
            "gpu": properties["name"].decode(),
            "compute_capability": [properties["major"], properties["minor"]],
            "vram_bytes": properties["totalGlobalMem"],
            "driver_cuda_version": cp.cuda.runtime.driverGetVersion(),
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        }
        bundle, metadata = export_bundle(args, hardware)

        workers = min(args.workers, (len(dataset) + args.batch_size - 1) // args.batch_size)
        backends = [ASR(bundle, device_id=args.device) for _ in range(workers)]
        if is_vcb:
            backends = [ChunkedASR(backend, args.batch_size) for backend in backends]

        results = benchmark(
            dataset, backends, args.batch_size, args.warmup_steps,
            (args.min_audio_seconds, args.opt_audio_seconds, args.max_audio_seconds),
        )
        if len(results["predictions"]) != len(dataset):
            raise RuntimeError("Incomplete evaluation: not every sample was transcribed.")

        extra_fields = None
        if is_vcb:
            extra_fields = {key: results[key] for key in voice_code_bench.MANIFEST_KEYS}
        elif data_utils.is_chunked_dataset(args.dataset_path):
            extra_fields = {key: results[key] for key in data_utils.CHUNK_METADATA_KEYS}

        manifest = Path(
            data_utils.write_manifest(
                results["references"],
                results["predictions"],
                args.report_name,
                args.dataset_path,
                args.dataset,
                args.split,
                audio_length=results["audio_length_s"],
                transcription_time=results["transcription_time_s"],
                audio_filepaths=results["audio_filepaths"],
                extra_fields=extra_fields,
            )
        )

        metadata = {
            **metadata,
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "dataset_fingerprint": dataset._fingerprint,
            "samples": len(dataset),
            "audio_seconds": sum(lengths) / SAMPLE_RATE,
            "platform": platform(),
            "active_workers": workers,
            "sort": "descending duration, ascending utterance ID",
            "warmup": "synthetic silence at fixed min/opt/max profiles, full batch, untimed",
            "timing": (
                "queue wall time, including scheduling and synchronized full ASR calls, "
                "including text and word timestamps"
            ),
        }
        with open(manifest.with_suffix(".metadata.json"), "w", encoding="utf-8") as destination:
            json.dump(metadata, destination, indent=2, sort_keys=True)

    if is_vcb:
        # CTEM and WER are computed from the manifest by the grading job.
        rtfx = sum(results["audio_length_s"]) / sum(results["transcription_time_s"])
        print(f"Results: {manifest}\nRTFx: {rtfx:.2f}")
        return

    sessions = data_utils.merge_chunked_manifest(data_utils.read_manifest(manifest))
    refs = [tuple(data_utils.normalizer(row["text"]).split()) for row in sessions]
    preds = [tuple(data_utils.normalizer(row["pred_text"]).split()) for row in sessions]

    wer = batch_error_rate(refs, preds, merge_compounds=True)["err_rate"] * 100
    rtfx = sum(results["audio_length_s"]) / sum(results["transcription_time_s"])
    print(f"Results: {manifest}\nWER: {wer:.2f}%  RTFx: {rtfx:.2f}")


if __name__ == "__main__":
    main()
