#!/usr/bin/env python3
"""Evaluate one English dataset with Fast GPU ASR's TensorRT runtime.

Checkpoints are resolved from the Hub and exported on the selected GPU. Audio
is resampled and sorted by decreasing length before five warm-ups and one
measured pass by default. Timings cover the synchronized full ASR call,
including text and timestamps, but exclude loading, export, audio preparation,
and scoring. RTFx divides real audio duration by total measured inference time.

Raw transcripts and per-utterance timings are written under ``./results`` with
an environment metadata sidecar. The upstream English scorer normalizes both
references and hypotheses, merges Earnings22 chunks, and scores compounds.
Use a separate working directory for each configuration: manifest filenames
identify the checkpoint and dataset, not the decoder, precision, or batch size.
"""

import json
import subprocess
from argparse import ArgumentParser, Namespace
from audioop import ratecv
from collections.abc import Iterable
from hashlib import sha256
from io import BytesIO
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
from huggingface_hub import snapshot_download
from kaldialign import batch_error_rate
from librosa import resample
from tqdm.auto import tqdm

from normalizer import data_utils

SAMPLE_RATE = 16000
PCM16_SCALE = 32768


def parse_args() -> Namespace:
    """Parse and validate arguments without downloading data or selecting a GPU.

    Model family, checkpoint filename, and decoder are explicit inputs; model
    selection belongs to the shell launchers. Greedy modes force beam one.
    Streaming is disabled so the full dataset can be sorted before inference.

    Returns
    -------
    Namespace
        Validated model, dataset, export, and evaluation settings, with
        ``streaming=False`` and ``beam=1`` for greedy decoder modes.
    """

    parser = ArgumentParser(
        description=(
            "Evaluate Zipformer or Parakeet with Fast GPU ASR and report "
            "English word error rate (WER) and end-to-end throughput (RTFx)."
        )
    )

    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Hugging Face repository containing the model checkpoint.",
    )
    parser.add_argument(
        "--model-family",
        type=str,
        choices=("zipformer", "parakeet"),
        required=True,
        help="Model family selecting the TensorRT exporter.",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        required=True,
        help="Checkpoint filename within the Hub repository, such as model.pt or model.nemo.",
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
        help="Fixed engine batch capacity; the final partial batch is included.",
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
        help="Requested precision for both encoder and decoder exports.",
    )
    parser.add_argument(
        "--decoder-type",
        type=str,
        choices=(
            "transducer_modified_beam_search",
            "transducer_greedy_search",
            "ctc_greedy_search",
        ),
        required=True,
        help="Decoder matching the checkpoint head; validated by the exporter.",
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
        help="Calls on the first real batch excluded from reported timings.",
    )
    parser.set_defaults(streaming=False)
    args = parser.parse_args()

    if args.batch_size < 1 or args.beam < 1:
        parser.error("batch_size and beam must be positive.")
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
        ``zipformer`` resamples through PCM16 with ``audioop.ratecv`` and a
        fresh state per clip. ``parakeet`` uses librosa's ``soxr_hq`` resampling,
        matching the earlier NumPy-based upstream loader.
        Clips already at ``SAMPLE_RATE`` bypass PCM16 conversion and resampling.

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
            pcm = np.clip(waveform * PCM16_SCALE, -PCM16_SCALE, PCM16_SCALE - 1).astype(np.int16)
            pcm, _ = ratecv(pcm.tobytes(), 2, 1, sampling_rate, SAMPLE_RATE, None)
            waveform = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / PCM16_SCALE

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
    """Return the bundle path and metadata for a completed target-GPU export.

    The cache key includes the resolved Hub revision, settings, and GPU properties.
    Clear the cache after TensorRT or plugin upgrades. A per-key lock and
    temporary directory keep incomplete builds out of the cache.
    The child exporter receives ``CUDA_VISIBLE_DEVICES=str(args.device)``
    as its environment and sees that device as logical device zero.
    Run without a remapped parent ``CUDA_VISIBLE_DEVICES`` so the selected
    index identifies the same GPU for export and inference.
    Builder resources are released before ASR loads the engines.

    Parameters
    ----------
    args : Namespace
        Checkpoint, export settings, and engine-cache directory from the CLI.
    hardware : dict[str, str | int | list[int]]
        GPU name, compute capability, memory, and CUDA versions.

    Returns
    -------
    tuple[Path, dict[str, str | dict[str, str | int | float | list[int]]]]
        Completed bundle directory and its checkpoint, settings, and hardware metadata.
    """

    patterns = (args.checkpoint_file,)
    if args.model_family == "zipformer":
        patterns += ("config.yaml", "bpe.model")

    snapshot = Path(snapshot_download(args.model_id, allow_patterns=patterns))
    checkpoint = snapshot / args.checkpoint_file
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

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
        "checkpoint_file": args.checkpoint_file,
        "revision": snapshot.name,
        "export_settings": settings,
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
            command = [
                executable,
                "-m",
                f"fast_gpu_asr.export.export_{args.model_family}",
                "--model-path",
                str(checkpoint),
                "--output-dir",
                str(output),
            ]
            for name, value in settings.items():
                command.extend((f"--{name.replace('_', '-')}", str(value)))
            subprocess.run(
                command, check=True, env={"CUDA_VISIBLE_DEVICES": str(args.device)}
            )

            with open(output / "export.json", "w", encoding="utf-8") as destination:
                json.dump(metadata, destination, indent=2, sort_keys=True)

            output.rename(bundle)

    return bundle, metadata


def benchmark(
    original_text: list[str],
    audio: Iterable[dict[str, int | list[float] | np.typing.NDArray[np.float32]]],
    backend: ASR,
) -> dict[str, list[str | float]]:
    """Transcribe one real batch and return references, predictions, and timings.

    Float32 conversion is excluded from timing. The ASR stream is synchronized
    before and after the call, so staging, transfers, decoding, text, and word
    timestamps are included. Real durations exclude padding; elapsed batch time
    is split equally across rows to preserve summed inference time, not to
    estimate individual-request latency.

    Parameters
    ----------
    original_text : list[str]
        Raw reference transcripts in the same order as the audio.
    audio : Iterable[dict[str, int | list[float] | np.typing.NDArray[np.float32]]]
        Prepared mono audio at ``SAMPLE_RATE``, with waveform values under ``array``.
    backend : ASR
        Loaded runtime and its synchronization stream.

    Returns
    -------
    dict[str, list[str | float]]
        References, predictions, real durations, and equally divided batch
        timings, with one entry per input clip.
    """

    audios = [np.ascontiguousarray(clip["array"], dtype=np.float32) for clip in audio]

    backend.stream.synchronize()
    start = perf_counter()
    predictions, timestamps = backend(audios)
    backend.stream.synchronize()
    elapsed = perf_counter() - start

    if len(predictions) != len(audios) or len(timestamps) != len(audios):
        raise RuntimeError("ASR must return one transcript and timestamp list per input.")

    return {
        "references": original_text,
        "predictions": predictions,
        "audio_length_s": [len(audio) / SAMPLE_RATE for audio in audios],
        "transcription_time_s": [elapsed / len(audios)] * len(audios),
    }


def main() -> None:
    """Prepare audio, run one measured pass, and save scored raw transcripts.

    Warm-ups are discarded; final partial batches contribute to pooled RTFx.
    WER uses upstream normalization and compound merging after reconstructing
    Earnings22 parent transcripts. Results are written under ``./results``.

    The metadata sidecar records arguments, resolved checkpoint revision, GPU
    properties, and dataset totals. ``driver_cuda_version`` is the driver's
    supported CUDA API version, not the NVIDIA driver release number.
    """

    args = parse_args()
    with cp.cuda.Device(args.device):
        dataset: Dataset = data_utils.load_data(args).cast_column(
            "audio", Audio(decode=False)
        )
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
                "max_samples": round(args.max_audio_seconds * SAMPLE_RATE),
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
        backend = ASR(bundle, device_id=args.device)

        if args.warmup_steps:
            first = next(iter(dataset.iter(batch_size=args.batch_size)))
            for _ in range(args.warmup_steps):
                benchmark(first["original_text"], first["audio"], backend)
            del first

        results: dict[str, list[str | int | float | None]] = {}
        for batch in tqdm(dataset.iter(batch_size=args.batch_size), desc="Batches"):
            batch_results = {
                **benchmark(batch["original_text"], batch["audio"], backend),
                "audio_filepaths": data_utils.extract_audio_filepaths_from_batch(batch),
                **{key: batch[key] for key in data_utils.CHUNK_METADATA_KEYS if key in batch},
            }
            for key, values in batch_results.items():
                results.setdefault(key, []).extend(values)

        if len(results["predictions"]) != len(dataset):
            raise RuntimeError("Incomplete evaluation: not every sample was transcribed.")

        extra_fields = None
        if data_utils.is_chunked_dataset(args.dataset_path):
            extra_fields = {key: results[key] for key in data_utils.CHUNK_METADATA_KEYS}

        manifest = Path(
            data_utils.write_manifest(
                results["references"],
                results["predictions"],
                args.model_id,
                args.dataset_path,
                args.dataset,
                args.split,
                audio_length=results["audio_length_s"],
                transcription_time=results["transcription_time_s"],
                audio_filepaths=results["audio_filepaths"],
                extra_fields=extra_fields,
            )
        )

        sessions = data_utils.merge_chunked_manifest(data_utils.read_manifest(manifest))
        refs = [tuple(data_utils.normalizer(row["text"]).split()) for row in sessions]
        preds = [tuple(data_utils.normalizer(row["pred_text"]).split()) for row in sessions]

        wer = batch_error_rate(refs, preds, merge_compounds=True)["err_rate"] * 100
        rtfx = sum(results["audio_length_s"]) / sum(results["transcription_time_s"])
        print(f"Results: {manifest}\nWER: {wer:.2f}%  RTFx: {rtfx:.2f}")

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
            "sort": "descending duration, ascending utterance ID",
            "timing": "synchronized full ASR call, including text and word timestamps",
        }
        with open(manifest.with_suffix(".metadata.json"), "w", encoding="utf-8") as destination:
            json.dump(metadata, destination, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
