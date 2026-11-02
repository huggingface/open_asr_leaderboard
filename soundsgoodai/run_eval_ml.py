#!/usr/bin/env python3
"""Evaluate multilingual datasets with Fast GPU ASR.

Use the upstream language-specific normalizer and compound-aware scoring.
Export, audio preparation, worker scheduling, and timing are shared with
``run_eval.py``. The language affects reference filtering and scoring, not
model decoding.
"""

import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from platform import platform

import cupy as cp
from datasets import Audio, Features, Value, concatenate_datasets
from fast_gpu_asr import ASR
from kaldialign import batch_error_rate
from run_eval import SAMPLE_RATE, benchmark, export_bundle, inspect_audio

from normalizer import data_utils

LANGUAGES = ("de", "fr", "it", "es", "pt", "nl")


def parse_args() -> Namespace:
    """Parse and validate multilingual evaluation settings without selecting a GPU.

    Model and export options match the English runner. Greedy modes force beam
    one, and streaming is disabled so the full dataset can be sorted first.

    Returns
    -------
    Namespace
        Validated evaluation settings. Unless explicitly supplied, ``language``
        is inferred from the dataset suffix, such as ``de`` in ``fleurs_de``.
        Unrecognized suffixes require an explicit supported language.
    """

    parser = ArgumentParser(
        description=(
            "Evaluate multilingual datasets with Fast GPU ASR and report WER and RTFx."
        )
    )

    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Hugging Face repository containing the model checkpoint.",
    )
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
        default="hf-audio/open-asr-leaderboard-multilingual-datasets",
        help="Hugging Face dataset repository or local dataset path.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset configuration name, such as fleurs_de; use '' if none is needed.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate, such as test, test.clean, or test.other.",
    )
    parser.add_argument(
        "--language",
        choices=LANGUAGES,
        help="Normalization language; defaults to the dataset's language suffix.",
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
        help="Calls on the first real batch per worker excluded from reported timings.",
    )
    parser.set_defaults(streaming=False)

    args = parser.parse_args()

    if not args.report_name:
        args.report_name = args.model_id
    if args.batch_size < 1 or args.beam < 1 or args.workers < 1:
        parser.error("batch_size, beam, and workers must be positive.")
    if (
        args.warmup_steps < 0
        or args.max_eval_samples == 0
        or args.max_eval_samples < -1
    ):
        parser.error(
            "warmup_steps must be nonnegative; max_eval_samples must be -1 or positive."
        )
    if (
        not 0
        < args.min_audio_seconds
        <= args.opt_audio_seconds
        <= args.max_audio_seconds
        < float("inf")
    ):
        parser.error(
            "Expected finite 0 < min_audio_seconds <= opt_audio_seconds <= max_audio_seconds."
        )
    if args.max_eval_samples > 0 and data_utils.is_chunked_dataset(args.dataset_path):
        parser.error(
            "Do not truncate a chunked dataset: WER requires complete parent sessions."
        )

    if args.decoder_type != "transducer_modified_beam_search":
        args.beam = 1

    if args.language is None:
        args.language = args.dataset.rsplit("_", 1)[-1]
        if args.language not in LANGUAGES:
            parser.error(
                "Cannot infer a supported language from --dataset; supply --language."
            )

    return args


def main() -> None:
    """Evaluate one dataset and report language-specific WER and pooled RTFx.

    Preserve raw transcripts in the manifest. Apply upstream language-specific
    normalization for filtering and scoring, and align compound boundaries
    before scoring. Both steps are outside the measured inference calls.
    Save raw transcripts and the same environment metadata as the English
    runner under ``./results``, including the selected normalization language.
    """

    args = parse_args()
    with cp.cuda.Device(args.device):
        dataset = data_utils.load_data(args).cast_column("audio", Audio(decode=False))
        # Normalize metadata without copying or decoding the encoded audio column.
        metadata = dataset.remove_columns("audio")
        references = [data_utils.get_text(sample) for sample in metadata]
        for col in ("original_text", "norm_text"):
            if col in metadata.column_names:
                metadata = metadata.remove_columns(col)
        metadata = metadata.add_column("original_text", references)
        metadata = metadata.add_column(
            "norm_text",
            [data_utils.ml_normalizer(text, lang=args.language) for text in references],
        )

        indices = [
            index
            for index, text in enumerate(metadata["norm_text"])
            if data_utils.is_target_text_in_range(text)
        ]
        if args.max_eval_samples > 0:
            indices = indices[: args.max_eval_samples]

        if not indices:
            raise ValueError("No evaluable samples remain after upstream filtering.")

        ids = (
            tuple(metadata["id"])
            if "id" in metadata.column_names
            else (None,) * len(metadata)
        )
        file_names = (
            tuple(metadata["file_name"])
            if "file_name" in metadata.column_names
            else (None,) * len(metadata)
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
                    "audio": {
                        "array": [Value("float32")],
                        "sampling_rate": Value("int32"),
                    },
                    "num_samples": Value("int64"),
                }
            ),
            fn_kwargs={
                "max_samples": round(args.max_audio_seconds * SAMPLE_RATE),
                "model_family": args.model_family,
            },
        )

        lengths, ids = tuple(dataset["num_samples"]), tuple(dataset["utterance_id"])
        order = sorted(range(len(dataset)), key=lambda i: (-lengths[i], ids[i]))
        # Python audio lists make garbage collection scan every sample during ASR.
        dataset = dataset.select(order).with_format(
            "numpy", columns=["audio"], output_all_columns=True
        )

        properties = cp.cuda.runtime.getDeviceProperties(args.device)
        hardware = {
            "gpu": properties["name"].decode(),
            "compute_capability": [properties["major"], properties["minor"]],
            "vram_bytes": properties["totalGlobalMem"],
            "driver_cuda_version": cp.cuda.runtime.driverGetVersion(),
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        }
        bundle, metadata = export_bundle(args, hardware)

        workers = min(
            args.workers, (len(dataset) + args.batch_size - 1) // args.batch_size
        )
        backends = [ASR(bundle, device_id=args.device) for _ in range(workers)]

        results = benchmark(dataset, backends, args.batch_size, args.warmup_steps)
        if len(results["predictions"]) != len(dataset):
            raise RuntimeError(
                "Incomplete evaluation: not every sample was transcribed."
            )

        extra_fields = None
        if data_utils.is_chunked_dataset(args.dataset_path):
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
            "timing": (
                "queue wall time, including scheduling and synchronized full ASR calls, "
                "including text and word timestamps"
            ),
        }
        with open(
            manifest.with_suffix(".metadata.json"), "w", encoding="utf-8"
        ) as destination:
            json.dump(metadata, destination, indent=2, sort_keys=True)

    sessions = data_utils.merge_chunked_manifest(data_utils.read_manifest(manifest))
    refs = [
        data_utils.ml_normalizer(row["text"], lang=args.language) for row in sessions
    ]
    preds = [
        data_utils.ml_normalizer(row["pred_text"], lang=args.language)
        for row in sessions
    ]
    refs, preds = data_utils.normalize_compound_pairs(refs, preds)
    refs = [tuple(ref.split()) for ref in refs]
    preds = [tuple(pred.split()) for pred in preds]

    wer = batch_error_rate(refs, preds, merge_compounds=True)["err_rate"] * 100
    rtfx = sum(results["audio_length_s"]) / sum(results["transcription_time_s"])

    print(f"Results: {manifest}\nWER: {wer:.2f}%  RTFx: {rtfx:.2f}")


if __name__ == "__main__":
    main()
