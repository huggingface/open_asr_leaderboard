"""Download selected datasets or evaluate one dataset against running vLLM engines."""

import argparse
import asyncio
import tempfile
from pathlib import Path

from reson8_rumble.dataset_config import DEFAULT_DATASET_PATH, LANGUAGE_NAMES, parse_datasets

BASE_PORT = 8000


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
    from reson8_rumble.vllm_async import results_from_rows, run_requests

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

        from reson8_rumble.vllm_async import AUDIO_ROOT, prepare_inputs

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
