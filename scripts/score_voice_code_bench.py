"""Score VoiceCodeBench CTEM from a leaderboard manifest or released predictions.

The benchmark's verifier and scoring code are loaded from its dataset repository,
so this script uses the same CTEM definition as the published baseline results.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


DATASET = "besimple-ai/voice-code-bench"
DEFAULT_REVISION = "bef2824f83ef1c796f3e79731a3b0741708730df"
PARAKEET_BASELINE = "baselines/predictions/modal_nvidia_parakeet_tdt_0_6b_v3.json"


def benchmark_root(local_root, revision):
    if local_root is not None:
        return local_root.resolve()
    return Path(snapshot_download(
        repo_id=DATASET,
        repo_type="dataset",
        revision=revision,
        allow_patterns=[
            "data/metadata.jsonl",
            "scripts/voice_code_bench/*.py",
            "scripts/voice_code_bench/verifiers/*.json",
        ],
    ))


def read_predictions(path):
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [
            {"audio_id": row["audio_id"], "model_transcript": row["pred_text"]}
            for row in rows
        ], None, None
    artifact = json.loads(path.read_text(encoding="utf-8"))
    run_metadata = artifact.get("run_metadata", {})
    return artifact["items"], run_metadata.get("model_id"), run_metadata.get("entity_verifier")


def score(args):
    root = benchmark_root(args.dataset_root, args.revision)
    sys.path.insert(0, str(root / "scripts"))
    from voice_code_bench.io import read_metadata
    from voice_code_bench.metrics import aggregate_entity_score_rows, score_entity_capture
    from voice_code_bench.verifier import VerifierCache, load_verifier_config, verify_entity_matches

    metadata = read_metadata(root / "data" / "metadata.jsonl")
    by_id = {row["audio_id"]: row for row in metadata}
    if args.released_parakeet_baseline:
        path = root / PARAKEET_BASELINE
        if not path.is_file():
            path = Path(hf_hub_download(
                repo_id=DATASET, repo_type="dataset", revision=args.revision,
                filename=PARAKEET_BASELINE,
            ))
    else:
        path = args.input.resolve()
    predictions, artifact_model_id, artifact_verifier = read_predictions(path)
    if not predictions:
        raise ValueError(f"No predictions found in {path}.")
    model_id = args.model_id or artifact_model_id
    if not model_id:
        raise ValueError("--model-id is required for a leaderboard manifest.")
    if artifact_model_id and args.model_id and args.model_id != artifact_model_id:
        raise ValueError(f"Model ID {args.model_id!r} does not match artifact model {artifact_model_id!r}.")

    prediction_by_id = {}
    for row in predictions:
        audio_id = row["audio_id"]
        if audio_id not in by_id:
            raise ValueError(f"Unknown audio_id: {audio_id}")
        if audio_id in prediction_by_id:
            raise ValueError(f"Duplicate audio_id: {audio_id}")
        if not isinstance(row.get("model_transcript"), str):
            raise ValueError(f"Missing model transcript for {audio_id}")
        prediction_by_id[audio_id] = row
    if not args.allow_partial and set(prediction_by_id) != set(by_id):
        missing = sorted(set(by_id) - set(prediction_by_id))
        raise ValueError(f"Expected all {len(by_id)} test items; missing {len(missing)} (first: {missing[:3]}).")

    config = load_verifier_config(args.verifier_id)
    if any("entity_matches" in row for row in predictions):
        if not artifact_verifier or (
            artifact_verifier.get("id") != config.id
            or artifact_verifier.get("config_digest") != config.digest
        ):
            raise ValueError("Published entity decisions need matching verifier provenance.")
    cache = VerifierCache(args.verifier_cache.resolve()) if args.verifier_cache else None

    def score_one(datapoint):
        row = prediction_by_id[datapoint["audio_id"]]
        if "entity_matches" not in row:
            matches = verify_entity_matches(
                datapoint, row["model_transcript"], config, {},
                cache=cache, cache_mode=args.verifier_mode,
            )
            row = {**row, "entity_matches": matches}
        return score_entity_capture(datapoint, model_id, row)

    selected = [row for row in metadata if row["audio_id"] in prediction_by_id]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        scores = list(pool.map(score_one, selected))
    aggregate = aggregate_entity_score_rows(scores)
    result = {
        "dataset": DATASET,
        "dataset_revision": args.revision if args.dataset_root is None else None,
        "model_id": model_id,
        "verifier_id": config.id,
        "verifier_config_digest": config.digest,
        "ctem": aggregate["entity_capture_rate"],
        "tsr": aggregate["task_success_rate"],
        "aggregate": aggregate,
        "items": scores,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"model={model_id} samples={len(scores)} CTEM={aggregate['entity_capture_rate']:.4%} "
          f"TSR={aggregate['task_success_rate']:.4%} "
          f"entities={aggregate['correct_token_count']}/{aggregate['gold_token_count']}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Score VoiceCodeBench entity recovery (CTEM).")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Leaderboard JSONL manifest or VoiceCodeBench predictions JSON.")
    source.add_argument("--released-parakeet-baseline", action="store_true", help="Score the published Parakeet predictions without an API key.")
    parser.add_argument("--model-id", help="Required for a leaderboard manifest.")
    parser.add_argument("--dataset-root", type=Path, help="Local VoiceCodeBench checkout; otherwise fetch code and metadata from the Hub.")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="VoiceCodeBench Hub revision to fetch.")
    parser.add_argument("--verifier-id", default="openai_gpt_5_5_v1")
    parser.add_argument("--verifier-cache", type=Path, default=Path("results/voice_code_bench_verifier_cache.json"))
    parser.add_argument("--verifier-mode", choices=["live", "replay", "live-fill"], default="live-fill")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--allow-partial", action="store_true", help="Score a subset for development; report its sample count.")
    parser.add_argument("--output", type=Path, help="Save per-item decisions and aggregate CTEM as JSON.")
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    score(args)


if __name__ == "__main__":
    main()
