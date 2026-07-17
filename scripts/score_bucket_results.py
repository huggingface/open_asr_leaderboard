#!/usr/bin/env python3
"""Download all results from an HF bucket and print a CSV summary.

Usage:
    python scripts/score_bucket_results.py
    python scripts/score_bucket_results.py --bucket bezzam/asr_leaderboard
    python scripts/score_bucket_results.py --bucket bezzam/asr_leaderboard --local_dir results
    python scripts/score_bucket_results.py --skip_sync   # re-score already-downloaded results
    python scripts/score_bucket_results.py --family appen --family dataocean   # non-default families
    python scripts/score_bucket_results.py --family all  # every detected family

    # Multilingual (FLEURS/MCV/MLS) results. Defaults to the
    # hf-audio/asr_leaderboard_multilingual bucket, and scores each language
    # separately (each with its own normalizer).
    python scripts/score_bucket_results.py --multilingual
    python scripts/score_bucket_results.py --multilingual --language fr --language de
"""

import argparse
import os
import subprocess
import sys

# Allow importing normalizer from the repo root regardless of where the script
# is called from.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from normalizer.eval_utils import score_results

# Languages covered by the multilingual (FLEURS/MCV/MLS) benchmarks.
ML_LANGUAGES = ["de", "fr", "it", "es", "pt"]


def sync_bucket(bucket: str, local_dir: str, hf_token: str | None = None) -> None:
    """Sync an HF bucket to a local directory using the `hf` CLI."""
    bucket_url = f"hf://buckets/{bucket}"
    print(f"Syncing {bucket_url}  \u2192  {local_dir} ...")
    os.makedirs(local_dir, exist_ok=True)
    env = os.environ.copy()
    if hf_token:
        env["HF_TOKEN"] = hf_token
    subprocess.run(
        ["hf", "buckets", "sync", bucket_url, local_dir],
        check=True,
        env=env,
    )
    print("Sync complete.\n")


def main():
    parser = argparse.ArgumentParser(description="Score all results from an HF bucket.")
    parser.add_argument(
        "--bucket",
        default=None,
        help="HF bucket name (without the hf://buckets/ prefix). Defaults to "
             "hf-audio/asr_leaderboard_multilingual if --multilingual is set, "
             "otherwise hf-audio/asr_leaderboard_h200.",
    )
    parser.add_argument(
        "--local_dir",
        default=None,
        help="Local directory to sync results into. Defaults to <repo_root>/results.",
    )
    parser.add_argument(
        "--skip_sync",
        action="store_true",
        help="Skip the bucket sync and score the already-downloaded results in --local_dir.",
    )
    parser.add_argument(
        "--hf_token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token for private buckets. Defaults to $HF_TOKEN env var.",
    )
    parser.add_argument(
        "--family",
        action="append",
        default=None,
        choices=["appen", "dataocean", "public", "extra", "all"],
        metavar="FAMILY",
        help="Dataset family to include in the CSV summary (can be repeated). "
             "Choices: appen, dataocean, public, extra, all. Defaults to public. "
             "Ignored when --multilingual is set.",
    )
    parser.add_argument(
        "--model_id",
        action="append",
        default=None,
        metavar="MODEL_ID",
        help="Score only this model (can be repeated for multiple models). "
             "E.g. --model_id zoom/scribe_v1 --model_id assembly/universal-3-pro. "
             "Defaults to scoring all models.",
    )
    parser.add_argument(
        "--multilingual",
        action="store_true",
        help="Score multilingual (FLEURS/MCV/MLS) results instead of the English "
             "public benchmarks. Scores each language separately, since each "
             "requires its own normalizer.",
    )
    parser.add_argument(
        "--language",
        action="append",
        default=None,
        choices=ML_LANGUAGES,
        metavar="LANGUAGE",
        help=f"Language(s) to score (can be repeated). Choices: {', '.join(ML_LANGUAGES)}. "
             "Only used with --multilingual. Defaults to all languages found.",
    )
    args = parser.parse_args()

    bucket = args.bucket or (
        "hf-audio/asr_leaderboard_multilingual" if args.multilingual else "hf-audio/asr_leaderboard_h200"
    )
    local_dir = args.local_dir or os.path.join(REPO_ROOT, "results")

    if not args.skip_sync:
        sync_bucket(bucket, local_dir, hf_token=args.hf_token)
    else:
        print(f"Skipping sync — scoring results in: {local_dir}\n")

    if not os.path.isdir(local_dir):
        print(f"ERROR: Local results directory not found: {local_dir}", file=sys.stderr)
        sys.exit(1)

    # Score the requested models; csv_only=True suppresses per-dataset and
    # composite output, printing only the CSV summary block.
    model_ids = args.model_id or [None]  # None means all models

    if args.multilingual:
        languages = args.language or ML_LANGUAGES
        for model_id in model_ids:
            for language in languages:
                try:
                    score_results(
                        local_dir,
                        model_id=model_id,
                        multilingual=True,
                        language=language,
                        families=[f"ml_{language}"],
                        csv_only=True,
                    )
                except ValueError as e:
                    print(f"Skipping language={language} model_id={model_id}: {e}")
    else:
        families = args.family or ["public"]
        if "all" in families:
            families = None  # None means all families

        for model_id in model_ids:
            score_results(local_dir, model_id=model_id, csv_only=True, families=families)


if __name__ == "__main__":
    main()
