#!/usr/bin/env python3
"""Download all results from an HF bucket and print a CSV summary.

Usage:
    python scripts/score_bucket_results.py
    python scripts/score_bucket_results.py --bucket bezzam/asr_leaderboard
    python scripts/score_bucket_results.py --bucket bezzam/asr_leaderboard --local_dir results
    python scripts/score_bucket_results.py --skip_sync   # re-score already-downloaded results
    python scripts/score_bucket_results.py --family appen --family dataocean   # non-default families
    python scripts/score_bucket_results.py --family all  # every detected family
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
        default="hf-audio/asr_leaderboard_h200",
        help="HF bucket name (without the hf://buckets/ prefix). Default: hf-audio/asr_leaderboard_h200",
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
        default="public",
        choices=["appen", "dataocean", "public", "extra", "all"],
        metavar="FAMILY",
        help="Dataset family to include in the CSV summary (can be repeated). "
             "Choices: appen, dataocean, public, extra, all. Defaults to public.",
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
    args = parser.parse_args()

    local_dir = args.local_dir or os.path.join(REPO_ROOT, "results")

    if not args.skip_sync:
        sync_bucket(args.bucket, local_dir, hf_token=args.hf_token)
    else:
        print(f"Skipping sync — scoring results in: {local_dir}\n")

    if not os.path.isdir(local_dir):
        print(f"ERROR: Local results directory not found: {local_dir}", file=sys.stderr)
        sys.exit(1)

    families = args.family or ["public"]
    if "all" in families:
        families = None  # None means all families

    # Score the requested models; csv_only=True suppresses per-dataset and
    # composite output, printing only the CSV summary block.
    model_ids = args.model_id or [None]  # None means all models
    for model_id in model_ids:
        score_results(local_dir, model_id=model_id, csv_only=True, families=families)


if __name__ == "__main__":
    main()
