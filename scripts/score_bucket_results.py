#!/usr/bin/env python3
"""Download all results from an HF bucket and print a CSV summary.

Usage:
    python scripts/score_bucket_results.py
    python scripts/score_bucket_results.py --bucket bezzam/asr_leaderboard
    python scripts/score_bucket_results.py --bucket bezzam/asr_leaderboard --local_dir /tmp/asr_results
    python scripts/score_bucket_results.py --skip_sync   # re-score already-downloaded results
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


def sync_bucket(bucket: str, local_dir: str) -> None:
    """Sync an HF bucket to a local directory using the `hf` CLI."""
    bucket_url = f"hf://buckets/{bucket}"
    print(f"Syncing {bucket_url}  →  {local_dir} ...")
    os.makedirs(local_dir, exist_ok=True)
    result = subprocess.run(
        ["hf", "buckets", "sync", bucket_url, local_dir],
        check=True,
    )
    print("Sync complete.\n")


def main():
    parser = argparse.ArgumentParser(description="Score all results from an HF bucket.")
    parser.add_argument(
        "--bucket",
        default="hf-audio/asr_leaderboard",
        help="HF bucket name (without the hf://buckets/ prefix). Default: hf-audio/asr_leaderboard",
    )
    parser.add_argument(
        "--local_dir",
        default=None,
        help="Local directory to sync results into. Defaults to <repo_root>/results_bucket.",
    )
    parser.add_argument(
        "--skip_sync",
        action="store_true",
        help="Skip the bucket sync and score the already-downloaded results in --local_dir.",
    )
    args = parser.parse_args()

    local_dir = args.local_dir or os.path.join(REPO_ROOT, "results_bucket")

    if not args.skip_sync:
        sync_bucket(args.bucket, local_dir)
    else:
        print(f"Skipping sync — scoring results in: {local_dir}\n")

    if not os.path.isdir(local_dir):
        print(f"ERROR: Local results directory not found: {local_dir}", file=sys.stderr)
        sys.exit(1)

    # Score all models at once; csv_only=True suppresses per-dataset and
    # composite output, printing only the CSV summary block.
    score_results(local_dir, model_id=None, csv_only=True)


if __name__ == "__main__":
    main()
