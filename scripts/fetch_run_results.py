#!/usr/bin/env python3
"""Download one submit run's results for a model from an HF bucket.

The submit_jobs*.sh scripts call this after their jobs finish, instead of
syncing the model's whole bucket folder: that folder accumulates the results
of every dataset the model was ever run on, and scoring all of them again
(alignment is quadratic in transcript length) can take far longer than the run.

A file belongs to this run if it was uploaded at or after --since (the time the
submit script started). Those files are downloaded into --local-dir and linked
into a run directory, whose path is printed on stdout so the caller can pass it
to score_results:

    RUN_RESULTS=$(python scripts/fetch_run_results.py \
        --bucket hf-audio/asr_leaderboard_h200 --model-folder openai-whisper-large-v3 \
        --local-dir ./results/openai-whisper-large-v3 --since "$RUN_START" --expected 2)

--since 0 takes every file in the folder (the old behaviour). Messages go to
stderr.
"""

import argparse
import os
import shutil
import sys
from datetime import datetime, timezone

from huggingface_hub import HfApi
from huggingface_hub.hf_api import BucketFile

# Next to the model's results; hidden, so a recursive glob over ./results (as
# score_results does) does not pick the links up a second time.
RUN_DIR_NAME = ".run"


def log(msg):
    print(msg, file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bucket", required=True, help="Bucket id, e.g. hf-audio/asr_leaderboard_h200.")
    parser.add_argument("--model-folder", required=True, help="Model's folder in the bucket.")
    parser.add_argument("--local-dir", required=True, help="Where the files are downloaded.")
    parser.add_argument(
        "--since", type=float, default=0,
        help="Unix time; only files uploaded at or after it are taken. 0 takes every file.",
    )
    parser.add_argument("--expected", type=int, default=None, help="Number of manifests the run should produce.")
    args = parser.parse_args()

    since = datetime.fromtimestamp(args.since, tz=timezone.utc)
    prefix = args.model_folder.strip("/") + "/"
    try:
        files = [
            f
            for f in HfApi().list_bucket_tree(args.bucket, prefix=prefix, recursive=True)
            if isinstance(f, BucketFile) and f.uploaded_at >= since
        ]
    except Exception as e:  # e.g. the folder does not exist yet: every job failed
        log(f"WARNING: could not list {args.bucket}/{prefix}: {e}")
        files = []

    local_dir = os.path.abspath(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    targets = [(f, os.path.join(local_dir, f.path[len(prefix):])) for f in files]
    for _, target in targets:
        os.makedirs(os.path.dirname(target), exist_ok=True)
    if targets:
        HfApi().download_bucket_files(args.bucket, targets)

    run_dir = os.path.join(os.path.dirname(local_dir), RUN_DIR_NAME, os.path.basename(local_dir))
    shutil.rmtree(run_dir, ignore_errors=True)
    os.makedirs(run_dir)
    # Every file of the run is linked (some scripts check per-manifest sidecars,
    # e.g. *.metadata.json); score_results only reads the *.jsonl among them.
    for _, target in targets:
        link = os.path.join(run_dir, os.path.relpath(target, local_dir))
        os.makedirs(os.path.dirname(link), exist_ok=True)
        os.symlink(target, link)

    found = sum(t.endswith(".jsonl") for _, t in targets)
    if args.expected is not None and found < args.expected:
        log(
            f"WARNING: expected {args.expected} result files but only {found} were uploaded "
            "during this run. Some jobs may have failed or not finished yet."
        )
    else:
        log(f"All {found} result files present.")
    print(run_dir)


if __name__ == "__main__":
    main()
