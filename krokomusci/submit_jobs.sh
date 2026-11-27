#!/usr/bin/env bash
set -euo pipefail

if (( $# > 1 )) || [[ "${1:-}" != "" && "${1:-}" != --dry-run ]]; then
    echo 'Usage: RESULTS_BUCKET=OWNER/BUCKET bash krokomusci/submit_jobs.sh [--dry-run]' >&2
    exit 2
fi

: "${RESULTS_BUCKET:?Set RESULTS_BUCKET to a private bucket you can write to, in OWNER/BUCKET form}"
if [[ ! "$RESULTS_BUCKET" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
    echo 'RESULTS_BUCKET must be in OWNER/BUCKET form.' >&2
    exit 2
fi

# A reviewer can set SPACE_ID to a copy built from the reference revision.
# Hugging Face Jobs pulls the latest build of the selected Space.
space_id="${SPACE_ID:-Banafo/KrokoMusci-OpenASR}"
run_id="krokomusci-openasr-$(date -u +%Y%m%dT%H%M%SZ)-$$-${RANDOM}"
manifest="${run_id}/voice_code_bench/results/MODEL_Banafo-KrokoMusci_DATASET_besimple-ai-voice-code-bench_default_test.jsonl"
options=(
    --flavor h200 --timeout 90m --name KrokoMusci-OpenASR
    --secrets HF_TOKEN
    --volume "hf://buckets/${RESULTS_BUCKET}:/results"
    --env "KM_RESULTS=/results/${run_id}"
    --env KM_SPEED_RECIPE=packed-retain
    --env KM_INCLUDE_VOICECODE=1
)
if [[ -n "${ORG_NAME:-}" ]]; then
    options+=(--namespace "$ORG_NAME")
fi
if [[ "${1:-}" == --dry-run ]]; then
    hf jobs run "${options[@]}" --dry-run -- "hf.co/spaces/$space_id" bash /app/job.sh
    printf 'After successful inference, grade VoiceCodeBench with the official BF16 judge on H200.\nManifest: hf://buckets/%s/%s\n' "$RESULTS_BUCKET" "$manifest"
    exit 0
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
upstream_dir=$(dirname "$script_dir")
source "$upstream_dir/scripts/submit_utils.sh"

printf 'Results directory: hf://buckets/%s/%s\n' "$RESULTS_BUCKET" "$run_id"
hf jobs run "${options[@]}" -- "hf.co/spaces/$space_id" bash /app/job.sh

# The job can finish just before the bucket mount makes its last writes visible.
python - "$RESULTS_BUCKET" "$manifest" <<'PY'
import sys
from time import sleep

from huggingface_hub import HfApi

bucket, manifest = sys.argv[1:]
api = HfApi()
for attempt in range(12):
    if any(item.path == manifest and getattr(item, "size", 0) > 0
           for item in api.list_bucket_tree(bucket, prefix=manifest)):
        break
    if attempt < 11:
        sleep(5)
else:
    raise SystemExit(f"VoiceCodeBench manifest did not appear at hf://buckets/{bucket}/{manifest}")
PY

# The suite writes into a nested run directory. Pass its exact manifest path to the shared grader.
output_dir="$script_dir/results/$run_id"
JOB_RECORD="$output_dir/voice-code-bench-grading-job.json" \
VCB_JUDGE=bf16 VERIFIER_MODE=live-fill ALLOW_PARTIAL=0 SCORING_FLAVOR=h200 \
SCORING_TIMEOUT="${SCORING_TIMEOUT:-45m}" vcb_submit_grading "$manifest"

PYTHONPATH="$upstream_dir" python "$upstream_dir/scripts/summarize_voice_code_bench.py" \
    --bucket "$RESULTS_BUCKET" --manifest-path "$manifest" --model-id Banafo/KrokoMusci \
    --run-id "$run_id" --judge bf16 --output-dir "$output_dir" --publish-to "${manifest%/*}"
