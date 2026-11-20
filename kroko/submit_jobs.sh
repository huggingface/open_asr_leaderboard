#!/usr/bin/env bash
# Sign in with `hf auth login` using an account with model and Space access.
# Set RESULTS_BUCKET to an owner/bucket you can write to, and optionally ORG_NAME.
# Usage: bash kroko/submit_jobs.sh [--dry-run]
set -euo pipefail

if (( $# > 1 )) || [[ -n "${1:-}" && "$1" != --dry-run ]]; then
    echo 'Usage: bash kroko/submit_jobs.sh [--dry-run]' >&2
    exit 2
fi

SPACE="Banafo/Kroko-ASR-XL-EN-Offline-Eval"
MODEL_ID="Banafo/Kroko-ASR-XL-EN-Offline"
MODEL_REVISION="8c1fb9b89831c6cd9f63d0295f2ccb2be6da2964"
RESULTS_BUCKET="${RESULTS_BUCKET:-Banafo/kroko-asr-xl-en-offline-eval-results}"
RUN_ID="kroko-openasr-$(date -u +%Y%m%dT%H%M%SZ)-$$-${RANDOM}"

options=(--flavor h200 --timeout 2h --name Kroko-ASR-XL-EN-Offline-OpenASR
    --env PYTHONUNBUFFERED=1 --env HF_HUB_DISABLE_TELEMETRY=1
    --secrets HF_TOKEN --volume "hf://buckets/${RESULTS_BUCKET}:/results")
if [[ -n "${ORG_NAME:-}" ]]; then
    options+=(--namespace "$ORG_NAME")
fi
if [[ "${1:-}" == --dry-run ]]; then
    options+=(--dry-run)
fi

# HF Jobs uses the latest build of this Space. The reference build used
# revision d787a9ea470a22ab9a520efc42c4ea534060f204 and loads fast-gpu-asr/.
hf jobs run "${options[@]}" -- "hf.co/spaces/${SPACE}" python /app/evaluate.py \
    --model-id "$MODEL_ID" --model-revision "$MODEL_REVISION" \
    --output-dir "/results/runs/${RUN_ID}" --mode full \
    --batch-size 256 --workers 3 --beam 6 --precision fp16 \
    --min-audio-seconds 0.1 --opt-audio-seconds 10.0 --max-audio-seconds 40.0 \
    --warmup-steps 5
