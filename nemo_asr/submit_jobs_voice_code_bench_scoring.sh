#!/usr/bin/env bash
# Grade an existing bucket manifest without running ASR again.
# Usage: RESULTS_BUCKET=owner/bucket bash <this-script> runs/<run>/<model>/<manifest>.jsonl
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_voice_code}"
VERIFIER_MODE="${VERIFIER_MODE:-live-fill}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"
MANIFEST_PATH="${1:?Pass a manifest path relative to RESULTS_BUCKET.}"

# Reject invalid paths/modes and missing credentials before submitting anything.
python - "$MANIFEST_PATH" "$VERIFIER_MODE" "$ALLOW_PARTIAL" <<'PY'
import os
from pathlib import PurePosixPath
import sys
from huggingface_hub import get_token
path = PurePosixPath(sys.argv[1])
if path.is_absolute() or '..' in path.parts or path.suffix != '.jsonl':
    raise SystemExit("Expected a relative bucket manifest path ending in .jsonl.")
if sys.argv[2] not in {'live-fill', 'replay'}:
    raise SystemExit("VERIFIER_MODE must be replay or live-fill.")
if sys.argv[3] not in {'0', '1'}:
    raise SystemExit("ALLOW_PARTIAL must be 0 or 1.")
if not get_token():
    raise SystemExit("Set HF_TOKEN or run hf auth login before submitting jobs.")
if sys.argv[2] == 'live-fill' and not os.environ.get('OPENAI_API_KEY'):
    raise SystemExit("Set OPENAI_API_KEY for live-fill grading.")
PY

# Ship the current scoring code, including unpushed changes, into the CPU image.
PAYLOAD=$(tar --exclude='__pycache__' --exclude='*.pyc' -czf - -C "$REPO_ROOT" \
    normalizer scripts/score_voice_code_bench_job.py \
    requirements/requirements_voice_code_bench_scoring.txt | base64 | tr -d '\n')
SECRET_ARGS=(--secret HF_TOKEN)
[[ "$VERIFIER_MODE" == "live-fill" ]] && SECRET_ARGS+=(--secret OPENAI_API_KEY)
JOB_RECORD="${JOB_RECORD:-${SCRIPT_DIR}/results/voice_code_bench/${MANIFEST_PATH%.jsonl}.grading-job.json}"

python "${REPO_ROOT}/scripts/run_voice_code_bench_job.py" \
    --image "${SCORING_IMAGE:-python:3.12-slim}" \
    --flavor "${SCORING_FLAVOR:-cpu-basic}" --timeout "${SCORING_TIMEOUT:-8h}" \
    --namespace "${ORG_NAME:-}" --bucket "$RESULTS_BUCKET" --stage grading \
    --record "$JOB_RECORD" "${SECRET_ARGS[@]}" \
    --env "MANIFEST_PATH=${MANIFEST_PATH}" --env "VERIFIER_MODE=${VERIFIER_MODE}" \
    --env "ALLOW_PARTIAL=${ALLOW_PARTIAL}" --env "RESULTS_BUCKET=${RESULTS_BUCKET}" \
    -- bash -ec "
        set -o pipefail
        mkdir -p /app
        echo '${PAYLOAD}' | base64 -d | tar -xzf - -C /app
        cd /app
        python -m pip install -r requirements/requirements_voice_code_bench_scoring.txt
        extra_args=()
        [[ \"\$ALLOW_PARTIAL\" == 1 ]] && extra_args+=(--allow-partial)
        PYTHONPATH=/app python scripts/score_voice_code_bench_job.py \\
            --manifest \"/results/\$MANIFEST_PATH\" \\
            --verifier-cache \"/results/\${MANIFEST_PATH%.jsonl}.verifier-cache.json\" \\
            --verifier-mode \"\$VERIFIER_MODE\" \"\${extra_args[@]}\"
    "
