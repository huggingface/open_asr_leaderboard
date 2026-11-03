#!/usr/bin/env bash
# Submit inference on GPU, then CTEM grading in a separate CPU HF Job.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/scripts/submit_utils.sh"

SPACE="${SPACE:-hf-audio/open-asr-leaderboard-nemo}"
export RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_voice_code}"
export ORG_NAME="${ORG_NAME:-}"
FLAVOR="${FLAVOR:-h200}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:--1}"
export VERIFIER_MODE="${VERIFIER_MODE:-live-fill}"
export ALLOW_PARTIAL=0
[[ "$MAX_EVAL_SAMPLES" != "-1" ]] && ALLOW_PARTIAL=1
# A fresh directory prevents grading manifests left behind by an earlier run.
RUN_ID="$(python -c 'import uuid; print(uuid.uuid4().hex)')"
LOCAL_RESULTS="${SCRIPT_DIR}/results/voice_code_bench/${RUN_ID}"

# Check credentials before spending GPU time. HF_TOKEN may also come from hf auth login.
python - "$VERIFIER_MODE" "$MAX_EVAL_SAMPLES" <<'PY'
import os
import sys
from huggingface_hub import get_token
if not get_token():
    raise SystemExit("Set HF_TOKEN or run hf auth login before submitting jobs.")
try:
    samples = int(sys.argv[2])
except ValueError:
    raise SystemExit("MAX_EVAL_SAMPLES must be -1 or a positive integer.")
if samples != -1 and samples <= 0:
    raise SystemExit("MAX_EVAL_SAMPLES must be -1 or a positive integer.")
if sys.argv[1] not in {"replay", "live-fill"}:
    raise SystemExit("VERIFIER_MODE must be replay or live-fill.")
if sys.argv[1] == "live-fill" and not os.environ.get("OPENAI_API_KEY"):
    raise SystemExit("Set OPENAI_API_KEY for live-fill grading.")
if sys.argv[1] == "replay":
    raise SystemExit("A fresh inference run has no grading cache. Use the grading-only script to replay an existing run.")
PY

LOCAL_SCRIPT_INJECT=$(local_script_inject "$SCRIPT_DIR" run_eval.py)
LOCAL_NORMALIZER_INJECT=$(local_normalizer_inject)

# Models: "model_id batch_size". Match the Parakeet WER evaluation.
MODEL_CONFIGS=(
    "nvidia/parakeet-tdt-0.6b-v3 128"
    # "nvidia/parakeet-tdt-0.6b-v2 128"
    # "nvidia/parakeet-rnnt-1.1b 128"
    # "nvidia/parakeet-ctc-1.1b 128"
)

run_model() {
    local model_id=$1 batch_size=$2
    local model_folder="${model_id//\//-}"
    local bucket_dir="runs/${RUN_ID}/${model_folder}"
    local manifest_name="MODEL_${model_folder}_DATASET_besimple-ai-voice-code-bench_default_test.jsonl"
    local manifest_path="${bucket_dir}/${manifest_name}"
    echo "Run ${RUN_ID}: ${model_id}"
    echo "Manifest: hf://buckets/${RESULTS_BUCKET}/${manifest_path}"
    python "${REPO_ROOT}/scripts/run_voice_code_bench_job.py" \
        --image "hf.co/spaces/${SPACE}" --flavor "$FLAVOR" --timeout 8h \
        --namespace "$ORG_NAME" --bucket "$RESULTS_BUCKET" --stage inference \
        --record "${LOCAL_RESULTS}/${model_folder}/inference-job.json" \
        --secret HF_TOKEN \
        --env "MODEL_ID=${model_id}" --env "BATCH_SIZE=${batch_size}" \
        --env "MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES}" \
        --env "BUCKET_DIR=${bucket_dir}" --env "MANIFEST_NAME=${manifest_name}" \
        -- bash -ec "
            set -o pipefail
            (${LOCAL_NORMALIZER_INJECT} ${LOCAL_SCRIPT_INJECT} true)
            cd /app
            mkdir -p results
            rm -f \"results/\$MANIFEST_NAME\"
            PYTHONPATH=/app python run_eval.py \\
                --model_id=\"\$MODEL_ID\" \\
                --dataset_path=besimple-ai/voice-code-bench \\
                --dataset=default --split=test --device=0 \\
                --batch_size=\"\$BATCH_SIZE\" \\
                --max_eval_samples=\"\$MAX_EVAL_SAMPLES\"
            mkdir -p \"/results/\$BUCKET_DIR\"
            cp \"results/\$MANIFEST_NAME\" \"/results/\$BUCKET_DIR/\"
        "
    # run_voice_code_bench_job.py returns success only for HF's COMPLETED status.
    JOB_RECORD="${LOCAL_RESULTS}/${model_folder}/grading-job.json" \
        bash "${SCRIPT_DIR}/submit_jobs_voice_code_bench_scoring.sh" "$manifest_path"
    if [[ "${DOWNLOAD_RESULTS:-1}" == "1" ]]; then
        hf buckets sync "hf://buckets/${RESULTS_BUCKET}/${bucket_dir}" \
            "${LOCAL_RESULTS}/${model_folder}"
    fi
}

pids=()
for model_cfg in "${MODEL_CONFIGS[@]}"; do
    read -r model_id batch_size <<< "$model_cfg"
    run_model "$model_id" "$batch_size" &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
exit "$failed"
