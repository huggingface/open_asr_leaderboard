#!/bin/bash
# Local script to submit HF Jobs for VoiceCodeBench (CTEM) evaluation of NeMo models.
# Usage: HF_TOKEN=hf_... bash submit_jobs_voice_code_bench.sh
#
# Inference runs on HF Jobs. CTEM scoring runs locally afterwards, since it needs
# the pinned voice_code_bench package and, for live-fill, OPENAI_API_KEY (see
# voice_code_bench.md). Results go to a dedicated bucket and, locally, to
# results/voice_code_bench/ so they never mix with WER result directories.
#
# Smoke test on two recordings:
#   MAX_EVAL_SAMPLES=2 HF_TOKEN=hf_... bash submit_jobs_voice_code_bench.sh

# ── Configuration ────────────────────────────────────────────────────────────
SPACE="${SPACE:-hf-audio/open-asr-leaderboard-nemo}"
RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_voice_code}"
ORG_NAME="${ORG_NAME:-}"
FLAVOR="${FLAVOR:-h200}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:--1}"
# replay (no API key, fails on cache misses) or live-fill (paid OpenAI requests).
VERIFIER_MODE="${VERIFIER_MODE:-live-fill}"

DATASET_PATH="besimple-ai/voice-code-bench"
DATASET_CONFIG="default"
SPLIT="test"
RESULTS_SUBDIR="voice_code_bench"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# The Space does not yet ship VoiceCodeBench support, so the local run_eval.py
# and normalizer/ package (which contains voice_code_bench.py) are injected.
USE_LOCAL_SCRIPT="${USE_LOCAL_SCRIPT:-1}"
LOCAL_SCRIPT_INJECT=""
if [[ "$USE_LOCAL_SCRIPT" == "1" ]]; then
    RUN_EVAL_B64=$(base64 -w0 "${SCRIPT_DIR}/run_eval.py")
    LOCAL_SCRIPT_INJECT="echo '${RUN_EVAL_B64}' | base64 -d > /app/run_eval.py &&"
fi

USE_LOCAL_NORMALIZER="${USE_LOCAL_NORMALIZER:-1}"
LOCAL_NORMALIZER_INJECT=""
if [[ "$USE_LOCAL_NORMALIZER" == "1" ]]; then
    NORMALIZER_B64=$(tar --exclude='__pycache__' --exclude='*.pyc' -czf - -C "${REPO_ROOT}" normalizer | base64 -w0)
    LOCAL_NORMALIZER_INJECT="echo '${NORMALIZER_B64}' | base64 -d | tar -xzf - -C /app &&"
fi

# ── Models: "model_id batch_size" ────────────────────────────────────────────
MODEL_CONFIGS=(
    "nvidia/parakeet-tdt-0.6b-v3 16"
    # "nvidia/parakeet-tdt-0.6b-v2 16"
    # "nvidia/parakeet-rnnt-1.1b 16"
    # "nvidia/parakeet-ctc-1.1b 16"
)

ALLOW_PARTIAL_ARG=""
[[ "$MAX_EVAL_SAMPLES" != "-1" ]] && ALLOW_PARTIAL_ARG="--allow-partial"

NAMESPACE_ARG=""
[ -n "$ORG_NAME" ] && NAMESPACE_ARG="--namespace ${ORG_NAME}"

# ── Submit one job per model ─────────────────────────────────────────────────
for model_cfg in "${MODEL_CONFIGS[@]}"; do
    read -r MODEL_ID BATCH_SIZE <<< "$model_cfg"
    MODEL_FOLDER="${MODEL_ID//\//-}"
    echo "Submitting job: model=${MODEL_ID} dataset_path=${DATASET_PATH} split=${SPLIT} batch_size=${BATCH_SIZE} max_eval_samples=${MAX_EVAL_SAMPLES}"

    hf jobs run \
        --flavor "$FLAVOR" \
        --timeout 8h \
        --env HF_TOKEN="$HF_TOKEN" \
        ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" \
        "hf.co/spaces/${SPACE}" \
        bash -c "
            ${LOCAL_NORMALIZER_INJECT}
            ${LOCAL_SCRIPT_INJECT}
            PYTHONPATH=/app python run_eval.py \
                --model_id=${MODEL_ID} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET_CONFIG} \
                --split=${SPLIT} \
                --device=0 \
                --batch_size=${BATCH_SIZE} \
                --max_eval_samples=${MAX_EVAL_SAMPLES} &&
            mkdir -p /results/${MODEL_FOLDER} &&
            cp results/*.jsonl /results/${MODEL_FOLDER}/
        " > /dev/null 2>&1 &
done
if [ -n "$ORG_NAME" ]; then
    echo "For live status see: https://huggingface.co/organizations/${ORG_NAME}/settings/jobs"
else
    echo "For live status see: https://huggingface.co/settings/jobs"
fi

wait
echo "All jobs finished."
sleep 10  # allow time for the last results to be flushed to the bucket

# ── Sync results and score CTEM locally ──────────────────────────────────────
LOCAL_RESULTS="${SCRIPT_DIR}/results/${RESULTS_SUBDIR}"
VERIFIER_CACHE="${LOCAL_RESULTS}/voice_code_bench.verifier-cache.json"

CAN_SCORE=1
if ! python -c "import voice_code_bench" > /dev/null 2>&1; then
    echo "WARNING: voice_code_bench package not installed; skipping CTEM scoring (see voice_code_bench.md)."
    CAN_SCORE=0
elif [[ "$VERIFIER_MODE" == "live-fill" && -z "${OPENAI_API_KEY:-}" ]]; then
    echo "WARNING: OPENAI_API_KEY not set for live-fill; skipping CTEM scoring."
    CAN_SCORE=0
fi

for model_cfg in "${MODEL_CONFIGS[@]}"; do
    read -r MODEL_ID _ <<< "$model_cfg"
    MODEL_FOLDER="${MODEL_ID//\//-}"

    mkdir -p "${LOCAL_RESULTS}/${MODEL_FOLDER}"
    hf buckets sync \
        "hf://buckets/${RESULTS_BUCKET}/${MODEL_FOLDER}" \
        "${LOCAL_RESULTS}/${MODEL_FOLDER}" > /dev/null 2>&1

    MANIFEST="${LOCAL_RESULTS}/${MODEL_FOLDER}/MODEL_${MODEL_FOLDER}_DATASET_${DATASET_PATH//\//-}_${DATASET_CONFIG}_${SPLIT}.jsonl"
    if [[ ! -f "$MANIFEST" ]]; then
        echo "WARNING: no manifest for ${MODEL_ID} at ${MANIFEST}. The job may have failed."
        continue
    fi

    SCORE_CMD=(python -m normalizer.voice_code_bench
        --manifest "$MANIFEST"
        --verifier-cache "$VERIFIER_CACHE"
        --verifier-mode "$VERIFIER_MODE"
        ${ALLOW_PARTIAL_ARG})
    if [[ "$CAN_SCORE" == "1" ]]; then
        echo "Scoring CTEM for ${MODEL_ID}"
        (cd "$REPO_ROOT" && PYTHONPATH="${REPO_ROOT}" "${SCORE_CMD[@]}")
    else
        echo "Score ${MODEL_ID} later from ${REPO_ROOT} with: ${SCORE_CMD[*]}"
    fi
done
