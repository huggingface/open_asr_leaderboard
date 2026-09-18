#!/bin/bash
# Submit HF Jobs to evaluate annotehrushi/whisper-large-v3-turbo-hindi
# on the Open ASR Leaderboard Hindi multilingual benchmark (Monsoon hi).
#
# This is a LoRA fine-tuned whisper-large-v3-turbo for Hindi ASR.
# Same architecture as the base model, uses the standard transformers Space.
#
# Usage:
#   RESULTS_BUCKET="<your-bucket>" HF_TOKEN=hf_... bash transformers/submit_jobs_whisper_hindi_ml.sh
#   ORG_NAME="<org>" RESULTS_BUCKET="<your-bucket>" HF_TOKEN=hf_... bash transformers/submit_jobs_whisper_hindi_ml.sh

SPACE="${SPACE:-hf-audio/open-asr-leaderboard-transformers}"
RESULTS_BUCKET="${RESULTS_BUCKET:-annotehrushi/asr-eval-results}"
MONSOON_DATASET_PATH="${MONSOON_DATASET_PATH:-VoiceArena/Monsoon_hi_test}"
FLAVOR="${FLAVOR:-h200}"
ORG_NAME="${ORG_NAME:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

USE_LOCAL_SCRIPT="${USE_LOCAL_SCRIPT:-1}"
LOCAL_SCRIPT_INJECT=""
if [[ "$USE_LOCAL_SCRIPT" == "1" ]]; then
    RUN_EVAL_B64=$(base64 < "${SCRIPT_DIR}/run_eval_ml.py" | tr -d '\n')
    LOCAL_SCRIPT_INJECT="echo '${RUN_EVAL_B64}' | base64 -d > /app/run_eval_ml.py &&"
fi

USE_LOCAL_NORMALIZER="${USE_LOCAL_NORMALIZER:-1}"
LOCAL_NORMALIZER_INJECT=""
if [[ "$USE_LOCAL_NORMALIZER" == "1" ]]; then
    NORMALIZER_B64=$(tar --exclude='__pycache__' --exclude='*.pyc' -czf - -C "${REPO_ROOT}" normalizer | base64 | tr -d '\n')
    LOCAL_NORMALIZER_INJECT="echo '${NORMALIZER_B64}' | base64 -d | tar -xzf - -C /app &&"
fi

MODEL_CONFIGS=(
    "annotehrushi/whisper-large-v3-turbo-hindi  64"
)

# "monsoon hi" uses the standalone VoiceArena/Monsoon_hi_test repo (no config);
DATASET_CONFIGS=(
    "monsoon hi"
)

for model_cfg in "${MODEL_CONFIGS[@]}"; do
    read -r MODEL_ID BATCH_SIZE <<< "$model_cfg"
    MODEL_FOLDER="${MODEL_ID//\//-}"

    echo "████████████████████████████████████████████████████████████████████████████████"
    echo "  Evaluating (Hindi): ${MODEL_ID}"
    echo "████████████████████████████████████████████████████████████████████████████████"

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET LANGUAGE <<< "$cfg"
        if [[ "$DATASET" == "monsoon" ]]; then
            JOB_DATASET="${MONSOON_DATASET_PATH}"
            CONFIG_ARG="--language=${LANGUAGE}"
            CONFIG_NAME="(none)"
        else
            echo "ERROR: unexpected dataset ${DATASET}" >&2
            exit 1
        fi
        echo "Submitting job: model=${MODEL_ID} dataset=${JOB_DATASET} config=${CONFIG_NAME} batch_size=${BATCH_SIZE}"

        NAMESPACE_ARG=""
        [ -n "$ORG_NAME" ] && NAMESPACE_ARG="--namespace ${ORG_NAME}"

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
                PYTHONPATH=/app python run_eval_ml.py \
                    --model_id=${MODEL_ID} \
                    --dataset=${JOB_DATASET} \
                    ${CONFIG_ARG} \
                    --split=test \
                    --device=0 \
                    --batch_size=${BATCH_SIZE} \
                    --max_eval_samples=-1 &&
                mkdir -p /results/${MODEL_FOLDER} &&
                cp results/*.jsonl /results/${MODEL_FOLDER}/
            " > /dev/null 2>&1 &
    done

    if [ -n "$ORG_NAME" ]; then
        echo "For live status: https://huggingface.co/organizations/${ORG_NAME}/settings/jobs"
    else
        echo "For live status: https://huggingface.co/settings/jobs"
    fi

    wait
    echo "All jobs finished."
    sleep 10

    mkdir -p "./results/${MODEL_FOLDER}"
    hf buckets sync \
        "hf://buckets/${RESULTS_BUCKET}/${MODEL_FOLDER}" \
        "./results/${MODEL_FOLDER}" > /dev/null 2>&1

    PYTHONPATH="${REPO_ROOT}" python3 -c "
from normalizer.eval_utils import score_results
score_results('$(pwd)/results/${MODEL_FOLDER}', '${MODEL_ID}', language='hi', families=['ml_hi'])
"
done
