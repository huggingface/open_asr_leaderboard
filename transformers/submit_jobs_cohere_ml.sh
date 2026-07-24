#!/bin/bash
# Local script to submit HF Jobs for multilingual Cohere ASR evaluation.
# Evaluates on FLEURS, MCV (Mozilla Common Voice), and MLS (Multilingual LibriSpeech).
# This script is NOT pushed to the HF Space — it runs on your local machine.
# Usage: HF_TOKEN=hf_... bash submit_jobs_cohere_ml.sh

SPACE="${SPACE:-hf-audio/open-asr-leaderboard-transformers}"
RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_multilingual}"
DATASET_PATH="${DATASET_PATH:-hf-audio/open-asr-leaderboard-multilingual-datasets}"
FLAVOR="${FLAVOR:-h200}"
ORG_NAME="${ORG_NAME:-}"
MAX_NEW_TOKENS=500

# Set USE_LOCAL_SCRIPT=1 to run your local run_eval_ml.py instead of the version
# committed to the Space (useful for iterating without pushing to the Space).
USE_LOCAL_SCRIPT="${USE_LOCAL_SCRIPT:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_SCRIPT_INJECT=""
if [[ "$USE_LOCAL_SCRIPT" == "1" ]]; then
    LOCAL_SCRIPT_B64=$(base64 -w0 "${SCRIPT_DIR}/run_eval_ml.py")
    LOCAL_SCRIPT_INJECT="echo '${LOCAL_SCRIPT_B64}' | base64 -d > /app/run_eval_ml.py &&"
fi

MODEL_CONFIGS=(
    "CohereLabs/cohere-transcribe-03-2026      64"
)

# Cohere ASR supports: en, es, fr, de, it, pt, nl, el, pl, ar, ja, ko, vi, zh
DATASET_CONFIGS=(
    "fleurs de"
    "fleurs fr"
    "fleurs it"
    "fleurs es"
    "fleurs pt"
    "mcv de"
    "mcv es"
    "mcv fr"
    "mcv it"
    "mls es"
    "mls fr"
    "mls it"
    "mls pt"
)

for model_cfg in "${MODEL_CONFIGS[@]}"; do
    read -r MODEL_ID BATCH_SIZE <<< "$model_cfg"
    MODEL_FOLDER="${MODEL_ID//\//-}"

    echo "████████████████████████████████████████████████████████████████████████████████"
    echo "  Evaluating: ${MODEL_ID}"
    echo "████████████████████████████████████████████████████████████████████████████████"

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET LANGUAGE <<< "$cfg"
        CONFIG_NAME="${DATASET}_${LANGUAGE}"
        echo "Submitting job: model=${MODEL_ID} config=${CONFIG_NAME} batch_size=${BATCH_SIZE}"

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
                ${LOCAL_SCRIPT_INJECT}
                PYTHONPATH=/app python run_eval_ml.py \
                    --model_id=${MODEL_ID} \
                    --dataset=${DATASET_PATH} \
                    --config_name=${CONFIG_NAME} \
                    --language=${LANGUAGE} \
                    --split=test \
                    --device=0 \
                    --batch_size=${BATCH_SIZE} \
                    --max_eval_samples=-1 \
                    --max_new_tokens=${MAX_NEW_TOKENS} &&
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
    sleep 10

    mkdir -p "./results/${MODEL_FOLDER}"
    hf buckets sync \
        "hf://buckets/${RESULTS_BUCKET}/${MODEL_FOLDER}" \
        "./results/${MODEL_FOLDER}" > /dev/null 2>&1

    EXPECTED=${#DATASET_CONFIGS[@]}
    ACTUAL=$(find "./results/${MODEL_FOLDER}" -name "*.jsonl" | wc -l)
    if [[ "$ACTUAL" -lt "$EXPECTED" ]]; then
        echo "WARNING: expected ${EXPECTED} result files but only found ${ACTUAL}. Some jobs may not have finished yet."
    else
        echo "All ${ACTUAL} result files present."
    fi

    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

    ALL_LANGUAGES=()
    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET LANGUAGE <<< "$cfg"
        if [[ ! " ${ALL_LANGUAGES[*]} " == *" ${LANGUAGE} "* ]]; then
            ALL_LANGUAGES+=("$LANGUAGE")
        fi
    done

    for LANGUAGE in "${ALL_LANGUAGES[@]}"; do
        PYTHONPATH="${REPO_ROOT}" python -c "
from normalizer.eval_utils import score_results
score_results('$(pwd)/results/${MODEL_FOLDER}', '${MODEL_ID}', multilingual=True, language='${LANGUAGE}', families=['ml_${LANGUAGE}'], csv_only=True)
"
    done

done
