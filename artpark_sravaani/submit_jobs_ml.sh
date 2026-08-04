#!/bin/bash
# Local script to submit HF Jobs for ARTPARK-IISc SraVaani-1.0 evaluation.
# SraVaani covers 63 Indian languages and dialects; of the leaderboard's
# multilingual sets only Hindi overlaps, via VoiceArena/Monsoon_hi_test.
# NOTE: the model repo is gated — HF_TOKEN must belong to an account that has
# accepted https://huggingface.co/ARTPARK-IISc/SraVaani-1.0 terms.
# This script is NOT pushed to the HF Space — it runs on your local machine.
# Usage: HF_TOKEN=hf_... bash submit_jobs_ml.sh

# ── Configuration ────────────────────────────────────────────────────────────
SPACE="${SPACE:-bezzam/open-asr-leaderboard-artpark-sravaani}"
RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_multilingual}"
DATASET_PATH="${DATASET_PATH:-hf-audio/open-asr-leaderboard-multilingual-datasets}"
MONSOON_DATASET_PATH="${MONSOON_DATASET_PATH:-VoiceArena/Monsoon_hi_test}"
FLAVOR="${FLAVOR:-h200}"
ORG_NAME="${ORG_NAME:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Set USE_LOCAL_SCRIPT=1 to run your local run_eval_ml.py instead of the version
# committed to the Space (useful for iterating without pushing to the Space).
USE_LOCAL_SCRIPT="${USE_LOCAL_SCRIPT:-1}"
LOCAL_SCRIPT_INJECT=""
if [[ "$USE_LOCAL_SCRIPT" == "1" ]]; then
    RUN_EVAL_B64=$(base64 -w0 "${SCRIPT_DIR}/run_eval_ml.py")
    LOCAL_SCRIPT_INJECT="echo '${RUN_EVAL_B64}' | base64 -d > /app/run_eval_ml.py &&"
fi

# Set USE_LOCAL_NORMALIZER=1 to inject your local normalizer/ package into the
# job (so normalizer changes take effect without updating the HF Space).
USE_LOCAL_NORMALIZER="${USE_LOCAL_NORMALIZER:-1}"
LOCAL_NORMALIZER_INJECT=""
if [[ "$USE_LOCAL_NORMALIZER" == "1" ]]; then
    NORMALIZER_B64=$(tar --exclude='__pycache__' --exclude='*.pyc' -czf - -C "${REPO_ROOT}" normalizer | base64 -w0)
    LOCAL_NORMALIZER_INJECT="echo '${NORMALIZER_B64}' | base64 -d | tar -xzf - -C /app &&"
fi

# ── Models: "model_id batch_size" ───────────────────────────────────────────
# The exported encoder is not length-masked, so heavy padding perturbs results
# slightly (batch_size=32 vs 1 measured at 0.35% WER drift). Use 1 for exact
# model-card fidelity at ~1/3 the throughput.
MODEL_CONFIGS=(
    "ARTPARK-IISc/SraVaani-1.0      64"
)

# ── Datasets/languages: "dataset language" (comment / uncomment to select) ──
# "monsoon hi" uses the standalone VoiceArena/Monsoon_hi_test repo (no config);
# all others would be configs of ${DATASET_PATH} (none currently cover the
# Indic languages SraVaani supports).
DATASET_CONFIGS=(
    "monsoon hi"
)

# ── Submit one job per model/dataset/language combination ───────────────────
for model_cfg in "${MODEL_CONFIGS[@]}"; do
    read -r MODEL_ID BATCH_SIZE <<< "$model_cfg"
    # Sanitize model ID for use as a folder name (e.g. "ARTPARK-IISc/SraVaani-1.0" -> "ARTPARK-IISc-SraVaani-1.0")
    MODEL_FOLDER="${MODEL_ID//\//-}"

    echo "████████████████████████████████████████████████████████████████████████████████"
    echo "  Evaluating: ${MODEL_ID}"
    echo "████████████████████████████████████████████████████████████████████████████████"

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET LANGUAGE <<< "$cfg"
        if [[ "$DATASET" == "monsoon" ]]; then
            # Standalone single-config dataset repo — no --config_name.
            JOB_DATASET="${MONSOON_DATASET_PATH}"
            CONFIG_ARG=""
            CONFIG_NAME="(none)"
        else
            JOB_DATASET="${DATASET_PATH}"
            CONFIG_NAME="${DATASET}_${LANGUAGE}"
            CONFIG_ARG="--config_name=${CONFIG_NAME}"
        fi
        echo "Submitting job: model=${MODEL_ID} dataset=${JOB_DATASET} config=${CONFIG_NAME} batch_size=${BATCH_SIZE}"

        NAMESPACE_ARG=""
        [ -n "$ORG_NAME" ] && NAMESPACE_ARG="--namespace ${ORG_NAME}"

        hf jobs run \
            --flavor "$FLAVOR" \
            --timeout 8h \
            --env HF_TOKEN="$HF_TOKEN" \
            --env HF_AUDIO_DECODER_BACKEND="soundfile" \
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
                    --language=${LANGUAGE} \
                    --split=test \
                    --device=0 \
                    --batch_size=${BATCH_SIZE} \
                    --max_eval_samples=-1 &&
                mkdir -p /results/${MODEL_FOLDER} &&
                cp results/*.jsonl /results/${MODEL_FOLDER}/
            " > /dev/null 2>&1 &    # suppress output and run in background
    done
    if [ -n "$ORG_NAME" ]; then
        echo "For live status see: https://huggingface.co/organizations/${ORG_NAME}/settings/jobs"
    else
        echo "For live status see: https://huggingface.co/settings/jobs"
    fi

    # Wait for all background job submissions to complete
    wait
    echo "All jobs finished."
    sleep 10  # allow time for the last results to be flushed to the bucket

    # Download results and score
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

    # Collect the set of languages actually evaluated (across all datasets)
    ALL_LANGUAGES=()
    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET LANGUAGE <<< "$cfg"
        if [[ ! " ${ALL_LANGUAGES[*]} " == *" ${LANGUAGE} "* ]]; then
            ALL_LANGUAGES+=("$LANGUAGE")
        fi
    done

    # Evaluate results: one call per language, so each is normalized with the
    # correct language-specific normalizer and only its "ml_<lang>" family
    # CSV block is printed.
    for LANGUAGE in "${ALL_LANGUAGES[@]}"; do
        PYTHONPATH="${REPO_ROOT}" python -c "
from normalizer.eval_utils import score_results
score_results('$(pwd)/results/${MODEL_FOLDER}', '${MODEL_ID}', multilingual=True, language='${LANGUAGE}', families=['ml_${LANGUAGE}'], csv_only=True)
"
    done

done
