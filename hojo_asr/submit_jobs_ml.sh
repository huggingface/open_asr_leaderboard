#!/bin/bash
# Local script to submit HF Jobs for multilingual Hojo ASR evaluation.
# Evaluates on FLEURS, MCV (Mozilla Common Voice), and MLS (Multilingual LibriSpeech).
# This script is NOT pushed to the HF Space — it runs on your local machine.
# Usage: HF_TOKEN=hf_... bash submit_ml_jobs.sh

# Shared helpers (scripts/submit_utils.sh): local script/normalizer injection,
# ONLY_DATASETS filtering, and fetching this run's results.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/submit_utils.sh"

# ── Configuration ────────────────────────────────────────────────────────────
SPACE="${SPACE:-hf-audio/open-asr-leaderboard-hojo-asr}"
RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_multilingual}"
DATASET_PATH="${DATASET_PATH:-hf-audio/open-asr-leaderboard-multilingual-datasets}"
FLAVOR="${FLAVOR:-h200}"
ORG_NAME="${ORG_NAME:-}"

# Set USE_LOCAL_SCRIPT=1 to run your local run_eval_ml.py instead of the version
# committed to the Space (useful for iterating without pushing to the Space).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_SCRIPT_INJECT=$(local_script_inject "${SCRIPT_DIR}" run_eval_ml.py) || exit 1

# ── Models: "model_id batch_size" ───────────────────────────────────────────
MODEL_CONFIGS=(
    "HojoAI/Hojo-ASR-Multi-V1      32"
)

# ── Datasets/languages: "dataset language" (comment / uncomment to select) ──
# German, French, Italian, Spanish, Portuguese
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

# ── Submit one job per model/dataset/language combination ───────────────────
for model_cfg in "${MODEL_CONFIGS[@]}"; do
    read -r MODEL_ID BATCH_SIZE <<< "$model_cfg"
    # Sanitize model ID for use as a folder name (e.g. "nvidia/parakeet" -> "nvidia-parakeet")
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
            --secrets HF_TOKEN="$HF_TOKEN" \
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
                    --batch_size=${BATCH_SIZE} &&
                mkdir -p /results/${MODEL_FOLDER} &&
                cp results/*.jsonl /results/${MODEL_FOLDER}/
            " > /dev/null 2>&1 &
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

    RUN_RESULTS=$(python "${FETCH_RUN_RESULTS}" \
        --bucket "${RESULTS_BUCKET}" --model-folder "${MODEL_FOLDER}" \
        --local-dir "./results/${MODEL_FOLDER}" --since "${RUN_START}" \
        --expected "${#DATASET_CONFIGS[@]}")

    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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
score_results('${RUN_RESULTS}', '${MODEL_ID}', multilingual=True, language='${LANGUAGE}', families=['ml_${LANGUAGE}'], csv_only=True)
"
    done

done
