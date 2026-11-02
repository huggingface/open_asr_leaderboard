#!/bin/bash
# Local script to submit HF Jobs for multilingual Gemma 4 evaluation.
# Evaluates on FLEURS, MCV (Mozilla Common Voice), MLS (Multilingual LibriSpeech)
# and Monsoon (Hindi).
#
# Gemma 4 is a general multimodal LLM, not a dedicated ASR model: run_eval_ml.py
# prompts it for a transcription through the chat template, names the target
# language so it neither translates nor romanises a non-Latin script, forces
# greedy decoding, and suppresses its reasoning channel. See
# gemma4_transcribe_prompt() in run_eval_ml.py.
#
# This script is NOT pushed to the HF Space — it runs on your local machine.
# Usage: HF_TOKEN=hf_... bash submit_jobs_gemma4_ml.sh
#        HF_TOKEN=hf_... ONLY_LANGUAGES="hi" bash submit_jobs_gemma4_ml.sh

# Shared helpers (scripts/submit_utils.sh): local script/normalizer injection,
# ONLY_DATASETS filtering, and fetching this run's results.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/submit_utils.sh"

# ── Configuration ────────────────────────────────────────────────────────────
SPACE="${SPACE:-hf-audio/open-asr-leaderboard-transformers}"
RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_multilingual}"
DATASET_PATH="${DATASET_PATH:-hf-audio/open-asr-leaderboard-multilingual-datasets}"
MONSOON_DATASET_PATH="${MONSOON_DATASET_PATH:-VoiceArena/Monsoon_hi_test}"
FLAVOR="${FLAVOR:-h200}"
ORG_NAME="${ORG_NAME:-}"
MAX_NEW_TOKENS=256
BATCH_SIZE="${BATCH_SIZE:-64}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Set USE_LOCAL_SCRIPT=1 to run your local run_eval_ml.py instead of the version
# committed to the Space (useful for iterating without pushing to the Space).
LOCAL_SCRIPT_INJECT=$(local_script_inject "${SCRIPT_DIR}" run_eval_ml.py) || exit 1

# Set USE_LOCAL_NORMALIZER=1 to inject your local normalizer/ package into the
# job (so normalizer changes take effect without updating the HF Space).
LOCAL_NORMALIZER_INJECT=$(local_normalizer_inject)

# ── Models: "model_id batch_size" ───────────────────────────────────────────
MODEL_IDs=(
    # "google/gemma-4-12B-it"
    "google/gemma-4-E4B-it"
    # "google/gemma-4-E2B-it"
)

# ── Datasets/languages: "dataset language" (comment / uncomment to select) ──
# German, French, Italian, Spanish, Portuguese, Dutch, Hindi
# "monsoon hi" uses the standalone VoiceArena/Monsoon_hi_test repo (no config);
# all others are configs of ${DATASET_PATH}.
DATASET_CONFIGS=(
    "fleurs de"
    "fleurs fr"
    "fleurs it"
    "fleurs es"
    "fleurs pt"
    "fleurs nl"
    "mcv de"
    "mcv es"
    "mcv fr"
    "mcv it"
    "mcv nl"
    "mls es"
    "mls fr"
    "mls it"
    "mls pt"
    "mls nl"
    "monsoon hi"
)

# Optional: restrict this run to specific datasets and/or languages, matched
# against the first and second field of each DATASET_CONFIGS entry, e.g.:
#   ONLY_LANGUAGES="nl" bash <this script>
#   ONLY_DATASETS="fleurs mcv" ONLY_LANGUAGES="nl de" bash <this script>
filter_only_datasets_languages || exit 1

# ── Submit one job per model/dataset/language combination ───────────────────
for MODEL_ID in "${MODEL_IDs[@]}"; do
    # Sanitize model ID for use as a folder name (e.g. "google/gemma-4-12B-it" -> "google-gemma-4-12B-it")
    MODEL_FOLDER="${MODEL_ID//\//-}"

    echo "████████████████████████████████████████████████████████████████████████████████"
    echo "  Evaluating: ${MODEL_ID}"
    echo "████████████████████████████████████████████████████████████████████████████████"

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET LANGUAGE <<< "$cfg"
        if [[ "$DATASET" == "monsoon" ]]; then
            # Standalone single-config dataset repo — no --config_name.
            JOB_DATASET="${MONSOON_DATASET_PATH}"
            CONFIG_ARG="--language=${LANGUAGE}"
            CONFIG_NAME="(none)"
        else
            JOB_DATASET="${DATASET_PATH}"
            CONFIG_NAME="${DATASET}_${LANGUAGE}"
            CONFIG_ARG="--config_name=${CONFIG_NAME} --language=${LANGUAGE}"
        fi
        echo "Submitting job: model=${MODEL_ID} dataset=${JOB_DATASET} config=${CONFIG_NAME} batch_size=${BATCH_SIZE}"

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
                ${LOCAL_NORMALIZER_INJECT}
                ${LOCAL_SCRIPT_INJECT}
                PYTHONPATH=/app python run_eval_ml.py \
                    --model_id=${MODEL_ID} \
                    --dataset=${JOB_DATASET} \
                    ${CONFIG_ARG} \
                    --split=test \
                    --device=0 \
                    --batch_size=${BATCH_SIZE} \
                    --max_eval_samples=-1 \
                    --max_new_tokens=${MAX_NEW_TOKENS} &&
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
