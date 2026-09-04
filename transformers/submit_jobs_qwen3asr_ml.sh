#!/bin/bash
# Local script to submit HF Jobs for multilingual Qwen3-ASR (Transformers-native
# hf checkpoints) evaluation.
# This script is NOT pushed to the HF Space — it runs on your local machine.
# Usage: HF_TOKEN=hf_... bash submit_jobs_qwen3asr_ml.sh
#        HF_TOKEN=hf_... ONLY_LANGUAGES="nl" bash submit_jobs_qwen3asr_ml.sh

# ── Configuration ────────────────────────────────────────────────────────────
SPACE="${SPACE:-hf-audio/open-asr-leaderboard-transformers}"
RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_multilingual}"
DATASET_PATH="${DATASET_PATH:-hf-audio/open-asr-leaderboard-multilingual-datasets}"
FLAVOR="${FLAVOR:-h200}"
ORG_NAME="${ORG_NAME:-}"
MAX_NEW_TOKENS=256

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
MODEL_CONFIGS=(
    "Qwen/Qwen3-ASR-0.6B-hf      64"
    "Qwen/Qwen3-ASR-1.7B-hf      64"
)

# ── Datasets/languages: "dataset language" (comment / uncomment to select) ──
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
)

# Optional: restrict this run to specific datasets and/or languages, matched
# against the first and second field of each DATASET_CONFIGS entry, e.g.:
#   ONLY_LANGUAGES="nl" bash <this script>
#   ONLY_DATASETS="fleurs mcv" ONLY_LANGUAGES="nl de" bash <this script>
if [[ -n "${ONLY_DATASETS:-}" || -n "${ONLY_LANGUAGES:-}" ]]; then
    _selected=()
    for _cfg in "${DATASET_CONFIGS[@]}"; do
        read -r _name _lang <<< "$_cfg"
        _keep_ds=1
        if [[ -n "${ONLY_DATASETS:-}" ]]; then
            _keep_ds=0
            for _want in ${ONLY_DATASETS}; do
                [[ "$_name" == "$_want" ]] && _keep_ds=1
            done
        fi
        _keep_lang=1
        if [[ -n "${ONLY_LANGUAGES:-}" ]]; then
            _keep_lang=0
            for _want in ${ONLY_LANGUAGES}; do
                [[ "$_lang" == "$_want" ]] && _keep_lang=1
            done
        fi
        [[ "$_keep_ds" == 1 && "$_keep_lang" == 1 ]] && _selected+=("$_cfg")
    done
    if [[ ${#_selected[@]} -eq 0 ]]; then
        echo "ERROR: ONLY_DATASETS='${ONLY_DATASETS:-}' ONLY_LANGUAGES='${ONLY_LANGUAGES:-}' matched no entry in DATASET_CONFIGS." >&2
        exit 1
    fi
    DATASET_CONFIGS=("${_selected[@]}")
    echo "Restricted to ${#DATASET_CONFIGS[@]} dataset/language combination(s): ${DATASET_CONFIGS[*]}"
fi

# ── Submit one job per model/dataset/language combination ───────────────────
for model_cfg in "${MODEL_CONFIGS[@]}"; do
    read -r MODEL_ID BATCH_SIZE <<< "$model_cfg"
    # Sanitize model ID for use as a folder name (e.g. "Qwen/Qwen3-ASR" -> "Qwen-Qwen3-ASR")
    MODEL_FOLDER="${MODEL_ID//\//-}"

    echo "████████████████████████████████████████████████████████████████████████████████"
    echo "  Evaluating: ${MODEL_ID}"
    echo "████████████████████████████████████████████████████████████████████████████████"

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET LANGUAGE <<< "$cfg"
        # --language is forced for every dataset so the model transcribes in the
        # known target language (consistent with the API models, which always
        # pass the language to the provider).
        JOB_DATASET="${DATASET_PATH}"
        CONFIG_NAME="${DATASET}_${LANGUAGE}"
        CONFIG_ARG="--config_name=${CONFIG_NAME} --language=${LANGUAGE}"
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
