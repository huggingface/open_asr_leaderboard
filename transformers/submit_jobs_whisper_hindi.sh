#!/bin/bash
# Submit HF Jobs to evaluate annotehrushi/whisper-large-v3-turbo-hindi
# on the Open ASR Leaderboard English short-form benchmarks.
#
# This is a LoRA fine-tuned whisper-large-v3-turbo for Hindi ASR.
# Same architecture as the base model, uses the standard transformers Space.
#
# Prerequisites:
#   1. hf auth login (with a WRITE token)
#   2. Create a storage bucket: hf buckets create <your-bucket>
#   3. Add billing credits (~$5 for a full run)
#
# Usage:
#   RESULTS_BUCKET="<your-bucket>" HF_TOKEN=hf_... bash transformers/submit_jobs_whisper_hindi.sh
#   ORG_NAME="<org>" RESULTS_BUCKET="<your-bucket>" HF_TOKEN=hf_... bash transformers/submit_jobs_whisper_hindi.sh

SPACE="${SPACE:-hf-audio/open-asr-leaderboard-transformers}"
RESULTS_BUCKET="${RESULTS_BUCKET:-annotehrushi/asr-eval-results}"
DEFAULT_DATASET_PATH="${DEFAULT_DATASET_PATH:-hf-audio/open-asr-leaderboard}"
FLAVOR="${FLAVOR:-h200}"
ORG_NAME="${ORG_NAME:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

USE_LOCAL_SCRIPT="${USE_LOCAL_SCRIPT:-1}"
LOCAL_SCRIPT_INJECT=""
if [[ "$USE_LOCAL_SCRIPT" == "1" ]]; then
    RUN_EVAL_B64=$(base64 < "${SCRIPT_DIR}/run_eval.py" | tr -d '\n')
    LOCAL_SCRIPT_INJECT="echo '${RUN_EVAL_B64}' | base64 -d > /app/run_eval.py &&"
fi

USE_LOCAL_NORMALIZER="${USE_LOCAL_NORMALIZER:-1}"
LOCAL_NORMALIZER_INJECT=""
if [[ "$USE_LOCAL_NORMALIZER" == "1" ]]; then
    NORMALIZER_B64=$(tar --exclude='__pycache__' --exclude='*.pyc' -czf - -C "${REPO_ROOT}" normalizer | base64 | tr -d '\n')
    LOCAL_NORMALIZER_INJECT="echo '${NORMALIZER_B64}' | base64 -d | tar -xzf - -C /app &&"
fi

MODEL_CONFIGS=(
    "annotehrushi/whisper-large-v3-turbo-hindi  1024"
)

DATASET_CONFIGS=(
    "ami_cleaned test"
    "gigaspeech_cleaned test"
    "voxpopuli_cleaned_aa test"
    "earnings22_cleaned_aa_chunked test ArtificialAnalysis/Earnings22-Cleaned-AA-chunked"
    "librispeech test.clean"
    "librispeech test.other"
    "spgispeech test"
    "monsoon_en_in test VoiceArena/Monsoon_en_IN_test"
)

if [[ -n "${ONLY_DATASETS:-}" ]]; then
    _selected=()
    for _cfg in "${DATASET_CONFIGS[@]}"; do
        read -r _name _ <<< "$_cfg"
        for _want in ${ONLY_DATASETS}; do
            if [[ "$_name" == "$_want" || "${_name##*/}" == "$_want" ]]; then
                _selected+=("$_cfg")
            fi
        done
    done
    if [[ ${#_selected[@]} -eq 0 ]]; then
        echo "ERROR: ONLY_DATASETS='${ONLY_DATASETS}' matched no active entry." >&2
        exit 1
    fi
    DATASET_CONFIGS=("${_selected[@]}")
fi

for model_cfg in "${MODEL_CONFIGS[@]}"; do
    read -r MODEL_ID BATCH_SIZE <<< "$model_cfg"
    MODEL_FOLDER="${MODEL_ID//\//-}"

    echo "████████████████████████████████████████████████████████████████████████████████"
    echo "  Evaluating: ${MODEL_ID}"
    echo "████████████████████████████████████████████████████████████████████████████████"

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET SPLIT DATASET_PATH <<< "$cfg"
        if [[ -n "$DATASET_PATH" ]]; then
            DATASET_CONFIG=""
        else
            DATASET_PATH="$DEFAULT_DATASET_PATH"
            DATASET_CONFIG="$DATASET"
        fi
        echo "Submitting job: model=${MODEL_ID} dataset=${DATASET} split=${SPLIT} batch_size=${BATCH_SIZE}"

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
                PYTHONPATH=/app python run_eval.py \
                    --model_id=${MODEL_ID} \
                    --dataset_path=${DATASET_PATH} \
                    --dataset=${DATASET_CONFIG} \
                    --split=${SPLIT} \
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

    EXPECTED=${#DATASET_CONFIGS[@]}
    ACTUAL=$(find "./results/${MODEL_FOLDER}" -name "*.jsonl" | wc -l)
    if [[ "$ACTUAL" -lt "$EXPECTED" ]]; then
        echo "WARNING: expected ${EXPECTED} result files but only found ${ACTUAL}."
    else
        echo "All ${ACTUAL} result files present."
    fi

    PYTHONPATH="${REPO_ROOT}" python3 -c "
from normalizer.eval_utils import score_results
score_results('$(pwd)/results/${MODEL_FOLDER}', '${MODEL_ID}')
"
done
