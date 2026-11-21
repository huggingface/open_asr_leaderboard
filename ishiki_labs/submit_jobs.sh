#!/usr/bin/env bash
# Official Open ASR Leaderboard HF Jobs Submission Script for Ishiki Labs (`ishiki-labs/ishiki-asr-0.5`)
# Usage:
#   HF_TOKEN=hf_... bash ishiki_labs/submit_jobs.sh
#   ONLY_DATASETS="ami_cleaned spgispeech" HF_TOKEN=hf_... bash ishiki_labs/submit_jobs.sh

set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/submit_utils.sh"

MODEL_ID="${MODEL_ID:-ishiki-labs/ishiki-asr-0.5}"
MODEL_FOLDER="${MODEL_ID//\//-}"
SPACE="${SPACE:-hf-audio/open-asr-leaderboard-transformers}"
RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_h200}"
DEFAULT_DATASET_PATH="${DEFAULT_DATASET_PATH:-hf-audio/open-asr-leaderboard}"
FLAVOR="${FLAVOR:-h200}"
ORG_NAME="${ORG_NAME:-}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:--1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LOCAL_SCRIPT_INJECT=$(local_script_inject "${SCRIPT_DIR}" run_eval.py) || exit 1
LOCAL_NORMALIZER_INJECT=$(local_normalizer_inject --no-same-owner)

DATASET_CONFIGS=(
    "ami_cleaned ${DEFAULT_DATASET_PATH} ami_cleaned test default"
    "gigaspeech_cleaned ${DEFAULT_DATASET_PATH} gigaspeech_cleaned test default"
    "voxpopuli_cleaned_aa ${DEFAULT_DATASET_PATH} voxpopuli_cleaned_aa test default"
    "earnings22_cleaned_aa_chunked ArtificialAnalysis/Earnings22-Cleaned-AA-chunked - test default"
    "spgispeech ${DEFAULT_DATASET_PATH} spgispeech test spgispeech"
    "urgent2024 ${DEFAULT_DATASET_PATH} urgent2024 test default"
    "urgent2024_clean ${DEFAULT_DATASET_PATH} urgent2024_clean test default"
    "monsoon_en_in VoiceArena/Monsoon_en_IN_test - test monsoon_en"
)

filter_only_datasets || exit 1

echo "████████████████████████████████████████████████████████████████████████████████"
echo "  Evaluating: ${MODEL_ID}"
echo "████████████████████████████████████████████████████████████████████████████████"

for cfg in "${DATASET_CONFIGS[@]}"; do
    read -r LABEL DATASET_PATH DATASET_CONFIG SPLIT PROMPT_PROFILE <<< "${cfg}"
    [[ "${DATASET_CONFIG}" == "-" ]] && DATASET_CONFIG=""

    echo "Submitting job: label=${LABEL} dataset_path=${DATASET_PATH} dataset=${DATASET_CONFIG} split=${SPLIT} prompt=${PROMPT_PROFILE}"

    NAMESPACE_ARG=()
    [[ -n "${ORG_NAME}" ]] && NAMESPACE_ARG=(--namespace "${ORG_NAME}")

    hf jobs run \
        --name "ishiki-asr-${LABEL}" \
        --flavor "${FLAVOR}" \
        --timeout 8h \
        --secrets HF_TOKEN="${HF_TOKEN}" \
        "${NAMESPACE_ARG[@]}" \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" \
        "hf.co/spaces/${SPACE}" \
        bash -c "
            python -c 'from transformers import Nemotron3_5TranscribeForConditionalGeneration' 2>/dev/null || pip install -q --no-deps git+https://github.com/huggingface/transformers.git hojo-asr omegaconf antlr4-python3-runtime==4.9.3 openai-whisper tiktoken &&
            ${LOCAL_NORMALIZER_INJECT}
            ${LOCAL_SCRIPT_INJECT}
            PYTHONPATH=/app python /app/run_eval.py \
                --model_id=${MODEL_ID} \
                --checkpoint_path=${MODEL_ID} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET_CONFIG} \
                --split=${SPLIT} \
                --prompt_profile=${PROMPT_PROFILE} \
                --device=0 \
                --batch_size=${BATCH_SIZE} \
                --max_eval_samples=${MAX_EVAL_SAMPLES} &&
            mkdir -p /results/${MODEL_FOLDER} &&
            cp results/*.jsonl /results/${MODEL_FOLDER}/
        " &
done

if [[ -n "${ORG_NAME}" ]]; then
    echo "For live status see: https://huggingface.co/organizations/${ORG_NAME}/settings/jobs"
else
    echo "For live status see: https://huggingface.co/settings/jobs"
fi

wait
echo "All jobs finished for ${MODEL_ID}."
sleep 10

mkdir -p "./results/${MODEL_FOLDER}"
RUN_RESULTS=$(python "${FETCH_RUN_RESULTS}" \
    --bucket "${RESULTS_BUCKET}" \
    --model-folder "${MODEL_FOLDER}" \
    --local-dir "./results/${MODEL_FOLDER}" \
    --since "${RUN_START}" \
    --expected "${#DATASET_CONFIGS[@]}")

PYTHONPATH="${REPO_ROOT}" python -c "
from normalizer.eval_utils import score_results
score_results('${RUN_RESULTS}', '${MODEL_ID}')
"
