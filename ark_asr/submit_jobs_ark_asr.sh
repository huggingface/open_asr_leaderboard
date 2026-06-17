#!/bin/bash
# Local script to submit HF Jobs for ARK-ASR evaluation.
# Usage: RESULTS_BUCKET="<your-bucket>" HF_TOKEN=hf_... bash submit_jobs_ark_asr.sh

set -e

SPACE="${SPACE:-AutoArk-AI/open-asr-leaderboard-ark-asr}"
RESULTS_BUCKET="${RESULTS_BUCKET:-}"
DATASET_PATH="${DATASET_PATH:-hf-audio/open-asr-leaderboard}"
FLAVOR="${FLAVOR:-h200}"
ORG_NAME="${ORG_NAME:-}"

MODEL_CONFIGS=(
    "AutoArk-AI/ARK-ASR-0.6B"
)

DATASET_CONFIGS=(
    "voxpopuli test 64"
    "ami test 64"
    "earnings22 test 64"
    "gigaspeech test 64"
    "librispeech test.clean 64"
    "librispeech test.other 64"
    "spgispeech test 64"
)

if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN is required." >&2
    exit 1
fi

if [ -z "${RESULTS_BUCKET}" ]; then
    echo "RESULTS_BUCKET is required, for example: RESULTS_BUCKET=\"your-org/asr-results\"." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

for MODEL_ID in "${MODEL_CONFIGS[@]}"; do
    MODEL_FOLDER="${MODEL_ID//\//-}"

    echo "Evaluating: ${MODEL_ID}"

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET SPLIT BATCH_SIZE <<< "${cfg}"

        echo "Submitting job: model=${MODEL_ID} dataset=${DATASET} split=${SPLIT} batch_size=${BATCH_SIZE}"

        NAMESPACE_ARG=""
        [ -n "${ORG_NAME}" ] && NAMESPACE_ARG="--namespace ${ORG_NAME}"

        hf jobs run \
            --flavor "${FLAVOR}" \
            --timeout 8h \
            --env HF_TOKEN="${HF_TOKEN}" \
            --env HF_AUDIO_DECODER_BACKEND="soundfile" \
            ${NAMESPACE_ARG} \
            --volume "hf://buckets/${RESULTS_BUCKET}:/results" \
            "hf.co/spaces/${SPACE}" \
            bash -c "
                PYTHONPATH=/app python run_eval.py \
                    --model_id=${MODEL_ID} \
                    --dataset_path=${DATASET_PATH} \
                    --dataset=${DATASET} \
                    --split=${SPLIT} \
                    --device=0 \
                    --batch_size=${BATCH_SIZE} \
                    --max_eval_samples=-1 &&
                mkdir -p /results/${MODEL_FOLDER} &&
                cp results/*.jsonl /results/${MODEL_FOLDER}/
            " > /dev/null 2>&1 &
    done

    if [ -n "${ORG_NAME}" ]; then
        echo "For live status see: https://huggingface.co/organizations/${ORG_NAME}/settings/jobs"
    else
        echo "For live status see: https://huggingface.co/settings/jobs"
    fi

    wait
    echo "All jobs finished for ${MODEL_ID}."
    sleep 10

    mkdir -p "${SCRIPT_DIR}/results/${MODEL_FOLDER}"
    hf buckets sync \
        "hf://buckets/${RESULTS_BUCKET}/${MODEL_FOLDER}" \
        "${SCRIPT_DIR}/results/${MODEL_FOLDER}" > /dev/null 2>&1

    EXPECTED=${#DATASET_CONFIGS[@]}
    ACTUAL=$(find "${SCRIPT_DIR}/results/${MODEL_FOLDER}" -name "*.jsonl" | wc -l)
    if [[ "${ACTUAL}" -lt "${EXPECTED}" ]]; then
        echo "WARNING: expected ${EXPECTED} result files but only found ${ACTUAL}. Some jobs may not have finished yet."
    else
        echo "All ${ACTUAL} result files present."
    fi

    PYTHONPATH="${REPO_ROOT}" python -c "
from normalizer.eval_utils import score_results
score_results('${SCRIPT_DIR}/results/${MODEL_FOLDER}', '${MODEL_ID}')
"
done
