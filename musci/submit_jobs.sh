#!/bin/bash
# Open ASR Leaderboard HF Jobs submission for Musci-ASR-2B.
# Usage: HF_TOKEN=hf_... bash submit_jobs.sh

set -euo pipefail

# Reuse the official Qwen3-ASR Space and the repository's shared result helpers.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/submit_utils.sh"
SPACE="${SPACE:-hf-audio/open-asr-leaderboard-qwen}"
RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_h200}"
DEFAULT_DATASET_PATH="${DEFAULT_DATASET_PATH:-hf-audio/open-asr-leaderboard}"
FLAVOR="${FLAVOR:-h200}"
ORG_NAME="${ORG_NAME:-}"
MODEL_CONFIGS=("Musci-research/Musci-ASR-2B")
DATASET_CONFIGS=(
    "ami_cleaned test 128"
    "earnings22_cleaned_aa_chunked test 128 ArtificialAnalysis/Earnings22-Cleaned-AA-chunked"
    "gigaspeech_cleaned test 128"
    "librispeech test.clean 128"
    "librispeech test.other 128"
    "spgispeech test 128"
    "voxpopuli_cleaned_aa test 128"
)

# The official Qwen Space already contains the compatible run_eval.py. Set
# USE_LOCAL_SCRIPT=1 only when testing a deliberately supplied local script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_SCRIPT_INJECT=""
if [[ "${USE_LOCAL_SCRIPT:-0}" == "1" ]]; then
    LOCAL_SCRIPT_INJECT="$(local_script_inject "${SCRIPT_DIR}" run_eval.py)"
fi
LOCAL_NORMALIZER_INJECT="$(local_normalizer_inject)"
filter_only_datasets || exit 1

for MODEL_ID in "${MODEL_CONFIGS[@]}"; do
    MODEL_FOLDER="${MODEL_ID//\//-}"
    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET SPLIT BATCH_SIZE DATASET_PATH <<< "$cfg"
        if [[ -n "${DATASET_PATH:-}" ]]; then
            DATASET_CONFIG=""
        else
            DATASET_PATH="$DEFAULT_DATASET_PATH"
            DATASET_CONFIG="$DATASET"
        fi
        NAMESPACE_ARG=""
        [[ -n "$ORG_NAME" ]] && NAMESPACE_ARG="--namespace ${ORG_NAME}"
        hf jobs run \
            --flavor "$FLAVOR" --timeout 8h \
            --secrets HF_TOKEN="$HF_TOKEN" \
            --env HF_AUDIO_DECODER_BACKEND=soundfile \
            ${NAMESPACE_ARG} \
            --volume "hf://buckets/${RESULTS_BUCKET}:/results" \
            "hf.co/spaces/${SPACE}" \
            bash -c "
                ${LOCAL_NORMALIZER_INJECT}
                ${LOCAL_SCRIPT_INJECT}
                PYTHONPATH=/app python run_eval.py \\
                    --model_id=${MODEL_ID} \\
                    --dataset_path=${DATASET_PATH} \\
                    --dataset=${DATASET_CONFIG} \\
                    --split=${SPLIT} \\
                    --device=0 --batch_size=${BATCH_SIZE} \\
                    --max_eval_samples=-1 --max_new_tokens=512 &&
                mkdir -p /results/${MODEL_FOLDER} &&
                cp results/*.jsonl /results/${MODEL_FOLDER}/
            " &
    done
    wait
    mkdir -p "./results/${MODEL_FOLDER}"
    RUN_RESULTS="$(python "${FETCH_RUN_RESULTS}" --bucket "${RESULTS_BUCKET}" \
        --model-folder "${MODEL_FOLDER}" --local-dir "./results/${MODEL_FOLDER}" \
        --since "${RUN_START}" --expected "${#DATASET_CONFIGS[@]}")"
    PYTHONPATH="${REPO_ROOT}" python -c "
from normalizer.eval_utils import score_results
score_results('${RUN_RESULTS}', '${MODEL_ID}')
"
done
