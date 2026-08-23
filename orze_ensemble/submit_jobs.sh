#!/bin/bash
# Submit one reproducible Orze-ASR-3Way evaluation job per public dataset.
# Usage: HF_TOKEN=hf_... RESULTS_BUCKET=owner/bucket [SPACE=owner/space|IMAGE=registry/image] bash submit_jobs.sh

set -euo pipefail

SPACE="${SPACE:-erik-at-boson/open-asr-leaderboard-orze-ensemble}"
IMAGE="${IMAGE:-hf.co/spaces/${SPACE}}"
RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_h200}"
DEFAULT_DATASET_PATH="${DEFAULT_DATASET_PATH:-hf-audio/open-asr-leaderboard}"
FLAVOR="${FLAVOR:-h200}"
ORG_NAME="${ORG_NAME:-}"
MODEL_ID="bosonai/Orze-ASR-3Way"
MODEL_FOLDER="${MODEL_ID//\//-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_CONFIGS=(
    "ami_cleaned test 256 32 128"
    "gigaspeech_cleaned test 256 32 128"
    "voxpopuli_cleaned_aa test 256 32 128"
    "earnings22_cleaned_aa_chunked test 256 32 128 ArtificialAnalysis/Earnings22-Cleaned-AA-chunked"
    "librispeech test.clean 256 32 128"
    "librispeech test.other 256 32 128"
    "spgispeech test 256 32 128"
)

for cfg in "${DATASET_CONFIGS[@]}"; do
    read -r DATASET SPLIT QWEN_BATCH HOJO_BATCH MOSS_BATCH DATASET_PATH <<< "$cfg"
    DATASET_PATH="${DATASET_PATH:-$DEFAULT_DATASET_PATH}"
    NAMESPACE_ARGS=()
    [[ -n "$ORG_NAME" ]] && NAMESPACE_ARGS=(--namespace "$ORG_NAME")

    echo "Submitting ${MODEL_ID}: ${DATASET_PATH}/${DATASET}/${SPLIT}"
    hf jobs run \
        --flavor "$FLAVOR" \
        --timeout 24h \
        --secrets HF_TOKEN \
        --env HF_AUDIO_DECODER_BACKEND=soundfile \
        "${NAMESPACE_ARGS[@]}" \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results-bucket" \
        "$IMAGE" \
        bash -c "
            cd /app &&
            /opt/venvs/qwen/bin/python run_eval.py \
                --model_id=${MODEL_ID} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=0 \
                --qwen_batch_size=${QWEN_BATCH} \
                --hojo_batch_size=${HOJO_BATCH} \
                --moss_batch_size=${MOSS_BATCH} &&
            mkdir -p /results-bucket/${MODEL_FOLDER} &&
            cp results/*.jsonl /results-bucket/${MODEL_FOLDER}/
        " &
done

wait
echo "All Orze-ASR-3Way jobs finished."
sleep 10

mkdir -p "${REPO_ROOT}/results/${MODEL_FOLDER}"
hf buckets sync \
    "hf://buckets/${RESULTS_BUCKET}/${MODEL_FOLDER}" \
    "${REPO_ROOT}/results/${MODEL_FOLDER}"

PYTHONPATH="${REPO_ROOT}" python -c "
from normalizer.eval_utils import score_results
score_results('${REPO_ROOT}/results/${MODEL_FOLDER}', '${MODEL_ID}')
"
