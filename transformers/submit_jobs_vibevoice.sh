#!/bin/bash
# Local script to submit HF Jobs for VibeVoice ASR evaluation.
# Usage: HF_TOKEN=hf_... bash submit_jobs_vibevoice.sh

# ── Configuration ────────────────────────────────────────────────────────────
SPACE="hf-audio/open-asr-leaderboard-transformers"
RESULTS_BUCKET="hf-audio/asr_leaderboard"
DATASET_PATH="hf-audio/open-asr-leaderboard"
FLAVOR="a100-large"
MAX_NEW_TOKENS=225  # 30s audio at 24000 Hz with 3200 compression

# ── Models (comment / uncomment to select) ──────────────────────────────────
MODEL_IDs=(
    "microsoft/VibeVoice-ASR-HF"
)

# ── Datasets: "name split batch_size" ────────────────────────────────────────
DATASET_CONFIGS=(
    "voxpopuli test 32"
    "ami test 64"
    "earnings22 test 64"
    "gigaspeech test 64"
    "librispeech test.clean 64"
    "librispeech test.other 64"
    "spgispeech test 64"
)

# ── Submit one job per model/dataset combination ─────────────────────────────
for MODEL_ID in "${MODEL_IDs[@]}"; do
    MODEL_FOLDER="${MODEL_ID//\//-}"

    echo "████████████████████████████████████████████████████████████████████████████████"
    echo "  Evaluating: ${MODEL_ID}"
    echo "████████████████████████████████████████████████████████████████████████████████"

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET SPLIT EFFECTIVE_BATCH_SIZE <<< "$cfg"
        if [[ -z "${EFFECTIVE_BATCH_SIZE}" ]]; then
            echo "ERROR: batch_size missing for '${DATASET} ${SPLIT}' in DATASET_CONFIGS" >&2
            exit 1
        fi

        echo "Submitting job: model=${MODEL_ID} dataset=${DATASET} split=${SPLIT} batch_size=${EFFECTIVE_BATCH_SIZE}"

        hf jobs run \
            --flavor "$FLAVOR" \
            --timeout 8h \
            --env HF_TOKEN="$HF_TOKEN" \
            --volume "hf://buckets/${RESULTS_BUCKET}:/results" \
            "hf.co/spaces/${SPACE}" \
            bash -c "
                PYTHONPATH=/app python run_eval.py \
                    --model_id=${MODEL_ID} \
                    --dataset_path=${DATASET_PATH} \
                    --dataset=${DATASET} \
                    --split=${SPLIT} \
                    --device=0 \
                    --batch_size=${EFFECTIVE_BATCH_SIZE} \
                    --max_eval_samples=-1 \
                    --max_new_tokens=${MAX_NEW_TOKENS} &&
                mkdir -p /results/${MODEL_FOLDER} &&
                cp results/*.jsonl /results/${MODEL_FOLDER}/
            " > /dev/null 2>&1 &
    done
    echo "For live status see: https://huggingface.co/settings/jobs"

    wait
    echo "All jobs finished for ${MODEL_ID}."
    sleep 10  # allow time for the last results to be flushed to the bucket

    mkdir -p "./results/${MODEL_FOLDER}"
    hf buckets sync \
        "hf://buckets/${RESULTS_BUCKET}/${MODEL_FOLDER}" \
        "./results/${MODEL_FOLDER}" > /dev/null 2>&1

    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/." && pwd)"
    PYTHONPATH="${REPO_ROOT}" python -c "
from normalizer.eval_utils import score_results
score_results('$(pwd)/results/${MODEL_FOLDER}', '${MODEL_ID}')
"

done
