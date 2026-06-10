#!/bin/bash
# Local script to submit HF Jobs for ASR evaluation.
# This script is NOT pushed to the HF Space — it runs on your local machine.
# Usage: HF_TOKEN=hf_... bash submit_jobs.sh

# ── Configuration ────────────────────────────────────────────────────────────
SPACE="hf-audio/open-asr-leaderboard-transformers"
RESULTS_BUCKET="hf-audio/asr_leaderboard"      # HF bucket repo for saving results
DATASET_PATH="hf-audio/open-asr-leaderboard"
FLAVOR="a100-large"

# ── Models: "model_id batch_size voxpopuli_batch_size" ──────────────────────
# batch_size           → used for all datasets except voxpopuli
# voxpopuli_batch_size → voxpopuli has longer audio, so use a smaller value may be better
MODEL_CONFIGS=(
    "openai/whisper-large-v3-turbo      1024 1024"
    "openai/whisper-large-v3            128 128"
    "distil-whisper/distil-large-v3.5   1024 1024"
    # "openai/whisper-tiny.en           128 64"
    # "openai/whisper-small.en          128 64"
    # "openai/whisper-base.en           128 64"
    # "openai/whisper-medium.en         128 64"
    # "openai/whisper-large             64  32"
    # "openai/whisper-large-v2          64  32"
    # "distil-whisper/distil-medium.en  128 64"
    # "distil-whisper/distil-large-v2   64  32"
    # "distil-whisper/distil-large-v3   64  32"
)

# ── Datasets: "name split" (comment / uncomment to select) ──────────────────
DATASET_CONFIGS=(
    "voxpopuli test"
    "ami test"
    "earnings22 test"
    "gigaspeech test"
    "librispeech test.clean"
    "librispeech test.other"
    "spgispeech test"
)

# ── Submit one job per model/dataset combination ─────────────────────────────
for model_cfg in "${MODEL_CONFIGS[@]}"; do
    read -r MODEL_ID BATCH_SIZE VOXPOPULI_BATCH_SIZE <<< "$model_cfg"
    # Sanitize model ID for use as a folder name (e.g. "openai/whisper" -> "openai-whisper")
    MODEL_FOLDER="${MODEL_ID//\//-}"

    echo "████████████████████████████████████████████████████████████████████████████████"
    echo "  Evaluating: ${MODEL_ID}"
    echo "████████████████████████████████████████████████████████████████████████████████"

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET SPLIT <<< "$cfg"
        if [[ "$DATASET" == "voxpopuli" ]]; then
            EFFECTIVE_BATCH_SIZE="${VOXPOPULI_BATCH_SIZE}"
        else
            EFFECTIVE_BATCH_SIZE="${BATCH_SIZE}"
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
                    --max_eval_samples=-1 &&
                mkdir -p /results/${MODEL_FOLDER} &&
                cp results/*.jsonl /results/${MODEL_FOLDER}/
            " > /dev/null 2>&1 &    # suppress output and run in background
    done
    echo "For live status see: https://huggingface.co/settings/jobs"

    # Wait for all background job submissions to complete
    wait
    echo "All jobs finished."
    sleep 10  # allow time for the last results to be flushed to the bucket

    # Download results and score
    mkdir -p "./results/${MODEL_FOLDER}"

    hf buckets sync \
        "hf://buckets/${RESULTS_BUCKET}/${MODEL_FOLDER}" \
        "./results/${MODEL_FOLDER}" > /dev/null 2>&1

    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    PYTHONPATH="${REPO_ROOT}" python -c "
from normalizer.eval_utils import score_results
score_results('$(pwd)/results/${MODEL_FOLDER}', '${MODEL_ID}')
"

done
