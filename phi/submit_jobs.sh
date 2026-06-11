#!/bin/bash
# Local script to submit HF Jobs for Phi-4 multimodal ASR evaluation.
# Usage: HF_TOKEN=hf_... bash submit_jobs.sh

# ── Configuration ────────────────────────────────────────────────────────────
SPACE="${SPACE:-hf-audio/open-asr-leaderboard-phi4}"
RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_h200}"
DATASET_PATH="${DATASET_PATH:-hf-audio/open-asr-leaderboard}"
FLAVOR="${FLAVOR:-h200}"
ORG_NAME="${ORG_NAME:-}"
NUM_BEAMS=1
MAX_NEW_TOKENS=128
USER_PROMPT="Transcribe the audio clip into text."

# ── Models ────────────────────────────────────────────────────────────────────
MODEL_CONFIGS=(
    "microsoft/Phi-4-multimodal-instruct"
)

# ── Datasets: "name split batch_size" ────────────────────────────────────────
DATASET_CONFIGS=(
    "voxpopuli test 64"
    "ami test 128"
    "earnings22 test 128"
    "gigaspeech test 128"
    "librispeech test.clean 128"
    "librispeech test.other 128"
    "spgispeech test 128"
)

# ── Submit one job per model/dataset combination ─────────────────────────────
for MODEL_ID in "${MODEL_CONFIGS[@]}"; do
    MODEL_FOLDER="${MODEL_ID//\//-}"

    echo "████████████████████████████████████████████████████████████████████████████████"
    echo "  Evaluating: ${MODEL_ID}"
    echo "████████████████████████████████████████████████████████████████████████████████"

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET SPLIT BATCH_SIZE <<< "$cfg"

        echo "Submitting job: model=${MODEL_ID} dataset=${DATASET} split=${SPLIT} batch_size=${BATCH_SIZE}"

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
                PYTHONPATH=/app python run_eval.py \
                    --model_id=${MODEL_ID} \
                    --dataset_path=${DATASET_PATH} \
                    --dataset=${DATASET} \
                    --split=${SPLIT} \
                    --device=0 \
                    --batch_size=${BATCH_SIZE} \
                    --num_beams=${NUM_BEAMS} \
                    --max_eval_samples=-1 \
                    --max_new_tokens=${MAX_NEW_TOKENS} \
                    --user_prompt='${USER_PROMPT}' &&
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

    EXPECTED=${#DATASET_CONFIGS[@]}
    ACTUAL=$(find "./results/${MODEL_FOLDER}" -name "*.jsonl" | wc -l)
    if [[ "$ACTUAL" -lt "$EXPECTED" ]]; then
        echo "WARNING: expected ${EXPECTED} result files but only found ${ACTUAL}. Some jobs may not have finished yet."
    else
        echo "All ${ACTUAL} result files present."
    fi

    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/." && pwd)"
    PYTHONPATH="${REPO_ROOT}" python -c "
from normalizer.eval_utils import score_results
score_results('$(pwd)/results/${MODEL_FOLDER}', '${MODEL_ID}')
"

done