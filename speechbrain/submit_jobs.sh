#!/bin/bash
# Local script to submit HF Jobs for SpeechBrain ASR evaluation.
# Usage: HF_TOKEN=hf_... bash submit_jobs.sh

# ── Configuration ────────────────────────────────────────────────────────────
SPACE="hf-audio/open-asr-leaderboard-speechbrain"
RESULTS_BUCKET="hf-audio/asr_leaderboard"
DATASET_PATH="hf-audio/open-asr-leaderboard"
FLAVOR="a100-large"

# ── Models: "source speechbrain_class batch_size" ────────────────────────────
# EncoderASR: wav2vec2 models
# EncoderDecoderASR: conformer, crdnn, transformer models
MODEL_CONFIGS=(
    "speechbrain/asr-wav2vec2-librispeech EncoderASR 32"
    "speechbrain/asr-conformer-largescaleasr EncoderDecoderASR 32"
)

# ── Datasets: "name split" ────────────────────────────────────────────────────
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
    read -r SOURCE CLASS_NAME BATCH_SIZE <<< "$model_cfg"
    MODEL_FOLDER="${SOURCE//\//-}"

    echo "████████████████████████████████████████████████████████████████████████████████"
    echo "  Evaluating: ${SOURCE} (${CLASS_NAME})"
    echo "████████████████████████████████████████████████████████████████████████████████"

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET SPLIT <<< "$cfg"

        # Adjust batch size for specific model/dataset combinations
        ACTUAL_BATCH_SIZE=$BATCH_SIZE
        if [[ "$SOURCE" == "speechbrain/asr-conformer-largescaleasr" && "$DATASET" == "voxpopuli" ]]; then
            ACTUAL_BATCH_SIZE=8
        fi

        echo "Submitting job: source=${SOURCE} dataset=${DATASET} split=${SPLIT} batch_size=${ACTUAL_BATCH_SIZE}"

        hf jobs run \
            --flavor "$FLAVOR" \
            --timeout 8h \
            --env HF_TOKEN="$HF_TOKEN" \
            --volume "hf://buckets/${RESULTS_BUCKET}:/results" \
            "hf.co/spaces/${SPACE}" \
            bash -c "
                PYTHONPATH=/app python run_eval.py \
                    --source=${SOURCE} \
                    --speechbrain_pretrained_class_name=${CLASS_NAME} \
                    --dataset_path=${DATASET_PATH} \
                    --dataset=${DATASET} \
                    --split=${SPLIT} \
                    --device=0 \
                    --batch_size=${ACTUAL_BATCH_SIZE} \
                    --max_eval_samples=-1 &&
                mkdir -p /results/${MODEL_FOLDER} &&
                cp results/*.jsonl /results/${MODEL_FOLDER}/
            " > /dev/null 2>&1 &
    done
    echo "For live status see: https://huggingface.co/settings/jobs"

    wait
    echo "All jobs finished for ${SOURCE}."
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
