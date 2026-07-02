#!/bin/bash
# Local script to submit HF Jobs for SpeechBrain ASR evaluation.
# Usage: HF_TOKEN=hf_... bash submit_jobs.sh

# ── Configuration ────────────────────────────────────────────────────────────
SPACE="${SPACE:-hf-audio/open-asr-leaderboard-speechbrain}"
RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_h200}"
DATASET_PATH="${DATASET_PATH:-hf-audio/open-asr-leaderboard}"
FLAVOR="${FLAVOR:-h200}"
ORG_NAME="${ORG_NAME:-}"

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
    if [ -n "$ORG_NAME" ]; then
        echo "For live status see: https://huggingface.co/organizations/${ORG_NAME}/settings/jobs"
    else
        echo "For live status see: https://huggingface.co/settings/jobs"
    fi

    wait
    echo "All jobs finished for ${SOURCE}."
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
score_results('$(pwd)/results/${MODEL_FOLDER}', '${SOURCE}')
"

done
