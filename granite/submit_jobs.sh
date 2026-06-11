#!/bin/bash
# Local script to submit HF Jobs for Granite ASR evaluation.
# Usage: HF_TOKEN=hf_... bash submit_jobs.sh

# ── Configuration ────────────────────────────────────────────────────────────
SPACE="${SPACE:-hf-audio/open-asr-leaderboard-granite}"
RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_h200}"
DATASET_PATH="${DATASET_PATH:-hf-audio/open-asr-leaderboard}"
FLAVOR="${FLAVOR:-h200}"
ORG_NAME="${ORG_NAME:-}"

# ── Models: "model_id type batch_size [additional_params]" ────────────────────
# Types: speculative, speculative_bpe, nar
MODEL_CONFIGS=(
    "ibm-granite/granite-4.0-1b-speech speculative 256"
    "ibm-granite/granite-speech-4.1-2b speculative_bpe 128"
    "ibm-granite/granite-speech-4.1-2b-nar nar 256"
)

# ── Datasets: "name split" ───────────────────────────────────────────────────
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
    read -r MODEL_ID MODEL_TYPE BATCH_SIZE <<< "$model_cfg"
    MODEL_FOLDER="${MODEL_ID//\//-}"

    echo "████████████████████████████████████████████████████████████████████████████████"
    echo "  Evaluating: ${MODEL_ID} (${MODEL_TYPE}, batch_size=${BATCH_SIZE})"
    echo "████████████████████████████████████████████████████████████████████████████████"

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET SPLIT <<< "$cfg"

        echo "Submitting job: model=${MODEL_ID} dataset=${DATASET} split=${SPLIT} type=${MODEL_TYPE}"

        # Build command based on model type
        if [[ "$MODEL_TYPE" == "speculative" ]]; then
            EVAL_SCRIPT="run_eval_speculative.py"
            EXTRA_ARGS="--num_beams=2 --max_new_tokens=200 --confidence_threshold=0.2 --ctc_threshold=0.7"
        elif [[ "$MODEL_TYPE" == "speculative_bpe" ]]; then
            EVAL_SCRIPT="run_eval_speculative_bpe.py"
            EXTRA_ARGS="--num_beams=2 --max_new_tokens=200 --confidence_threshold=0.4 --ctc_threshold=0.0"
        elif [[ "$MODEL_TYPE" == "nar" ]]; then
            EVAL_SCRIPT="run_eval_nar.py"
            EXTRA_ARGS=""
        else
            echo "ERROR: Unknown model type: ${MODEL_TYPE}" >&2
            exit 1
        fi

        NAMESPACE_ARG=""
        [ -n "$ORG_NAME" ] && NAMESPACE_ARG="--namespace ${ORG_NAME}"

        hf jobs run \
            --flavor "$FLAVOR" \
            --timeout 8h \
            --env HF_TOKEN="$HF_TOKEN" \
            --env PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
            --env PYTORCH_ALLOC_CONF="expandable_segments:True" \
            ${NAMESPACE_ARG} \
            --volume "hf://buckets/${RESULTS_BUCKET}:/results" \
            "hf.co/spaces/${SPACE}" \
            bash -c "
                PYTHONPATH=/app python ${EVAL_SCRIPT} \
                    --model_id=${MODEL_ID} \
                    --dataset_path=${DATASET_PATH} \
                    --dataset=${DATASET} \
                    --split=${SPLIT} \
                    --device=0 \
                    --batch_size=${BATCH_SIZE} \
                    --max_eval_samples=-1 \
                    ${EXTRA_ARGS} &&
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