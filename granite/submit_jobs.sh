#!/bin/bash
# Local script to submit HF Jobs for Granite ASR evaluation.
# Usage: HF_TOKEN=hf_... bash submit_jobs.sh

# ── Configuration ────────────────────────────────────────────────────────────
SPACE="${SPACE:-hf-audio/open-asr-leaderboard-granite}"
RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_h200}"
DEFAULT_DATASET_PATH="${DEFAULT_DATASET_PATH:-hf-audio/open-asr-leaderboard}"
FLAVOR="${FLAVOR:-h200}"
ORG_NAME="${ORG_NAME:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Set USE_LOCAL_SCRIPT=1 to run your local eval script instead of the version
# committed to the Space (useful for iterating without pushing to the Space).
# The script is picked per model type below, so the injection is built there.
USE_LOCAL_SCRIPT="${USE_LOCAL_SCRIPT:-1}"

# Set USE_LOCAL_NORMALIZER=1 to inject your local normalizer/ package into the
# job (so normalizer changes take effect without updating the HF Space).
USE_LOCAL_NORMALIZER="${USE_LOCAL_NORMALIZER:-1}"
LOCAL_NORMALIZER_INJECT=""
if [[ "$USE_LOCAL_NORMALIZER" == "1" ]]; then
    NORMALIZER_B64=$(tar --exclude='__pycache__' --exclude='*.pyc' -czf - -C "${REPO_ROOT}" normalizer | base64 -w0)
    LOCAL_NORMALIZER_INJECT="echo '${NORMALIZER_B64}' | base64 -d | tar -xzf - -C /app &&"
fi

# ── Models: "model_id type batch_size [additional_params]" ────────────────────
# Types: speculative, speculative_bpe, nar
MODEL_CONFIGS=(
    "ibm-granite/granite-4.0-1b-speech speculative 256"
    "ibm-granite/granite-speech-4.1-2b speculative_bpe 128"
)

# ── Datasets: "name split [dataset_path]" ─────────────────────────────────────
# dataset_path defaults to $DEFAULT_DATASET_PATH when omitted.
DATASET_CONFIGS=(
    "ami_cleaned test"
    "gigaspeech_cleaned test"
    "voxpopuli_cleaned_aa test"
    "earnings22_cleaned_aa_chunked test ArtificialAnalysis/Earnings22-Cleaned-AA-chunked"
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
        read -r DATASET SPLIT DATASET_PATH <<< "$cfg"
        DATASET_PATH="${DATASET_PATH:-$DEFAULT_DATASET_PATH}"

        echo "Submitting job: model=${MODEL_ID} dataset_path=${DATASET_PATH} dataset=${DATASET} split=${SPLIT} type=${MODEL_TYPE}"

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

        LOCAL_SCRIPT_INJECT=""
        if [[ "$USE_LOCAL_SCRIPT" == "1" ]]; then
            RUN_EVAL_B64=$(base64 -w0 "${SCRIPT_DIR}/${EVAL_SCRIPT}")
            LOCAL_SCRIPT_INJECT="echo '${RUN_EVAL_B64}' | base64 -d > /app/${EVAL_SCRIPT} &&"
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
                ${LOCAL_NORMALIZER_INJECT}
                ${LOCAL_SCRIPT_INJECT}
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

    PYTHONPATH="${REPO_ROOT}" python -c "
from normalizer.eval_utils import score_results
score_results('$(pwd)/results/${MODEL_FOLDER}', '${MODEL_ID}')
"

done