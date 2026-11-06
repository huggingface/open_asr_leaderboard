#!/bin/bash
# Local script to submit HF Jobs for English long-form evaluation of the NeMo SALM models (Canary-Qwen).
# Evaluates on earnings21 and earnings22 (hf-audio/asr-leaderboard-longform) and the CORAAL subsets (bezzam/coraal).
# This script is NOT pushed to the HF Space — it runs on your local machine.
# Usage: HF_TOKEN=hf_... bash submit_jobs_salm_longform.sh

# Shared helpers (scripts/submit_utils.sh): local script/normalizer injection,
# ONLY_DATASETS filtering, and fetching this run's results.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/submit_utils.sh"

# ── Configuration ────────────────────────────────────────────────────────────
SPACE="${SPACE:-hf-audio/open-asr-leaderboard-nemo}"
RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_longform}"      # HF bucket repo for saving results
FLAVOR="${FLAVOR:-h200}"
ORG_NAME="${ORG_NAME:-}"
CHUNK_LENGTH_S=30   # recordings are transcribed in 30-second chunks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Set USE_LOCAL_SCRIPT=1 to run your local run_eval_salm.py instead of the version
# committed to the Space (useful for iterating without pushing to the Space).
LOCAL_SCRIPT_INJECT=$(local_script_inject "${SCRIPT_DIR}" run_eval_salm.py) || exit 1

# Set USE_LOCAL_NORMALIZER=1 to inject your local normalizer/ package into the
# job (so normalizer changes take effect without updating the HF Space).
LOCAL_NORMALIZER_INJECT=$(local_normalizer_inject)

# ── Models: "model_id batch_size" (comment / uncomment to select) ────────────
# batch_size is the number of CHUNK_LENGTH_S chunks decoded together.
MODEL_CONFIGS=(
    "nvidia/canary-qwen-2.5b 192"
)

# ── Datasets: "dataset_path dataset" (comment / uncomment to select) ─────────
DATASET_CONFIGS=(
    "hf-audio/asr-leaderboard-longform earnings21"
    "hf-audio/asr-leaderboard-longform earnings22"
    "bezzam/coraal ATL"
    "bezzam/coraal DCA"
    "bezzam/coraal DCB"
    "bezzam/coraal DTA"
    "bezzam/coraal LES"
    "bezzam/coraal PRV"
    "bezzam/coraal ROC"
    "bezzam/coraal VLD"
)
# Optional: restrict this run to specific datasets, matched against the second
# field of each DATASET_CONFIGS entry, e.g.:
#   ONLY_DATASETS="earnings21 earnings22" bash <this script>
filter_only_datasets 2 || exit 1


# ── Submit one job per model/dataset combination ─────────────────────────────
for model_cfg in "${MODEL_CONFIGS[@]}"; do
    read -r MODEL_ID BATCH_SIZE <<< "$model_cfg"
    # Sanitize model ID for use as a folder name (e.g. "nvidia/canary-qwen-2.5b" -> "nvidia-canary-qwen-2.5b")
    MODEL_FOLDER="${MODEL_ID//\//-}"

    echo "████████████████████████████████████████████████████████████████████████████████"
    echo "  Evaluating: ${MODEL_ID}"
    echo "████████████████████████████████████████████████████████████████████████████████"

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET_PATH DATASET <<< "$cfg"
        echo "Submitting job: model=${MODEL_ID} dataset_path=${DATASET_PATH} dataset=${DATASET} batch_size=${BATCH_SIZE}"

        NAMESPACE_ARG=""
        [ -n "$ORG_NAME" ] && NAMESPACE_ARG="--namespace ${ORG_NAME}"

        # --streaming: the recordings are up to about an hour long, and loading them in full overflows an Arrow block
        # --longform: split each recording into CHUNK_LENGTH_S chunks and join their transcripts
        hf jobs run \
            --flavor "$FLAVOR" \
            --timeout 8h \
            --secrets HF_TOKEN="$HF_TOKEN" \
            ${NAMESPACE_ARG} \
            --volume "hf://buckets/${RESULTS_BUCKET}:/results" \
            "hf.co/spaces/${SPACE}" \
            bash -c "
                ${LOCAL_NORMALIZER_INJECT}
                ${LOCAL_SCRIPT_INJECT}
                PYTHONPATH=/app python run_eval_salm.py \
                    --model_id=${MODEL_ID} \
                    --dataset_path=${DATASET_PATH} \
                    --dataset=${DATASET} \
                    --split=test \
                    --device=0 \
                    --batch_size=${BATCH_SIZE} \
                    --max_eval_samples=-1 \
                    --longform \
                    --chunk_length_s=${CHUNK_LENGTH_S} \
                    --streaming &&
                mkdir -p /results/${MODEL_FOLDER} &&
                cp results/*.jsonl /results/${MODEL_FOLDER}/
            " > /dev/null 2>&1 &    # suppress output and run in background
    done
    if [ -n "$ORG_NAME" ]; then
        echo "For live status see: https://huggingface.co/organizations/${ORG_NAME}/settings/jobs"
    else
        echo "For live status see: https://huggingface.co/settings/jobs"
    fi

    # Wait for all background job submissions to complete
    wait
    echo "All jobs finished for ${MODEL_ID}."
    sleep 10  # allow time for the last results to be flushed to the bucket

    # Download results and score
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

    # Long-form average as on the leaderboard: mean of earnings21, earnings22 and the CORAAL average.
    python "${REPO_ROOT}/scripts/score_bucket_results.py" \
        --family longform \
        --model_id "${MODEL_ID}" \
        --bucket "${RESULTS_BUCKET}" \
        --local_dir "$(pwd)/results" \
        --skip_sync

done
