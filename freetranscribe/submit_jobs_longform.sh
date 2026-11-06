#!/bin/bash
# Local script to submit HF Jobs for English long-form evaluation of freetranscribe/parakeet-ft-v1.
# Evaluates on earnings21 and earnings22 (hf-audio/asr-leaderboard-longform) and the CORAAL subsets (bezzam/coraal).
# This script is NOT pushed to the HF Space — it runs on your local machine.
# Usage: HF_TOKEN=hf_... bash submit_jobs_longform.sh

# Shared helpers (scripts/submit_utils.sh): local script/normalizer injection,
# ONLY_DATASETS filtering, and fetching this run's results.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/submit_utils.sh"

# ── Configuration ────────────────────────────────────────────────────────────
# The NeMo image is used as is. The decoding package is installed when the job starts.
SPACE="${SPACE:-hf-audio/open-asr-leaderboard-nemo}"
RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_longform}"      # HF bucket repo for saving results
FLAVOR="${FLAVOR:-h200}"
ORG_NAME="${ORG_NAME:-}"
FREETRANSCRIBE_VERSION="0.1.0"   # pins the decoding: 256-frame local attention, pause cuts of at most 90 s, 5 s of overlap

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# This folder's run_eval_long.py always replaces the Space's copy in the job: the Space's own
# run_eval_long.py decodes each recording whole.
LOCAL_SCRIPT_INJECT=$(USE_LOCAL_SCRIPT=1 local_script_inject "${SCRIPT_DIR}" run_eval_long.py) || exit 1

# Set USE_LOCAL_NORMALIZER=1 to inject your local normalizer/ package into the
# job (so normalizer changes take effect without updating the HF Space).
LOCAL_NORMALIZER_INJECT=$(local_normalizer_inject)

# ── Models: "model_id batch_size" ────────────────────────────────────────────
# batch_size is the number of pieces decoded together; a piece is at most 100 s of audio.
MODEL_CONFIGS=(
    "freetranscribe/parakeet-ft-v1 32"
)

# ── Datasets: "dataset_path dataset" ─────────────────────────────────────────
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
# Optional: restrict this run to specific datasets, matched against the dataset
# name (second field), e.g.:  ONLY_DATASETS="earnings21 ATL" bash <this script>
filter_only_datasets 2 || exit 1


# ── Submit one job per model/dataset combination ─────────────────────────────
for model_cfg in "${MODEL_CONFIGS[@]}"; do
    read -r MODEL_ID BATCH_SIZE <<< "$model_cfg"
    MODEL_FOLDER="${MODEL_ID//\//-}"

    echo "████████████████████████████████████████████████████████████████████████████████"
    echo "  Evaluating: ${MODEL_ID}"
    echo "████████████████████████████████████████████████████████████████████████████████"

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET_PATH DATASET <<< "$cfg"

        echo "Submitting job: model=${MODEL_ID} dataset_path=${DATASET_PATH} dataset=${DATASET} batch_size=${BATCH_SIZE}"

        # --streaming: the recordings are up to about an hour long, and loading them in full overflows an Arrow block
        NAMESPACE_ARG=""
        [ -n "$ORG_NAME" ] && NAMESPACE_ARG="--namespace ${ORG_NAME}"

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
                pip install --no-cache-dir freetranscribe==${FREETRANSCRIBE_VERSION} &&
                PYTHONPATH=/app python run_eval_long.py \
                    --model_id=${MODEL_ID} \
                    --dataset_path=${DATASET_PATH} \
                    --dataset=${DATASET} \
                    --split=test \
                    --device=0 \
                    --batch_size=${BATCH_SIZE} \
                    --max_eval_samples=-1 \
                    --longform \
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
    wait
    echo "All jobs finished."
    sleep 10  # allow time for the last results to be flushed to the bucket

    # ── Fetch this run's results from bucket ─────────────────────────────────
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

    python "${REPO_ROOT}/scripts/score_bucket_results.py" \
        --family longform \
        --model_id "${MODEL_ID}" \
        --bucket "${RESULTS_BUCKET}" \
        --local_dir "$(pwd)/results" \
        --skip_sync

done
