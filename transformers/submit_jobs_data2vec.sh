#!/bin/bash
# Local script to submit HF Jobs for Data2Vec ASR evaluation.
# Usage: HF_TOKEN=hf_... bash submit_jobs_data2vec.sh

# Shared helpers (scripts/submit_utils.sh): local script/normalizer injection,
# ONLY_DATASETS filtering, and fetching this run's results.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/submit_utils.sh"

# ── Configuration ────────────────────────────────────────────────────────────
SPACE="${SPACE:-hf-audio/open-asr-leaderboard-transformers}"
RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_h200}"
DEFAULT_DATASET_PATH="${DEFAULT_DATASET_PATH:-hf-audio/open-asr-leaderboard}"
FLAVOR="${FLAVOR:-h200}"
ORG_NAME="${ORG_NAME:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Set USE_LOCAL_SCRIPT=1 to run your local run_eval.py instead of the version
# committed to the Space (useful for iterating without pushing to the Space).
LOCAL_SCRIPT_INJECT=$(local_script_inject "${SCRIPT_DIR}" run_eval.py) || exit 1

# Set USE_LOCAL_NORMALIZER=1 to inject your local normalizer/ package into the
# job (so normalizer changes take effect without updating the HF Space).
LOCAL_NORMALIZER_INJECT=$(local_normalizer_inject)

# ── Models (comment / uncomment to select) ──────────────────────────────────
MODEL_IDs=(
    # "facebook/data2vec-audio-large-960h"
    # "facebook/data2vec-audio-base-960h"
)

# ── Datasets: "name split batch_size [dataset_path]" ──────────────────────────
# dataset_path defaults to $DEFAULT_DATASET_PATH when omitted.
# An entry that names its own repo (e.g. VoiceArena/Monsoon_en_IN_test) passes no
# config name: the first field is only a label for selection and result files.
DATASET_CONFIGS=(
    "ami_cleaned test 8"
    "gigaspeech_cleaned test 8"
    "voxpopuli_cleaned_aa test 8"
    "earnings22_cleaned_aa_chunked test 8 ArtificialAnalysis/Earnings22-Cleaned-AA-chunked"
    "spgispeech test 8"
    "urgent2024 test 8"
    "urgent2024_clean test 8"
    "monsoon_en_in test 8 VoiceArena/Monsoon_en_IN_test"
)
# Optional: restrict this run to specific datasets, matched against the first
# field of each DATASET_CONFIGS entry, e.g.:
#   ONLY_DATASETS="monsoon_en_in" bash <this script>
#   ONLY_DATASETS="librispeech spgispeech" bash <this script>
filter_only_datasets || exit 1


# ── Submit one job per model/dataset combination ─────────────────────────────
for MODEL_ID in "${MODEL_IDs[@]}"; do
    MODEL_FOLDER="${MODEL_ID//\//-}"

    echo "████████████████████████████████████████████████████████████████████████████████"
    echo "  Evaluating: ${MODEL_ID}"
    echo "████████████████████████████████████████████████████████████████████████████████"

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET SPLIT EFFECTIVE_BATCH_SIZE DATASET_PATH <<< "$cfg"
        if [[ -n "$DATASET_PATH" ]]; then
            # Entry names its own repo: pass no config. Such repos hold a single
            # (default) config, and the name here is just a label.
            DATASET_CONFIG=""
        else
            DATASET_PATH="$DEFAULT_DATASET_PATH"
            DATASET_CONFIG="$DATASET"
        fi
        if [[ -z "${EFFECTIVE_BATCH_SIZE}" ]]; then
            echo "ERROR: batch_size missing for '${DATASET} ${SPLIT}' in DATASET_CONFIGS" >&2
            exit 1
        fi

        echo "Submitting job: model=${MODEL_ID} dataset_path=${DATASET_PATH} dataset=${DATASET} split=${SPLIT} batch_size=${EFFECTIVE_BATCH_SIZE}"

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
                PYTHONPATH=/app python run_eval.py \
                    --model_id=${MODEL_ID} \
                    --dataset_path=${DATASET_PATH} \
                    --dataset=${DATASET_CONFIG} \
                    --split=${SPLIT} \
                    --device=0 \
                    --batch_size=${EFFECTIVE_BATCH_SIZE} \
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
    echo "All jobs finished for ${MODEL_ID}."
    sleep 10  # allow time for the last results to be flushed to the bucket

    mkdir -p "./results/${MODEL_FOLDER}"
    RUN_RESULTS=$(python "${FETCH_RUN_RESULTS}" \
        --bucket "${RESULTS_BUCKET}" --model-folder "${MODEL_FOLDER}" \
        --local-dir "./results/${MODEL_FOLDER}" --since "${RUN_START}" \
        --expected "${#DATASET_CONFIGS[@]}")

    PYTHONPATH="${REPO_ROOT}" python -c "
from normalizer.eval_utils import score_results
score_results('${RUN_RESULTS}', '${MODEL_ID}')
"

done
