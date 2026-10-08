#!/bin/bash
# Local script to submit HF Jobs for ASR evaluation.
# This script is NOT pushed to the HF Space — it runs on your local machine.
# Usage: HF_TOKEN=hf_... bash submit_jobs.sh
# The TheStage AI token for the compiled engines is set below (THESTAGE_AUTH_TOKEN).

# Shared helpers (scripts/submit_utils.sh): local script/normalizer injection,
# ONLY_DATASETS filtering, and fetching this run's results.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/submit_utils.sh"

# ── Configuration ────────────────────────────────────────────────────────────
SPACE="${SPACE:-hf-audio/open-asr-leaderboard-thewhisper}"
RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_h200}"      # HF bucket repo for saving results
DEFAULT_DATASET_PATH="${DEFAULT_DATASET_PATH:-hf-audio/open-asr-leaderboard}"
FLAVOR="${FLAVOR:-h200}"  # compiled engines are published for H200 (and H100, A100, L40S, RTX 4090/5090)
ORG_NAME="${ORG_NAME:-}"
# TheStage AI token for downloading the compiled engines; passed to every job as a secret.
export THESTAGE_AUTH_TOKEN="${THESTAGE_AUTH_TOKEN:-th_Har1cDb3EWLfNLtrfXaPwBc9cev6BtSNNUYiZd8r}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Set USE_LOCAL_SCRIPT=1 to run your local run_eval.py instead of the version
# committed to the Space (useful for iterating without pushing to the Space).
LOCAL_SCRIPT_INJECT=$(local_script_inject "${SCRIPT_DIR}" run_eval.py) || exit 1

# Set USE_LOCAL_NORMALIZER=1 to inject your local normalizer/ package into the
# job (so normalizer changes take effect without updating the HF Space).
LOCAL_NORMALIZER_INJECT=$(local_normalizer_inject)

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_ID="TheStageAI/thewhisper-large-v3-turbo"
REVISION="6592b6933656513345c5e7a65523a0de3e30d5a3"  # pins the configs and the compiled engines
MODE="XL"             # engine size
CHUNK_LENGTH=30       # engine input window, seconds
BATCH_SIZE=256        # largest batch of the H200 engines (built for batches 1-256; the H100 ones for 1-128)
MODEL_CONFIGS=("${MODEL_ID} ${BATCH_SIZE}")

# ── Datasets: "name split [dataset_path]" (comment / uncomment to select) ─────
# dataset_path defaults to $DEFAULT_DATASET_PATH when omitted.
# An entry that names its own repo (e.g. VoiceArena/Monsoon_en_IN_test) passes no
# config name: the first field is only a label for selection and result files.
DATASET_CONFIGS=(
    "ami_cleaned test"
    "gigaspeech_cleaned test"
    "voxpopuli_cleaned_aa test"
    "earnings22_cleaned_aa_chunked test ArtificialAnalysis/Earnings22-Cleaned-AA-chunked"
    "spgispeech test"
    "urgent2024 test"
    "urgent2024_clean test"
    "monsoon_en_in test VoiceArena/Monsoon_en_IN_test"
)
# Optional: restrict this run to specific datasets, matched against the first
# field of each DATASET_CONFIGS entry, e.g.:
#   ONLY_DATASETS="monsoon_en_in" bash <this script>
#   ONLY_DATASETS="librispeech spgispeech" bash <this script>
filter_only_datasets || exit 1


# ── Submit one job per model/dataset combination ─────────────────────────────
for model_cfg in "${MODEL_CONFIGS[@]}"; do
    read -r MODEL_ID BATCH_SIZE <<< "$model_cfg"
    # Sanitize model ID for use as a folder name (e.g. "openai/whisper" -> "openai-whisper")
    MODEL_FOLDER="${MODEL_ID//\//-}"

    echo "████████████████████████████████████████████████████████████████████████████████"
    echo "  Evaluating: ${MODEL_ID}"
    echo "████████████████████████████████████████████████████████████████████████████████"

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET SPLIT DATASET_PATH <<< "$cfg"
        if [[ -n "$DATASET_PATH" ]]; then
            # Entry names its own repo: pass no config. Such repos hold a single
            # (default) config, and the name here is just a label.
            DATASET_CONFIG=""
        else
            DATASET_PATH="$DEFAULT_DATASET_PATH"
            DATASET_CONFIG="$DATASET"
        fi
        echo "Submitting job: model=${MODEL_ID} dataset_path=${DATASET_PATH} dataset=${DATASET} split=${SPLIT} batch_size=${BATCH_SIZE}"

        NAMESPACE_ARG=""
        [ -n "$ORG_NAME" ] && NAMESPACE_ARG="--namespace ${ORG_NAME}"

        hf jobs run \
            --flavor "$FLAVOR" \
            --timeout 8h \
            --secrets HF_TOKEN="$HF_TOKEN" \
            --secrets THESTAGE_AUTH_TOKEN \
            ${NAMESPACE_ARG} \
            --volume "hf://buckets/${RESULTS_BUCKET}:/results" \
            "hf.co/spaces/${SPACE}" \
            bash -c "
                ${LOCAL_NORMALIZER_INJECT}
                ${LOCAL_SCRIPT_INJECT}
                PYTHONPATH=/app python run_eval.py \
                    --model_id=${MODEL_ID} \
                    --revision=${REVISION} \
                    --mode=${MODE} \
                    --chunk_length=${CHUNK_LENGTH} \
                    --dataset_path=${DATASET_PATH} \
                    --dataset=${DATASET_CONFIG} \
                    --split=${SPLIT} \
                    --device=0 \
                    --batch_size=${BATCH_SIZE} \
                    --max_eval_samples=-1 &&
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
    echo "All jobs finished."
    sleep 10  # allow time for the last results to be flushed to the bucket

    # Download results and score
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
