#!/bin/bash
# Local script to submit HF Jobs for ABR ASR evaluation.
# Usage: HF_TOKEN=hf_... bash submit_jobs.sh

# Shared helpers (scripts/submit_utils.sh): local script/normalizer injection,
# ONLY_DATASETS filtering, and fetching this run's results.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/submit_utils.sh"

# ── Configuration ────────────────────────────────────────────────────────────
SPACE="${SPACE:-hf-audio/open-asr-leaderboard-abr}"
RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_h200}"
DEFAULT_DATASET_PATH="${DEFAULT_DATASET_PATH:-hf-audio/open-asr-leaderboard}"
FLAVOR="${FLAVOR:-h200}"
ORG_NAME="${ORG_NAME:-}"
BATCH_SIZE=2048
WARMUP_STEPS=5
SUBBATCH_SAMPLES=30000000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Set USE_LOCAL_SCRIPT=1 to run your local run_eval.py instead of the version
# committed to the Space (useful for iterating without pushing to the Space).
LOCAL_SCRIPT_INJECT=$(local_script_inject "${SCRIPT_DIR}" run_eval.py) || exit 1

# Set USE_LOCAL_NORMALIZER=1 to inject your local normalizer/ package into the
# job (so normalizer changes take effect without updating the HF Space).
LOCAL_NORMALIZER_INJECT=$(local_normalizer_inject)

# ── Models: "model_id revision" ──────────────────────────────────────────────
MODEL_CONFIGS=(
    "abr-ai/niagara-9m-batch.en 1521edf95a146d06e3c7c1ad18a7209a899bc570"
    "abr-ai/niagara-19m-batch.en d0276b85317389bc679d0206f60d88779cfbd15a"
    "abr-ai/niagara-38m-batch.en 7bfe48fb7fb065484419b1860c6b3d4e4f817c0c"
    "abr-ai/niagara-84m-batch.en ed93390475b146ad5412c668f680609a3ffca1b0"
)

# ── Datasets: "name split [dataset_path]" ─────────────────────────────────────
# dataset_path defaults to $DEFAULT_DATASET_PATH when omitted.
# An entry that names its own repo (e.g. VoiceArena/Monsoon_en_IN_test) passes no
# config name: the first field is only a label for selection and result files.
DATASET_CONFIGS=(
    "ami_cleaned test"
    "earnings22_cleaned_aa_chunked test ArtificialAnalysis/Earnings22-Cleaned-AA-chunked"
    "gigaspeech_cleaned test"
    "spgispeech test"
    "urgent2024 test"
    "urgent2024_clean test"
    "voxpopuli_cleaned_aa test"
    "monsoon_en_in test VoiceArena/Monsoon_en_IN_test"
)
# Optional: restrict this run to specific datasets, matched against the first
# field of each DATASET_CONFIGS entry, e.g.:
#   ONLY_DATASETS="monsoon_en_in" bash <this script>
#   ONLY_DATASETS="librispeech spgispeech" bash <this script>
filter_only_datasets || exit 1


# ── Submit one job per model/dataset combination ─────────────────────────────
for model_cfg in "${MODEL_CONFIGS[@]}"; do
    read -r MODEL_ID REVISION <<< "$model_cfg"
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

        echo "Submitting job: model=${MODEL_ID} dataset_path=${DATASET_PATH} dataset=${DATASET} split=${SPLIT}"

        NAMESPACE_ARG=""
        [ -n "$ORG_NAME" ] && NAMESPACE_ARG="--namespace ${ORG_NAME}"

        hf jobs run \
            --flavor "$FLAVOR" \
            --timeout 8h \
            --secrets HF_TOKEN \
            ${NAMESPACE_ARG} \
            --volume "hf://buckets/${RESULTS_BUCKET}:/results" \
            "hf.co/spaces/${SPACE}" \
            bash -c "
                ${LOCAL_NORMALIZER_INJECT}
                ${LOCAL_SCRIPT_INJECT}
                PYTHONPATH=/app python run_eval.py \
                    --model_id=${MODEL_ID} \
                    --revision=${REVISION} \
                    --dataset_path=${DATASET_PATH} \
                    --dataset=${DATASET_CONFIG} \
                    --split=${SPLIT} \
                    --batch_size=${BATCH_SIZE} \
                    --warmup_steps=${WARMUP_STEPS} \
                    --subbatch_samples=${SUBBATCH_SAMPLES} \
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
