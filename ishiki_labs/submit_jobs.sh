#!/bin/bash
# Local script to submit HF Jobs for Ishiki-ASR-1.0 (`ishiki-labs/ishiki-asr-1.0`) English short-form evaluation.
# Usage:
#   HF_TOKEN=hf_... bash ishiki_labs/submit_jobs.sh
#   ONLY_DATASETS="ami_cleaned spgispeech" HF_TOKEN=hf_... bash ishiki_labs/submit_jobs.sh

# Shared helpers (scripts/submit_utils.sh): local script/normalizer injection,
# ONLY_DATASETS filtering, and fetching this run's results.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/submit_utils.sh"

# ── Configuration ────────────────────────────────────────────────────────────
SPACE="${SPACE:-hf-audio/open-asr-leaderboard-transformers}"
RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_h200}"
DEFAULT_DATASET_PATH="${DEFAULT_DATASET_PATH:-hf-audio/open-asr-leaderboard}"
FLAVOR="${FLAVOR:-h200}"
ORG_NAME="${ORG_NAME:-}"
JOB_TIMEOUT="${JOB_TIMEOUT:-4h}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Runs your local run_eval.py (the default; set USE_LOCAL_SCRIPT=0 to run the
# version committed to the Space instead).
LOCAL_SCRIPT_INJECT=$(local_script_inject "${SCRIPT_DIR}" run_eval.py) || exit 1

# Injects your local normalizer/ package into the job (the default; set
# USE_LOCAL_NORMALIZER=0 to use the Space's copy).
LOCAL_NORMALIZER_INJECT=$(local_normalizer_inject)

# ── Models: "model_id revision batch_size" ───────────────────────────────────
# Pins the immutable Hub commit SHA of the single-file 13.31B BF16 checkpoint (`model.safetensors`).
MODEL_CONFIGS=(
    "ishiki-labs/ishiki-asr-1.0 902d5c85ad46066c214895c0b8f2200c1800661a ${BATCH_SIZE:-256}"
)

# ── Datasets: "name split [dataset_path]" ─────────────────────────────────────
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
    "voice_code_bench test besimple-ai/voice-code-bench"
)
# Optional: restrict this run to specific datasets, matched against the first
# field of each DATASET_CONFIGS entry, e.g.:
#   ONLY_DATASETS="monsoon_en_in" bash <this script>
#   ONLY_DATASETS="ami_cleaned spgispeech" bash <this script>
filter_only_datasets || exit 1
vcb_init


# ── Submit one job per model/dataset combination ─────────────────────────────
for model_cfg in "${MODEL_CONFIGS[@]}"; do
    read -r MODEL_ID REVISION BATCH_SIZE <<< "$model_cfg"
    MODEL_FOLDER="${MODEL_ID//\//-}"

    echo "████████████████████████████████████████████████████████████████████████████████"
    echo "  Evaluating: ${MODEL_ID}"
    echo "████████████████████████████████████████████████████████████████████████████████"

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET SPLIT DATASET_PATH <<< "$cfg"
        if [[ -n "$DATASET_PATH" ]]; then
            DATASET_CONFIG=""
        else
            DATASET_PATH="$DEFAULT_DATASET_PATH"
            DATASET_CONFIG="$DATASET"
        fi
        vcb_dataset_config
        JOB_BATCH_SIZE="$BATCH_SIZE"
        [[ "$DATASET_PATH" == "$VCB_DATASET_PATH" ]] && JOB_BATCH_SIZE=32  # 35-123 s clips: smaller batches

        echo "Submitting job: model=${MODEL_ID}@${REVISION:0:7} dataset_path=${DATASET_PATH} dataset=${DATASET} split=${SPLIT} batch_size=${BATCH_SIZE}"

        NAMESPACE_ARG=""
        [ -n "$ORG_NAME" ] && NAMESPACE_ARG="--namespace ${ORG_NAME}"

        hf jobs run \
            --flavor "$FLAVOR" \
            --timeout "${JOB_TIMEOUT}" \
            --secrets HF_TOKEN="$HF_TOKEN" \
            --env TOKENIZERS_PARALLELISM=false \
            --env HF_AUDIO_DECODER_BACKEND=soundfile \
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
                    --device=0 \
                    --batch_size=${JOB_BATCH_SIZE} \
                    --dtype=bfloat16 \
                    --warmup_steps=2 \
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

    vcb_track_model "$MODEL_ID" "$MODEL_FOLDER"
done

vcb_grade
