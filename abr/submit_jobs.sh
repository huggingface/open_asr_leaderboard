#!/bin/bash
# Local script to submit HF Jobs for ABR ASR evaluation.
# Usage: HF_TOKEN=hf_... bash submit_jobs.sh

# ── Configuration ────────────────────────────────────────────────────────────
SPACE="${SPACE:-hf-audio/open-asr-leaderboard-abr}"
RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_h200}"
DATASET_PATH="${DATASET_PATH:-hf-audio/open-asr-leaderboard}"
MONSOON_EN_IN_DATASET_PATH="${MONSOON_EN_IN_DATASET_PATH:-VoiceArena/Monsoon_en_IN_test}"
FLAVOR="${FLAVOR:-h200}"
ORG_NAME="${ORG_NAME:-}"
BATCH_SIZE=512
WARMUP_STEPS=5
SUBBATCH_SAMPLES=30000000

# ── Models: "model_id revision" ──────────────────────────────────────────────
MODEL_CONFIGS=(
    "abr-ai/niagara-19m-batch.en dab6545337495482f2fc05455432a7a05c88d3cc"
    "abr-ai/niagara-38m-batch.en 4f3ec18d377b1fd01e94d15dc9b9db0a8cd74bd2"
)

# ── Datasets: "name split" ────────────────────────────────────────────────────
DATASET_CONFIGS=(
    "ami_cleaned test"
    "earnings22 test"
    "gigaspeech_cleaned test"
    "librispeech test.clean"
    "librispeech test.other"
    "spgispeech test"
    "voxpopuli_cleaned_aa test"
    "monsoon_en_in test"
)
# Optional: restrict this run to specific datasets, matched against the first
# field of each DATASET_CONFIGS entry (a dataset name here, a dataset path in the
# private scripts). A repo basename also matches, so both "HF_English_Private_Set"
# and "hf-audio/HF_English_Private_Set" work. Lets you evaluate a subset without
# commenting out the rest of the list, e.g.:
#   ONLY_DATASETS="monsoon_en_in" bash <this script>
#   ONLY_DATASETS="librispeech spgispeech" bash <this script>
if [[ -n "${ONLY_DATASETS:-}" ]]; then
    _selected=()
    if [[ ${#DATASET_CONFIGS[@]} -gt 0 ]]; then
        for _cfg in "${DATASET_CONFIGS[@]}"; do
            read -r _name _ <<< "$_cfg"
            for _want in ${ONLY_DATASETS}; do
                if [[ "$_name" == "$_want" || "${_name##*/}" == "$_want" ]]; then
                    _selected+=("$_cfg")
                fi
            done
        done
    fi
    if [[ ${#_selected[@]} -eq 0 ]]; then
        echo "ERROR: ONLY_DATASETS='${ONLY_DATASETS}' matched no active entry in DATASET_CONFIGS." >&2
        exit 1
    fi
    DATASET_CONFIGS=("${_selected[@]}")
fi


# ── Submit one job per model/dataset combination ─────────────────────────────
for model_cfg in "${MODEL_CONFIGS[@]}"; do
    read -r MODEL_ID REVISION <<< "$model_cfg"
    MODEL_FOLDER="${MODEL_ID//\//-}"

    echo "████████████████████████████████████████████████████████████████████████████████"
    echo "  Evaluating: ${MODEL_ID}"
    echo "████████████████████████████████████████████████████████████████████████████████"

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET SPLIT <<< "$cfg"
        if [[ "$DATASET" == "monsoon_en_in" ]]; then
            # Standalone single-config repo: pass an empty --dataset, which
            # resolves to the repo's default config.
            JOB_DATASET_PATH="${MONSOON_EN_IN_DATASET_PATH}"
            DATASET_NAME=""
        else
            JOB_DATASET_PATH="${DATASET_PATH}"
            DATASET_NAME="${DATASET}"
        fi

        echo "Submitting job: model=${MODEL_ID} dataset=${DATASET} split=${SPLIT}"

        NAMESPACE_ARG=""
        [ -n "$ORG_NAME" ] && NAMESPACE_ARG="--namespace ${ORG_NAME}"

        hf jobs run \
            --flavor "$FLAVOR" \
            --timeout 8h \
            --env HF_TOKEN="$HF_TOKEN" \
            --env HF_AUDIO_DECODER_BACKEND=soundfile \
            ${NAMESPACE_ARG} \
            --volume "hf://buckets/${RESULTS_BUCKET}:/results" \
            "hf.co/spaces/${SPACE}" \
            bash -c "
                PYTHONPATH=/app python run_eval.py \
                    --model_id=${MODEL_ID} \
                    --revision=${REVISION} \
                    --dataset_path=${JOB_DATASET_PATH} \
                    --dataset=${DATASET_NAME} \
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
