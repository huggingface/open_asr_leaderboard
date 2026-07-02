#!/bin/bash
# Local script to submit HF Jobs for wav2vec2-conformer ASR evaluation.
# Usage: HF_TOKEN=hf_... bash submit_jobs_wav2vec2_conformer.sh

# ── Configuration ────────────────────────────────────────────────────────────
SPACE="${SPACE:-hf-audio/open-asr-leaderboard-transformers}"
RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_h200}"
DATASET_PATH="${DATASET_PATH:-hf-audio/open-asr-leaderboard}"
FLAVOR="${FLAVOR:-h200}"
ORG_NAME="${ORG_NAME:-}"

# ── Models (comment / uncomment to select) ──────────────────────────────────
MODEL_IDs=(
    "facebook/wav2vec2-conformer-rel-pos-large-960h-ft"
    "facebook/wav2vec2-conformer-rope-large-960h-ft"
)

# ── Datasets: "name split batch_size" ────────────────────────────────────────
DATASET_CONFIGS=(
    "voxpopuli test 24"
    "ami test 24"
    "earnings22 test 24"
    "gigaspeech test 24"
    "librispeech test.clean 24"
    "librispeech test.other 24"
    "spgispeech test 24"
)

# ── Submit one job per model/dataset combination ─────────────────────────────
for MODEL_ID in "${MODEL_IDs[@]}"; do
    MODEL_FOLDER="${MODEL_ID//\//-}"

    echo "████████████████████████████████████████████████████████████████████████████████"
    echo "  Evaluating: ${MODEL_ID}"
    echo "████████████████████████████████████████████████████████████████████████████████"

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET SPLIT EFFECTIVE_BATCH_SIZE <<< "$cfg"
        if [[ -z "${EFFECTIVE_BATCH_SIZE}" ]]; then
            echo "ERROR: batch_size missing for '${DATASET} ${SPLIT}' in DATASET_CONFIGS" >&2
            exit 1
        fi

        echo "Submitting job: model=${MODEL_ID} dataset=${DATASET} split=${SPLIT} batch_size=${EFFECTIVE_BATCH_SIZE}"

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
                    --model_id=${MODEL_ID} \
                    --dataset_path=${DATASET_PATH} \
                    --dataset=${DATASET} \
                    --split=${SPLIT} \
                    --device=0 \
                    --batch_size=${EFFECTIVE_BATCH_SIZE} \
                    --max_eval_samples=-1 \
                    --attn_implementation=eager &&
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
