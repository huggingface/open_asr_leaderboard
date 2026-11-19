#!/bin/bash
# Local script to submit HF Jobs for ASR Consilium evaluation.
# Usage: HF_TOKEN=hf_... bash submit_jobs.sh

# ── Configuration ────────────────────────────────────────────────────────────
SPACE="${SPACE:-ZFTurbo/open-asr-leaderboard-asr-consilium}"
RESULTS_BUCKET="${RESULTS_BUCKET:-ZFTurbo/asr_leaderboard_h200}"
DATASET_PATH="${DATASET_PATH:-hf-audio/open-asr-leaderboard}"
FLAVOR="${FLAVOR:-h200}"
ORG_NAME="${ORG_NAME:-}"
USE_MODEL_CACHE="${USE_MODEL_CACHE:-false}"
MODEL_LIST="${MODEL_LIST:-nvidia/parakeet-tdt-0.6b-v2,nvidia/parakeet-tdt-0.6b-v3,Qwen/Qwen3-ASR-1.7B,nvidia/canary-qwen-2.5b,ibm-granite/granite-speech-3.3-8b,ibm-granite/granite-4.0-1b-speech,ibm-granite/granite-speech-4.1-2b,ZFTurbo/Phi-4-multimodal-instruct}"
WEIGHTS_LIST="${WEIGHTS_LIST:-4.5,4.2,8.4,9.8,8.7,3.5,8.9,9.4}"

USE_MODEL_CACHE="${USE_MODEL_CACHE:-false}"


# ── Models ────────────────────────────────────────────────────────────────────
# model_list=[
#     'nvidia/parakeet-tdt-0.6b-v2',
#     'nvidia/parakeet-tdt-0.6b-v3',
#     'Qwen/Qwen3-ASR-1.7B',
#     'nvidia/canary-qwen-2.5b',
#     'ibm-granite/granite-speech-3.3-8b',
#     'ibm-granite/granite-4.0-1b-speech',
#     'ibm-granite/granite-speech-4.1-2b',
#     'ZFTurbo/Phi-4-multimodal-instruct',
# ],
# weights=[4.5, 4.2, 8.4, 9.8, 8.7, 3.5, 8.9, 9.4],
# Total number of parameters: 22B

MODEL_CONFIGS=(
    "ZFTurbo/asr-consilium-2026-06"
)

# ── Datasets: "name split batch_size" ────────────────────────────────────────
DATASET_CONFIGS=(
    "voxpopuli test 64"
    "ami test 128"
    "earnings22 test 128"
    "gigaspeech test 128"
    "librispeech test.clean 128"
    "librispeech test.other 128"
    "spgispeech test 128"
)

DATASET_CONFIGS=(
    "earnings22 test 128"
)

# ── Submit one job per model/dataset combination ─────────────────────────────
for MODEL_ID in "${MODEL_CONFIGS[@]}"; do
    MODEL_FOLDER="${MODEL_ID//\//-}"

    echo "████████████████████████████████████████████████████████████████████████████████"
    echo "  Evaluating: ${MODEL_ID}"
    echo "████████████████████████████████████████████████████████████████████████████████"

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET SPLIT BATCH_SIZE <<< "$cfg"

        echo "Submitting job: model=${MODEL_ID} dataset=${DATASET} split=${SPLIT} batch_size=${BATCH_SIZE}"

        CACHE_ARGS=()
        if [ "$USE_MODEL_CACHE" = "true" ]; then
            CACHE_ARGS=(
                "--env" "HF_HOME=/cache/huggingface"
                "--env" "XDG_CACHE_HOME=/cache"
                "--volume" "hf://buckets/${RESULTS_BUCKET}:/cache"
            )
        fi

        NAMESPACE_ARG=""
        [ -n "$ORG_NAME" ] && NAMESPACE_ARG="--namespace ${ORG_NAME}"

        hf jobs run \
            --flavor "$FLAVOR" \
            --timeout 8h \
            --env HF_TOKEN="$HF_TOKEN" \
            --env HF_AUDIO_DECODER_BACKEND="soundfile" \
            ${NAMESPACE_ARG} \
            "${CACHE_ARGS[@]}" \
            --volume "hf://buckets/${RESULTS_BUCKET}:/results" \
            "hf.co/spaces/${SPACE}" \
            bash -c "
                PYTHONPATH=/app python run_eval.py \
                    --model_id=${MODEL_ID} \
                    --dataset_path=${DATASET_PATH} \
                    --dataset=${DATASET} \
                    --split=${SPLIT} \
                    --ensemble_models=${MODEL_LIST} \
                    --ensemble_weights=${WEIGHTS_LIST} \
                    --device=0 \
                    --batch_size=${BATCH_SIZE} \
                    --max_eval_samples=-1 &&
                mkdir -p /results/${MODEL_FOLDER} &&
                cp results/*.jsonl /results/${MODEL_FOLDER}/
            " > "job_error_${DATASET}.log" 2>&1 &
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