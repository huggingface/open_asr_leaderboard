#!/usr/bin/env bash
set -euo pipefail

SPACE="${SPACE:-Cb1ock/open-asr-leaderboard-moss-transcribe-diarize}"
RESULTS_BUCKET="${RESULTS_BUCKET:-Cb1ock/open-asr-leaderboard-moss-transcribe-diarize-results}"
DATASET_PATH="${DATASET_PATH:-hf-audio/open-asr-leaderboard}"
FLAVOR="${FLAVOR:-h200}"
ORG_NAME="${ORG_NAME:-}"
MODEL_ID="${MODEL_ID:-OpenMOSS-Team/MOSS-Transcribe-Diarize}"
MODEL_REVISION="${MODEL_REVISION:-e5118b411bf5a77d7a90c4941066bec93c967312}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
BATCH_MAX_NEW_TOKENS="${BATCH_MAX_NEW_TOKENS:-512}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:--1}"
WARMUP_STEPS="${WARMUP_STEPS:-1}"
BATCH_SIZE="${BATCH_SIZE:-256}"
JOB_TIMEOUT="${JOB_TIMEOUT:-8h}"

DATASET_CONFIGS=(
    "ami_cleaned test"
    "earnings22 test"
    "gigaspeech_cleaned test"
    "librispeech test.clean"
    "librispeech test.other"
    "spgispeech test"
    "voxpopuli_cleaned_aa test"
)

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is required" >&2
    exit 1
fi

MODEL_FOLDER="${MODEL_ID//\//-}"
NAMESPACE_ARGS=()
if [[ -n "${ORG_NAME}" ]]; then
    NAMESPACE_ARGS=(--namespace "${ORG_NAME}")
fi

pids=()
for config in "${DATASET_CONFIGS[@]}"; do
    read -r dataset split <<< "${config}"
    echo "Submitting model=${MODEL_ID} dataset=${dataset} split=${split} batch_size=${BATCH_SIZE}"

    (
        hf jobs run \
            --flavor "${FLAVOR}" \
            --timeout "${JOB_TIMEOUT}" \
            --secrets HF_TOKEN \
            --env HF_AUDIO_DECODER_BACKEND=soundfile \
            "${NAMESPACE_ARGS[@]}" \
            --volume "hf://buckets/${RESULTS_BUCKET}:/results" \
            "hf.co/spaces/${SPACE}" \
            bash -c "
                set -euo pipefail
                PYTHONPATH=/app python run_eval.py \
                    --model_id='${MODEL_ID}' \
                    --model_revision='${MODEL_REVISION}' \
                    --dataset_path='${DATASET_PATH}' \
                    --dataset='${dataset}' \
                    --split='${split}' \
                    --device=0 \
                    --batch_size='${BATCH_SIZE}' \
                    --batch_max_new_tokens='${BATCH_MAX_NEW_TOKENS}' \
                    --max_new_tokens='${MAX_NEW_TOKENS}' \
                    --max_eval_samples='${MAX_EVAL_SAMPLES}' \
                    --warmup_steps='${WARMUP_STEPS}'
                mkdir -p '/results/${MODEL_FOLDER}'
                cp results/*.jsonl '/results/${MODEL_FOLDER}/'
            "
    ) &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        failed=1
    fi
done
if [[ "${failed}" -ne 0 ]]; then
    echo "One or more HF Jobs failed" >&2
    exit 1
fi

sleep 10

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
local_results="${repo_root}/results/${MODEL_FOLDER}"
mkdir -p "${local_results}"
hf buckets sync \
    "hf://buckets/${RESULTS_BUCKET}/${MODEL_FOLDER}" \
    "${local_results}"

actual="$(find "${local_results}" -name '*.jsonl' | wc -l)"
expected="${#DATASET_CONFIGS[@]}"
if [[ "${actual}" -ne "${expected}" ]]; then
    echo "Expected ${expected} result files, found ${actual}" >&2
    exit 1
fi

PYTHONPATH="${repo_root}" python - <<PY
from normalizer.eval_utils import score_results

score_results("${local_results}", "${MODEL_ID}")
PY

echo "All MOSS-Transcribe-Diarize public HF Jobs completed and scored."
