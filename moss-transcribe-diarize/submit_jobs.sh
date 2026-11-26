#!/usr/bin/env bash

# Shared helpers (scripts/submit_utils.sh): local script/normalizer injection,
# ONLY_DATASETS filtering, and fetching this run's results.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/submit_utils.sh"


set -euo pipefail

SPACE="${SPACE:-hf-audio/open-asr-leaderboard-moss-transcribe-diarize}"
RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_h200}"
DEFAULT_DATASET_PATH="${DEFAULT_DATASET_PATH:-hf-audio/open-asr-leaderboard}"
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

# Datasets: "name split [dataset_path]"; dataset_path defaults to
# $DEFAULT_DATASET_PATH when omitted.
DATASET_CONFIGS=(
    "ami_cleaned test"
    "earnings22_cleaned_aa_chunked test ArtificialAnalysis/Earnings22-Cleaned-AA-chunked"
    "gigaspeech_cleaned test"
    "spgispeech test"
    "urgent2024 test"
    "urgent2024_clean test"
    "voxpopuli_cleaned_aa test"
    "monsoon_en_in test VoiceArena/Monsoon_en_IN_test"
    "voice_code_bench test besimple-ai/voice-code-bench"
)
# Optional: restrict this run to specific datasets, matched against the first
# field of each DATASET_CONFIGS entry, e.g.:
#   ONLY_DATASETS="monsoon_en_in" bash <this script>
#   ONLY_DATASETS="ami_cleaned spgispeech" bash <this script>
filter_only_datasets || exit 1
vcb_init


if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is required" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Set USE_LOCAL_SCRIPT=1 to run your local run_eval.py instead of the version
# committed to the Space (useful for iterating without pushing to the Space).
LOCAL_SCRIPT_INJECT=$(local_script_inject "${SCRIPT_DIR}" run_eval.py) || exit 1

# Set USE_LOCAL_NORMALIZER=1 to inject your local normalizer/ package into the
# job (so normalizer changes take effect without updating the HF Space).
LOCAL_NORMALIZER_INJECT=$(local_normalizer_inject)

MODEL_FOLDER="${MODEL_ID//\//-}"
NAMESPACE_ARGS=()
if [[ -n "${ORG_NAME}" ]]; then
    NAMESPACE_ARGS=(--namespace "${ORG_NAME}")
fi

pids=()
for config in "${DATASET_CONFIGS[@]}"; do
    read -r dataset split dataset_path <<< "${config}"
    if [[ -n "$dataset_path" ]]; then
        # Entry names its own repo: pass no config. Such repos hold a single
        # (default) config, and the name here is just a label.
        dataset_config=""
    else
        dataset_path="$DEFAULT_DATASET_PATH"
        dataset_config="$dataset"
    fi
    job_batch_size="${BATCH_SIZE}"
    job_batch_max_new_tokens="${BATCH_MAX_NEW_TOKENS}"
    DATASET_PATH="${dataset_path}"
    DATASET_CONFIG="${dataset_config}"
    vcb_dataset_config
    if [[ "${DATASET_PATH}" == "${VCB_DATASET_PATH}" ]]; then
        dataset="${DATASET_CONFIG}"  # VoiceCodeBench's one config, the name its manifest uses
        # Every VoiceCodeBench recording is longer than 30 s (35-123 s). The model's processor splits long audio
        # into encoder chunks itself, so no chunking is added here. Use a smaller batch than the short-form one
        # (long inputs are much larger) and lift the batched generation cap (512) so that the per-sample budget
        # (12 tokens per audio second + 64, i.e. <= ~1540 for 123 s) applies, rather than forcing an early EOS.
        job_batch_size=32
        job_batch_max_new_tokens=2048
    fi
    echo "Submitting model=${MODEL_ID} dataset_path=${dataset_path} dataset=${dataset} split=${split} batch_size=${job_batch_size}"

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
                ${LOCAL_NORMALIZER_INJECT}
                ${LOCAL_SCRIPT_INJECT}
                PYTHONPATH=/app python run_eval.py \
                    --model_id='${MODEL_ID}' \
                    --model_revision='${MODEL_REVISION}' \
                    --dataset_path='${dataset_path}' \
                    --dataset='${dataset}' \
                    --split='${split}' \
                    --device=0 \
                    --batch_size='${job_batch_size}' \
                    --batch_max_new_tokens='${job_batch_max_new_tokens}' \
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
RUN_RESULTS=$(python "${FETCH_RUN_RESULTS}" \
    --bucket "${RESULTS_BUCKET}" --model-folder "${MODEL_FOLDER}" \
    --local-dir "${local_results}" --since "${RUN_START}")

actual="$(find "${RUN_RESULTS}" -name '*.jsonl' | wc -l)"
expected="${#DATASET_CONFIGS[@]}"
if [[ "${actual}" -ne "${expected}" ]]; then
    echo "Expected ${expected} result files, found ${actual}" >&2
    exit 1
fi

PYTHONPATH="${repo_root}" python - <<PY
from normalizer.eval_utils import score_results

score_results("${RUN_RESULTS}", "${MODEL_ID}")
PY

echo "All MOSS-Transcribe-Diarize public HF Jobs completed and scored."

vcb_track_model "${MODEL_ID}" "${MODEL_FOLDER}"

# VoiceCodeBench CTEM, if it was evaluated: one grading job per model, then CTEM and WER are printed.
vcb_grade
