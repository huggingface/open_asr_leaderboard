#!/usr/bin/env bash
# Local script to submit HF Jobs for multilingual MOSS-Transcribe-Diarize evaluation.
# Usage: HF_TOKEN=hf_... bash submit_jobs_ml.sh
#        HF_TOKEN=hf_... ONLY_LANGUAGES="nl" bash submit_jobs_ml.sh
set -euo pipefail

SPACE="${SPACE:-hf-audio/open-asr-leaderboard-moss-transcribe-diarize}"
RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_multilingual}"
DATASET_PATH="${DATASET_PATH:-hf-audio/open-asr-leaderboard-multilingual-datasets}"
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
    "fleurs de"
    "fleurs fr"
    "fleurs it"
    "fleurs es"
    "fleurs pt"
    "fleurs nl"
    "mcv de"
    "mcv es"
    "mcv fr"
    "mcv it"
    "mcv nl"
    "mls es"
    "mls fr"
    "mls it"
    "mls pt"
    "mls nl"
)

# Optional smoke-test override: DATASETS="fleurs:pt mcv:de" bash ...
if [[ -n "${DATASETS:-}" ]]; then
    DATASET_CONFIGS=()
    for pair in ${DATASETS}; do
        DATASET_CONFIGS+=("${pair/:/ }")
    done
fi

# Optional: restrict this run to specific datasets and/or languages, matched
# against the first and second field of each DATASET_CONFIGS entry, e.g.:
#   ONLY_LANGUAGES="nl" bash <this script>
#   ONLY_DATASETS="fleurs mcv" ONLY_LANGUAGES="nl de" bash <this script>
if [[ -n "${ONLY_DATASETS:-}" || -n "${ONLY_LANGUAGES:-}" ]]; then
    _selected=()
    for _cfg in "${DATASET_CONFIGS[@]}"; do
        read -r _name _lang <<< "$_cfg"
        _keep_ds=1
        if [[ -n "${ONLY_DATASETS:-}" ]]; then
            _keep_ds=0
            for _want in ${ONLY_DATASETS}; do
                [[ "$_name" == "$_want" ]] && _keep_ds=1
            done
        fi
        _keep_lang=1
        if [[ -n "${ONLY_LANGUAGES:-}" ]]; then
            _keep_lang=0
            for _want in ${ONLY_LANGUAGES}; do
                [[ "$_lang" == "$_want" ]] && _keep_lang=1
            done
        fi
        [[ "$_keep_ds" == 1 && "$_keep_lang" == 1 ]] && _selected+=("$_cfg")
    done
    if [[ ${#_selected[@]} -eq 0 ]]; then
        echo "ERROR: ONLY_DATASETS='${ONLY_DATASETS:-}' ONLY_LANGUAGES='${ONLY_LANGUAGES:-}' matched no entry in DATASET_CONFIGS." >&2
        exit 1
    fi
    DATASET_CONFIGS=("${_selected[@]}")
    echo "Restricted to ${#DATASET_CONFIGS[@]} dataset/language combination(s): ${DATASET_CONFIGS[*]}"
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is required" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_FOLDER="${MODEL_ID//\//-}"

# Use the local scripts while iterating; set USE_LOCAL_SCRIPT=0 to use the
# versions committed to the evaluator Space.
USE_LOCAL_SCRIPT="${USE_LOCAL_SCRIPT:-1}"
LOCAL_SCRIPT_INJECT=""
if [[ "${USE_LOCAL_SCRIPT}" == "1" ]]; then
    RUN_EVAL_B64="$(base64 -w0 "${SCRIPT_DIR}/run_eval.py")"
    RUN_EVAL_ML_B64="$(base64 -w0 "${SCRIPT_DIR}/run_eval_ml.py")"
    LOCAL_SCRIPT_INJECT="echo '${RUN_EVAL_B64}' | base64 -d > /app/run_eval.py && echo '${RUN_EVAL_ML_B64}' | base64 -d > /app/run_eval_ml.py &&"
fi

# Keep scoring identical to the local repository while iterating.
USE_LOCAL_NORMALIZER="${USE_LOCAL_NORMALIZER:-1}"
LOCAL_NORMALIZER_INJECT=""
if [[ "${USE_LOCAL_NORMALIZER}" == "1" ]]; then
    NORMALIZER_B64="$(tar --exclude='__pycache__' --exclude='*.pyc' -czf - -C "${REPO_ROOT}" normalizer | base64 -w0)"
    LOCAL_NORMALIZER_INJECT="echo '${NORMALIZER_B64}' | base64 -d | tar -xzf - -C /app &&"
fi

NAMESPACE_ARGS=()
if [[ -n "${ORG_NAME}" ]]; then
    NAMESPACE_ARGS=(--namespace "${ORG_NAME}")
fi

pids=()
for config in "${DATASET_CONFIGS[@]}"; do
    read -r dataset language <<< "${config}"
    config_name="${dataset}_${language}"
    echo "Submitting model=${MODEL_ID} config=${config_name} batch_size=${BATCH_SIZE}"

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
                PYTHONPATH=/app python run_eval_ml.py \
                    --model_id='${MODEL_ID}' \
                    --model_revision='${MODEL_REVISION}' \
                    --dataset='${DATASET_PATH}' \
                    --config_name='${config_name}' \
                    --language='${language}' \
                    --split='test' \
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

local_results="${REPO_ROOT}/results/${MODEL_FOLDER}"
mkdir -p "${local_results}"
hf buckets sync \
    "hf://buckets/${RESULTS_BUCKET}/${MODEL_FOLDER}" \
    "${local_results}"

missing=0
for config in "${DATASET_CONFIGS[@]}"; do
    read -r dataset language <<< "${config}"
    config_name="${dataset}_${language}"
    expected="MODEL_${MODEL_FOLDER}_DATASET_${DATASET_PATH//\//-}_${config_name}_test.jsonl"
    if [[ ! -f "${local_results}/${expected}" ]]; then
        echo "Missing result: ${expected}" >&2
        missing=1
    fi
done
if [[ "${missing}" -ne 0 ]]; then
    exit 1
fi

languages=()
for config in "${DATASET_CONFIGS[@]}"; do
    read -r _ language <<< "${config}"
    if [[ ! " ${languages[*]} " == *" ${language} "* ]]; then
        languages+=("${language}")
    fi
done

for language in "${languages[@]}"; do
    PYTHONPATH="${REPO_ROOT}" python -c "
from normalizer.eval_utils import score_results
score_results('${local_results}', '${MODEL_ID}', multilingual=True, language='${language}', families=['ml_${language}'], csv_only=True)
"
done

echo "All MOSS-Transcribe-Diarize multilingual HF Jobs completed and scored."
