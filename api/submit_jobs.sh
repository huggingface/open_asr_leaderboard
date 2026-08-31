#!/bin/bash
# Submit HF Jobs for API-based ASR evaluation.
# Usage:
#   HF_TOKEN=hf_... \
#   OPENAI_API_KEY=... \
#   ASSEMBLYAI_API_KEY=... \
#   ELEVENLABS_API_KEY=... \
#   REVAI_API_KEY=... \
#   SPEECHMATICS_API_KEY=... \
#   AQUAVOICE_API_KEY=... \
#   ZOOM_API_KEY=... \
#   AZURE_API_KEY=... \
#   bash submit_jobs.sh

# Global defaults (can be left as-is; per-model `max_workers` will override)
SPACE="${SPACE:-hf-audio/open-asr-leaderboard-apis}"
RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_h200}"
DEFAULT_DATASET_PATH="${DEFAULT_DATASET_PATH:-hf-audio/open-asr-leaderboard}"
# API jobs are CPU-only (no model weights loaded locally)
FLAVOR="${FLAVOR:-cpu-basic}"
ORG_NAME="${ORG_NAME:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Set USE_LOCAL_SCRIPT=1 to run your local run_eval.py instead of the version
# committed to the Space (useful for iterating without pushing to the Space).
USE_LOCAL_SCRIPT="${USE_LOCAL_SCRIPT:-1}"
LOCAL_SCRIPT_INJECT=""
if [[ "$USE_LOCAL_SCRIPT" == "1" ]]; then
    RUN_EVAL_B64=$(base64 -w0 "${SCRIPT_DIR}/run_eval.py")
    LOCAL_SCRIPT_INJECT="echo '${RUN_EVAL_B64}' | base64 -d > /app/run_eval.py &&"
fi

# Set USE_LOCAL_NORMALIZER=1 to inject your local normalizer/ package into the
# job (so normalizer changes take effect without updating the HF Space).
USE_LOCAL_NORMALIZER="${USE_LOCAL_NORMALIZER:-1}"
LOCAL_NORMALIZER_INJECT=""
if [[ "$USE_LOCAL_NORMALIZER" == "1" ]]; then
    NORMALIZER_B64=$(tar --exclude='__pycache__' --exclude='*.pyc' -czf - -C "${REPO_ROOT}" normalizer | base64 -w0)
    LOCAL_NORMALIZER_INJECT="echo '${NORMALIZER_B64}' | base64 -d | tar -xzf - -C /app &&"
fi


# ── Models: "model_id max_workers" ──────────────────────────────────────────
# Fields:
#  - model_id:    provider-prefixed model name (e.g. 'elevenlabs/scribe_v1')
#  - max_workers: number of concurrent threads for this model (required)
MODEL_CONFIGS=(
    # "openai/gpt-4o-transcribe      16"
    # "openai/gpt-4o-mini-transcribe 16"
    # "openai/whisper-1              16"
    # "assembly/universal-3-pro      4"   # `cpu-xl` needed for spgispeech
    # "assembly/universal-3-5-pro    4"   # `cpu-xl` needed for spgispeech
    # "elevenlabs/scribe_v1          16"
    # "revai/machine                 8"
    # "revai/fusion                  8"
    # "speechmatics/enhanced         8"    # `cpu-xl` needed for spgispeech
    # "aquavoice/avalon-v1-en        16"
    # "zoom/scribe_v1                32"
    # "microsoft/azure-speech-06-2026  4"
    # "reson8/resonant-1             16"
    # "reson8/resonant-1-flash       16"
)

# ── Datasets: "name split [dataset_path]" ─────────────────────────────────────
# dataset_path defaults to $DEFAULT_DATASET_PATH when omitted.
# An entry that names its own repo (e.g. VoiceArena/Monsoon_en_IN_test) passes no
# config name: the first field is only a label for selection and result files.
DATASET_CONFIGS=(
    "ami_cleaned test"
    "earnings22_cleaned_aa_chunked test ArtificialAnalysis/Earnings22-Cleaned-AA-chunked"
    "gigaspeech_cleaned test"
    "librispeech test.clean"
    "librispeech test.other"
    "spgispeech test"
    "voxpopuli_cleaned_aa test"
    "monsoon_en_in test VoiceArena/Monsoon_en_IN_test"
)
# Optional: restrict this run to specific datasets, matched against the first
# field of each DATASET_CONFIGS entry, e.g.:
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


# Datasets that require a lexical-format prompt for microsoft models
LEXICAL_DATASETS="librispeech gigaspeech"

# ── Submit one job per model/dataset combination ─────────────────────────────
for model_cfg in "${MODEL_CONFIGS[@]}"; do
    read -r MODEL_ID MODEL_MAX_WORKERS <<< "$model_cfg"
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

        PROMPT_ARG=""
        if [[ "$MODEL_ID" == microsoft/* ]] && [[ " $LEXICAL_DATASETS " == *" $DATASET "* ]]; then
            PROMPT_ARG="--prompt 'Output must be in lexical format.'"
        fi

        echo "Submitting job: model=${MODEL_ID} dataset_path=${DATASET_PATH} dataset=${DATASET} split=${SPLIT}"

        NAMESPACE_ARG=""
        [ -n "$ORG_NAME" ] && NAMESPACE_ARG="--namespace ${ORG_NAME}"

        hf jobs run \
            --flavor "$FLAVOR" \
            --timeout 8h \
            --env HF_TOKEN="$HF_TOKEN" \
            --env OPENAI_API_KEY="$OPENAI_API_KEY" \
            --env ASSEMBLYAI_API_KEY="$ASSEMBLYAI_API_KEY" \
            --env ELEVENLABS_API_KEY="$ELEVENLABS_API_KEY" \
            --env REVAI_API_KEY="$REVAI_API_KEY" \
            --env SPEECHMATICS_API_KEY="$SPEECHMATICS_API_KEY" \
            --env AQUAVOICE_API_KEY="$AQUAVOICE_API_KEY" \
            --env ZOOM_API_KEY="$ZOOM_API_KEY" \
            --env AZURE_API_KEY="$AZURE_API_KEY" \
            --env AZURE_REGION="$AZURE_REGION" \
            --env HF_AUDIO_DECODER_BACKEND="soundfile" \
            ${NAMESPACE_ARG} \
            --volume "hf://buckets/${RESULTS_BUCKET}:/results" \
            "hf.co/spaces/${SPACE}" \
            bash -c "
                ${LOCAL_NORMALIZER_INJECT}
                ${LOCAL_SCRIPT_INJECT}
                PYTHONPATH=/app python run_eval.py \
                    --dataset_path=${DATASET_PATH} \
                    --dataset=${DATASET_CONFIG} \
                    --split=${SPLIT} \
                    --model_name=${MODEL_ID} \
                    --max_workers=${MODEL_MAX_WORKERS} \
                    ${PROMPT_ARG} &&
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

    PYTHONPATH="${REPO_ROOT}" python -c "
from normalizer.eval_utils import score_results
score_results('$(pwd)/results/${MODEL_FOLDER}', '${MODEL_ID}')
"
done
