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
DATASET_PATH="${DATASET_PATH:-hf-audio/open-asr-leaderboard}"
# API jobs are CPU-only (no model weights loaded locally)
FLAVOR="${FLAVOR:-cpu-basic}"
ORG_NAME="${ORG_NAME:-}"


# ── Models: "model_id use_url max_workers" ─────────────────────────────────
# Fields:
#  - model_id:    provider-prefixed model name (e.g. 'elevenlabs/scribe_v1')
#  - use_url:     true|false — whether the provider requires a remote audio URL
#  - max_workers: number of concurrent threads for this model (required)
MODEL_CONFIGS=(
    # "openai/gpt-4o-transcribe      false  16"
    # "openai/gpt-4o-mini-transcribe false  16"
    # "openai/whisper-1              false  16"
    # "assembly/universal-3-pro      false  4"   # `cpu-xl` needed for spgispeech
    # "assembly/universal-3-5-pro    false  4"   # `cpu-xl` needed for spgispeech
    # "elevenlabs/scribe_v1          false  16"
    # "revai/machine                 false  8"
    # "revai/fusion                  false  8"
    # "speechmatics/enhanced         false  16"    # # `cpu-xl` needed for spgispeech
    # "aquavoice/avalon-v1-en        false  16"
    # "zoom/scribe_v1                false  32"
    # "microsoft/azure-speech-05-2026  false  4"
    # "reson8/resonant-1               false 16"
    # "reson8/resonant-1-flash         false 16"
)

# ── Datasets ──────────────────────────────────────────────────────────────────
DATASET_CONFIGS=(
    "ami test"
    "earnings22 test"
    "gigaspeech test"
    "librispeech test.clean"
    "librispeech test.other"
    "spgispeech test"
    "voxpopuli test"
)

# Datasets that require a lexical-format prompt for microsoft models
LEXICAL_DATASETS="librispeech gigaspeech"

# ── Submit one job per model/dataset combination ─────────────────────────────
for model_cfg in "${MODEL_CONFIGS[@]}"; do
    # parse: model_id use_url max_workers
    read -r MODEL_ID USE_URL MODEL_MAX_WORKERS <<< "$model_cfg"
    MODEL_FOLDER="${MODEL_ID//\//-}"
    USE_URL_FLAG=$([[ "$USE_URL" == "true" ]] && echo "--use_url" || echo "")

    echo "████████████████████████████████████████████████████████████████████████████████"
    echo "  Evaluating: ${MODEL_ID}"
    echo "████████████████████████████████████████████████████████████████████████████████"

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET SPLIT <<< "$cfg"

        PROMPT_ARG=""
        if [[ "$MODEL_ID" == microsoft/* ]] && [[ " $LEXICAL_DATASETS " == *" $DATASET "* ]]; then
            PROMPT_ARG="--prompt 'Output must be in lexical format.'"
        fi

        echo "Submitting job: model=${MODEL_ID} dataset=${DATASET} split=${SPLIT}"

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
                PYTHONPATH=/app python run_eval.py \
                    --dataset_path=${DATASET_PATH} \
                    --dataset=${DATASET} \
                    --split=${SPLIT} \
                    --model_name=${MODEL_ID} \
                    --max_workers=${MODEL_MAX_WORKERS} \
                    ${USE_URL_FLAG} \
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

    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/." && pwd)"
    PYTHONPATH="${REPO_ROOT}" python -c "
from normalizer.eval_utils import score_results
score_results('$(pwd)/results/${MODEL_FOLDER}', '${MODEL_ID}')
"
done
