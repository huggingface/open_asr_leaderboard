#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

export OPENAI_API_KEY="your_api_key"
export ASSEMBLYAI_API_KEY="your_api_key"
export ELEVENLABS_API_KEY="your_api_key"
export REVAI_API_KEY="your_api_key"
export AQUAVOICE_API_KEY="your_api_key"
export ZOOM_API_KEY="your_api_key"
export SMALLESTAI_API_KEY="your_api_key"
export RESON8_API_KEY="your_api_key"
export AZURE_API_KEY="your_api_key"

export HF_TOKEN="hf_your_key"

# ── Models: "model_id use_url" ───────────────────────────────────────────────
# use_url=true  → provider receives a remote audio URL (revai, zoom)
# use_url=false → provider receives a local audio file (all others)
MODEL_CONFIGS=(
    # "openai/gpt-4o-transcribe      false"
    # "openai/gpt-4o-mini-transcribe false"
    # "openai/whisper-1              false"
    # "assembly/universal-3-pro      false"
    # "elevenlabs/scribe_v1            false"
    # "revai/machine                 true"
    # "revai/fusion                  true"
    # "speechmatics/enhanced         false"
    # "aquavoice/avalon-v1-en        false"
    # "zoom/scribe_v1                true"
    "reson8/resonant-1               true"
    "reson8/resonant-1-flash         true"
    "microsoft/azure-speech-05-2026  false"
)

MAX_WORKERS=20
DATASET_PATH="hf-audio/open-asr-leaderboard"

declare -a EVAL_DATASETS=(
    "ami:test"
    "earnings22:test"
    "gigaspeech:test"
    "librispeech:test.clean"
    "librispeech:test.other"
    "spgispeech:test"
    "voxpopuli:test"
)

# Datasets that require lexical format prompt
LEXICAL_DATASETS="librispeech gigaspeech"
num_models=${#MODEL_IDs[@]}

for model_cfg in "${MODEL_CONFIGS[@]}"; do
    read -r MODEL_ID USE_URL <<< "$model_cfg"
    USE_URL_FLAG=$([[ "$USE_URL" == "true" ]] && echo "--use_url" || echo "")

    for entry in "${EVAL_DATASETS[@]}"; do
        DATASET="${entry%%:*}"
        SPLIT="${entry##*:}"

        PROMPT_ARGS=()
        if [[ "$MODEL_ID" == microsoft/* ]] && [[ " $LEXICAL_DATASETS " == *" $DATASET "* ]]; then
            PROMPT_ARGS=(--prompt "Output must be in lexical format.")
        fi

        python run_eval.py \
            --dataset_path="$DATASET_PATH" \
            --dataset="$DATASET" \
            --split="$SPLIT" \
            --model_name ${MODEL_ID} \
            --max_workers ${MAX_WORKERS} \
            ${USE_URL_FLAG} \
            "${PROMPT_ARGS[@]}"
    done

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done