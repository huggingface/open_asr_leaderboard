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

MODEL_IDs=(
    # "openai/gpt-4o-transcribe"
    # "openai/gpt-4o-mini-transcribe"
    # "openai/whisper-1"
    # "assembly/universal-3-pro"
    # "elevenlabs/scribe_v1"
    # "revai/machine" # please use --use_url=True
    # "revai/fusion" # please use --use_url=True
    # "speechmatics/enhanced"
    # "aquavoice/avalon-v1-en"
    # "zoom/scribe_v1" # please use --use_url
    # "smallestai/pulse" # please use --use_url
    "reson8/resonant-1" # please use --use_url
    "reson8/resonant-1-flash" # please use --use_url
    "microsoft/azure-speech-05-2026"
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
    "tedlium:test"
    "voxpopuli:test"
)

# Datasets that require lexical format prompt
LEXICAL_DATASETS="librispeech gigaspeech tedlium"

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}

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
            "${PROMPT_ARGS[@]}"
    done

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done