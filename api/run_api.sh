#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

export OPENAI_API_KEY="your_api_key"
export ASSEMBLYAI_API_KEY="your_api_key"
export ELEVENLABS_API_KEY="your_api_key"
export REVAI_API_KEY="your_api_key"
export AQUAVOICE_API_KEY="your_api_key"

MODEL_IDs=(
    "openai/gpt-4o-transcribe"
    "openai/gpt-4o-mini-transcribe"
    "openai/whisper-1"
    "assembly/best"
    "elevenlabs/scribe_v2"
    "revai/machine" # please use --use_url=True
    "revai/fusion" # please use --use_url=True
    "speechmatics/enhanced"
    "aquavoice/avalon-v1-en"
)

MAX_WORKERS=10
DATASET_PATH="hf-audio/esb-datasets-test-only-sorted"

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

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}

    for entry in "${EVAL_DATASETS[@]}"; do
        DATASET="${entry%%:*}"
        SPLIT="${entry##*:}"

        python run_eval.py \
            --dataset_path="$DATASET_PATH" \
            --dataset="$DATASET" \
            --split="$SPLIT" \
            --model_name ${MODEL_ID} \
            --max_workers ${MAX_WORKERS}
    done

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
