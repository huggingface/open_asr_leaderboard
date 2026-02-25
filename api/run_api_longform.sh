#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

export HF_TOKEN="your_api_key"
export OPENAI_API_KEY="your_api_key"
export ASSEMBLYAI_API_KEY="your_api_key"
export ELEVENLABS_API_KEY="your_api_key"
export REVAI_API_KEY="your_api_key"
export AQUAVOICE_API_KEY="your_api_key"
export SPEECHMATICS_API_KEY="your_api_key"

MODEL_IDs=(
    "openai/gpt-4o-transcribe"
    "openai/gpt-4o-mini-transcribe"
    "openai/whisper-1"
    "assembly/universal-3-pro"
    "elevenlabs/scribe_v2"
    "revai/machine" # please use --use_url=True
    "revai/fusion" # please use --use_url=True
    "speechmatics/enhanced"
    "aquavoice/avalon-v1-en"
)

MAX_WORKERS=10

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}
    python run_eval.py \
        --dataset_path="hf-audio/asr-leaderboard-longform" \
        --dataset="earnings21" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_workers ${MAX_WORKERS}

    python run_eval.py \
        --dataset_path="hf-audio/asr-leaderboard-longform" \
        --dataset="earnings22" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_workers ${MAX_WORKERS}

    python run_eval.py \
        --dataset_path="hf-audio/asr-leaderboard-longform" \
        --dataset="tedlium" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_workers ${MAX_WORKERS}

    # CORAAL evaluation ATL DCA DCB DTA LES PRV ROC VLD
    for SUBSET in PRV ROC VLD; do
        python run_eval.py \
            --dataset_path="bezzam/coraal" \
            --dataset=${SUBSET} \
            --split="test" \
            --model_name ${MODEL_ID} \
            --max_workers ${MAX_WORKERS}
    done

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
