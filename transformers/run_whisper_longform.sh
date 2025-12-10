#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

MODEL_IDs=("openai/whisper-large" "openai/whisper-large-v2" "openai/whisper-large-v3" "openai/whisper-large-v3-turbo" "distil-whisper/distil-medium.en" "distil-whisper/distil-large-v2" "distil-whisper/distil-large-v3" "distil-whisper/distil-large-v3.5")
BATCH_SIZE=32

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/asr-leaderboard-longform" \
        --dataset="earnings21" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --longform

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/asr-leaderboard-longform" \
        --dataset="earnings22" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --longform

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/asr-leaderboard-longform" \
        --dataset="tedlium" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --longform

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
