#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

MODEL_IDs=("kyutai/stt-2.6b-en")
BATCH_SIZE=16

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
        --max_eval_samples=-1

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/asr-leaderboard-longform" \
        --dataset="earnings22" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/asr-leaderboard-longform" \
        --dataset="tedlium" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
