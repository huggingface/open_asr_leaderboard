#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

MODEL_IDs=("nvidia/canary-1b-v2")  # options: "nvidia/canary-1b" "nvidia/canary-1b-flash" "nvidia/canary-180m-flash" "nvidia/canary-1b-v2"
BATCH_SIZE=128
DEVICE_ID=0

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}
    
    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/open-asr-leaderboard" \
        --dataset="ami" \
        --split="test" \
        --device=${DEVICE_ID} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 
    
    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/open-asr-leaderboard" \
        --dataset="earnings22" \
        --split="test" \
        --device=${DEVICE_ID} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/open-asr-leaderboard" \
        --dataset="gigaspeech" \
        --split="test" \
        --device=${DEVICE_ID} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/open-asr-leaderboard" \
        --dataset="librispeech" \
        --split="test.clean" \
        --device=${DEVICE_ID} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/open-asr-leaderboard" \
        --dataset="librispeech" \
        --split="test.other" \
        --device=${DEVICE_ID} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/open-asr-leaderboard" \
        --dataset="spgispeech" \
        --split="test" \
        --device=${DEVICE_ID} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/open-asr-leaderboard" \
        --dataset="tedlium" \
        --split="test" \
        --device=${DEVICE_ID} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/open-asr-leaderboard" \
        --dataset="voxpopuli" \
        --split="test" \
        --device=${DEVICE_ID} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
