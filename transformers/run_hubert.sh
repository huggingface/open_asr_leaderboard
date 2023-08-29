#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

MODEL_IDs=( "facebook/hubert-large-ls960-ft" "facebook/hubert-xlarge-ls960-ft" "patrickvonplaten/hubert-xlarge-ls960-ft-4-gram")
BATCH_SIZE=8

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="ami" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=8


    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="earnings22" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=8

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="gigaspeech" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=8

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="librispeech" \
        --split="test.clean" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=8

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="librispeech" \
        --split="test.other" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=8

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="spgispeech" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=8

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="tedlium" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=8

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="voxpopuli" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=8

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="common_voice" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=8

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done