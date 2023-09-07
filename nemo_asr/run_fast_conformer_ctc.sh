#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

#considering FC-XL, FC-XXL, FC-L, C-L, C-S CTC models
MODEL_IDs=("nvidia/stt_en_fastconformer_ctc_xxlarge" "nvidia/stt_en_fastconformer_ctc_xlarge" "nvidia/stt_en_fastconformer_ctc_large" "nvidia/stt_en_conformer_ctc_large" "nvidia/stt_en_conformer_ctc_small")
BATCH_SIZE=8
DEVICE_ID=0

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}

    
    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="ami" \
        --split="test" \
        --device=${DEVICE_ID} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --no-streaming
    
    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="earnings22" \
        --split="test" \
        --device=${DEVICE_ID} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --no-streaming

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="gigaspeech" \
        --split="test" \
        --device=${DEVICE_ID} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --no-streaming

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="librispeech" \
        --split="test.clean" \
        --device=${DEVICE_ID} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --no-streaming

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="librispeech" \
        --split="test.other" \
        --device=${DEVICE_ID} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --no-streaming

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="spgispeech" \
        --split="test" \
        --device=${DEVICE_ID} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --no-streaming

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="tedlium" \
        --split="test" \
        --device=${DEVICE_ID} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --no-streaming

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="voxpopuli" \
        --split="test" \
        --device=${DEVICE_ID} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --no-streaming

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="common_voice" \
        --split="test" \
        --device=${DEVICE_ID} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --no-streaming

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
