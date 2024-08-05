#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

MODEL_IDs=("tiny.en" "small.en" "base.en" "medium.en" "large-v1" "large-v2" "large-v3")
BATCH_SIZE=1
DEVICE_INDEX=0

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="https://huggingface.co/datasets/hf-audio/esb-datasets-test-only" \
        --dataset="ami" \
        --split="test" \
        --device=${DEVICE_INDEX} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="https://huggingface.co/datasets/hf-audio/esb-datasets-test-only" \
        --dataset="earnings22" \
        --split="test" \
        --device=${DEVICE_INDEX} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="https://huggingface.co/datasets/hf-audio/esb-datasets-test-only" \
        --dataset="gigaspeech" \
        --split="test" \
        --device=${DEVICE_INDEX} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="https://huggingface.co/datasets/hf-audio/esb-datasets-test-only" \
        --dataset="librispeech" \
        --split="test.clean" \
        --device=${DEVICE_INDEX} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="https://huggingface.co/datasets/hf-audio/esb-datasets-test-only" \
        --dataset="librispeech" \
        --split="test.other" \
        --device=${DEVICE_INDEX} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="https://huggingface.co/datasets/hf-audio/esb-datasets-test-only" \
        --dataset="spgispeech" \
        --split="test" \
        --device=${DEVICE_INDEX} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="https://huggingface.co/datasets/hf-audio/esb-datasets-test-only" \
        --dataset="tedlium" \
        --split="test" \
        --device=${DEVICE_INDEX} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="https://huggingface.co/datasets/hf-audio/esb-datasets-test-only" \
        --dataset="voxpopuli" \
        --split="test" \
        --device=${DEVICE_INDEX} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="https://huggingface.co/datasets/hf-audio/esb-datasets-test-only" \
        --dataset="common_voice" \
        --split="test" \
        --device=${DEVICE_INDEX} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
