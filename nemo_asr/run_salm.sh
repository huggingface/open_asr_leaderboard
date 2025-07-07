#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH
export TOKENIZERS_PARALLELISM=false

MODEL_IDs=(
    nvidia/canary-qwen-2.5b
)
BATCH_SIZE=192
DEVICE_ID=0

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}

    python run_eval_salm.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="ami" \
        --split="test" \
        --device=${DEVICE_ID} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1

    python run_eval_salm.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="earnings22" \
        --split="test" \
        --device=${DEVICE_ID} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1

    python run_eval_salm.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="gigaspeech" \
        --split="test" \
        --device=${DEVICE_ID} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1

    python run_eval_salm.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="librispeech" \
        --split="test.clean" \
        --device=${DEVICE_ID} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1

    python run_eval_salm.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="librispeech" \
        --split="test.other" \
        --device=${DEVICE_ID} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1

    python run_eval_salm.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="spgispeech" \
        --split="test" \
        --device=${DEVICE_ID} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1

    python run_eval_salm.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="tedlium" \
        --split="test" \
        --device=${DEVICE_ID} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1

    python run_eval_salm.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
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
