#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

MODEL_IDs=("facebook/wav2vec2-base-960h" "facebook/wav2vec2-large-960h" "facebook/wav2vec2-large-960h-lv60-self" "facebook/wav2vec2-large-robust-ft-libri-960h" "facebook/wav2vec2-conformer-rel-pos-large-960h-ft" "facebook/wav2vec2-conformer-rope-large-960h-ft")
BATCH_SIZE=64

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="ami" \
        --split="test" \
        --device=1 \
        --batch_size=${BATCH_SIZE} \
        --torch_compile \
        --max_eval_samples=-1


    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="earnings22" \
        --split="test" \
        --device=1 \
        --batch_size=${BATCH_SIZE} \
        --torch_compile \
        --max_eval_samples=-1

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="gigaspeech" \
        --split="test" \
        --device=1 \
        --batch_size=${BATCH_SIZE} \
        --torch_compile \
        --max_eval_samples=-1

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="librispeech" \
        --split="test.clean" \
        --device=1 \
        --batch_size=${BATCH_SIZE} \
        --torch_compile \
        --max_eval_samples=-1

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="librispeech" \
        --split="test.other" \
        --device=1 \
        --batch_size=${BATCH_SIZE} \
        --torch_compile \
        --max_eval_samples=-1

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="spgispeech" \
        --split="test" \
        --device=1 \
        --batch_size=${BATCH_SIZE} \
        --torch_compile \
        --max_eval_samples=-1

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="tedlium" \
        --split="test" \
        --device=1 \
        --batch_size=${BATCH_SIZE} \
        --torch_compile \
        --max_eval_samples=-1

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="voxpopuli" \
        --split="test" \
        --device=1 \
        --batch_size=${BATCH_SIZE} \
        --torch_compile \
        --max_eval_samples=-1

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="common_voice" \
        --split="test" \
        --device=1 \
        --batch_size=${BATCH_SIZE} \
        --torch_compile \
        --max_eval_samples=-1

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
