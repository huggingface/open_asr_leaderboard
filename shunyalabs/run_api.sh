#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH


export SHUNYA_API_KEY="your_api_key"

MODEL_IDs=(
    "shunyalabs/Pingala-V1-Verbatim"
)

num_models=${#MODEL_IDs[@]}
for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}
    python run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="ami" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_workers 8

    python run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="earnings22" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_workers 8

    python run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="gigaspeech" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_workers 8

    python run_eval.py \
        --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
        --dataset "librispeech" \
        --split "test.clean" \
        --model_name ${MODEL_ID} \
        --max_workers 8

    python run_eval.py \
        --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
        --dataset "librispeech" \
        --split "test.other" \
        --model_name ${MODEL_ID} \
        --max_workers 8

    python run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="spgispeech" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_workers 8

    python run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="tedlium" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_workers 8

    python run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="voxpopuli" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_workers 8
    
    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
