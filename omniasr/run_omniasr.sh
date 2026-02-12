#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

# Available omniASR-LLM models
MODEL_IDs=(
    "facebook/omniASR-CTC-300M" "facebook/omniASR-CTC-1B" "facebook/omniASR-CTC-3B" "facebook/omniASR-CTC-7B"
    "facebook/omniASR-LLM-300M" "facebook/omniASR-LLM-1B" "facebook/omniASR-LLM-3B" "facebook/omniASR-LLM-7B"
    )
BATCH_SIZE=64  # Conservative batch size due to LLM memory requirements
LANGUAGE="eng_Latn"  # English language code for omniASR

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="voxpopuli" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --language=${LANGUAGE}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="ami" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --language=${LANGUAGE}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="earnings22" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --language=${LANGUAGE}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="gigaspeech" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --language=${LANGUAGE}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="librispeech" \
        --split="test.clean" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --language=${LANGUAGE}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="librispeech" \
        --split="test.other" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --language=${LANGUAGE}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="spgispeech" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --language=${LANGUAGE}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="tedlium" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --language=${LANGUAGE}

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
