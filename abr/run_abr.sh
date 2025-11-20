#!/bin/bash
set -e

export PYTHONPATH="..":$PYTHONPATH

MODEL_IDs=("abr-ai/asr-19m-v2-en-32b")
BATCH_SIZE=256
MAX_EVAL_SAMPLES=-1
WARMUP_STEPS=5
SUBBATCH_SAMPLES=30000000

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="voxpopuli" \
        --split="test" \
        --batch_size=${BATCH_SIZE} \
        --warmup_steps=${WARMUP_STEPS} \
        --subbatch_samples=${SUBBATCH_SAMPLES} \
        --max_eval_samples=${MAX_EVAL_SAMPLES}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="ami" \
        --split="test" \
        --batch_size=${BATCH_SIZE} \
        --warmup_steps=${WARMUP_STEPS} \
        --subbatch_samples=${SUBBATCH_SAMPLES} \
        --max_eval_samples=${MAX_EVAL_SAMPLES}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="earnings22" \
        --split="test" \
        --batch_size=${BATCH_SIZE} \
        --warmup_steps=${WARMUP_STEPS} \
        --subbatch_samples=${SUBBATCH_SAMPLES} \
        --max_eval_samples=${MAX_EVAL_SAMPLES}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="gigaspeech" \
        --split="test" \
        --batch_size=${BATCH_SIZE} \
        --warmup_steps=${WARMUP_STEPS} \
        --subbatch_samples=${SUBBATCH_SAMPLES} \
        --max_eval_samples=${MAX_EVAL_SAMPLES}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="librispeech" \
        --split="test.clean" \
        --batch_size=${BATCH_SIZE} \
        --warmup_steps=${WARMUP_STEPS} \
        --subbatch_samples=${SUBBATCH_SAMPLES} \
        --max_eval_samples=${MAX_EVAL_SAMPLES}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="librispeech" \
        --split="test.other" \
        --batch_size=${BATCH_SIZE} \
        --warmup_steps=${WARMUP_STEPS} \
        --subbatch_samples=${SUBBATCH_SAMPLES} \
        --max_eval_samples=${MAX_EVAL_SAMPLES}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="spgispeech" \
        --split="test" \
        --batch_size=${BATCH_SIZE} \
        --warmup_steps=${WARMUP_STEPS} \
        --subbatch_samples=${SUBBATCH_SAMPLES} \
        --max_eval_samples=${MAX_EVAL_SAMPLES}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="tedlium" \
        --split="test" \
        --batch_size=${BATCH_SIZE} \
        --warmup_steps=${WARMUP_STEPS} \
        --subbatch_samples=${SUBBATCH_SAMPLES} \
        --max_eval_samples=${MAX_EVAL_SAMPLES}

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
