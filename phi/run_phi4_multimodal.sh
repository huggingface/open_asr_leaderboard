#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

MODEL_IDs=("microsoft/Phi-4-multimodal-instruct")
BATCH_SIZE=32
NUM_BEAMS=1
MAX_NEW_TOKENS=512

num_models=${#MODEL_IDs[@]}
default_user_prompt="Transcribe the audio clip into text."

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
        --num_beams=${NUM_BEAMS} \
        --max_eval_samples=-1 \
        --max_new_tokens=${MAX_NEW_TOKENS} \
        --user_prompt="${default_user_prompt}"

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="ami" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --num_beams=${NUM_BEAMS} \
        --max_eval_samples=-1 \
        --max_new_tokens=${MAX_NEW_TOKENS} \
        --user_prompt="${default_user_prompt}"

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="earnings22" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --num_beams=${NUM_BEAMS} \
        --max_eval_samples=-1 \
        --max_new_tokens=${MAX_NEW_TOKENS} \
        --user_prompt="Transcribe the audio clip to English text."

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="gigaspeech" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --num_beams=${NUM_BEAMS} \
        --max_eval_samples=-1 \
        --max_new_tokens=${MAX_NEW_TOKENS} \
        --user_prompt="${default_user_prompt}"

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="librispeech" \
        --split="test.clean" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --num_beams=${NUM_BEAMS} \
        --max_eval_samples=-1 \
        --max_new_tokens=${MAX_NEW_TOKENS} \
        --user_prompt="${default_user_prompt}"

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="librispeech" \
        --split="test.other" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --num_beams=${NUM_BEAMS} \
        --max_eval_samples=-1 \
        --max_new_tokens=${MAX_NEW_TOKENS} \
        --user_prompt="${default_user_prompt}"

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="spgispeech" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --num_beams=${NUM_BEAMS} \
        --max_eval_samples=-1 \
        --max_new_tokens=${MAX_NEW_TOKENS} \
        --user_prompt="${default_user_prompt}"

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="tedlium" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --num_beams=${NUM_BEAMS} \
        --max_eval_samples=-1 \
        --max_new_tokens=${MAX_NEW_TOKENS} \
        --user_prompt="${default_user_prompt}"

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
