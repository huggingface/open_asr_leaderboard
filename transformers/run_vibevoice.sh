#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

MODEL_IDs=("microsoft/VibeVoice-ASR-HF")
BATCH_SIZE=64
MAX_NEW_TOKENS=225 # 30 seconds of audio at 24000 kHz with 3200 compression

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/open-asr-leaderboard" \
        --dataset="voxpopuli" \
        --split="test" \
        --device=0 \
        --batch_size=32 \
        --max_eval_samples=-1 \
        --max_new_tokens=${MAX_NEW_TOKENS}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/open-asr-leaderboard" \
        --dataset="ami" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --max_new_tokens=${MAX_NEW_TOKENS}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/open-asr-leaderboard" \
        --dataset="earnings22" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --max_new_tokens=${MAX_NEW_TOKENS}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/open-asr-leaderboard" \
        --dataset="gigaspeech" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --max_new_tokens=${MAX_NEW_TOKENS}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/open-asr-leaderboard" \
        --dataset="librispeech" \
        --split="test.clean" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --max_new_tokens=${MAX_NEW_TOKENS}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/open-asr-leaderboard" \
        --dataset="librispeech" \
        --split="test.other" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --max_new_tokens=${MAX_NEW_TOKENS}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/open-asr-leaderboard" \
        --dataset="spgispeech" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --max_new_tokens=${MAX_NEW_TOKENS}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/open-asr-leaderboard" \
        --dataset="tedlium" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --max_new_tokens=${MAX_NEW_TOKENS}

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done