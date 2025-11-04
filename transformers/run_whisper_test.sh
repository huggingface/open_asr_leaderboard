#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

MODEL_ID="openai/whisper-tiny.en"
BATCH_SIZE=1
MAX_EVAL_SAMPLES=4

python run_eval.py \
    --model_id=${MODEL_ID} \
    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    --dataset="gigaspeech" \
    --split="test" \
    --device=0 \
    --batch_size=${BATCH_SIZE} \
    --max_eval_samples=${MAX_EVAL_SAMPLES}