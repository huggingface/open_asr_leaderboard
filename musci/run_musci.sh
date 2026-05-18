#!/bin/bash
# Open ASR Leaderboard: Musci-ASR-2.4B
# Identical decoding hyper-parameters across every subset.

set -euo pipefail

export PYTHONPATH="..":$PYTHONPATH

MODEL_ID="${MODEL_ID:-Musci-research/Musci-ASR-2.4B}"
BATCH_SIZE=128
DEVICE=0
WARMUP=2
DATASET_PATH="hf-audio/esb-datasets-test-only-sorted"

run() {
    python run_eval.py \
        --model_id="$MODEL_ID" \
        --dataset_path="$DATASET_PATH" \
        --dataset="$1" \
        --split="$2" \
        --device=$DEVICE \
        --batch_size=$BATCH_SIZE \
        --warmup_steps=$WARMUP \
        --max_eval_samples=-1
}

run ami               test
run earnings22        test
run gigaspeech        test
run librispeech       test.clean
run librispeech       test.other
run spgispeech        test
run tedlium           test
run voxpopuli         test
