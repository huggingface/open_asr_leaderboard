#!/bin/bash
# Open ASR Leaderboard runner for Xiaomi MiMo-V2.5-ASR.
#
# Default dataset = Earnings22; override via $DATASET to run other ESB sets
# (ami / earnings22 / gigaspeech / librispeech / spgispeech / tedlium /
# voxpopuli) — same hyperparameters across all per ESB rules.
#
# Prereqs (see mimo/README.md):
#   1. Clone XiaomiMiMo/MiMo-V2.5-ASR repo, install its requirements
#   2. hf download XiaomiMiMo/MiMo-V2.5-ASR --local-dir ./models/MiMo-V2.5-ASR
#   3. hf download XiaomiMiMo/MiMo-Audio-Tokenizer --local-dir ./models/MiMo-Audio-Tokenizer
#   4. export MIMO_REPO_PATH=/abs/path/to/MiMo-V2.5-ASR (parent of src/)
set -euo pipefail

export PYTHONPATH="..":$PYTHONPATH

MODEL_ID="XiaomiMiMo/MiMo-V2.5-ASR"
MODEL_PATH="${MODEL_PATH:-./models/MiMo-V2.5-ASR}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./models/MiMo-Audio-Tokenizer}"
DEVICE_ID="${DEVICE_ID:-0}"
DATASET="${DATASET:-earnings22}"

python run_eval.py \
    --model_id="$MODEL_ID" \
    --model_path="$MODEL_PATH" \
    --tokenizer_path="$TOKENIZER_PATH" \
    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    --dataset="$DATASET" \
    --split="test" \
    --device="$DEVICE_ID" \
    --audio_tag="<english>" \
    --max_eval_samples=-1
