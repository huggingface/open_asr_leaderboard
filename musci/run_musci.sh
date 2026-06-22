#!/bin/bash
# Open ASR Leaderboard: Musci-ASR-2.4B (local single-GPU eval).
# Identical decoding hyper-parameters across every subset. TED-LIUM is no longer
# on the leaderboard, so it is excluded. Raw outputs are saved; score with the
# standardized normalizer afterwards (see note at the bottom).

set -euo pipefail

export PYTHONPATH="..":$PYTHONPATH

MODEL_ID="${MODEL_ID:-Musci-research/Musci-ASR-2.4B}"
DEVICE=0
WARMUP=2
DATASET_PATH="hf-audio/esb-datasets-test-only-sorted"

run() {  # dataset split batch_size
    python run_eval.py \
        --model_id="$MODEL_ID" \
        --dataset_path="$DATASET_PATH" \
        --dataset="$1" \
        --split="$2" \
        --device=$DEVICE \
        --batch_size="$3" \
        --max_new_tokens=512 \
        --warmup_steps=$WARMUP \
        --max_eval_samples=-1
}

run ami               test          128
run earnings22        test          128
run gigaspeech        test          128
run librispeech       test.clean    128
run librispeech       test.other    128
run spgispeech        test          128
run voxpopuli         test          64

# Score all manifests with the standardized normalizer:
#   python -c "from normalizer.eval_utils import score_results; score_results('results', '$MODEL_ID')"
