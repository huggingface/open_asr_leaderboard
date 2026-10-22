#!/usr/bin/env bash
set -euo pipefail

# Run from any directory. Install the dependencies documented in voice_code_bench.md.
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

python nemo_asr/run_eval.py \
    --model_id nvidia/parakeet-tdt-0.6b-v3 \
    --dataset_path besimple-ai/voice-code-bench \
    --dataset default --split test \
    --device 0 --batch_size 1 "$@"

# Scoring is separate so inference requires no verifier credential. See the
# printed CTEM command and voice_code_bench.md for live-fill and offline replay.
