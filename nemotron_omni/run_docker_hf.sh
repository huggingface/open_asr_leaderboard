#!/bin/bash
# Docker entrypoint for the HF transformers path: runs the 8-dataset Open ASR
# Leaderboard sweep through `model.generate()` (run_eval.py + run_nemotron_omni.sh).
# No server is started; HF transformers loads the model in-process per script.
#
# Usage (from the repo root, after `docker build -t open-asr-nemotron-omni
# -f nemotron_omni/Dockerfile .`):
#
#     docker run --gpus all \
#         -v $(pwd):/app \
#         -v $HF_HOME:/root/.cache/huggingface \
#         open-asr-nemotron-omni run_docker_hf.sh
#
# Override batch size / model with env vars (see run_nemotron_omni.sh for the
# full list): `-e BATCH_SIZE=64`, `-e MODEL_ID=...`, etc.

set -euo pipefail

cd "$(dirname "$0")"

exec ./run_nemotron_omni.sh "$@"
