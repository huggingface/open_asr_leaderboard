#!/bin/bash

set -euo pipefail

MODEL_ID="soundsgoodai/Zipformer-transducer-XL-290M"
BATCH_SIZE=64
DEVICE_ID=0
ICEFALL_PATH="./icefall"

export PYTHONPATH="${ICEFALL_PATH}:${ICEFALL_PATH}/egs/librispeech/ASR/zipformer:..:${PYTHONPATH:-}"

ARGS=(
    --model_id="${MODEL_ID}"
    --dataset_path="hf-audio/open-asr-leaderboard"
    --device="${DEVICE_ID}"
    --batch_size="${BATCH_SIZE}"
    --max_eval_samples=-1
)

python run_eval.py "${ARGS[@]}" --dataset="ami" --split="test"
python run_eval.py "${ARGS[@]}" --dataset="earnings22" --split="test"
python run_eval.py "${ARGS[@]}" --dataset="gigaspeech" --split="test"
python run_eval.py "${ARGS[@]}" --dataset="librispeech" --split="test.clean"
python run_eval.py "${ARGS[@]}" --dataset="librispeech" --split="test.other"
python run_eval.py "${ARGS[@]}" --dataset="spgispeech" --split="test"
python run_eval.py "${ARGS[@]}" --dataset="tedlium" --split="test"
python run_eval.py "${ARGS[@]}" --dataset="voxpopuli" --split="test"

RUNDIR=$(pwd)
cd ../normalizer
python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')"
cd "${RUNDIR}"
