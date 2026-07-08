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

python run_eval.py "${ARGS[@]}" --dataset="ami_cleaned" --split="test"
python run_eval.py "${ARGS[@]}" --dataset="earnings22" --split="test"
python run_eval.py "${ARGS[@]}" --dataset="gigaspeech_cleaned" --split="test"
python run_eval.py "${ARGS[@]}" --dataset="librispeech" --split="test.clean"
python run_eval.py "${ARGS[@]}" --dataset="librispeech" --split="test.other"
python run_eval.py "${ARGS[@]}" --dataset="spgispeech" --split="test"
python run_eval.py "${ARGS[@]}" --dataset="voxpopuli_cleaned_aa" --split="test"

RUNDIR=$(pwd)
PYTHONPATH="${RUNDIR}/..:${PYTHONPATH}" python -c "from normalizer.eval_utils import score_results; score_results('${RUNDIR}/results', '${MODEL_ID}')"
