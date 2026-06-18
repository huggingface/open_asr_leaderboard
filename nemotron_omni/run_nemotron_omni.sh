#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH
export TOKENIZERS_PARALLELISM=false

MODEL_ID="${MODEL_ID:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16}"
MODEL_PATH="${MODEL_PATH:-}"
REVISION="${REVISION:-}"
BATCH_SIZE="${BATCH_SIZE:-128}"
# voxpopuli has the longest clips in the leaderboard test set; bs=128 OOMs on
# a single 96GB GPU. Use a smaller batch only for that dataset.
BATCH_SIZE_VOXPOPULI="${BATCH_SIZE_VOXPOPULI:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
DEVICE_ID="${DEVICE_ID:-0}"
USER_PROMPT="Transcribe the audio clip into text. Return only the transcription."

COMMON_ARGS=(
    --model_id="${MODEL_ID}"
    --dataset_path="hf-audio/esb-datasets-test-only-sorted"
    --device=${DEVICE_ID}
    --max_eval_samples=-1
    --max_new_tokens=${MAX_NEW_TOKENS}
    --warmup_steps=${WARMUP_STEPS}
    --user_prompt="${USER_PROMPT}"
)
if [[ -n "${MODEL_PATH}" ]]; then
    COMMON_ARGS+=(--model_path="${MODEL_PATH}")
fi
if [[ -n "${REVISION}" ]]; then
    COMMON_ARGS+=(--revision="${REVISION}")
fi

python run_eval.py "${COMMON_ARGS[@]}" --batch_size=${BATCH_SIZE_VOXPOPULI} --dataset="voxpopuli"   --split="test"
python run_eval.py "${COMMON_ARGS[@]}" --batch_size=${BATCH_SIZE}           --dataset="ami"         --split="test"
python run_eval.py "${COMMON_ARGS[@]}" --batch_size=${BATCH_SIZE}           --dataset="earnings22"  --split="test"
python run_eval.py "${COMMON_ARGS[@]}" --batch_size=${BATCH_SIZE}           --dataset="gigaspeech"  --split="test"
python run_eval.py "${COMMON_ARGS[@]}" --batch_size=${BATCH_SIZE}           --dataset="librispeech" --split="test.clean"
python run_eval.py "${COMMON_ARGS[@]}" --batch_size=${BATCH_SIZE}           --dataset="librispeech" --split="test.other"
python run_eval.py "${COMMON_ARGS[@]}" --batch_size=${BATCH_SIZE}           --dataset="spgispeech"  --split="test"
python run_eval.py "${COMMON_ARGS[@]}" --batch_size=${BATCH_SIZE}           --dataset="tedlium"     --split="test"

RUNDIR=$(pwd) && \
cd ../normalizer && \
python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
cd "${RUNDIR}"
