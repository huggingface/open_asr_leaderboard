#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

export GLADIA_API_KEY="your_api_key"
export HF_TOKEN="hf_your_key"

MODEL_ID="gladia/solaria-3"
MAX_WORKERS=20

declare -a EVAL_DATASETS=(
    "ami:test"
    "earnings22:test"
    "gigaspeech:test"
    "librispeech:test.clean"
    "librispeech:test.other"
    "spgispeech:test"
    "tedlium:test"
    "voxpopuli:test"
)

for entry in "${EVAL_DATASETS[@]}"; do
    DATASET="${entry%%:*}"
    SPLIT="${entry##*:}"

    python run_eval.py \
        --model_id="${MODEL_ID}" \
        --dataset_path="hf-audio/open-asr-leaderboard" \
        --dataset="${DATASET}" \
        --split="${SPLIT}" \
        --max_workers="${MAX_WORKERS}"
done

RUNDIR=$(pwd)
cd ../normalizer
python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')"
cd "$RUNDIR"
