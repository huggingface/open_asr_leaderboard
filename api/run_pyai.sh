#!/bin/bash
# Evaluate PyAI Hear on the Open ASR Leaderboard benchmarks.
# Sign up for a free API key at https://api.pyai.com (1M free min/month).

export PYTHONPATH="..":$PYTHONPATH
export PYAI_API_KEY="your_pyai_api_key"
export HF_TOKEN="hf_your_key"

MODEL_IDs=(
    "pyai/hear-v4"
)

MAX_WORKERS=10
DATASET_PATH="hf-audio/open-asr-leaderboard"

declare -a EVAL_DATASETS=(
    "ami:test"
    "earnings22:test"
    "gigaspeech:test"
    "librispeech:test.clean"
    "librispeech:test.other"
    "spgispeech:test"
    "tedlium:test"
    "voxpopuli:test"
    "common_voice:test"
)

for MODEL_ID in "${MODEL_IDs[@]}"; do
    for DATASET_SPLIT in "${EVAL_DATASETS[@]}"; do
        DATASET=$(echo $DATASET_SPLIT | cut -d: -f1)
        SPLIT=$(echo $DATASET_SPLIT | cut -d: -f2)

        echo "Evaluating $MODEL_ID on $DATASET ($SPLIT)..."
        python run_eval.py \
            --model_id "$MODEL_ID" \
            --dataset_path "$DATASET_PATH" \
            --dataset "$DATASET" \
            --split "$SPLIT" \
            --max_workers "$MAX_WORKERS" \
            --device "cpu" \
            2>&1 | tee "logs/${MODEL_ID//\//_}_${DATASET}_${SPLIT}.log"
    done
done
