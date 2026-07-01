#!/bin/bash

export PYTHONPATH="..:../api":$PYTHONPATH

export GLADIA_API_KEY="your_api_key"
export HF_TOKEN="hf_your_key"

# Add new Gladia models here as gladia/<variant> (see api/providers/gladia_provider.py).
MODEL_IDs=(
    "gladia/solaria-3"
    # "gladia/solaria-4"
)

MAX_WORKERS=20
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
)

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}

    for entry in "${EVAL_DATASETS[@]}"; do
        DATASET="${entry%%:*}"
        SPLIT="${entry##*}"

        python ../api/run_eval.py \
            --dataset_path="${DATASET_PATH}" \
            --dataset="${DATASET}" \
            --split="${SPLIT}" \
            --model_name="${MODEL_ID}" \
            --max_workers="${MAX_WORKERS}"
    done

    RUNDIR=$(pwd)
    cd ../normalizer
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')"
    cd "$RUNDIR"
done
