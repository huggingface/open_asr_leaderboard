#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH
RUNDIR=`pwd`
MODEL_IDs=(
    "collab-external/cohere-model-testing-TODO-SWITCH"
)
BATCH_SIZE=256
DEVICE_ID=0

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}
    if [[ -d "$MODEL_ID" ]]; then # local path
        short_model_id="${MODEL_ID##*/}"
    else # HF repo id
        short_model_id="${MODEL_ID#*/}"
    fi
    RESULTS_DIR="${RUNDIR}/benchmarks/results-${short_model_id}/longform"
    python run_eval.py \
    --model_id=${MODEL_ID} \
    --dataset_path="hf-audio/asr-leaderboard-longform" \
    --dataset="earnings21" \
    --split="test" \
    --device=${DEVICE_ID} \
    --batch_size=${BATCH_SIZE} \
    --max_eval_samples=-1 \
    --basedir="$RESULTS_DIR"

    python run_eval.py \
    --model_id=${MODEL_ID} \
    --dataset_path="hf-audio/asr-leaderboard-longform" \
    --dataset="earnings22" \
    --split="test" \
    --device=${DEVICE_ID} \
    --batch_size=${BATCH_SIZE} \
    --max_eval_samples=-1 \
    --basedir="$RESULTS_DIR"

    python run_eval.py \
    --model_id=${MODEL_ID} \
    --dataset_path="hf-audio/asr-leaderboard-longform" \
    --dataset="tedlium" \
    --split="test" \
    --device=${DEVICE_ID} \
    --batch_size=${BATCH_SIZE} \
    --max_eval_samples=-1 \
    --basedir="$RESULTS_DIR"

    for SUBSET in ATL DCA DCB DTA LES PRV ROC VLD; do
        python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="bezzam/coraal" \
        --dataset=${SUBSET} \
        --split="test" \
        --device=${DEVICE_ID} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --basedir="$RESULTS_DIR"
    done

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
