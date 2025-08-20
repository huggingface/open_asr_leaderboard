#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH
export SONIOX_API_KEY="3ca7a2f013f953471042c2075107d993bc0d7265fcd3694f1e0f91b1ea763664"

DATASETS_PATH="nithinraok/asr-leaderboard-datasets"
MODEL_NAME="soniox/speech-to-text"
MODES=("async" "realtime")

declare -A EVAL_DATASETS
EVAL_DATASETS["fleurs"]="de fr it es pt"
EVAL_DATASETS["mcv"]="de fr it es"
EVAL_DATASETS["mls"]="fr it es pt"

for MODE in "${MODES[@]}"
do
    MODEL_ID="${MODEL_NAME}-${MODE}"
    for dataset in ${!EVAL_DATASETS[@]}; do
        for language in ${EVAL_DATASETS[$dataset]}; do
            config_name="${dataset}_${language}"
            echo "Running evaluation for ${config_name} with mode ${MODE}"
            PYTHONPATH="." python soniox/run_eval_ml.py \
                --dataset="${DATASETS_PATH}" \
                --config_name="${config_name}" \
                --split="test" \
                --model_name="${MODEL_ID}" \
                --language="${language}" \
                --mode="${MODE}"

            # Evaluate results
            RUNDIR=`pwd` && \
            cd "${RUNDIR}/normalizer" && \
            python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/soniox/results', '${MODEL_ID}')" && \
            cd $RUNDIR
        done
    done
done
