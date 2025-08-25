#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

# Set your Soniox API key as an environment variable before running this script:
# export SONIOX_API_KEY="your-api-key-here"
if [ -z "$SONIOX_API_KEY" ]; then
    echo "Error: SONIOX_API_KEY environment variable is not set"
    echo "Please set your API key: export SONIOX_API_KEY=\"your-api-key-here\""
    exit 1
fi

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
