#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

# Set your Soniox API key as an environment variable before running this script:
# export SONIOX_API_KEY="your-api-key-here"
if [ -z "$SONIOX_API_KEY" ]; then
    echo "Error: SONIOX_API_KEY environment variable is not set"
    echo "Please set your API key: export SONIOX_API_KEY=\"your-api-key-here\""
    exit 1
fi

MODES=("async" "realtime")
MODEL_NAME="soniox/speech-to-text"

for MODE in "${MODES[@]}"
do
    MODEL_ID="${MODEL_NAME}-${MODE}"
    PYTHONPATH="." python soniox/run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="ami" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --mode ${MODE} \
        --use_url

    PYTHONPATH="." python soniox/run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="earnings22" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --mode ${MODE} \
        --use_url

    PYTHONPATH="." python soniox/run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="gigaspeech" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --mode ${MODE} \
        --use_url

    PYTHONPATH="." python soniox/run_eval.py \
        --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
        --dataset "librispeech" \
        --split "test.clean" \
        --model_name ${MODEL_ID} \
        --mode ${MODE} \
        --use_url

    PYTHONPATH="." python soniox/run_eval.py \
        --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
        --dataset "librispeech" \
        --split "test.other" \
        --model_name ${MODEL_ID} \
        --mode ${MODE} \
        --use_url

    PYTHONPATH="." python soniox/run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="spgispeech" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --mode ${MODE} \
        --use_url

    PYTHONPATH="." python soniox/run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="tedlium" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --mode ${MODE} \
        --use_url

    PYTHONPATH="." python soniox/run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="voxpopuli" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --mode ${MODE} \
        --use_url

    # Evaluate results
    RUNDIR=`pwd` && \
    cd "${RUNDIR}/normalizer" && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/soniox/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
