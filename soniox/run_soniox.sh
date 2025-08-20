#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH
export SONIOX_API_KEY="3ca7a2f013f953471042c2075107d993bc0d7265fcd3694f1e0f91b1ea763664"

MODES=("async" "realtime")
MODEL_NAME="soniox/speech-to-text"

for MODE in "${MODES[@]}"
do
    MODEL_ID="${MODEL_NAME}-${MODE}"
    python soniox/run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="ami" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --mode ${MODE} \
        --use_url

    python soniox/run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="earnings22" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --mode ${MODE} \
        --use_url

    python soniox/run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="gigaspeech" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --mode ${MODE} \
        --use_url

    python soniox/run_eval.py \
        --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
        --dataset "librispeech" \
        --split "test.clean" \
        --model_name ${MODEL_ID} \
        --mode ${MODE} \
        --use_url

    python soniox/run_eval.py \
        --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
        --dataset "librispeech" \
        --split "test.other" \
        --model_name ${MODEL_ID} \
        --mode ${MODE} \
        --use_url

    python soniox/run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="spgispeech" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --mode ${MODE} \
        --use_url

    python soniox/run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="tedlium" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --mode ${MODE} \
        --use_url

    python soniox/run_eval.py \
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
