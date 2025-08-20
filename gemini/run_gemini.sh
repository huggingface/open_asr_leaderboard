#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

MODEL_IDs=(
    "gemini/gemini-2.5-pro"
    "gemini/gemini-2.5-flash"
)

for MODEL_ID in "${MODEL_IDs[@]}"
do
    echo "--- Running Benchmarks for $MODEL_ID ---"

    # --- English Benchmarks ---
    echo "--- Running English Benchmarks ---"
    python gemini/run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="ami" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_samples 2

    python gemini/run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="earnings22" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_samples 2

    python gemini/run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="gigaspeech" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_samples 2

    python gemini/run_eval.py \
        --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
        --dataset "librispeech" \
        --split "test.clean" \
        --model_name ${MODEL_ID} \
        --max_samples 2

    python gemini/run_eval.py \
        --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
        --dataset "librispeech" \
        --split "test.other" \
        --model_name ${MODEL_ID} \
        --max_samples 2

    python gemini/run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="spgispeech" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_samples 2

    python gemini/run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="tedlium" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_samples 2

    python gemini/run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="voxpopuli" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_samples 2

    # --- Multilingual Benchmarks ---
    echo "--- Running Multilingual Benchmarks ---"
    declare -A EVAL_DATASETS
    EVAL_DATASETS["fleurs"]="en de fr it es pt"
    EVAL_DATASETS["mcv"]="en de es fr it"
    EVAL_DATASETS["mls"]="es fr it pt"

    for dataset in ${!EVAL_DATASETS[@]}; do
        for language in ${EVAL_DATASETS[$dataset]}; do
            config_name="${dataset}_${language}"
            echo "Running evaluation for $config_name"
            python gemini/run_eval_ml.py \
                --model_name="$MODEL_ID" \
                --dataset="nithinraok/asr-leaderboard-datasets" \
                --config_name="$config_name" \
                --language="$language" \
                --split="test" \
                --max_samples 2
        done
    done

    # --- Scoring ---
    echo "--- Scoring results for $MODEL_ID ---"
    RUNDIR=$(pwd)
    cd normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done

echo "--- All benchmarks complete ---"
