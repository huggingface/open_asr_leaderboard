#!/bin/bash

# Multilingual ASR Evaluation Script for Cohere ASR
# Evaluates on FLEURS, MCV (Mozilla Common Voice), and MLS (Multilingual LibriSpeech)

export PYTHONPATH="..":$PYTHONPATH
RUNDIR=`pwd`

# Configuration
MODEL_IDs=(
    "collab-external/cohere-model-testing-TODO-SWITCH"
)
BATCH_SIZE=256
DEVICE_ID=0

# Available datasets and languages
DATASETS="nithinraok/asr-leaderboard-datasets"

# Cohere ASR supports 14 languages: en, es, fr, de, it, pt, nl, el, pl, ar, ja, ko, vi, zh
declare -A EVAL_DATASETS
EVAL_DATASETS["fleurs"]="de fr it es pt"
EVAL_DATASETS["mcv"]="de es fr it"
EVAL_DATASETS["mls"]="es fr it pt"

# Function to run evaluation
run_evaluation() {
    local model_id=$1
    local dataset=$2
    local language=$3
    local config_name="${dataset}_${language}"

    echo ""
    echo "Running evaluation: $config_name"
    echo "   Model: $model_id"
    echo "   Dataset: $dataset"
    echo "   Language: $language"
    echo "   Device: $DEVICE_ID"
    echo "   Batch Size: $BATCH_SIZE"
    echo "   Time: $(date)"
    echo "----------------------------------------"

    python run_eval.py \
        --model_id="$model_id" \
        --dataset_path="$DATASETS" \
        --dataset="$config_name" \
        --split="test" \
        --device="$DEVICE_ID" \
        --batch_size="$BATCH_SIZE" \
        --max_eval_samples=-1 \
        --language="$language" \
        --basedir="$RESULTS_DIR"

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "Evaluation completed successfully for $config_name"
    else
        echo "Evaluation failed for $config_name (exit code: $exit_code)"
    fi

    echo "----------------------------------------"
    return $exit_code
}

# Main execution
echo "========================================================"
echo "Starting Cohere ASR Multilingual Evaluation"
echo "Batch Size: $BATCH_SIZE"
echo "Device: $DEVICE_ID"
echo ""

# Run evaluations for all models
for MODEL_ID in "${MODEL_IDs[@]}"; do
    echo ""
    echo "Processing Model: $MODEL_ID"
    echo "========================================================"

    if [[ -d "$MODEL_ID" ]]; then # local path 
        short_model_id="${MODEL_ID##*/}"
    else # HF repo id
        short_model_id="${MODEL_ID#*/}"
    fi
    
    RESULTS_DIR="${RUNDIR}/benchmarks/results-${short_model_id}/ml"

    # Run evaluations for all datasets and languages
    for dataset in "${!EVAL_DATASETS[@]}"; do
        if [[ ${EVAL_DATASETS[$dataset]} ]]; then
            languages=${EVAL_DATASETS[$dataset]}

            echo ""
            echo "Processing dataset: $dataset"
            echo "   Languages: $languages"
            echo ""

            for language in $languages; do
                run_evaluation "$MODEL_ID" "$dataset" "$language"
            done
        fi
    done

    echo ""
    echo "========================================================"
    echo "Evaluating results for $MODEL_ID"
    echo "========================================================"

    # Evaluate results
    cd ../normalizer
    python -c "import eval_utils; eval_utils.score_results('${RESULTS_DIR}', '${MODEL_ID}')"
    cd $RUNDIR

done

echo ""
echo "========================================================"
echo "All evaluations completed!"
echo "========================================================"
