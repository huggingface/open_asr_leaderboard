#!/bin/bash

# Multilingual ASR Evaluation Script for Phi-4 Multimodal
# Evaluates on FLEURS, MCV (Mozilla Common Voice), and MLS (Multilingual LibriSpeech)

export PYTHONPATH="..":$PYTHONPATH

# Configuration
MODEL_IDs=("microsoft/Phi-4-multimodal-instruct")

BATCH_SIZE=64
NUM_BEAMS=1
MAX_NEW_TOKENS=128
DEVICE_ID=0

# Available datasets and languages
DATASETS="nithinraok/asr-leaderboard-datasets"

default_user_prompt="Transcribe the audio clip into text."

# German, French, Italian, Spanish, Portuguese
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

    python run_eval_ml.py \
        --model_id="$model_id" \
        --dataset="$DATASETS" \
        --config_name="$config_name" \
        --split="test" \
        --device="$DEVICE_ID" \
        --batch_size="$BATCH_SIZE" \
        --num_beams="$NUM_BEAMS" \
        --max_eval_samples=-1 \
        --max_new_tokens="$MAX_NEW_TOKENS" \
        --user_prompt="$default_user_prompt"

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
echo "Starting Phi-4 Multimodal Multilingual Evaluation"
echo "Batch Size: $BATCH_SIZE"
echo "Device: $DEVICE_ID"
echo ""

# Run evaluations for all models
for MODEL_ID in "${MODEL_IDs[@]}"; do
    echo ""
    echo "Processing Model: $MODEL_ID"
    echo "========================================================"

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
    RUNDIR=`pwd`
    cd ../normalizer
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')"
    cd "$RUNDIR"

    echo ""
done

echo ""
echo "========================================================"
echo "All evaluations completed!"
echo "========================================================"
