#!/bin/bash

# ASR Evaluation Script for Current Dataset Format
# Runs all possible combinations for each dataset automatically
# Usage: ./run_canary_new.sh

export PYTHONPATH="..":$PYTHONPATH

# Configuration
MODEL_ID="nvidia/parakeet-tdt-0.6b-v3"  #"nvidia/canary-1b-v2"

BATCH_SIZE=64

DEVICE_ID=0

# Available datasets and languages
DATASETS="nithinraok/asr-leaderboard-datasets"

# German, French, Italian, Spanish, Portuguese
declare -A EVAL_DATASETS
EVAL_DATASETS["fleurs"]="en de fr it es pt"
#cs da de el en es et fi fr hr hu it lt lv mt nl pl pt ro ru sk sl sv uk"
EVAL_DATASETS["mcv"]="en de es fr it"
# "de en es et fr it lv nl pt ru sl sv uk"
EVAL_DATASETS["mls"]="es fr it pt"
# "es" #fr it nl pl pt"

# Function to run evaluation
run_evaluation() {
    local dataset=$1
    local language=$2
    local config_name="${dataset}_${language}"
    
    echo ""
    echo "🚀 Running evaluation: $config_name"
    echo "   Model: $MODEL_ID"
    echo "   Dataset: $dataset"
    echo "   Language: $language"
    echo "   Device: $DEVICE_ID"
    echo "   Batch Size: $BATCH_SIZE"
    echo "   Time: $(date)"
    echo "----------------------------------------"
    
    python run_eval_ml.py \
        --model_id="$MODEL_ID" \
        --dataset="$DATASETS" \
        --config_name="$config_name" \
        --language="$language" \
        --split="test" \
        --device="$DEVICE_ID" \
        --batch_size="$BATCH_SIZE" \
        --max_eval_samples=-1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ Evaluation completed successfully for $config_name"
    else
        echo "❌ Evaluation failed for $config_name (exit code: $exit_code)"
    fi
    
    echo "----------------------------------------"
    return $exit_code
}

# Main execution
echo "========================================================"
echo "Model: $MODEL_ID"
echo "Batch Size: $BATCH_SIZE"
echo "Device: $DEVICE_ID"
echo ""

# Run evaluations for all datasets and languages
for dataset in ${!EVAL_DATASETS[@]}; do  # Process in specific order
    if [[ ${EVAL_DATASETS[$dataset]} ]]; then
        languages=${EVAL_DATASETS[$dataset]}
        
        echo "🗂️  Processing dataset: $dataset"
        echo "   Languages: $languages"
        echo ""
        
        for language in $languages; do
            run_evaluation "$dataset" "$language"
        done
    fi
done


echo "========================================================"

# Evaluate results 
RUNDIR=`pwd`
cd ../normalizer
python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')"
cd "$RUNDIR"
