#!/bin/bash

# ASR Evaluation Script for Current Dataset Format
# Runs all possible combinations for each dataset automatically

export PYTHONPATH="..":$PYTHONPATH

# Configuration
MODEL_IDS=(
    "nvidia/parakeet-tdt-0.6b-v3"
    "nvidia/canary-1b-v2"
)

BATCH_SIZE=64

DEVICE_ID=0

# Available datasets and languages
DATASETS="nithinraok/asr-leaderboard-datasets"

DATASET_NAMES=("fleurs" "mcv" "mls")
DATASET_LANGS_fleurs="de fr it es pt"
DATASET_LANGS_mcv="de es fr it"
DATASET_LANGS_mls="es fr it pt"

# Function to run evaluation
run_evaluation() {
    local model_id=$1
    local dataset=$2
    local language=$3
    local config_name="${dataset}_${language}"
    
    echo ""
    echo "🚀 Running evaluation: $config_name"
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
RUNDIR=$(pwd)

for MODEL_ID in "${MODEL_IDS[@]}"; do
    echo "========================================================"
    echo "Model: $MODEL_ID"
    echo "Batch Size: $BATCH_SIZE"
    echo "Device: $DEVICE_ID"
    echo "========================================================"
    echo ""

    for dataset in "${DATASET_NAMES[@]}"; do
        varname="DATASET_LANGS_${dataset}"
        languages="${!varname}"
        if [[ -n "$languages" ]]; then
            
            echo "🗂️  Processing dataset: $dataset"
            echo "   Languages: $languages"
            echo ""
            
            for language in $languages; do
                run_evaluation "$MODEL_ID" "$dataset" "$language"
            done
        fi
    done

    echo ""
    echo "📊 Scoring results for $MODEL_ID"
    cd ../normalizer
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}', multilingual=True)"
    cd "$RUNDIR"
    echo "========================================================"
    echo ""
done
