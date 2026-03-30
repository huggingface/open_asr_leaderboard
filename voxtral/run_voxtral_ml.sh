#!/bin/bash

# Multilingual ASR Evaluation Script for Voxtral
# Evaluates on FLEURS, MCV (Mozilla Common Voice), and MLS (Multilingual LibriSpeech)

export PYTHONPATH="..":$PYTHONPATH

# Configuration
MODEL_IDs=(
    "mistralai/Voxtral-Mini-3B-2507"
    "mistralai/Voxtral-Small-24B-2507"
)

DEVICE_ID=0

# Available datasets and languages
DATASETS="nithinraok/asr-leaderboard-datasets"

# Voxtral supports: English, Spanish, French, Portuguese, Hindi, German, Dutch, Italian
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
    echo "   Batch Size: $batch_size"
    echo "   Time: $(date)"
    echo "----------------------------------------"

    python run_eval_ml.py \
        --model_id="$model_id" \
        --dataset="$DATASETS" \
        --config_name="$config_name" \
        --split="test" \
        --device="$DEVICE_ID" \
        --batch_size="$batch_size" \
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
echo "Starting Voxtral Multilingual Evaluation"
echo "Device: $DEVICE_ID"
echo ""

# Run evaluations for all models
for MODEL_ID in "${MODEL_IDs[@]}"; do
    # Per-model batch size
    if [[ "$MODEL_ID" == *"24B"* ]]; then
        batch_size=24
    else
        batch_size=64
    fi

    echo ""
    echo "Processing Model: $MODEL_ID (batch_size=$batch_size)"
    echo "========================================================"

    # Run evaluations for all datasets and languages
    for dataset in "${DATASET_NAMES[@]}"; do
        varname="DATASET_LANGS_${dataset}"
        languages="${!varname}"

        if [[ -n "$languages" ]]; then
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
    echo "📊 Evaluating results for $MODEL_ID"
    echo "========================================================"

    # Evaluate results
    RUNDIR=`pwd`
    cd ../normalizer
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}', multilingual=True)"
    cd "$RUNDIR"

    echo ""
done

echo ""
echo "========================================================"
echo "✅ All evaluations completed!"
echo "========================================================"
