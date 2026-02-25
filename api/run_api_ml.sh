#!/bin/bash

# Multilingual API ASR Evaluation Script
# Evaluates on FLEURS, MCV (Mozilla Common Voice), and MLS (Multilingual LibriSpeech)

export PYTHONPATH="..":$PYTHONPATH

export OPENAI_API_KEY="your_api_key"
export ASSEMBLYAI_API_KEY="your_api_key"
export ELEVENLABS_API_KEY="your_api_key"
export REVAI_API_KEY="your_api_key"
export AQUAVOICE_API_KEY="your_api_key"

# Configuration
MODEL_IDs=(
    "openai/gpt-4o-transcribe"
    "openai/gpt-4o-mini-transcribe"
    "openai/whisper-1"
    "assembly/universal-3-pro"
    "elevenlabs/scribe_v1"
    "speechmatics/enhanced"
)

MAX_WORKERS=10
DATASET_PATH="nithinraok/asr-leaderboard-datasets"

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
    echo "   Time: $(date)"
    echo "----------------------------------------"

    python run_eval_ml.py \
        --dataset_path="$DATASET_PATH" \
        --config_name="$config_name" \
        --language="$language" \
        --split="test" \
        --model_name="$model_id" \
        --max_workers="$MAX_WORKERS"

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
echo "Starting Multilingual API ASR Evaluation"
echo "========================================================"

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
