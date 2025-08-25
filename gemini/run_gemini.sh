#!/bin/bash

# Set PYTHONPATH to include parent directory for proper imports
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/..:$PYTHONPATH"

# Load environment variables from .env if present (not required but convenient)
if [ -f "${SCRIPT_DIR}/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    . "${SCRIPT_DIR}/.env"
    set +a
fi

# Use Python from PATH by default (override with PYTHON_CMD if desired)
PYTHON_CMD=${PYTHON_CMD:-python3}

# Check if Google API key is set
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "Error: GOOGLE_API_KEY environment variable not set"
    echo "Please set it with: export GOOGLE_API_KEY='your_api_key_here'"
    exit 1
fi

# Verify Python installation
if ! command -v "$PYTHON_CMD" &> /dev/null; then
    echo "Warning: Python not found at $PYTHON_CMD"
    if command -v python3 &> /dev/null; then
        echo "Falling back to python3"
        PYTHON_CMD="python3"
    else
        echo "Falling back to python"
        PYTHON_CMD="python"
    fi
fi

MODEL_IDs=(
    "gemini/gemini-2.5-pro"
    "gemini/gemini-2.5-flash"
)

# Test with small samples first
TEST_SAMPLES=2

for MODEL_ID in "${MODEL_IDs[@]}"
do
    echo "--- Running Benchmarks for $MODEL_ID ---"
    echo "Using Python: $PYTHON_CMD"
    
    # Test one sample first to verify setup
    echo "--- Testing setup with AMI dataset ---"
    if ! "$PYTHON_CMD" run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="ami" \
        --split="test" \
        --model_id "${MODEL_ID}" \
        --max_eval_samples 1; then
        echo "Error: Failed to run test evaluation for $MODEL_ID"
        echo "Skipping this model..."
        continue
    fi
    
    echo "--- Setup verified, continuing with full English benchmarks ---"

    # --- English Benchmarks ---
    echo "--- Running English Benchmarks ---"
    
    for dataset in "earnings22" "gigaspeech" "librispeech" "spgispeech" "tedlium" "voxpopuli"; do
        echo "Processing dataset: $dataset"
        
        if [ "$dataset" = "librispeech" ]; then
            # LibriSpeech has multiple splits
            for split in "test.clean" "test.other"; do
                echo "Running $dataset/$split..."
                "$PYTHON_CMD" run_eval.py \
                    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
                    --dataset="$dataset" \
                    --split="$split" \
                    --model_id "${MODEL_ID}" \
                    --max_eval_samples $TEST_SAMPLES || echo "Warning: Failed to process $dataset/$split"
            done
        else
            echo "Running $dataset..."
            "$PYTHON_CMD" run_eval.py \
                --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
                --dataset="$dataset" \
                --split="test" \
                --model_id "${MODEL_ID}" \
                --max_eval_samples $TEST_SAMPLES || echo "Warning: Failed to process $dataset"
        fi
    done

    # --- Multilingual Benchmarks ---
    echo "--- Running Multilingual Benchmarks ---"
    declare -A EVAL_DATASETS
    EVAL_DATASETS["fleurs"]="en de fr it es pt"
    EVAL_DATASETS["mcv"]="en de es fr it"
    EVAL_DATASETS["mls"]="es fr it pt"

    for dataset in ${!EVAL_DATASETS[@]}; do
        echo "Processing multilingual dataset: $dataset"
        for language in ${EVAL_DATASETS[$dataset]}; do
            config_name="${dataset}_${language}"
            echo "Running evaluation for $config_name"
            "$PYTHON_CMD" run_eval_ml.py \
                --model_id="$MODEL_ID" \
                --dataset="nithinraok/asr-leaderboard-datasets" \
                --config_name="$config_name" \
                --language="$language" \
                --split="test" \
                --max_eval_samples $TEST_SAMPLES || echo "Warning: Failed to process $config_name"
        done
    done

    # --- Scoring ---
    echo "--- Scoring results for $MODEL_ID ---"
    RUNDIR=$(pwd)
    if cd ../normalizer; then
        "$PYTHON_CMD" -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" || echo "Warning: Scoring failed for $MODEL_ID"
        cd "$RUNDIR" || echo "Warning: Could not return to original directory"
    else
        echo "Warning: Could not access normalizer directory for scoring"
    fi
    
    echo "--- Completed benchmarks for $MODEL_ID ---"
done

echo "--- All benchmarks complete ---"
