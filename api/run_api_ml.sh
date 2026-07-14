#!/bin/bash

# Multilingual API ASR Evaluation Script
# Evaluates on FLEURS, MCV (Mozilla Common Voice), and MLS (Multilingual LibriSpeech)

export PYTHONPATH="..":$PYTHONPATH

export OPENAI_API_KEY="your_api_key"
export ASSEMBLYAI_API_KEY="your_api_key"
export ELEVENLABS_API_KEY="your_api_key"
export REVAI_API_KEY="your_api_key"
export AQUAVOICE_API_KEY="your_api_key"
export SPEECHMATICS_API_KEY="your_api_key"
export RESON8_API_KEY="your_api_key"
export AZURE_API_KEY="your_api_key"
export SONIOX_API_KEY="${SONIOX_API_KEY:-}"

# Configuration
MODEL_IDs=(
    # "openai/gpt-4o-transcribe"
    # "openai/gpt-4o-mini-transcribe"
    # "openai/whisper-1"
    # "assembly/universal-3-pro"
    # "elevenlabs/scribe_v2"
    # "speechmatics/enhanced"
    # "soniox/stt-async-v5"
    "reson8/resonant-1"
    "reson8/resonant-1-flash"
    "microsoft/azure-speech-05-2026"
)

MAX_WORKERS="${MAX_WORKERS:-20}"
DATASET_PATH="nithinraok/asr-leaderboard-datasets"

# German, French, Italian, Spanish, Portuguese
DATASET_NAMES=("fleurs" "mls" "mcv")
DATASET_LANGS_fleurs="de fr it es pt"
DATASET_LANGS_mcv="de es fr it"
DATASET_LANGS_mls="es fr it pt"

# Run a single provider without editing this file, for example:
#   SONIOX_API_KEY=... MODEL_ID=soniox/stt-async-v5 bash run_api_ml.sh
if [[ -n "${MODEL_ID:-}" ]]; then
    MODEL_IDs=("$MODEL_ID")
fi

# Datasets that require lexical format prompt (azure only)
LEXICAL_DATASETS="mls-it"
RESUME_ARGS=()
if [[ "${RESUME:-1}" != "0" ]]; then
    RESUME_ARGS=(--resume)
fi

# Function to run evaluation
run_evaluation() {
    local model_id=$1
    local dataset=$2
    local language=$3
    local config_name="${dataset}_${language}"

    # Build prompt args for azure + lexical datasets
    local prompt_args=()
    if [[ "$model_id" == microsoft/* ]] && [[ " $LEXICAL_DATASETS " == *" ${dataset}-${language} "* ]]; then
        prompt_args=(--prompt "Output must be in lexical format.")
    fi

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
        --max_workers="$MAX_WORKERS" \
        ${MAX_SAMPLES:+--max_samples="$MAX_SAMPLES"} \
        ${USE_URL:+--use_url} \
        "${RESUME_ARGS[@]}" \
        "${prompt_args[@]}"

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
    for dataset in "${DATASET_NAMES[@]}"; do
        varname="DATASET_LANGS_${dataset}"
        languages="${!varname}"

        if [[ -n "$languages" ]]; then
            echo ""
            echo "Processing dataset: $dataset"
            echo "   Languages: $languages"
            echo ""

            for language in $languages; do
                if ! run_evaluation "$MODEL_ID" "$dataset" "$language"; then
                    echo "Stopping after failed evaluation: ${dataset}_${language}" >&2
                    exit 1
                fi
            done
        fi
    done

    echo ""
    echo "========================================================"
    echo "Evaluating results for $MODEL_ID"
    echo "========================================================"

    # Score every config with its own language normalizer and print a row in
    # the exact column order expected by scripts/data/multilingual.csv.
    python score_multilingual_results.py \
        --results-dir "$(pwd)/results" \
        --model-id "$MODEL_ID" \
        --dataset-path "$DATASET_PATH"

    echo ""
done

echo ""
echo "========================================================"
echo "All evaluations completed!"
echo "========================================================"
