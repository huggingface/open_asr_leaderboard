#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

# Available omniASR models
MODEL_IDs=(
    "facebook/omniASR-CTC-300M" "facebook/omniASR-CTC-1B" "facebook/omniASR-CTC-3B" "facebook/omniASR-CTC-7B"
    "facebook/omniASR-CTC-300M-v2" "facebook/omniASR-CTC-1B-v2" "facebook/omniASR-CTC-3B-v2" "facebook/omniASR-CTC-7B-v2"
    "facebook/omniASR-LLM-300M" "facebook/omniASR-LLM-1B" "facebook/omniASR-LLM-3B" "facebook/omniASR-LLM-7B"
    "facebook/omniASR-LLM-300M-v2" "facebook/omniASR-LLM-1B-v2" "facebook/omniASR-LLM-3B-v2" "facebook/omniASR-LLM-7B-v2"
    )
BATCH_SIZE=64  # Conservative batch size due to LLM memory requirements

# Multilingual datasets and languages
DATASETS="nithinraok/asr-leaderboard-datasets"

DATASET_NAMES=("fleurs" "mcv" "mls")
DATASET_LANGS_fleurs="de fr it es pt"
DATASET_LANGS_mcv="de es fr it"
DATASET_LANGS_mls="es fr it pt"

# Function to run multilingual evaluation
run_evaluation() {
    local model_id=$1
    local dataset=$2
    local language=$3
    local config_name="${dataset}_${language}"

    echo ""
    echo "Running multilingual evaluation: $config_name"
    echo "   Model: $model_id"
    echo "   Dataset: $dataset"
    echo "   Language: $language"
    echo "   Time: $(date)"
    echo "----------------------------------------"

    python run_eval_ml.py \
        --model_id="$model_id" \
        --dataset="$DATASETS" \
        --config_name="$config_name" \
        --language="$language" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "Evaluation completed successfully for $config_name"
    else
        echo "Evaluation failed for $config_name (exit code: $exit_code)"
    fi

    echo "----------------------------------------"
    return $exit_code
}

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}

    echo "========================================================"
    echo "Model: $MODEL_ID"
    echo "========================================================"

    for dataset in "${DATASET_NAMES[@]}"; do
        varname="DATASET_LANGS_${dataset}"
        languages="${!varname}"
        if [[ -n "$languages" ]]; then

            echo "Processing multilingual dataset: $dataset"
            echo "   Languages: $languages"
            echo ""

            for language in $languages; do
                run_evaluation "$MODEL_ID" "$dataset" "$language"
            done
        fi
    done

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}', multilingual=True)" && \
    cd $RUNDIR

done
