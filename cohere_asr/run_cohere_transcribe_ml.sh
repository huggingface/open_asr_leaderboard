#!/bin/bash

export PYTHONPATH="../transformers_cohere/src:..":$PYTHONPATH
export LD_LIBRARY_PATH="/opt/conda/envs/cohere_trans/lib:${LD_LIBRARY_PATH}"

MODEL_IDs=("CohereLabs/cohere-transcribe-03-2026")
BATCH_SIZE=256
DEVICE_ID=0
DATASETS="nithinraok/asr-leaderboard-datasets"

DATASET_NAMES=("fleurs" "mcv" "mls")
DATASET_LANGS_fleurs="de fr it es pt"
DATASET_LANGS_mcv="de es fr it"
DATASET_LANGS_mls="es fr it pt"

run_evaluation() {
    local model_id=$1
    local dataset=$2
    local language=$3
    local config_name="${dataset}_${language}"

    echo ""
    echo "Running evaluation: ${config_name}"
    echo "  Model: ${model_id}"
    echo "  Dataset: ${dataset}"
    echo "  Language: ${language}"
    echo "  Device: ${DEVICE_ID}"
    echo "  Batch Size: ${BATCH_SIZE}"
    echo "----------------------------------------"

    python run_eval_ml.py \
        --model_id="${model_id}" \
        --dataset="${DATASETS}" \
        --config_name="${config_name}" \
        --language="${language}" \
        --split="test" \
        --device="${DEVICE_ID}" \
        --batch_size="${BATCH_SIZE}" \
        --max_eval_samples=-1

    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "Evaluation failed for ${config_name} (exit code: ${exit_code})"
    fi

    echo "----------------------------------------"
    return $exit_code
}

for MODEL_ID in "${MODEL_IDs[@]}"; do
    for dataset in "${DATASET_NAMES[@]}"; do
        varname="DATASET_LANGS_${dataset}"
        languages="${!varname}"

        if [[ -n "$languages" ]]; then
            for language in $languages; do
                run_evaluation "$MODEL_ID" "$dataset" "$language"
            done
        fi
    done

    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}',multilingual=True)" && \
    cd $RUNDIR
done
