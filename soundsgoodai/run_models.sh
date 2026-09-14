#!/bin/bash
# Run the shared evaluation matrix locally.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/config.sh"

RUN_ID=fast-gpu-asr-$(date -u +%Y%m%dT%H%M%S)

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    return
fi

export PYTHONPATH=${SCRIPT_DIR}/..:${PYTHONPATH:-}
ENGINE_CACHE=$(realpath -m "${ENGINE_CACHE:-${SCRIPT_DIR}/engines}")
RUN_DIR=${SCRIPT_DIR}/runs/${RUN_ID}
for MODEL_CONFIG in "${MODEL_CONFIGS[@]}"; do
    read -r MODEL_ID MODEL_TYPE CHECKPOINT_FILE DECODER_TYPE BEAM BATCH_SIZE <<< ${MODEL_CONFIG}

    MODEL_DIR=${RUN_DIR}/${MODEL_ID//\//-}/${DECODER_TYPE}
    mkdir -p "${MODEL_DIR}"
    cd "${MODEL_DIR}"

    echo "Results: ${MODEL_DIR}/results"
    for CONFIG in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET SPLIT DATASET_PATH <<< ${CONFIG}

        if [[ -n ${DATASET_PATH} ]]; then
            DATASET=
        fi

        python "${SCRIPT_DIR}/run_eval.py" "${COMMON_ARGS[@]}" \
            --engine-cache="${ENGINE_CACHE}" \
            --model-id="${MODEL_ID}" \
            --model-family="${MODEL_TYPE}" \
            --checkpoint-file="${CHECKPOINT_FILE}" \
            --decoder-type="${DECODER_TYPE}" \
            --beam="${BEAM}" \
            --batch-size="${BATCH_SIZE}" \
            --dataset-path="${DATASET_PATH:-${DEFAULT_DATASET_PATH}}" \
            --dataset="${DATASET}" \
            --split="${SPLIT}"
    done

    python -c 'import sys; from normalizer.eval_utils import score_results; score_results("results", sys.argv[1])' "${MODEL_ID}"
done
