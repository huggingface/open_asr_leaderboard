#!/bin/bash
# Run the shared evaluation matrix locally.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONFIG=${CONFIG:-config.sh}
[[ ${CONFIG} == */* ]] || CONFIG=${SCRIPT_DIR}/${CONFIG}
if [[ ! -f ${CONFIG} ]]; then
    echo "Config not found: ${CONFIG}" >&2
    exit 1
fi
source "${CONFIG}"

RUN_ID=${RUN_ID:-fast-gpu-asr-$(date -u +%Y%m%dT%H%M%S)}

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    return
fi

MULTILINGUAL=${MULTILINGUAL:-0}
EVAL_SCRIPT=run_eval.py
LANGUAGES=(en)
if [[ ${MULTILINGUAL} == 1 ]]; then
    EVAL_SCRIPT=run_eval_ml.py
    LANGUAGES=()
    for CONFIG in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET _ <<< "${CONFIG}"
        LANGUAGES+=("${DATASET##*_}")
    done
fi

export PYTHONPATH=${SCRIPT_DIR}/..:${PYTHONPATH:-}
ENGINE_CACHE=$(realpath -m "${ENGINE_CACHE:-${SCRIPT_DIR}/engines}")
RUN_DIR=${SCRIPT_DIR}/runs/${RUN_ID}
for MODEL_CONFIG in "${MODEL_CONFIGS[@]}"; do
    read -r MODEL_ID MODEL_TYPE CHECKPOINT_FILE DECODER_TYPE BEAM BATCH_SIZE WORKERS REPORT_SUFFIX <<< ${MODEL_CONFIG}
    # Reported name; also names the result folder.
    REPORT_NAME=${MODEL_ID}${REPORT_SUFFIX:+ ${REPORT_SUFFIX}}

    MODEL_DIR=${RUN_DIR}/$(model_folder "${REPORT_NAME}")
    mkdir -p "${MODEL_DIR}"
    cd "${MODEL_DIR}"

    echo "Results: ${MODEL_DIR}/results"
    for CONFIG in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET SPLIT DATASET_PATH <<< ${CONFIG}

        LANGUAGE_ARGS=()
        if [[ ${MULTILINGUAL} == 1 ]]; then
            LANGUAGE_ARGS=(--language="${DATASET##*_}")
        fi
        if [[ -n ${DATASET_PATH} ]]; then
            DATASET=
        fi

        python "${SCRIPT_DIR}/${EVAL_SCRIPT}" "${COMMON_ARGS[@]}" "${LANGUAGE_ARGS[@]}" \
            --engine-cache="${ENGINE_CACHE}" \
            --model-id="${MODEL_ID}" \
            --report-name="${REPORT_NAME}" \
            --model-family="${MODEL_TYPE}" \
            --checkpoint-file="${CHECKPOINT_FILE}" \
            --decoder-type="${DECODER_TYPE}" \
            --beam="${BEAM}" \
            --batch-size="${BATCH_SIZE}" \
            --workers="${WORKERS}" \
            --dataset-path="${DATASET_PATH:-${DEFAULT_DATASET_PATH}}" \
            --dataset="${DATASET}" \
            --split="${SPLIT}"
    done

    python - "${REPORT_NAME}" "${LANGUAGES[@]}" <<'PY'
import sys
from normalizer.eval_utils import score_results

for language in set(sys.argv[2:]):
    multilingual = language != "en"
    lang_family = [f"ml_{language}"] if multilingual else None
    score_results(
        "results", sys.argv[1], multilingual, language=language, families=lang_family
    )
PY
done
