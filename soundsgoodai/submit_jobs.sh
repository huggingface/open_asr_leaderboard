#!/bin/bash
# Submit one HF job per model, exporting once and evaluating all datasets.
# Usage: HF_TOKEN=hf_... bash soundsgoodai/submit_jobs.sh
set -euo pipefail
shopt -s nullglob

SPACE=${SPACE:-hf-audio/open-asr-leaderboard-zipformer}
RESULTS_BUCKET=${RESULTS_BUCKET:-hf-audio/asr_leaderboard_h200}
FLAVOR=${FLAVOR:-h200}
ORG_NAME=${ORG_NAME:-}
RUN_ID=${RUN_ID:-fast-gpu-asr-$(date -u +%Y%m%dT%H%M%S)}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/config.sh"
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
LOCAL_DIR=${SCRIPT_DIR}/results/${RUN_ID}
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}

# Set USE_LOCAL_NORMALIZER=1 to inject your local normalizer/ package into the
# job (so normalizer changes take effect without updating the HF Space).
LOCAL_NORMALIZER_INJECT=
if [[ ${USE_LOCAL_NORMALIZER:-0} == 1 ]]; then
    NORMALIZER_B64=$(tar --exclude='__pycache__' --exclude='*.pyc' -czf - -C "${REPO_ROOT}" normalizer | base64 -w0)
    LOCAL_NORMALIZER_INJECT="echo '${NORMALIZER_B64}' | base64 -d | tar -xzf - -C /app"
fi

# Set USE_LOCAL_SCRIPT=1 to inject your local run_eval.py.
LOCAL_SCRIPT_INJECT=
if [[ ${USE_LOCAL_SCRIPT:-0} == 1 ]]; then
    RUN_EVAL_B64=$(base64 -w0 "${SCRIPT_DIR}/run_eval.py")
    LOCAL_SCRIPT_INJECT="echo '${RUN_EVAL_B64}' | base64 -d > /app/run_eval.py"
fi

# Optional: restrict this run to specific datasets, matched against the first
# field of each DATASET_CONFIGS entry, e.g.:
#   ONLY_DATASETS="monsoon_en_in" bash <this script>
#   ONLY_DATASETS="librispeech spgispeech" bash <this script>
if [[ -n ${ONLY_DATASETS:-} ]]; then
    read -ra WANTED <<< ${ONLY_DATASETS}
    SELECTED=()
    for CONFIG in "${DATASET_CONFIGS[@]}"; do
        read -r NAME _ <<< ${CONFIG}
        for WANT in "${WANTED[@]}"; do
            if [[ ${NAME} == "${WANT}" || ${NAME##*/} == "${WANT}" ]]; then
                SELECTED+=("${CONFIG}")
                break
            fi
        done
    done
    if (( ${#SELECTED[@]} == 0 )); then
        echo "ONLY_DATASETS='${ONLY_DATASETS}' matched no datasets." >&2
        exit 1
    fi
    DATASET_CONFIGS=("${SELECTED[@]}")
fi

NAMESPACE_ARGS=()
if [[ -n ${ORG_NAME} ]]; then
    NAMESPACE_ARGS=(--namespace "${ORG_NAME}")
fi
mkdir -p "${SCRIPT_DIR}/results"
mkdir "${LOCAL_DIR}"  # Do not mix the current run with existing local results.

for MODEL_CONFIG in "${MODEL_CONFIGS[@]}"; do
    read -r MODEL_ID MODEL_TYPE CHECKPOINT_FILE DECODER_TYPE BEAM BATCH_SIZE <<< ${MODEL_CONFIG}
    MODEL_FOLDER=${MODEL_ID//\//-}/${DECODER_TYPE}
    MODEL_DIR=${LOCAL_DIR}/${MODEL_FOLDER}
    DESTINATION=/results/${RUN_ID}/${MODEL_FOLDER}
    mkdir -p "${MODEL_DIR}"

    # Serialize configuration safely; the quoted body expands variables in the job.
    JOB_COMMAND=$(
        declare -p MODEL_ID MODEL_TYPE CHECKPOINT_FILE DECODER_TYPE BEAM BATCH_SIZE \
            DEFAULT_DATASET_PATH DATASET_CONFIGS COMMON_ARGS DESTINATION
        echo "${LOCAL_NORMALIZER_INJECT}"
        echo "${LOCAL_SCRIPT_INJECT}"
        cat <<'JOB'
export PYTHONPATH=/app
mkdir -p /app/evaluation "${DESTINATION}"
cd /app/evaluation
for CONFIG in "${DATASET_CONFIGS[@]}"; do
    read -r DATASET SPLIT DATASET_PATH <<< ${CONFIG}
    DATASET_CONFIG=${DATASET}
    if [[ -n ${DATASET_PATH} ]]; then
        DATASET_CONFIG=
    fi
    echo "Evaluating ${MODEL_ID}: ${DATASET} ${SPLIT}, batch ${BATCH_SIZE}, beam ${BEAM}"
    python /app/run_eval.py "${COMMON_ARGS[@]}" \
        --engine-cache=/app/engines \
        --model-id="${MODEL_ID}" --model-family="${MODEL_TYPE}" \
        --checkpoint-file="${CHECKPOINT_FILE}" --decoder-type="${DECODER_TYPE}" \
        --beam="${BEAM}" --batch-size="${BATCH_SIZE}" \
        --dataset-path="${DATASET_PATH:-${DEFAULT_DATASET_PATH}}" \
        --dataset="${DATASET_CONFIG}" --split="${SPLIT}" \
        2>&1 | tee "${DESTINATION}/${DATASET}-${SPLIT}.log"
    cp results/*.jsonl results/*.metadata.json "${DESTINATION}/"
done
JOB
    )

    echo "Submitting ${MODEL_ID} ${DECODER_TYPE}: ${#DATASET_CONFIGS[@]} datasets in one job"
    hf jobs run \
        --flavor "${FLAVOR}" \
        --timeout "${TIMEOUT:-8h}" \
        --secrets HF_TOKEN \
        "${NAMESPACE_ARGS[@]}" \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" \
        -- "hf.co/spaces/${SPACE}" \
        bash -euo pipefail -c "${JOB_COMMAND}" \
        2>&1 | tee "${MODEL_DIR}/job.log"

    hf buckets sync "hf://buckets/${RESULTS_BUCKET}/${RUN_ID}/${MODEL_FOLDER}" "${MODEL_DIR}"

    MANIFESTS=("${MODEL_DIR}"/*.jsonl)
    if (( ${#MANIFESTS[@]} != ${#DATASET_CONFIGS[@]} )); then
        echo "Expected ${#DATASET_CONFIGS[@]} manifests, found ${#MANIFESTS[@]}; refusing to score." >&2
        exit 1
    fi
    for MANIFEST in "${MANIFESTS[@]}"; do
        if [[ ! -f ${MANIFEST%.jsonl}.metadata.json ]]; then
            echo "Missing metadata for ${MANIFEST}; refusing to score." >&2
            exit 1
        fi
    done

    python -c 'import sys; from normalizer.eval_utils import score_results; score_results(sys.argv[1], sys.argv[2])' \
        "${MODEL_DIR}" "${MODEL_ID}" 2>&1 | tee "${MODEL_DIR}/scores.log"
done
