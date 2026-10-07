#!/bin/bash
# Submit HF evaluations, with dataset parallelism controlled by the config.
# Usage: HF_TOKEN=hf_... bash soundsgoodai/submit_jobs.sh
set -euo pipefail
shopt -s nullglob

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Evaluation matrix
CONFIG=${CONFIG:-config.sh}
[[ ${CONFIG} == */* ]] || CONFIG=${SCRIPT_DIR}/${CONFIG}
if [[ ! -f ${CONFIG} ]]; then
    echo "Config not found: ${CONFIG}" >&2
    exit 1
fi
source "${CONFIG}"
MULTILINGUAL=${MULTILINGUAL:-0}
EVAL_SCRIPT=run_eval.py
if [[ ${MULTILINGUAL} == 1 ]]; then
    EVAL_SCRIPT=run_eval_ml.py
fi

# Defaults come after the config so it can set them; the environment wins over both.
SPACE=${SPACE:-hf-audio/fast-gpu-asr-eval}
RESULTS_BUCKET=${RESULTS_BUCKET:-hf-audio/asr_leaderboard_h200}
FLAVOR=${FLAVOR:-h200}
ORG_NAME=${ORG_NAME:-}
RUN_ID=${RUN_ID:-fast-gpu-asr-$(date -u +%Y%m%dT%H%M%S)}  # Local results only.

REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
LOCAL_DIR=${SCRIPT_DIR}/results/${RUN_ID}
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}

# Local normalizer/ package is injected into the job by default, so normalizer
# changes take effect without updating the HF Space; USE_LOCAL_NORMALIZER=0 uses
# the image's version instead.
LOCAL_NORMALIZER_INJECT=
if [[ ${USE_LOCAL_NORMALIZER:-1} == 1 ]]; then
    NORMALIZER_B64=$(tar --exclude='__pycache__' --exclude='*.pyc' -czf - -C "${REPO_ROOT}" normalizer | base64 -w0)
    LOCAL_NORMALIZER_INJECT="echo '${NORMALIZER_B64}' | base64 -d | tar -xzf - -C /app"
fi

# Local runners are injected by default; USE_LOCAL_SCRIPT=0 uses the image's.
LOCAL_SCRIPT_INJECT=
if [[ ${USE_LOCAL_SCRIPT:-1} == 1 ]]; then
    for SCRIPT in run_eval.py run_eval_ml.py; do
        RUN_EVAL_B64=$(base64 -w0 "${SCRIPT_DIR}/${SCRIPT}")
        LOCAL_SCRIPT_INJECT+="echo '${RUN_EVAL_B64}' | base64 -d > /app/${SCRIPT}"$'\n'
    done
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
            if [[ ${NAME} == "${WANT}" || ${NAME##*/} == "${WANT}" ||
                ( ${MULTILINGUAL} == 1 && ${NAME%_*} == "${WANT}" ) ]]; then
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

# Multilingual configs use <dataset>_<language>; keep filtering and scoring aligned.
if [[ -n ${ONLY_LANGUAGES:-} ]]; then
    if [[ ${MULTILINGUAL} != 1 ]]; then
        echo "ONLY_LANGUAGES requires a multilingual config (CONFIG=config_ml.sh)." >&2
        exit 1
    fi

    read -ra WANTED <<< "${ONLY_LANGUAGES}"
    SELECTED=()
    for CONFIG in "${DATASET_CONFIGS[@]}"; do
        read -r NAME _ <<< "${CONFIG}"
        if [[ " ${WANTED[*]} " == *" ${NAME##*_} "* ]]; then
            SELECTED+=("${CONFIG}")
        fi
    done

    if (( ${#SELECTED[@]} == 0 )); then
        echo "ONLY_LANGUAGES='${ONLY_LANGUAGES}' matched no selected datasets." >&2
        exit 1
    fi

    DATASET_CONFIGS=("${SELECTED[@]}")
fi

LANGUAGES=(en)
if [[ ${MULTILINGUAL} == 1 ]]; then
    LANGUAGES=()
    for CONFIG in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET _ <<< "${CONFIG}"
        LANGUAGES+=("${DATASET##*_}")
    done
fi

NAMESPACE_ARGS=()
if [[ -n ${ORG_NAME} ]]; then
    NAMESPACE_ARGS=(--namespace "${ORG_NAME}")
fi
mkdir -p "${SCRIPT_DIR}/results"
mkdir "${LOCAL_DIR}"  # Do not mix the current run with existing local results.

# One job's script: evaluate the given dataset configs in order. The quoted
# heredoc body is expanded inside the job, against the configuration serialized
# above it; the local DATASET_CONFIGS is what that job evaluates.
job_command() {
    local -a DATASET_CONFIGS=("$@")
    declare -p MODEL_ID REPORT_NAME MODEL_TYPE CHECKPOINT_FILE DECODER_TYPE BEAM BATCH_SIZE WORKERS \
        DEFAULT_DATASET_PATH DATASET_CONFIGS COMMON_ARGS DESTINATION MULTILINGUAL EVAL_SCRIPT
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
    LANGUAGE_ARGS=()
    if [[ ${MULTILINGUAL} == 1 ]]; then
        LANGUAGE_ARGS=(--language="${DATASET##*_}")
    fi
    echo "Evaluating ${REPORT_NAME}: ${DATASET} ${SPLIT}, batch ${BATCH_SIZE}, workers ${WORKERS}, beam ${BEAM}"
    python "/app/${EVAL_SCRIPT}" "${COMMON_ARGS[@]}" "${LANGUAGE_ARGS[@]}" \
        --engine-cache=/app/engines \
        --model-id="${MODEL_ID}" --report-name="${REPORT_NAME}" \
        --model-family="${MODEL_TYPE}" \
        --checkpoint-file="${CHECKPOINT_FILE}" --decoder-type="${DECODER_TYPE}" \
        --beam="${BEAM}" --batch-size="${BATCH_SIZE}" \
        --workers="${WORKERS}" \
        --dataset-path="${DATASET_PATH:-${DEFAULT_DATASET_PATH}}" \
        --dataset="${DATASET_CONFIG}" --split="${SPLIT}" \
        2>&1 | tee "${DESTINATION}/${DATASET}-${SPLIT}.log"
    cp results/*.jsonl results/*.metadata.json "${DESTINATION}/"
done
JOB
}

# Run one job to completion; $1 names its log file, the rest are dataset
# configs. The job's output (export logs, per-dataset progress) goes to the log
# only. A failed job is reported and never fails this script, so the results it
# did produce are still scored and the remaining jobs still run.
submit_job() {
    local LOG=$1
    shift
    local JOB_COMMAND
    JOB_COMMAND=$(job_command "$@")
    if ! hf jobs run \
        --flavor "${FLAVOR}" \
        --timeout "${TIMEOUT:-8h}" \
        --secrets HF_TOKEN \
        "${NAMESPACE_ARGS[@]}" \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" \
        -- "hf.co/spaces/${SPACE}" \
        bash -euo pipefail -c "${JOB_COMMAND}" \
        > "${LOG}" 2>&1
    then
        echo "WARNING: job for ${REPORT_NAME} exited non-zero; see ${LOG}" >&2
    fi
}

for MODEL_CONFIG in "${MODEL_CONFIGS[@]}"; do
    read -r MODEL_ID MODEL_TYPE CHECKPOINT_FILE DECODER_TYPE BEAM BATCH_SIZE WORKERS REPORT_SUFFIX <<< ${MODEL_CONFIG}
    # Reported name; also names the result folder and the bucket path.
    REPORT_NAME=${MODEL_ID}${REPORT_SUFFIX:+ ${REPORT_SUFFIX}}
    MODEL_FOLDER=$(model_folder "${REPORT_NAME}")
    MODEL_DIR=${LOCAL_DIR}/${MODEL_FOLDER}
    DESTINATION=/results/${MODEL_FOLDER}  # Bucket path; reruns overwrite it.
    mkdir -p "${MODEL_DIR}"

    if [[ ${PARALLEL_DATASETS:-0} == 1 ]]; then
        echo "Submitting ${REPORT_NAME} ${DECODER_TYPE}: ${#DATASET_CONFIGS[@]} datasets, one job each"
        for CONFIG in "${DATASET_CONFIGS[@]}"; do
            read -r DATASET SPLIT _ <<< ${CONFIG}
            echo "  ${DATASET} ${SPLIT} -> ${MODEL_DIR}/job-${DATASET}-${SPLIT}.log"
            submit_job "${MODEL_DIR}/job-${DATASET}-${SPLIT}.log" "${CONFIG}" &
        done
        wait  # submit_job absorbs job failures, so this cannot abort the run.
    else
        echo "Submitting ${REPORT_NAME} ${DECODER_TYPE}: ${#DATASET_CONFIGS[@]} datasets in one job"
        echo "Job output: ${MODEL_DIR}/job.log"
        submit_job "${MODEL_DIR}/job.log" "${DATASET_CONFIGS[@]}"
    fi

    sleep 10  # Allow the last results to be flushed to the bucket.
    hf buckets sync "hf://buckets/${RESULTS_BUCKET}/${MODEL_FOLDER}" "${MODEL_DIR}"

    # Report what is missing but still score what arrived, so the summary prints
    # and the remaining models keep running.
    MANIFESTS=("${MODEL_DIR}"/*.jsonl)
    if (( ${#MANIFESTS[@]} != ${#DATASET_CONFIGS[@]} )); then
        echo "WARNING: expected ${#DATASET_CONFIGS[@]} manifests, found ${#MANIFESTS[@]}; scoring the ones present." >&2
    fi
    for MANIFEST in "${MANIFESTS[@]}"; do
        if [[ ! -f ${MANIFEST%.jsonl}.metadata.json ]]; then
            echo "WARNING: missing metadata for ${MANIFEST}." >&2
        fi
    done
    if (( ${#MANIFESTS[@]} == 0 )); then
        echo "No manifests for ${REPORT_NAME}; nothing to score." >&2
        continue
    fi

    echo "████████████████████████████████████████████████████████████████████████████████"
    echo "  Summary: ${REPORT_NAME} (${#MANIFESTS[@]}/${#DATASET_CONFIGS[@]} datasets)"
    echo "████████████████████████████████████████████████████████████████████████████████"
    python - "${MODEL_DIR}" "${REPORT_NAME}" "${LANGUAGES[@]}" <<'PY' 2>&1 | tee "${MODEL_DIR}/scores.log"
import sys
from pathlib import Path

from normalizer.eval_utils import score_results

for language in set(sys.argv[3:]):
    if language != "en" and not any(
        Path(sys.argv[1]).glob(
            f"MODEL_{sys.argv[2].replace('/', '-')}_DATASET_*_{language}_test.jsonl"
        )
    ):
        print(f"WARNING: no result manifests for {language}; skipping scoring.", file=sys.stderr)
        continue

    multilingual = language != "en"
    lang_family = [f"ml_{language}"] if multilingual else None
    score_results(
        sys.argv[1], sys.argv[2], multilingual, language=language, families=lang_family
    )
PY
done
