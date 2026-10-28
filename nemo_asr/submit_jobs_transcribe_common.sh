#!/usr/bin/env bash

umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/scripts/submit_utils.sh"

MODEL_ID="nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
MODEL_FOLDER="${MODEL_ID//\//-}"
SPACE="${SPACE:-nvidia/open-asr-nemotron-3dot5-transcribe}"
FLAVOR="${FLAVOR:-h200}"
JOB_TIMEOUT="${JOB_TIMEOUT:-8h}"
ORG_NAME="${ORG_NAME:-}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:--1}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-3600}"
MAX_RETRIES="${MAX_RETRIES:-2}"
SPECULATIVE="${SPECULATIVE:-mtp8}"

JOB_PIDS=()
JOB_LABELS=()

require_submission_environment() {
    local missing=0
    for name in HF_TOKEN SPACE RESULTS_BUCKET; do
        if [[ -z "${!name:-}" ]]; then
            echo "ERROR: ${name} must be set." >&2
            missing=1
        fi
    done
    [[ ${missing} -eq 0 ]] || return 2
    command -v hf >/dev/null || {
        echo "ERROR: hf CLI is not installed." >&2
        return 2
    }
    command -v python >/dev/null || {
        echo "ERROR: python is not installed." >&2
        return 2
    }
    case "${SPECULATIVE}" in
        off|mtp2|mtp8|dflash2-k7|dflash2-k15) ;;
        *) echo "ERROR: unsupported SPECULATIVE mode: ${SPECULATIVE}" >&2; return 2 ;;
    esac
}

prepare_injection() {
    LOCAL_SCRIPT_INJECT=$(
        local_script_inject "${SCRIPT_DIR}" \
            run_eval_transcribe.py \
            transcribe_eval_helpers.py \
            transcribe_model_card_contract.py \
            run_transcribe.sh
    ) || return 1
    LOCAL_NORMALIZER_INJECT=$(local_normalizer_inject --no-same-owner) || return 1
}

start_submission_logs() {
    local track=$1
    SUBMISSION_LOG_DIR="${SUBMISSION_LOG_DIR:-${SCRIPT_DIR}/submission_logs/${track}-$(date -u +%Y%m%dT%H%M%SZ)}"
    mkdir -p "${SUBMISSION_LOG_DIR}"
    echo "Job logs: ${SUBMISSION_LOG_DIR}"
}

submit_hf_job() {
    local label=$1
    local job_command=$2
    local safe_label=${label//[^a-zA-Z0-9_.-]/_}
    local job_name=${safe_label//[._]/-}
    local log_file="${SUBMISSION_LOG_DIR}/${safe_label}.log"
    local namespace_args=()
    [[ -n "${ORG_NAME}" ]] && namespace_args=(--namespace "${ORG_NAME}")

    hf jobs run \
        --name "transcribe-${job_name}" \
        --flavor "${FLAVOR}" \
        --timeout "${JOB_TIMEOUT}" \
        --secrets HF_TOKEN \
        --env HF_AUDIO_DECODER_BACKEND=soundfile \
        --env TOKENIZERS_PARALLELISM=false \
        --env SPECULATIVE="${SPECULATIVE}" \
        "${namespace_args[@]}" \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" \
        "hf.co/spaces/${SPACE}" \
        bash -c "${job_command}" >"${log_file}" 2>&1 &
    JOB_PIDS+=("$!")
    JOB_LABELS+=("${label}")
}

wait_for_submitted_jobs() {
    local failed=0
    local index pid label safe_label log_file
    for index in "${!JOB_PIDS[@]}"; do
        pid=${JOB_PIDS[$index]}
        label=${JOB_LABELS[$index]}
        safe_label=${label//[^a-zA-Z0-9_.-]/_}
        log_file="${SUBMISSION_LOG_DIR}/${safe_label}.log"
        if wait "${pid}"; then
            echo "Completed: ${label}"
        else
            echo "ERROR: HF Job failed: ${label}" >&2
            tail -n 100 "${log_file}" >&2 || true
            failed=1
        fi
    done
    [[ ${failed} -eq 0 ]]
}

fetch_this_run() {
    local expected=$1
    local local_dir="${REPO_ROOT}/results/${MODEL_FOLDER}"
    local attempts="${FETCH_ATTEMPTS:-12}"
    local interval="${FETCH_INTERVAL_S:-10}"
    local attempt run_dir found
    mkdir -p "${local_dir}"
    for ((attempt = 1; attempt <= attempts; attempt++)); do
        run_dir=$(python "${FETCH_RUN_RESULTS}" \
            --bucket "${RESULTS_BUCKET}" \
            --model-folder "${MODEL_FOLDER}" \
            --local-dir "${local_dir}" \
            --since "${RUN_START}" \
            --expected "${expected}") || return 1
        found=$(find -L "${run_dir}" -type f -name '*.jsonl' -print | wc -l)
        if [[ "${found}" -eq "${expected}" ]]; then
            printf '%s\n' "${run_dir}"
            return 0
        fi
        if [[ "${found}" -gt "${expected}" ]]; then
            echo "ERROR: fetched ${found} manifests, expected ${expected}." >&2
            return 1
        fi
        if [[ "${attempt}" -lt "${attempts}" ]]; then
            sleep "${interval}"
        fi
    done
    echo "ERROR: only ${found}/${expected} manifests became visible." >&2
    return 1
}

print_jobs_url() {
    if [[ -n "${ORG_NAME}" ]]; then
        echo "Live status: https://huggingface.co/organizations/${ORG_NAME}/settings/jobs"
    else
        echo "Live status: https://huggingface.co/settings/jobs"
    fi
}
