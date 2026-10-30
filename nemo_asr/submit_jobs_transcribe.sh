#!/usr/bin/env bash
set -Eeuo pipefail

RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_h200}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_jobs_transcribe_common.sh"

DEFAULT_DATASET_PATH="${DEFAULT_DATASET_PATH:-hf-audio/open-asr-leaderboard}"
SHORTFORM_CONCURRENCY="${SHORTFORM_CONCURRENCY:-192}"
SHORTFORM_WARMUP_SAMPLES="${SHORTFORM_WARMUP_SAMPLES:-192}"

DATASET_CONFIGS=(
    "ami_cleaned ${DEFAULT_DATASET_PATH} ami_cleaned test default ${SHORTFORM_CONCURRENCY}"
    "gigaspeech_cleaned ${DEFAULT_DATASET_PATH} gigaspeech_cleaned test default ${SHORTFORM_CONCURRENCY}"
    "voxpopuli_cleaned_aa ${DEFAULT_DATASET_PATH} voxpopuli_cleaned_aa test default ${SHORTFORM_CONCURRENCY}"
    "earnings22_cleaned_aa_chunked ArtificialAnalysis/Earnings22-Cleaned-AA-chunked - test default ${SHORTFORM_CONCURRENCY}"
    "librispeech_clean ${DEFAULT_DATASET_PATH} librispeech test.clean default ${SHORTFORM_CONCURRENCY}"
    "librispeech_other ${DEFAULT_DATASET_PATH} librispeech test.other default ${SHORTFORM_CONCURRENCY}"
    "spgispeech ${DEFAULT_DATASET_PATH} spgispeech test spgispeech ${SHORTFORM_CONCURRENCY}"
    "urgent2024 ${DEFAULT_DATASET_PATH} urgent2024 test default ${SHORTFORM_CONCURRENCY}"
    "urgent2024_clean ${DEFAULT_DATASET_PATH} urgent2024_clean test default ${SHORTFORM_CONCURRENCY}"
    "monsoon_en_in VoiceArena/Monsoon_en_IN_test - test monsoon_en ${SHORTFORM_CONCURRENCY}"
)

filter_only_datasets || exit 1

if [[ "${PRINT_PLAN:-0}" == "1" ]]; then
    printf 'track\tlabel\tdataset_path\tconfig\tsplit\tlanguage\tprompt_profile\tconcurrency\tstreaming\n'
    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r label dataset_path config split prompt_profile concurrency <<<"${cfg}"
        [[ "${config}" == "-" ]] && config=""
        printf 'english\t%s\t%s\t%s\t%s\ten\t%s\t%s\tfalse\n' \
            "${label}" "${dataset_path}" "${config}" "${split}" "${prompt_profile}" "${concurrency}"
    done
    exit 0
fi

require_submission_environment
prepare_injection
start_submission_logs english

for cfg in "${DATASET_CONFIGS[@]}"; do
    read -r label dataset_path config split prompt_profile concurrency <<<"${cfg}"
    [[ "${config}" == "-" ]] && config=""
    echo "Submitting English job: ${label}"

    job_command="${LOCAL_NORMALIZER_INJECT} ${LOCAL_SCRIPT_INJECT} \
RESULTS_DEST_DIR=/results/${MODEL_FOLDER} bash /app/run_transcribe.sh \
--dataset_path=${dataset_path} --dataset=${config} --split=${split} --language=en \
--prompt_profile=${prompt_profile} --batch_size=${concurrency} \
--warmup_samples=${SHORTFORM_WARMUP_SAMPLES} --max_eval_samples=${MAX_EVAL_SAMPLES} \
--request_timeout_s=${REQUEST_TIMEOUT_S} --max_retries=${MAX_RETRIES}"
    submit_hf_job "english-${label}-${split}" "${job_command}"
done

print_jobs_url
wait_for_submitted_jobs
sleep 10

RUN_RESULTS="$(fetch_this_run "${#DATASET_CONFIGS[@]}")"
RUN_RESULTS="${RUN_RESULTS}" MODEL_ID="${MODEL_ID}" PYTHONPATH="${REPO_ROOT}" \
    python -c 'import os; from normalizer.eval_utils import score_results; score_results(os.environ["RUN_RESULTS"], os.environ["MODEL_ID"], families=["public", "extra"])'
