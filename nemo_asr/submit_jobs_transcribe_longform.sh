#!/usr/bin/env bash
set -Eeuo pipefail

RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_longform}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_jobs_transcribe_common.sh"

LONGFORM_CONCURRENCY="${LONGFORM_CONCURRENCY:-192}"
LONGFORM_WARMUP_SAMPLES="${LONGFORM_WARMUP_SAMPLES:-192}"
LONGFORM_CHUNK_SECONDS="${LONGFORM_CHUNK_SECONDS:-3300}"
LONGFORM_CHUNK_OVERLAP_SECONDS="${LONGFORM_CHUNK_OVERLAP_SECONDS:-5}"
LONGFORM_RECORD_BATCH_SIZE="${LONGFORM_RECORD_BATCH_SIZE:-2}"

DATASET_CONFIGS=(
    "earnings21 hf-audio/asr-leaderboard-longform earnings21 test default ${LONGFORM_CONCURRENCY}"
    "earnings22 hf-audio/asr-leaderboard-longform earnings22 test default ${LONGFORM_CONCURRENCY}"
    "ATL bezzam/coraal ATL test default ${LONGFORM_CONCURRENCY}"
    "DCA bezzam/coraal DCA test default ${LONGFORM_CONCURRENCY}"
    "DCB bezzam/coraal DCB test default ${LONGFORM_CONCURRENCY}"
    "DTA bezzam/coraal DTA test default ${LONGFORM_CONCURRENCY}"
    "LES bezzam/coraal LES test default ${LONGFORM_CONCURRENCY}"
    "PRV bezzam/coraal PRV test default ${LONGFORM_CONCURRENCY}"
    "ROC bezzam/coraal ROC test default ${LONGFORM_CONCURRENCY}"
    "VLD bezzam/coraal VLD test default ${LONGFORM_CONCURRENCY}"
)

filter_only_datasets || exit 1

if [[ "${PRINT_PLAN:-0}" == "1" ]]; then
    printf 'track\tlabel\tdataset_path\tconfig\tsplit\tlanguage\tprompt_profile\tconcurrency\tstreaming\n'
    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r label dataset_path config split prompt_profile concurrency <<<"${cfg}"
        printf 'longform\t%s\t%s\t%s\t%s\ten\t%s\t%s\ttrue\n' \
            "${label}" "${dataset_path}" "${config}" "${split}" "${prompt_profile}" "${concurrency}"
    done
    exit 0
fi

require_submission_environment
prepare_injection
start_submission_logs longform

for cfg in "${DATASET_CONFIGS[@]}"; do
    read -r label dataset_path config split prompt_profile concurrency <<<"${cfg}"
    echo "Submitting long-form job: ${label}"

    job_command="${LOCAL_NORMALIZER_INJECT} ${LOCAL_SCRIPT_INJECT} \
RESULTS_DEST_DIR=/results/${MODEL_FOLDER} bash /app/run_transcribe.sh \
--dataset_path=${dataset_path} --dataset=${config} --split=${split} --language=en \
--prompt_profile=${prompt_profile} --batch_size=${concurrency} \
--warmup_samples=${LONGFORM_WARMUP_SAMPLES} --max_eval_samples=${MAX_EVAL_SAMPLES} \
--request_timeout_s=${REQUEST_TIMEOUT_S} --max_retries=${MAX_RETRIES} \
--streaming --longform_chunk_seconds=${LONGFORM_CHUNK_SECONDS} \
--longform_chunk_overlap_seconds=${LONGFORM_CHUNK_OVERLAP_SECONDS} \
--longform_record_batch_size=${LONGFORM_RECORD_BATCH_SIZE}"
    submit_hf_job "longform-${label}" "${job_command}"
done

print_jobs_url
wait_for_submitted_jobs
sleep 10

RUN_RESULTS="$(fetch_this_run "${#DATASET_CONFIGS[@]}")"
python "${REPO_ROOT}/scripts/score_bucket_results.py" \
    --family longform \
    --model_id "${MODEL_ID}" \
    --local_dir "${RUN_RESULTS}" \
    --skip_sync \
    --no_upload_scores
