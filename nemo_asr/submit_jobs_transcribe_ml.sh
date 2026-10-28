#!/usr/bin/env bash
set -Eeuo pipefail

RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_multilingual}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_jobs_transcribe_common.sh"

ML_DATASET_PATH="${ML_DATASET_PATH:-hf-audio/open-asr-leaderboard-multilingual-datasets}"
ML_CONCURRENCY="${ML_CONCURRENCY:-192}"
ML_WARMUP_SAMPLES="${ML_WARMUP_SAMPLES:-192}"

DATASET_CONFIGS=(
    "fleurs de ${ML_DATASET_PATH} fleurs_de fleurs ${ML_CONCURRENCY}"
    "mcv de ${ML_DATASET_PATH} mcv_de default ${ML_CONCURRENCY}"
    "fleurs fr ${ML_DATASET_PATH} fleurs_fr fleurs ${ML_CONCURRENCY}"
    "mcv fr ${ML_DATASET_PATH} mcv_fr default ${ML_CONCURRENCY}"
    "mls fr ${ML_DATASET_PATH} mls_fr default ${ML_CONCURRENCY}"
    "fleurs it ${ML_DATASET_PATH} fleurs_it fleurs ${ML_CONCURRENCY}"
    "mcv it ${ML_DATASET_PATH} mcv_it default ${ML_CONCURRENCY}"
    "mls it ${ML_DATASET_PATH} mls_it default ${ML_CONCURRENCY}"
    "fleurs es ${ML_DATASET_PATH} fleurs_es fleurs ${ML_CONCURRENCY}"
    "mcv es ${ML_DATASET_PATH} mcv_es default ${ML_CONCURRENCY}"
    "mls es ${ML_DATASET_PATH} mls_es default ${ML_CONCURRENCY}"
    "mls pt ${ML_DATASET_PATH} mls_pt default ${ML_CONCURRENCY}"
    "fleurs pt ${ML_DATASET_PATH} fleurs_pt fleurs ${ML_CONCURRENCY}"
    "fleurs nl ${ML_DATASET_PATH} fleurs_nl fleurs ${ML_CONCURRENCY}"
    "mcv nl ${ML_DATASET_PATH} mcv_nl default ${ML_CONCURRENCY}"
    "mls nl ${ML_DATASET_PATH} mls_nl default ${ML_CONCURRENCY}"
    "monsoon hi VoiceArena/Monsoon_hi_test - monsoon_hi ${ML_CONCURRENCY}"
)

filter_only_datasets_languages >&2 || exit 1

if [[ "${PRINT_PLAN:-0}" == "1" ]]; then
    printf 'track\tlabel\tdataset_path\tconfig\tsplit\tlanguage\tprompt_profile\tconcurrency\tstreaming\n'
    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r dataset language dataset_path config prompt_profile concurrency <<<"${cfg}"
        [[ "${config}" == "-" ]] && config=""
        printf 'multilingual\t%s\t%s\t%s\ttest\t%s\t%s\t%s\tfalse\n' \
            "${dataset}" "${dataset_path}" "${config}" "${language}" "${prompt_profile}" "${concurrency}"
    done
    exit 0
fi

require_submission_environment
prepare_injection
start_submission_logs multilingual

for cfg in "${DATASET_CONFIGS[@]}"; do
    read -r dataset language dataset_path config prompt_profile concurrency <<<"${cfg}"
    [[ "${config}" == "-" ]] && config=""
    echo "Submitting multilingual job: ${dataset}/${language}"

    job_command="${LOCAL_NORMALIZER_INJECT} ${LOCAL_SCRIPT_INJECT} \
RESULTS_DEST_DIR=/results/${MODEL_FOLDER} bash /app/run_transcribe.sh \
--dataset_path=${dataset_path} --dataset=${config} --split=test --language=${language} \
--prompt_profile=${prompt_profile} --batch_size=${concurrency} \
--warmup_samples=${ML_WARMUP_SAMPLES} --max_eval_samples=${MAX_EVAL_SAMPLES} \
--request_timeout_s=${REQUEST_TIMEOUT_S} --max_retries=${MAX_RETRIES}"
    submit_hf_job "multilingual-${dataset}-${language}" "${job_command}"
done

print_jobs_url
wait_for_submitted_jobs
sleep 10

RUN_RESULTS="$(fetch_this_run "${#DATASET_CONFIGS[@]}")"
LANGUAGES=()
for cfg in "${DATASET_CONFIGS[@]}"; do
    read -r _dataset language _rest <<<"${cfg}"
    if [[ ! " ${LANGUAGES[*]} " == *" ${language} "* ]]; then
        LANGUAGES+=("${language}")
    fi
done

for language in "${LANGUAGES[@]}"; do
    RUN_RESULTS="${RUN_RESULTS}" MODEL_ID="${MODEL_ID}" LANGUAGE="${language}" PYTHONPATH="${REPO_ROOT}" \
        python -c 'import os; from normalizer.eval_utils import score_results; lang=os.environ["LANGUAGE"]; score_results(os.environ["RUN_RESULTS"], os.environ["MODEL_ID"], multilingual=True, language=lang, families=[f"ml_{lang}"], csv_only=True)'
done
