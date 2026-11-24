#!/usr/bin/env bash
# Submit the HF Job for the English short-form evaluation of the Nemotron Transcribe models (MODEL_ID selects BF16 or -FP8).
# VoiceCodeBench is scored with CTEM and WER; ONLY_DATASETS="voice_code_bench" runs just that dataset.
# At the end a separate GPU job grades the model's transcripts (see scripts/submit_utils.sh, voice_code_bench.md).
set -Eeuo pipefail

RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_h200}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_jobs_transcribe_common.sh"

DEFAULT_DATASET_PATH="${DEFAULT_DATASET_PATH:-hf-audio/open-asr-leaderboard}"
SHORTFORM_CONCURRENCY="${SHORTFORM_CONCURRENCY:-192}"
SHORTFORM_WARMUP_SAMPLES="${SHORTFORM_WARMUP_SAMPLES:-192}"
# VoiceCodeBench recordings are 35-123 s long; the server takes them whole (audio limit 120 s+ per prompt in
# run_transcribe.sh), so no chunking, only fewer concurrent requests than for short clips.
VCB_CONCURRENCY="${VCB_CONCURRENCY:-64}"

DATASET_CONFIGS=(
    "ami_cleaned ${DEFAULT_DATASET_PATH} ami_cleaned test default ${SHORTFORM_CONCURRENCY}"
    "gigaspeech_cleaned ${DEFAULT_DATASET_PATH} gigaspeech_cleaned test default ${SHORTFORM_CONCURRENCY}"
    "voxpopuli_cleaned_aa ${DEFAULT_DATASET_PATH} voxpopuli_cleaned_aa test default ${SHORTFORM_CONCURRENCY}"
    "earnings22_cleaned_aa_chunked ArtificialAnalysis/Earnings22-Cleaned-AA-chunked - test default ${SHORTFORM_CONCURRENCY}"
    "spgispeech ${DEFAULT_DATASET_PATH} spgispeech test spgispeech ${SHORTFORM_CONCURRENCY}"
    "urgent2024 ${DEFAULT_DATASET_PATH} urgent2024 test default ${SHORTFORM_CONCURRENCY}"
    "urgent2024_clean ${DEFAULT_DATASET_PATH} urgent2024_clean test default ${SHORTFORM_CONCURRENCY}"
    "monsoon_en_in VoiceArena/Monsoon_en_IN_test - test monsoon_en ${SHORTFORM_CONCURRENCY}"
    "voice_code_bench besimple-ai/voice-code-bench default test default ${VCB_CONCURRENCY}"
)

filter_only_datasets || exit 1

# Is VoiceCodeBench among the datasets of this run? (its entry already names its one config, "default",
# so vcb_dataset_config is not needed here)
vcb_init

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

if [[ "${SINGLE_JOB:-1}" == "1" ]]; then
    # One HF Job for the whole track: vLLM starts (and JIT-compiles) once and every dataset
    # is evaluated against the same server. Set SINGLE_JOB=0 for one job per dataset.
    ARGS_QUOTED=""
    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r label dataset_path config split prompt_profile concurrency <<<"${cfg}"
        [[ "${config}" == "-" ]] && config=""
        line="--dataset_path=${dataset_path} --dataset=${config} --split=${split} --language=en --prompt_profile=${prompt_profile} --batch_size=${concurrency} --warmup_samples=${SHORTFORM_WARMUP_SAMPLES} --max_eval_samples=${MAX_EVAL_SAMPLES} --request_timeout_s=${REQUEST_TIMEOUT_S} --max_retries=${MAX_RETRIES}"
        ARGS_QUOTED+=" '${line}'"
    done
    echo "Submitting one english job for ${#DATASET_CONFIGS[@]} datasets"
    job_command="${LOCAL_NORMALIZER_INJECT} ${LOCAL_SCRIPT_INJECT} \
printf '%s\\n'${ARGS_QUOTED} > /tmp/eval_args.txt; \
EVAL_ARGS_FILE=/tmp/eval_args.txt RESULTS_DEST_DIR=/results/${MODEL_FOLDER} bash /app/run_transcribe.sh"
    submit_hf_job "english-all" "${job_command}"
else

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
fi

print_jobs_url
wait_for_submitted_jobs
sleep 10

RUN_RESULTS="$(fetch_this_run "${#DATASET_CONFIGS[@]}")"
RUN_RESULTS="${RUN_RESULTS}" MODEL_ID="${MODEL_ID}" PYTHONPATH="${REPO_ROOT}" \
    python -c 'import os; from normalizer.eval_utils import score_results; score_results(os.environ["RUN_RESULTS"], os.environ["MODEL_ID"], families=["public", "extra"])'

vcb_track_model "${MODEL_ID}" "${MODEL_FOLDER}"

# VoiceCodeBench CTEM, if it was evaluated: one grading job per model, then CTEM and WER are printed.
vcb_grade
