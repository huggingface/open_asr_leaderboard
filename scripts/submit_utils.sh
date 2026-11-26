# Shared helpers for the */submit_jobs*.sh scripts. Source it near the top:
#
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/submit_utils.sh"
#
# Sourcing also records when the run started (RUN_START), which
# scripts/fetch_run_results.py (FETCH_RUN_RESULTS) uses to fetch only this run's
# results; RESULTS_SINCE=0 takes every result in the model's bucket folder instead.

SUBMIT_UTILS_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_START="${RESULTS_SINCE:-$(date +%s)}"
FETCH_RUN_RESULTS="${SUBMIT_UTILS_REPO_ROOT}/scripts/fetch_run_results.py"

# _b64 FILE: FILE base64-encoded on one line (`base64 -w0` is GNU-only).
_b64() {
    base64 < "$1" | tr -d '\n'
}

# local_script_inject DIR FILE[:DEST]...
#
# With USE_LOCAL_SCRIPT=1 (the default), prints a command prefix that writes each
# local DIR/FILE to /app/DEST in the job (DEST defaults to FILE), so the job runs
# your local eval script(s) rather than the ones baked into the Space image.
# Prints nothing with USE_LOCAL_SCRIPT=0. Use as:
#
#   LOCAL_SCRIPT_INJECT=$(local_script_inject "${SCRIPT_DIR}" run_eval.py) || exit 1
local_script_inject() {
    [[ "${USE_LOCAL_SCRIPT:-1}" == "1" ]] || return 0
    local dir=$1 spec src dest out=""
    shift
    for spec in "$@"; do
        src=${spec%%:*}
        dest=${spec#*:}
        if [[ ! -f "${dir}/${src}" ]]; then
            echo "ERROR: ${dir}/${src} not found (set USE_LOCAL_SCRIPT=0 to use the Space's copy)." >&2
            return 1
        fi
        out+="echo '$(_b64 "${dir}/${src}")' | base64 -d > /app/${dest} && "
    done
    printf '%s' "${out% }"
}

# local_normalizer_inject [TAR_FLAG]...
#
# With USE_LOCAL_NORMALIZER=1 (the default), prints a command prefix that unpacks
# your local normalizer/ package into /app in the job, so normalizer changes take
# effect without updating the Space. TAR_FLAGs are added to the extracting tar
# (e.g. --no-same-owner). Prints nothing with USE_LOCAL_NORMALIZER=0.
#
# The whole job command, this prefix included, is one argument of `hf jobs run`, and Linux caps an argument at
# 131072 bytes (the command then fails to start, silently where output is discarded). With a large run_eval.py
# (transformers/) that leaves little room, so files that only the VoiceCodeBench grading job uses are left out:
# it ships its own copy of normalizer/ (vcb_submit_grading in this file).
local_normalizer_inject() {
    [[ "${USE_LOCAL_NORMALIZER:-1}" == "1" ]] || return 0
    local b64
    b64=$(tar --exclude='__pycache__' --exclude='*.pyc' \
              --exclude='normalizer/voice_code_bench_judge.py' --exclude='normalizer/voice_code_bench_prompt.txt' \
              -czf - -C "${SUBMIT_UTILS_REPO_ROOT}" normalizer | base64 | tr -d '\n')
    printf '%s' "echo '${b64}' | base64 -d | tar ${*:+$* }-xzf - -C /app &&"
}

# filter_only_datasets [FIELD]
#
# If ONLY_DATASETS is set (space-separated names), keeps only the DATASET_CONFIGS
# entries whose FIELD-th word (default 1) -- or its part after the last "/" --
# is one of them. Returns 1, with an error, when nothing matches. Use as:
#
#   filter_only_datasets || exit 1
filter_only_datasets() {
    [[ -n "${ONLY_DATASETS:-}" ]] || return 0
    local field=${1:-1} cfg name want words selected=()
    for cfg in ${DATASET_CONFIGS[@]+"${DATASET_CONFIGS[@]}"}; do
        read -ra words <<< "$cfg"
        name=${words[field - 1]:-}
        for want in ${ONLY_DATASETS}; do
            if [[ "$name" == "$want" || "${name##*/}" == "$want" ]]; then
                selected+=("$cfg")
            fi
        done
    done
    if [[ ${#selected[@]} -eq 0 ]]; then
        echo "ERROR: ONLY_DATASETS='${ONLY_DATASETS}' matched no active entry in DATASET_CONFIGS." >&2
        return 1
    fi
    DATASET_CONFIGS=("${selected[@]}")
}

# filter_only_datasets_languages
#
# For multilingual DATASET_CONFIGS ("dataset language ..."): if ONLY_DATASETS
# and/or ONLY_LANGUAGES are set (space-separated), keeps only the entries whose
# dataset (1st word) and language (2nd word) are both allowed. Returns 1, with
# an error, when nothing matches. Use as:
#
#   filter_only_datasets_languages || exit 1
filter_only_datasets_languages() {
    [[ -n "${ONLY_DATASETS:-}" || -n "${ONLY_LANGUAGES:-}" ]] || return 0
    local cfg name lang want keep_ds keep_lang selected=()
    for cfg in ${DATASET_CONFIGS[@]+"${DATASET_CONFIGS[@]}"}; do
        read -r name lang _ <<< "$cfg"
        keep_ds=1
        if [[ -n "${ONLY_DATASETS:-}" ]]; then
            keep_ds=0
            for want in ${ONLY_DATASETS}; do [[ "$name" == "$want" ]] && keep_ds=1; done
        fi
        keep_lang=1
        if [[ -n "${ONLY_LANGUAGES:-}" ]]; then
            keep_lang=0
            for want in ${ONLY_LANGUAGES}; do [[ "$lang" == "$want" ]] && keep_lang=1; done
        fi
        [[ $keep_ds == 1 && $keep_lang == 1 ]] && selected+=("$cfg")
    done
    if [[ ${#selected[@]} -eq 0 ]]; then
        echo "ERROR: ONLY_DATASETS='${ONLY_DATASETS:-}' ONLY_LANGUAGES='${ONLY_LANGUAGES:-}' matched no entry in DATASET_CONFIGS." >&2
        return 1
    fi
    DATASET_CONFIGS=("${selected[@]}")
    echo "Restricted to ${#DATASET_CONFIGS[@]} dataset/language combination(s): ${DATASET_CONFIGS[*]}"
}

# ── VoiceCodeBench ───────────────────────────────────────────────────────────
# VoiceCodeBench is evaluated like any other dataset: its DATASET_CONFIGS entry names the repo as its dataset path,
# its transcripts go to the model's folder of $RESULTS_BUCKET, and its WER is scored with the other datasets' (reported,
# but not part of the default average). Its entity metric, CTEM (higher is better, never averaged with WERs), is graded afterwards by
# one H200 GPU job per model (all models at once) that writes the CTEM report, the judge's
# responses and a summary beside the transcripts. See voice_code_bench.md.
#
# A submit script uses these helpers, besides an entry in DATASET_CONFIGS such as
#   "voice_code_bench test besimple-ai/voice-code-bench"      (plus a batch size where its format has one):
#
#   vcb_init                                after filter_only_datasets: is it among the datasets? (sets VCB_COUNT)
#   vcb_dataset_config                      in the job loop, once DATASET_PATH and DATASET_CONFIG are set
#   vcb_track_model MODEL_ID MODEL_FOLDER   once a model's jobs are done: remember it if its transcripts were written
#                                           during this run (an old manifest must not be graded for a failed job)
#   vcb_grade                               at the end of the script: grade the remembered models, then print each
#                                           one's CTEM and WER and store its summary in its bucket folder
#
# VCB_SCORING=0 only transcribes (grade later with vcb_submit_grading MANIFEST_PATH), PUBLISH_RESULTS=0
# skips the summary upload, and VCB_JUDGE=fp8 grades with the FP8 judge experiment instead of the pinned BF16 judge.
VCB_DATASET_PATH="besimple-ai/voice-code-bench"
VCB_SCORING="${VCB_SCORING:-1}"
VCB_COUNT=0
VCB_MODELS=()  # "model_id model_folder manifest_path" of the models whose transcripts exist

vcb_init() {
    local cfg
    VCB_COUNT=0
    VCB_MODELS=()
    for cfg in ${DATASET_CONFIGS[@]+"${DATASET_CONFIGS[@]}"}; do
        [[ " ${cfg} " == *" ${VCB_DATASET_PATH} "* ]] && VCB_COUNT=$((VCB_COUNT + 1))
    done
    return 0
}

# VoiceCodeBench is run with its one config, the name its manifests and grading use.
vcb_dataset_config() {
    [[ "$DATASET_PATH" == "$VCB_DATASET_PATH" ]] && DATASET_CONFIG="default"
    return 0
}

vcb_track_model() {
    local model_id=$1 model_folder=$2 manifest
    (( VCB_COUNT > 0 )) || return 0
    manifest="${model_folder}/MODEL_${model_folder}_DATASET_besimple-ai-voice-code-bench_default_test.jsonl"
    if python - "$RESULTS_BUCKET" "$manifest" "$RUN_START" <<'PY'
import sys
from datetime import datetime, timezone
from huggingface_hub import HfApi
bucket, path, since = sys.argv[1], sys.argv[2], datetime.fromtimestamp(float(sys.argv[3]), tz=timezone.utc)
sys.exit(0 if any(getattr(f, "path", None) == path and f.uploaded_at is not None and f.uploaded_at >= since
                  for f in HfApi().list_bucket_tree(bucket, prefix=path)) else 1)
PY
    then
        VCB_MODELS+=("${model_id} ${model_folder} ${manifest}")
    else
        echo "WARNING: no VoiceCodeBench transcripts written for ${model_id} (hf://buckets/${RESULTS_BUCKET}/${manifest}); not graded."
    fi
}

# Grade one transcript manifest (path relative to $RESULTS_BUCKET) in a GPU job; waits for it and returns its status.
# Env: VCB_JUDGE (bf16 default, or fp8), VERIFIER_MODE (live-fill default, or replay on a CPU job), ALLOW_PARTIAL=1 for
# smoke tests, SCORING_IMAGE / SCORING_FLAVOR / SCORING_TIMEOUT, ORG_NAME, JOB_RECORD.
# The FP8 judge writes its own cache and report beside the BF16 ones, so it never replaces or reuses them.
vcb_submit_grading() {
    local manifest=${1:?Pass a manifest path relative to RESULTS_BUCKET.}
    local mode="${VERIFIER_MODE:-live-fill}" allow_partial="${ALLOW_PARTIAL:-0}" judge="${VCB_JUDGE:-bf16}"
    local tag cache_name report_name image flavor payload record
    case "$judge" in
        bf16) tag="";     cache_name="qwen27b-verifier-cache.json";     report_name="ctem.json" ;;
        fp8)  tag="-fp8"; cache_name="qwen27b-fp8-verifier-cache.json"; report_name="ctem-fp8.json" ;;
        *) echo "VCB_JUDGE must be bf16 or fp8." >&2; return 1 ;;
    esac
    python - "$manifest" "$mode" "$allow_partial" <<'PY' || return 1
from pathlib import PurePosixPath
import sys
from huggingface_hub import get_token
path = PurePosixPath(sys.argv[1])
if path.is_absolute() or '..' in path.parts or path.suffix != '.jsonl':
    raise SystemExit("Expected a relative bucket manifest path ending in .jsonl.")
if sys.argv[2] not in {'live-fill', 'replay'}:
    raise SystemExit("VERIFIER_MODE must be replay or live-fill.")
if sys.argv[3] not in {'0', '1'}:
    raise SystemExit("ALLOW_PARTIAL must be 0 or 1.")
if not get_token():
    raise SystemExit("Set HF_TOKEN or run hf auth login before submitting jobs.")
PY
    # Ship the current scoring code, including unpushed changes, into the grading image.
    payload=$(tar --exclude='__pycache__' --exclude='*.pyc' -czf - -C "$SUBMIT_UTILS_REPO_ROOT" \
        normalizer scripts/score_voice_code_bench_job.py \
        requirements/requirements_voice_code_bench_scoring.txt | base64 | tr -d '\n')
    if [[ "$mode" == "live-fill" ]]; then
        image="${SCORING_IMAGE:-vllm/vllm-openai:v0.28.0-cu129}"; flavor="${SCORING_FLAVOR:-h200}"
    else
        image="${SCORING_IMAGE:-python:3.12-slim}"; flavor="${SCORING_FLAVOR:-cpu-basic}"
    fi
    record="${JOB_RECORD:-${SUBMIT_UTILS_REPO_ROOT}/nemo_asr/results/voice_code_bench/${manifest%.jsonl}.grading-job${tag}.json}"
    python "${SUBMIT_UTILS_REPO_ROOT}/scripts/run_voice_code_bench_job.py" \
        --image "$image" --flavor "$flavor" --timeout "${SCORING_TIMEOUT:-8h}" \
        --namespace "${ORG_NAME:-}" --bucket "$RESULTS_BUCKET" --stage grading \
        --record "$record" --secret HF_TOKEN \
        --env "MANIFEST_PATH=${manifest}" --env "VERIFIER_MODE=${mode}" \
        --env "ALLOW_PARTIAL=${allow_partial}" --env "RESULTS_BUCKET=${RESULTS_BUCKET}" \
        --env "VCB_JUDGE=${judge}" --env "CACHE_NAME=${cache_name}" --env "REPORT_NAME=${report_name}" \
        -- bash -ec "
        set -o pipefail
        mkdir -p /app
        echo '${payload}' | base64 -d | tar -xzf - -C /app
        cd /app
        python3 -m pip install -r requirements/requirements_voice_code_bench_scoring.txt
        extra_args=()
        [[ \"\$ALLOW_PARTIAL\" == 1 ]] && extra_args+=(--allow-partial)
        PYTHONPATH=/app python3 scripts/score_voice_code_bench_job.py \\
            --manifest \"/results/\$MANIFEST_PATH\" \\
            --output \"/results/\${MANIFEST_PATH%.jsonl}.\$REPORT_NAME\" \\
            --verifier-cache \"/results/\${MANIFEST_PATH%.jsonl}.\$CACHE_NAME\" \\
            --verifier-mode \"\$VERIFIER_MODE\" \"\${extra_args[@]}\"
    "
}

vcb_grade() {
    local entry model folder manifest pids=() i failed=0 publish_args
    (( ${#VCB_MODELS[@]} > 0 )) && [[ "$VCB_SCORING" == "1" ]] || return 0
    echo "████████████████████████████████████████████████████████████████████████████████"
    echo "  VoiceCodeBench: grading ${#VCB_MODELS[@]} model(s) with CTEM"
    echo "████████████████████████████████████████████████████████████████████████████████"
    for entry in "${VCB_MODELS[@]}"; do
        read -r model folder manifest <<< "$entry"
        (
            vcb_submit_grading "$manifest" 2>&1 \
                | sed -u "s|^|[${model}] |"
            exit "${PIPESTATUS[0]}"
        ) &
        pids+=("$!")
    done
    for i in "${!VCB_MODELS[@]}"; do
        read -r model folder manifest <<< "${VCB_MODELS[$i]}"
        if wait "${pids[$i]}"; then
            publish_args=()
            [[ "${PUBLISH_RESULTS:-1}" == "1" ]] && publish_args=(--publish-to "$folder")
            PYTHONPATH="${SUBMIT_UTILS_REPO_ROOT}" python "${SUBMIT_UTILS_REPO_ROOT}/scripts/summarize_voice_code_bench.py" \
                --bucket "$RESULTS_BUCKET" --manifest-path "$manifest" --model-id "$model" \
                --output-dir "${SUBMIT_UTILS_REPO_ROOT}/nemo_asr/results/voice_code_bench/${folder}" \
                ${publish_args[@]+"${publish_args[@]}"} || failed=1
        else
            echo "ERROR: VoiceCodeBench grading failed for ${model}; its transcripts are in hf://buckets/${RESULTS_BUCKET}/${manifest}"
            failed=1
        fi
    done
    return "$failed"
}
