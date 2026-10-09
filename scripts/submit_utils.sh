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
local_normalizer_inject() {
    [[ "${USE_LOCAL_NORMALIZER:-1}" == "1" ]] || return 0
    local b64
    b64=$(tar --exclude='__pycache__' --exclude='*.pyc' -czf - -C "${SUBMIT_UTILS_REPO_ROOT}" normalizer | base64 | tr -d '\n')
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
