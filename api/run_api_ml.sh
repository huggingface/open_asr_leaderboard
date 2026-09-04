#!/bin/bash

# Multilingual API ASR Evaluation Script
# Evaluates on FLEURS, MCV (Mozilla Common Voice), and MLS (Multilingual LibriSpeech)

RESULTS_BUCKET="${RESULTS_BUCKET:-}"
IMAGE_TAG="api-eval"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/." && pwd)"

if [[ -n "${RESULTS_BUCKET}" && -z "${HF_TOKEN}" ]]; then
    echo "ERROR: RESULTS_BUCKET is set but HF_TOKEN is not. Cannot write to bucket." >&2
    exit 1
fi

# ── Models: "model_id max_workers" ───────────────────────────────────────────
MODEL_CONFIGS=(
    # "openai/gpt-4o-transcribe      16"
    # "openai/gpt-4o-mini-transcribe 16"
    # "openai/whisper-1              16"
    # "assembly/universal-3-pro      4"
    # "assembly/universal-3-5-pro    4"
    # "elevenlabs/scribe_v2          8"
    # "speechmatics/enhanced         4"
    # "reson8/resonant-1             16"
    # "reson8/resonant-1-flash       16"
    # "microsoft/azure-speech        4"
    # "modulate/multilingual         25"
    # "soniox/stt-async-v5           20"
)

DATASET_PATH="hf-audio/open-asr-leaderboard-multilingual-datasets"
MONSOON_DATASET_PATH="${MONSOON_DATASET_PATH:-VoiceArena/Monsoon_hi_test}"

# ── Datasets/languages: "dataset language" (comment / uncomment to select) ──
# German, French, Italian, Spanish, Portuguese, Dutch, Hindi
# "monsoon hi" uses the standalone VoiceArena/Monsoon_hi_test repo (no config);
# all others are configs of ${DATASET_PATH}. Hindi is scored with voi_oiwer over
# the dataset's reference lattice (see OIWER_LANGUAGES in normalizer/eval_utils.py).
DATASET_CONFIGS=(
    "fleurs de"
    "fleurs fr"
    "fleurs it"
    "fleurs es"
    "fleurs pt"
    "fleurs nl"
    "mcv de"
    "mcv es"
    "mcv fr"
    "mcv it"
    "mcv nl"
    "mls es"
    "mls fr"
    "mls it"
    "mls pt"
    "mls nl"
    "monsoon hi"
)

# Optional: restrict this run with DATASETS and/or LANGUAGES.
#
# DATASETS selects datasets, space-separated, in either of two forms (mixable):
#   - a bare name filters DATASET_CONFIGS to that dataset's entries. It matches
#     the first field, and a repo basename also matches, so both
#     "HF_English_Private_Set" and "hf-audio/HF_English_Private_Set" work.
#   - a "dataset:language" pair names that exact combination directly, whether or
#     not it is listed (or commented out) in DATASET_CONFIGS above.
# LANGUAGES filters whatever DATASETS left by language, matched against the
# second field. E.g.:
#   DATASETS="HF_English_Private_Set" MODEL="modulate/vfast 25" bash <this script>
#   LANGUAGES="nl" MODEL="modulate/multilingual 25" bash <this script>
#   DATASETS="fleurs mcv" LANGUAGES="nl de" bash <this script>
#   DATASETS="fleurs:de mls:it" bash <this script>
if [[ -n "${DATASETS:-}" ]]; then
    _selected=()
    for _want in ${DATASETS}; do
        if [[ "$_want" == *:* ]]; then
            # Explicit "dataset:language" pair — taken as given.
            _selected+=("${_want/:/ }")
        elif [[ ${#DATASET_CONFIGS[@]} -gt 0 ]]; then
            # Bare dataset name — keep its entries from DATASET_CONFIGS.
            for _cfg in "${DATASET_CONFIGS[@]}"; do
                read -r _name _ <<< "$_cfg"
                if [[ "$_name" == "$_want" || "${_name##*/}" == "$_want" ]]; then
                    _selected+=("$_cfg")
                fi
            done
        fi
    done
    if [[ ${#_selected[@]} -eq 0 ]]; then
        echo "ERROR: DATASETS='${DATASETS}' matched no active entry in DATASET_CONFIGS." >&2
        exit 1
    fi
    DATASET_CONFIGS=("${_selected[@]}")
fi
if [[ -n "${LANGUAGES:-}" ]]; then
    _selected=()
    if [[ ${#DATASET_CONFIGS[@]} -gt 0 ]]; then
        for _cfg in "${DATASET_CONFIGS[@]}"; do
            read -r _ _lang <<< "$_cfg"
            for _want in ${LANGUAGES}; do
                [[ "$_lang" == "$_want" ]] && _selected+=("$_cfg") && break
            done
        done
    fi
    if [[ ${#_selected[@]} -eq 0 ]]; then
        echo "ERROR: LANGUAGES='${LANGUAGES}' matched no active entry in DATASET_CONFIGS${DATASETS:+ (after DATASETS='${DATASETS}')}." >&2
        exit 1
    fi
    DATASET_CONFIGS=("${_selected[@]}")
fi
if [[ -n "${DATASETS:-}" || -n "${LANGUAGES:-}" ]]; then
    echo "Restricted to ${#DATASET_CONFIGS[@]} dataset/language combination(s): ${DATASET_CONFIGS[*]}"
fi

# Override MODEL_CONFIGS from the environment for quick runs, e.g.:
#   MODEL="modulate/multilingual 25" bash run_api_ml.sh
if [[ -n "${MODEL:-}" ]]; then
    MODEL_CONFIGS=("$MODEL")
fi

# Datasets that require lexical format prompt (azure only)
LEXICAL_DATASETS="mls-it"

# Resolve a "dataset language" pair to the repo it lives in and its config name.
# Sets DS_PATH and CONFIG_NAME (empty for standalone single-config repos).
resolve_dataset() {
    local dataset="$1" language="$2"
    if [[ "$dataset" == "monsoon" ]]; then
        DS_PATH="${MONSOON_DATASET_PATH}"
        CONFIG_NAME=""
    else
        DS_PATH="${DATASET_PATH}"
        CONFIG_NAME="${dataset}_${language}"
    fi
}

RUNDIR="${REPO_ROOT}"
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
# The API image pins datasets==2.19.0, which cannot read a dataset_info.json
# written by a newer datasets (e.g. "_type": "List", added in 4.x). Give it its
# own arrow cache so it never reads one the host wrote; it rebuilds there on the
# first run of each dataset.
DATASETS_CACHE_DIR="/hf_cache/datasets_api"

# Create the bind-mount sources up front: Docker would otherwise create them
# as root, and the containers run as the current user (see --user below).
mkdir -p "${RUNDIR}/results" "${HF_CACHE_DIR}"

echo "Building Docker image ${IMAGE_TAG} (context: ${REPO_ROOT})..."
docker build -f "${REPO_ROOT}/Dockerfile" -t "${IMAGE_TAG}" "${REPO_ROOT}"

echo "========================================================"
echo "Starting Multilingual API ASR Evaluation"
echo "========================================================"

for model_cfg in "${MODEL_CONFIGS[@]}"; do
    read -r MODEL_ID MAX_WORKERS <<< "$model_cfg"
    MODEL_FOLDER="${MODEL_ID//\//-}"

    echo ""
    echo "Processing Model: $MODEL_ID"
    echo "========================================================"

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r dataset language <<< "$cfg"
        resolve_dataset "$dataset" "$language"
        CONFIG_ARG=""
        [[ -n "$CONFIG_NAME" ]] && CONFIG_ARG="--config_name=${CONFIG_NAME}"

        PROMPT_FLAG=""
        if [[ "$MODEL_ID" == microsoft/* ]] && [[ " $LEXICAL_DATASETS " == *" ${dataset}-${language} "* ]]; then
            PROMPT_FLAG="--prompt 'Output must be in lexical format.'"
        fi

        echo ""
        echo "Running evaluation: ${CONFIG_NAME:-$dataset}"
        echo "   Model: $MODEL_ID"
        echo "   Dataset: $DS_PATH"
        echo "   Config: ${CONFIG_NAME:-(none)}"
        echo "   Language: $language"
        echo "   Time: $(date)"
        echo "----------------------------------------"

        docker run --rm \
            --user "$(id -u):$(id -g)" \
            -e HF_TOKEN="${HF_TOKEN:-}" \
            -e HF_HOME=/tmp/hf_home \
            -e HF_DATASETS_CACHE="${DATASETS_CACHE_DIR}" \
            -e NUMBA_CACHE_DIR=/tmp/numba_cache \
            -e MODULATE_API_KEY="${MODULATE_API_KEY:-}" \
            -e GLADIA_API_KEY="${GLADIA_API_KEY:-}" \
            -e OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
            -e ASSEMBLYAI_API_KEY="${ASSEMBLYAI_API_KEY:-}" \
            -e ELEVENLABS_API_KEY="${ELEVENLABS_API_KEY:-}" \
            -e REVAI_API_KEY="${REVAI_API_KEY:-}" \
            -e SPEECHMATICS_API_KEY="${SPEECHMATICS_API_KEY:-}" \
            -e AQUAVOICE_API_KEY="${AQUAVOICE_API_KEY:-}" \
            -e ZOOM_API_KEY="${ZOOM_API_KEY:-}" \
            -e SMALLESTAI_API_KEY="${SMALLESTAI_API_KEY:-}" \
            -e RESON8_API_KEY="${RESON8_API_KEY:-}" \
            -e AZURE_API_KEY="${AZURE_API_KEY:-}" \
            -e SONIOX_API_KEY="${SONIOX_API_KEY:-}" \
            -v "${RUNDIR}/results:/app/results" \
            -v "${REPO_ROOT}/../normalizer:/app/normalizer" \
            -v "${HF_CACHE_DIR}:/hf_cache" \
            "${IMAGE_TAG}" -c "
                cd /app && PYTHONPATH=/app python run_eval_ml.py \
                    --dataset_path=${DS_PATH} \
                    ${CONFIG_ARG} \
                    --language=${language} \
                    --split=test \
                    --model_name=${MODEL_ID} \
                    --max_workers=${MAX_WORKERS} \
                    ${PROMPT_FLAG}
            "

        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "Evaluation completed successfully for ${CONFIG_NAME:-$dataset}"
        else
            echo "Evaluation failed for ${CONFIG_NAME:-$dataset} (exit code: $exit_code)"
        fi
        echo "----------------------------------------"
    done

    echo ""
    echo "========================================================"
    echo "Evaluating results for $MODEL_ID"
    echo "========================================================"

    # Move this model's result files into their own folder
    MODEL_RESULTS_DIR="${RUNDIR}/results/${MODEL_FOLDER}"
    mkdir -p "${MODEL_RESULTS_DIR}"
    model_files=("${RUNDIR}/results/MODEL_${MODEL_FOLDER}_DATASET_"*.jsonl)
    if [[ -e "${model_files[0]}" ]]; then
        mv "${model_files[@]}" "${MODEL_RESULTS_DIR}/"
    else
        echo "WARNING: no result files found for ${MODEL_ID}"
    fi

    # Collect the set of languages actually evaluated (across all datasets)
    ALL_LANGUAGES=()
    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r dataset language <<< "$cfg"
        if [[ ! " ${ALL_LANGUAGES[*]} " == *" ${language} "* ]]; then
            ALL_LANGUAGES+=("$language")
        fi
    done

    # Evaluate results: one call per language, so each is normalized with the
    # correct language-specific normalizer and only its "ml_<lang>" family
    # CSV block is printed.
    for language in "${ALL_LANGUAGES[@]}"; do
        docker run --rm \
            --user "$(id -u):$(id -g)" \
            -e HF_HOME=/hf_cache \
            -v "${RUNDIR}/results:/app/results" \
            -v "${REPO_ROOT}/../normalizer:/app/normalizer" \
            -v "${HF_CACHE_DIR}:/hf_cache" \
            "${IMAGE_TAG}" -c "
                cd /app && PYTHONPATH=/app python -c \"from normalizer.eval_utils import score_results; score_results('/app/results/${MODEL_FOLDER}', '${MODEL_ID}', multilingual=True, language='${language}', families=['ml_${language}'], csv_only=True)\"
            "
    done

    if [[ -n "${RESULTS_BUCKET}" ]]; then
        # Only upload the specific files for the datasets in DATASET_CONFIGS
        INCLUDE_ARGS=()
        for cfg in "${DATASET_CONFIGS[@]}"; do
            read -r dataset language <<< "$cfg"
            resolve_dataset "$dataset" "$language"
            # Manifest names are "MODEL_<model>_DATASET_<repo-slug>_<config>_<split>.jsonl";
            # <config> is empty for standalone repos, leaving a double underscore.
            FNAME="MODEL_${MODEL_FOLDER}_DATASET_${DS_PATH//\//-}_${CONFIG_NAME}_test.jsonl"
            if [[ -f "${MODEL_RESULTS_DIR}/${FNAME}" ]]; then
                INCLUDE_ARGS+=(--include "${FNAME}")
            else
                echo "WARNING: result file not found, skipping upload: ${MODEL_RESULTS_DIR}/${FNAME}"
            fi
        done
        if [[ ${#INCLUDE_ARGS[@]} -gt 0 ]]; then
            hf buckets sync "${MODEL_RESULTS_DIR}" "hf://buckets/${RESULTS_BUCKET}/${MODEL_FOLDER}" \
                "${INCLUDE_ARGS[@]}" > /dev/null 2>&1
        fi
    fi

    echo ""
done

echo ""
echo "========================================================"
echo "All evaluations completed!"
echo "========================================================"
