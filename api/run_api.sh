#!/bin/bash

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
    # "revai/machine                 4"
    # "revai/fusion                  4"
    # "speechmatics/enhanced         4"
    # "aquavoice/avalon-v1-en        5"
    # "zoom/scribe_v1                32"
    # "smallestai/pulse              16"
    # "reson8/resonant-1             16"
    # "reson8/resonant-1-flash       16"
    # "microsoft/azure-speech-06-2026  4"
    # "modulate/vfast                25"
    # "gladia/solaria-3             20"
    # "soniox/stt-async-v5           20"
)
DATASET_PATH="hf-audio/open-asr-leaderboard"
MONSOON_EN_IN_DATASET_PATH="${MONSOON_EN_IN_DATASET_PATH:-VoiceArena/Monsoon_en_IN_test}"

EVAL_DATASETS=(
    "ami_cleaned:test"
    "earnings22:test"
    "gigaspeech_cleaned:test"
    "librispeech:test.clean"
    "librispeech:test.other"
    "spgispeech:test"
    "voxpopuli_cleaned_aa:test"
    # Standalone single-config repo, not a config of ${DATASET_PATH}
    "monsoon_en_in:test"
)

# Override EVAL_DATASETS or MODEL_CONFIGS from the environment for quick runs, e.g.:
#   DATASETS="librispeech:test.clean" MODEL="modulate/vfast 25" bash run_api.sh
if [[ -n "${DATASETS:-}" ]]; then
    read -ra EVAL_DATASETS <<< "$DATASETS"
fi
if [[ -n "${MODEL:-}" ]]; then
    MODEL_CONFIGS=("$MODEL")
fi

# Datasets that require lexical format prompt
LEXICAL_DATASETS="librispeech gigaspeech"

# Resolve a dataset name to the repo it lives in and its config name.
# Sets DS_PATH and DATASET_NAME (empty for standalone single-config repos).
resolve_dataset() {
    local dataset="$1"
    if [[ "$dataset" == "monsoon_en_in" ]]; then
        DS_PATH="${MONSOON_EN_IN_DATASET_PATH}"
        DATASET_NAME=""
    else
        DS_PATH="${DATASET_PATH}"
        DATASET_NAME="${dataset}"
    fi
}

RUNDIR="${REPO_ROOT}"
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
# The API image pins datasets==2.19.0, which cannot read a dataset_info.json
# written by a newer datasets (e.g. "_type": "List", added in 4.x). Give it its
# own arrow cache so it never reads one the host wrote; it rebuilds there on the
# first run of each dataset.
DATASETS_CACHE_DIR="/hf_cache/datasets_api"

echo "Building Docker image ${IMAGE_TAG} (context: ${REPO_ROOT})..."
docker build -f "${REPO_ROOT}/Dockerfile" -t "${IMAGE_TAG}" "${REPO_ROOT}"

for model_cfg in "${MODEL_CONFIGS[@]}"; do
    read -r MODEL_ID MAX_WORKERS <<< "$model_cfg"
    MODEL_FOLDER="${MODEL_ID//\//-}"

    for entry in "${EVAL_DATASETS[@]}"; do
        DATASET="${entry%%:*}"
        SPLIT="${entry##*:}"
        resolve_dataset "$DATASET"

        PROMPT_FLAG=""
        if [[ "$MODEL_ID" == microsoft/* ]] && [[ " $LEXICAL_DATASETS " == *" $DATASET "* ]]; then
            PROMPT_FLAG="--prompt 'Output must be in lexical format.'"
        fi

        docker run --rm \
            --user "$(id -u):$(id -g)" \
            -e HF_TOKEN="${HF_TOKEN:-}" \
            -e HF_HOME=/tmp/hf_home \
            -e HF_DATASETS_CACHE="${DATASETS_CACHE_DIR}" \
            -e NUMBA_CACHE_DIR=/tmp/numba_cache \
            -e MODULATE_API_KEY="${MODULATE_API_KEY:-}" \
            -e GLADIA_API_KEY="${GLADIA_API_KEY:-}" \
            -e OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
            -e SONIOX_API_KEY="${SONIOX_API_KEY:-}" \
            -e ASSEMBLYAI_API_KEY="${ASSEMBLYAI_API_KEY:-}" \
            -e ELEVENLABS_API_KEY="${ELEVENLABS_API_KEY:-}" \
            -e REVAI_API_KEY="${REVAI_API_KEY:-}" \
            -e SPEECHMATICS_API_KEY="${SPEECHMATICS_API_KEY:-}" \
            -e AQUAVOICE_API_KEY="${AQUAVOICE_API_KEY:-}" \
            -e ZOOM_API_KEY="${ZOOM_API_KEY:-}" \
            -e SMALLESTAI_API_KEY="${SMALLESTAI_API_KEY:-}" \
            -e RESON8_API_KEY="${RESON8_API_KEY:-}" \
            -e AZURE_API_KEY="${AZURE_API_KEY:-}" \
            -v "${RUNDIR}/results:/app/results" \
            -v "${REPO_ROOT}/../normalizer:/app/normalizer" \
            -v "${HF_CACHE_DIR}:/hf_cache" \
            "${IMAGE_TAG}" -c "
                cd /app && PYTHONPATH=/app python run_eval.py \
                    --dataset_path=${DS_PATH} \
                    --dataset=${DATASET_NAME} \
                    --split=${SPLIT} \
                    --model_name=${MODEL_ID} \
                    --max_workers=${MAX_WORKERS} \
                    ${PROMPT_FLAG}
            "
    done

    MODEL_RESULTS_DIR="${RUNDIR}/results/${MODEL_FOLDER}"
    mkdir -p "${MODEL_RESULTS_DIR}"
    model_files=("${RUNDIR}/results/MODEL_${MODEL_FOLDER}_DATASET_"*.jsonl)
    if [[ -e "${model_files[0]}" ]]; then
        mv "${model_files[@]}" "${MODEL_RESULTS_DIR}/"
    else
        echo "WARNING: no result files found for ${MODEL_ID}"
    fi

    docker run --rm \
        --user "$(id -u):$(id -g)" \
        -e HF_HOME=/hf_cache \
        -v "${RUNDIR}/results:/app/results" \
        -v "${REPO_ROOT}/../normalizer:/app/normalizer" \
        -v "${HF_CACHE_DIR}:/hf_cache" \
        "${IMAGE_TAG}" -c "
            cd /app && PYTHONPATH=/app python -c \"from normalizer.eval_utils import score_results; score_results('/app/results/${MODEL_FOLDER}', '${MODEL_ID}')\"
        "

    if [[ -n "${RESULTS_BUCKET}" ]]; then
        # Only upload the specific files for the datasets in EVAL_DATASETS
        INCLUDE_ARGS=()
        for entry in "${EVAL_DATASETS[@]}"; do
            _DS="${entry%%:*}"
            _SP="${entry##*:}"
            resolve_dataset "$_DS"
            # Manifest names are "MODEL_<model>_DATASET_<repo-slug>_<config>_<split>.jsonl";
            # <config> is empty for standalone repos, leaving a double underscore.
            FNAME="MODEL_${MODEL_FOLDER}_DATASET_${DS_PATH//\//-}_${DATASET_NAME}_${_SP}.jsonl"
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
done
