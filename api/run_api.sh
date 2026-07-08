#!/bin/bash

RESULTS_BUCKET="${RESULTS_BUCKET:-}"
IMAGE_TAG="api-eval"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/." && pwd)"

if [[ -n "${RESULTS_BUCKET}" && -z "${HF_TOKEN}" ]]; then
    echo "ERROR: RESULTS_BUCKET is set but HF_TOKEN is not. Cannot write to bucket." >&2
    exit 1
fi

# ── Models: "model_id use_url max_workers" ───────────────────────────────────
# use_url=true  → provider receives a remote audio URL (revai, zoom)
# use_url=false → provider receives a local audio file (all others)
MODEL_CONFIGS=(
    # "openai/gpt-4o-transcribe      false  16"
    # "openai/gpt-4o-mini-transcribe false  16"
    # "openai/whisper-1              false  16"
    # "assembly/universal-3-pro      false  4"
    # "assembly/universal-3-5-pro    false  4"
    # "elevenlabs/scribe_v2          false  8"
    # "revai/machine                 false  4"
    # "revai/fusion                  false  4"
    # "speechmatics/enhanced         false  4"
    # "aquavoice/avalon-v1-en        false  5"
    # "zoom/scribe_v1                false  32"
    # "smallestai/pulse              false  16"
    # "reson8/resonant-1             false  16"
    # "reson8/resonant-1-flash       false  16"
    # "microsoft/azure-speech-05-2026  false  4"
    # "modulate/vfast   false  25"
)
DATASET_PATH="hf-audio/open-asr-leaderboard"

declare -a EVAL_DATASETS=(
    "ami_cleaned:test"
    "earnings22:test"
    "gigaspeech_cleaned:test"
    "librispeech:test.clean"
    "librispeech:test.other"
    "spgispeech:test"
    "voxpopuli_cleaned_aa:test"
)

# Datasets that require lexical format prompt
LEXICAL_DATASETS="librispeech gigaspeech"

RUNDIR="${REPO_ROOT}"
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"

echo "Building Docker image ${IMAGE_TAG} (context: ${REPO_ROOT})..."
docker build -f "${REPO_ROOT}/Dockerfile" -t "${IMAGE_TAG}" "${REPO_ROOT}"

for model_cfg in "${MODEL_CONFIGS[@]}"; do
    read -r MODEL_ID USE_URL MAX_WORKERS <<< "$model_cfg"
    USE_URL_FLAG=$([[ "$USE_URL" == "true" ]] && echo "--use_url" || echo "")
    MODEL_FOLDER="${MODEL_ID//\//-}"

    for entry in "${EVAL_DATASETS[@]}"; do
        DATASET="${entry%%:*}"
        SPLIT="${entry##*:}"

        PROMPT_FLAG=""
        if [[ "$MODEL_ID" == microsoft/* ]] && [[ " $LEXICAL_DATASETS " == *" $DATASET "* ]]; then
            PROMPT_FLAG="--prompt 'Output must be in lexical format.'"
        fi

        docker run --rm \
            --user "$(id -u):$(id -g)" \
            -e HF_TOKEN="${HF_TOKEN:-}" \
            -e HF_HOME=/tmp/hf_home \
            -e HF_DATASETS_CACHE=/hf_cache/datasets \
            -e NUMBA_CACHE_DIR=/tmp/numba_cache \
            -e MODULATE_API_KEY="${MODULATE_API_KEY:-}" \
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
            -v "${RUNDIR}/results:/app/results" \
            -v "${REPO_ROOT}/../normalizer:/app/normalizer" \
            -v "${HF_CACHE_DIR}:/hf_cache" \
            "${IMAGE_TAG}" -c "
                cd /app && PYTHONPATH=/app python run_eval.py \
                    --dataset_path=${DATASET_PATH} \
                    --dataset=${DATASET} \
                    --split=${SPLIT} \
                    --model_name=${MODEL_ID} \
                    --max_workers=${MAX_WORKERS} \
                    ${USE_URL_FLAG} \
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
        # Only upload the specific files for the datasets in EVAL_DATASETS to
        # avoid accidentally pushing private dataset results to the public bucket.
        # hf buckets sync is directory-based, so we use --include to whitelist
        # only the exact filenames we want to upload.
        DATASET_PATH_SLUG="${DATASET_PATH//\//-}"
        INCLUDE_ARGS=()
        for entry in "${EVAL_DATASETS[@]}"; do
            _DS="${entry%%:*}"
            _SP="${entry##*:}"
            FNAME="MODEL_${MODEL_FOLDER}_DATASET_${DATASET_PATH_SLUG}_${_DS}_${_SP}.jsonl"
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