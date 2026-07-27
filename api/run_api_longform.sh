#!/bin/bash

set -euo pipefail

RESULTS_BUCKET="${RESULTS_BUCKET:-}"
IMAGE_TAG="api-eval"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/." && pwd)"

if [[ -n "${RESULTS_BUCKET}" && -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: RESULTS_BUCKET is set but HF_TOKEN is not. Cannot write to bucket." >&2
    exit 1
fi

# ── Models: "model_id max_workers" ───────────────────────────────────────────
# Worker limits match the corresponding entries in run_api.sh.
MODEL_CONFIGS=(
    # "openai/gpt-4o-transcribe      16"
    # "openai/gpt-4o-mini-transcribe 16"
    # "openai/whisper-1              16"
    # "assembly/universal-3-pro      4"
    # "elevenlabs/scribe_v2          8"
    # "revai/machine                 4"
    # "revai/fusion                  4"
    # "speechmatics/enhanced         4"
    # "aquavoice/avalon-v1-en        5"
    # "reson8/resonant-1             16"
    # "reson8/resonant-1-flash       16"
)

# ── Datasets: "dataset_path:config:split" ────────────────────────────────────
# TED-LIUM is loaded from its original repository because it is no longer in
# the consolidated long-form dataset.
DATASET_CONFIGS=(
    "hf-audio/asr-leaderboard-longform:earnings21:test"
    "hf-audio/asr-leaderboard-longform:earnings22:test"
    "distil-whisper/tedlium-long-form:default:test"
    "bezzam/coraal:ATL:test"
    "bezzam/coraal:DCA:test"
    "bezzam/coraal:DCB:test"
    "bezzam/coraal:DTA:test"
    "bezzam/coraal:LES:test"
    "bezzam/coraal:PRV:test"
    "bezzam/coraal:ROC:test"
    "bezzam/coraal:VLD:test"
)

# Override either matrix for a focused run, for example:
#   MODEL="elevenlabs/scribe_v2 8" \
#   DATASETS="hf-audio/asr-leaderboard-longform:earnings21:test" \
#   bash api/run_api_longform.sh
if [[ -n "${DATASETS:-}" ]]; then
    read -ra DATASET_CONFIGS <<< "${DATASETS}"
fi
if [[ -n "${MODEL:-}" ]]; then
    MODEL_CONFIGS=("${MODEL}")
fi

RUNDIR="${REPO_ROOT}"
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "${RUNDIR}/results"

echo "Building Docker image ${IMAGE_TAG} (context: ${REPO_ROOT})..."
docker build -f "${REPO_ROOT}/Dockerfile" -t "${IMAGE_TAG}" "${REPO_ROOT}"

for model_cfg in "${MODEL_CONFIGS[@]}"; do
    read -r MODEL_ID MAX_WORKERS <<< "${model_cfg}"
    MODEL_FOLDER="${MODEL_ID//\//-}"

    for dataset_cfg in "${DATASET_CONFIGS[@]}"; do
        IFS=':' read -r DATASET_PATH DATASET SPLIT <<< "${dataset_cfg}"

        docker run --rm \
            --user "$(id -u):$(id -g)" \
            -e HF_TOKEN="${HF_TOKEN:-}" \
            -e HF_HOME=/tmp/hf_home \
            -e HF_DATASETS_CACHE=/hf_cache/datasets \
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
                cd /app && PYTHONPATH=/app python run_eval.py \
                    --dataset_path=${DATASET_PATH} \
                    --dataset=${DATASET} \
                    --split=${SPLIT} \
                    --model_name=${MODEL_ID} \
                    --max_workers=${MAX_WORKERS}
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
        -v "${REPO_ROOT}/../scripts:/app/scripts:ro" \
        -v "${HF_CACHE_DIR}:/hf_cache" \
        "${IMAGE_TAG}" -c "
            cd /app && PYTHONPATH=/app python /app/scripts/score_longform_results.py \
                /app/results/${MODEL_FOLDER} \
                --model-id=${MODEL_ID} \
                --current-csv=/app/scripts/data/en_longform.csv
        "

    if [[ -n "${RESULTS_BUCKET}" ]]; then
        # Upload only this run's public long-form manifests. The JSONL files
        # contain the raw, unnormalized references and predictions.
        INCLUDE_ARGS=()
        for dataset_cfg in "${DATASET_CONFIGS[@]}"; do
            IFS=':' read -r DATASET_PATH DATASET SPLIT <<< "${dataset_cfg}"
            DATASET_PATH_SLUG="${DATASET_PATH//\//-}"
            FNAME="MODEL_${MODEL_FOLDER}_DATASET_${DATASET_PATH_SLUG}_${DATASET}_${SPLIT}.jsonl"
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
