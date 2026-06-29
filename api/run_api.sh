#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

export AQUAVOICE_API_KEY="your_api_key"
export ELEVENLABS_API_KEY="your_api_key"
export ASSEMBLYAI_API_KEY="your_api_key"
export SMALLESTAI_API_KEY="your_api_key"
export SPEECHMATICS_API_KEY="your_api_key"
export RESON8_API_KEY="your_api_key"
export AZURE_API_KEY="your_api_key"
export REVAI_API_KEY="your_api_key"
export ZOOM_API_KEY="your_api_key"
export OPENAI_API_KEY="your_api_key"

export HF_TOKEN="your_api_key"
RESULTS_BUCKET="${RESULTS_BUCKET:-}"

# ── Models: "model_id use_url max_workers" ───────────────────────────────────
# use_url=true  → provider receives a remote audio URL (revai, zoom)
# use_url=false → provider receives a local audio file (all others)
MODEL_CONFIGS=(
    # "openai/gpt-4o-transcribe      false  16"
    # "openai/gpt-4o-mini-transcribe false  16"
    # "openai/whisper-1              false  16"
    # "assembly/universal-3-pro      false  4"
    # "elevenlabs/scribe_v1          false  16"
    # "revai/machine                 false  4"
    # "revai/fusion                  false  4"
    # "speechmatics/enhanced         false  4"
    # "aquavoice/avalon-v1-en        false  5"
    # "zoom/scribe_v1                false  32"
    # "smallestai/pulse              false  16"
    # "reson8/resonant-1             false  16"
    # "reson8/resonant-1-flash       false  16"
    # "microsoft/azure-speech-05-2026  false  4"
    "modulate/vfast   false  16"
)
DATASET_PATH="hf-audio/open-asr-leaderboard"

declare -a EVAL_DATASETS=(
    "ami:test"
    "earnings22:test"
    "gigaspeech:test"
    "librispeech:test.clean"
    "voxpopuli:test"
    "librispeech:test.other"
    "spgispeech:test"
)

# Datasets that require lexical format prompt
LEXICAL_DATASETS="librispeech gigaspeech"

RUNDIR=$(pwd)

for model_cfg in "${MODEL_CONFIGS[@]}"; do
    read -r MODEL_ID USE_URL MAX_WORKERS <<< "$model_cfg"
    USE_URL_FLAG=$([[ "$USE_URL" == "true" ]] && echo "--use_url" || echo "")
    MODEL_FOLDER="${MODEL_ID//\//-}"

    for entry in "${EVAL_DATASETS[@]}"; do
        DATASET="${entry%%:*}"
        SPLIT="${entry##*:}"

        PROMPT_ARGS=()
        if [[ "$MODEL_ID" == microsoft/* ]] && [[ " $LEXICAL_DATASETS " == *" $DATASET "* ]]; then
            PROMPT_ARGS=(--prompt "Output must be in lexical format.")
        fi

        python run_eval.py \
            --dataset_path="$DATASET_PATH" \
            --dataset="$DATASET" \
            --split="$SPLIT" \
            --model_name "$MODEL_ID" \
            --max_workers "$MAX_WORKERS" \
            ${USE_URL_FLAG} \
            "${PROMPT_ARGS[@]}"
    done

    MODEL_RESULTS_DIR="${RUNDIR}/results/${MODEL_FOLDER}"
    mkdir -p "${MODEL_RESULTS_DIR}"
    model_files=("${RUNDIR}/results/MODEL_${MODEL_FOLDER}_DATASET_"*.jsonl)
    if [[ -e "${model_files[0]}" ]]; then
        mv "${model_files[@]}" "${MODEL_RESULTS_DIR}/"
    else
        echo "WARNING: no result files found for ${MODEL_ID}"
    fi

    PYTHONPATH="${RUNDIR}/..:${PYTHONPATH}" python -c "from normalizer.eval_utils import score_results; score_results('${MODEL_RESULTS_DIR}', '${MODEL_ID}')"

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