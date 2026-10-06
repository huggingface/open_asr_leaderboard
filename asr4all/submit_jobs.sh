#!/bin/bash
# Local script to submit HF Jobs for FUTO asr4all evaluation.
# Usage: ORG_NAME=<org> RESULTS_BUCKET=<bucket> bash submit_jobs.sh

# ── Configuration ────────────────────────────────────────────────────────────
SPACE="${SPACE:-futo-org/open-asr-leaderboard-asr4all}"
RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_h200}"
DEFAULT_DATASET_PATH="${DEFAULT_DATASET_PATH:-hf-audio/open-asr-leaderboard}"
FLAVOR="${FLAVOR:-h200}"
HF_CLI="${HF_CLI:-hf}"
ORG_NAME="${ORG_NAME:-}"
REVISION="${REVISION:-open-asr-leaderboard}"  # trust_remote_code: pinned tag on each model repo

# One decoding configuration for every dataset. --trunk_op 256,64 decodes clips up to 10.24 s in a
# single window (full context); longer audio uses a 100-back / 64-ahead sliding window, computed
# block-sparse (--block_sparse 1, FlexAttention).  The same flags apply to every dataset.
EVAL_FLAGS="--dtype bfloat16 --kernels ldsa,fir --cudagraphs 0 --manual_graphs 1 \
--manual_graphs_pcec 1 --torch_compile max-autotune --compile_static 1 --pad_multiple 32000 \
--timed_passes 5 --cooldown_c 0 --trunk_op 256,64 --block_sparse 1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Set USE_LOCAL_SCRIPT=1 to run your local run_eval.py instead of the version
# committed to the Space (useful for iterating without pushing to the Space).
USE_LOCAL_SCRIPT="${USE_LOCAL_SCRIPT:-1}"
LOCAL_SCRIPT_INJECT=""
if [[ "$USE_LOCAL_SCRIPT" == "1" ]]; then
    RUN_EVAL_B64=$(base64 -w0 "${SCRIPT_DIR}/run_eval.py")
    LOCAL_SCRIPT_INJECT="echo '${RUN_EVAL_B64}' | base64 -d > /app/run_eval.py &&"
fi

# Set USE_LOCAL_NORMALIZER=1 to inject your local normalizer/ package into the
# job (so normalizer changes take effect without updating the HF Space).
USE_LOCAL_NORMALIZER="${USE_LOCAL_NORMALIZER:-1}"
LOCAL_NORMALIZER_INJECT=""
if [[ "$USE_LOCAL_NORMALIZER" == "1" ]]; then
    NORMALIZER_B64=$(tar --exclude='__pycache__' --exclude='*.pyc' -czf - -C "${REPO_ROOT}" normalizer | base64 -w0)
    LOCAL_NORMALIZER_INJECT="echo '${NORMALIZER_B64}' | base64 -d | tar -xzf - -C /app &&"
fi

# ── Models ────────────────────────────────────────────────────────────────────
MODEL_CONFIGS=(
    "futo-org/asr4all-s"
    "futo-org/asr4all-m"
    "futo-org/asr4all-l"
)

# ── Datasets: "name split batch_size [dataset_path]" ──────────────────────────
# dataset_path defaults to $DEFAULT_DATASET_PATH when omitted.
DATASET_CONFIGS=(
    "ami_cleaned test 256"
    "gigaspeech_cleaned test 256"
    "voxpopuli_cleaned_aa test 256"
    "earnings22_cleaned_aa_chunked test 256 ArtificialAnalysis/Earnings22-Cleaned-AA-chunked"
    "librispeech test.clean 256"
    "librispeech test.other 256"
    "spgispeech test 1024"
    "monsoon_en_in test 256 VoiceArena/Monsoon_en_IN_test"
)
# Per-model batch overrides ("model dataset batch"): the dataset table above is the default for every
# model; a larger model may need a smaller maximum batch to fit in H200 memory.
BATCH_OVERRIDES=(
    "futo-org/asr4all-m spgispeech 512"
    "futo-org/asr4all-l spgispeech 512"
)

if [[ -n "${ONLY_DATASETS:-}" ]]; then
    _selected=()
    for _cfg in "${DATASET_CONFIGS[@]}"; do
        read -r _name _ <<< "$_cfg"
        for _want in ${ONLY_DATASETS}; do
            [[ "$_name" == "$_want" || "${_name##*/}" == "$_want" ]] && _selected+=("$_cfg")
        done
    done
    [[ ${#_selected[@]} -eq 0 ]] && { echo "ERROR: ONLY_DATASETS='${ONLY_DATASETS}' matched nothing." >&2; exit 1; }
    DATASET_CONFIGS=("${_selected[@]}")
fi
if [[ -n "${ONLY_MODELS:-}" ]]; then
    MODEL_CONFIGS=(${ONLY_MODELS})
fi

# ── Submit one job per model/dataset combination ─────────────────────────────
for MODEL_ID in "${MODEL_CONFIGS[@]}"; do
    MODEL_FOLDER="${MODEL_ID//\//-}"
    echo "Evaluating: ${MODEL_ID} @ ${REVISION}"

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET SPLIT BATCH_SIZE DATASET_PATH <<< "$cfg"
        for _ov in "${BATCH_OVERRIDES[@]}"; do
            read -r _m _d _b <<< "$_ov"
            [[ "$_m" == "$MODEL_ID" && "$_d" == "$DATASET" ]] && BATCH_SIZE="$_b"
        done
        if [[ -n "$DATASET_PATH" ]]; then
            DATASET_CONFIG=""  # own-repo datasets: empty config, as the scorer and upstream scripts expect
        else
            DATASET_PATH="$DEFAULT_DATASET_PATH"
            DATASET_CONFIG="$DATASET"
        fi
        [[ "$DATASET" == earnings22* ]] && DATASET_CONFIG="$DATASET"
        echo "Submitting job: model=${MODEL_ID} dataset_path=${DATASET_PATH} dataset=${DATASET_CONFIG} split=${SPLIT}"

        NAMESPACE_ARG=""
        [ -n "$ORG_NAME" ] && NAMESPACE_ARG="--namespace ${ORG_NAME}"

        ${HF_CLI} jobs run \
            --flavor "$FLAVOR" \
            --name "asr4all-${MODEL_FOLDER##*-}-${DATASET}-${SPLIT//./_}" \
            --timeout 8h \
            --secrets HF_TOKEN \
            --env HF_AUDIO_DECODER_BACKEND="soundfile" \
            ${NAMESPACE_ARG} \
            --volume "hf://buckets/${RESULTS_BUCKET}:/results" \
            "hf.co/spaces/${SPACE}" \
            bash -c "
                ${LOCAL_NORMALIZER_INJECT}
                ${LOCAL_SCRIPT_INJECT}
                PYTHONPATH=/app python run_eval.py \
                    --model_id=${MODEL_ID} \
                    --revision=${REVISION} \
                    --dataset_path=${DATASET_PATH} \
                    --dataset=${DATASET_CONFIG} \
                    --split=${SPLIT} \
                    --device=0 \
                    --batch_size=${BATCH_SIZE} \
                    ${EVAL_FLAGS} &&
                mkdir -p /results/${MODEL_FOLDER} &&
                cp results/*.jsonl /results/${MODEL_FOLDER}/
            " > /dev/null 2>&1 &
    done
    [ -n "$ORG_NAME" ] && echo "For live status see: https://huggingface.co/organizations/${ORG_NAME}/settings/jobs"

    wait
    echo "All jobs finished for ${MODEL_ID}."
    sleep 10

    mkdir -p "./results/${MODEL_FOLDER}"
    ${HF_CLI} buckets sync "hf://buckets/${RESULTS_BUCKET}/${MODEL_FOLDER}" "./results/${MODEL_FOLDER}" > /dev/null 2>&1

    EXPECTED=${#DATASET_CONFIGS[@]}
    ACTUAL=$(find "./results/${MODEL_FOLDER}" -name "*.jsonl" | wc -l)
    if [[ "$ACTUAL" -lt "$EXPECTED" ]]; then
        echo "WARNING: expected ${EXPECTED} result files but only found ${ACTUAL}."
    else
        echo "All ${ACTUAL} result files present."
    fi

    PYTHONPATH="${REPO_ROOT}" python -c "
from normalizer.eval_utils import score_results
score_results('$(pwd)/results/${MODEL_FOLDER}', '${MODEL_ID}')
"
done
