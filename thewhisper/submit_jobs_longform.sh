#!/bin/bash
# Local script to submit HF Jobs for English long-form ASR evaluation.
# Evaluates on earnings21 and earnings22 (hf-audio/asr-leaderboard-longform) and the CORAAL subsets (bezzam/coraal).
# This script is NOT pushed to the HF Space — it runs on your local machine.
# Usage: HF_TOKEN=hf_... bash submit_jobs_longform.sh
# The TheStage AI token for the compiled engines is set below (THESTAGE_AUTH_TOKEN).

# ── Configuration ────────────────────────────────────────────────────────────
SPACE="${SPACE:-TheStageAI/open-asr-leaderboard-thewhisper}"
RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_longform}"      # HF bucket repo for saving results
FLAVOR="${FLAVOR:-h200}"  # compiled engines are published for H200 (and H100, A100, L40S, RTX 4090/5090)
ORG_NAME="${ORG_NAME:-}"
# TheStage AI token for downloading the compiled engines; passed to every job as a secret.
export THESTAGE_AUTH_TOKEN="${THESTAGE_AUTH_TOKEN:-th_Har1cDb3EWLfNLtrfXaPwBc9cev6BtSNNUYiZd8r}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Set USE_LOCAL_SCRIPT=1 to run your local run_eval_longform.py instead of the version
# committed to the Space (useful for iterating without pushing to the Space).
USE_LOCAL_SCRIPT="${USE_LOCAL_SCRIPT:-1}"
LOCAL_SCRIPT_INJECT=""
if [[ "$USE_LOCAL_SCRIPT" == "1" ]]; then
    RUN_EVAL_B64=$(base64 < "${SCRIPT_DIR}/run_eval_longform.py" | tr -d '\n')
    LOCAL_SCRIPT_INJECT="echo '${RUN_EVAL_B64}' | base64 -d > /app/run_eval_longform.py &&"
fi

# Set USE_LOCAL_NORMALIZER=1 to inject your local normalizer/ package into the
# job (so normalizer changes take effect without updating the HF Space).
USE_LOCAL_NORMALIZER="${USE_LOCAL_NORMALIZER:-1}"
LOCAL_NORMALIZER_INJECT=""
if [[ "$USE_LOCAL_NORMALIZER" == "1" ]]; then
    NORMALIZER_B64=$(tar --exclude='__pycache__' --exclude='*.pyc' -czf - -C "${REPO_ROOT}" normalizer | base64 | tr -d '\n')
    LOCAL_NORMALIZER_INJECT="echo '${NORMALIZER_B64}' | base64 -d | tar -xzf - -C /app &&"
fi

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_ID="TheStageAI/thewhisper-large-v3-turbo"
REVISION="6592b6933656513345c5e7a65523a0de3e30d5a3"  # pins the configs and the compiled engines
MODE="XL"                 # engine size
CHUNK_LENGTH=30           # engine input window, seconds
BATCH_SIZE=32             # recordings handed to the pipeline per call
PIPELINE_BATCH_SIZE=256   # speech windows decoded together: largest batch of the H200 engines (1-256; the H100 ones 1-128)
MODEL_CONFIGS=("${MODEL_ID} ${BATCH_SIZE}")

# ── Datasets: "dataset_path dataset" (comment / uncomment to select) ─────────
DATASET_CONFIGS=(
    "hf-audio/asr-leaderboard-longform earnings21"
    "hf-audio/asr-leaderboard-longform earnings22"
    "bezzam/coraal ATL"
    "bezzam/coraal DCA"
    "bezzam/coraal DCB"
    "bezzam/coraal DTA"
    "bezzam/coraal LES"
    "bezzam/coraal PRV"
    "bezzam/coraal ROC"
    "bezzam/coraal VLD"
)
# Optional: restrict this run to specific datasets, matched against the second
# field of each DATASET_CONFIGS entry, e.g.:
#   ONLY_DATASETS="earnings21 earnings22" bash <this script>
if [[ -n "${ONLY_DATASETS:-}" ]]; then
    _selected=()
    for _cfg in "${DATASET_CONFIGS[@]}"; do
        read -r _ _name <<< "$_cfg"
        for _want in ${ONLY_DATASETS}; do
            [[ "$_name" == "$_want" ]] && _selected+=("$_cfg")
        done
    done
    if [[ ${#_selected[@]} -eq 0 ]]; then
        echo "ERROR: ONLY_DATASETS='${ONLY_DATASETS}' matched no entry in DATASET_CONFIGS." >&2
        exit 1
    fi
    DATASET_CONFIGS=("${_selected[@]}")
fi


# ── Submit one job per model/dataset combination ─────────────────────────────
for model_cfg in "${MODEL_CONFIGS[@]}"; do
    read -r MODEL_ID BATCH_SIZE <<< "$model_cfg"
    # Sanitize model ID for use as a folder name (e.g. "openai/whisper" -> "openai-whisper")
    MODEL_FOLDER="${MODEL_ID//\//-}"

    echo "████████████████████████████████████████████████████████████████████████████████"
    echo "  Evaluating: ${MODEL_ID}"
    echo "████████████████████████████████████████████████████████████████████████████████"

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET_PATH DATASET <<< "$cfg"
        echo "Submitting job: model=${MODEL_ID} dataset_path=${DATASET_PATH} dataset=${DATASET} batch_size=${BATCH_SIZE}"

        NAMESPACE_ARG=""
        [ -n "$ORG_NAME" ] && NAMESPACE_ARG="--namespace ${ORG_NAME}"

        # --streaming: the recordings are up to about an hour long, and loading them in full overflows an Arrow block
        hf jobs run \
            --flavor "$FLAVOR" \
            --timeout 8h \
            --env HF_TOKEN="$HF_TOKEN" \
            --secrets THESTAGE_AUTH_TOKEN \
            ${NAMESPACE_ARG} \
            --volume "hf://buckets/${RESULTS_BUCKET}:/results" \
            "hf.co/spaces/${SPACE}" \
            bash -c "
                ${LOCAL_NORMALIZER_INJECT}
                ${LOCAL_SCRIPT_INJECT}
                PYTHONPATH=/app python run_eval_longform.py \
                    --model_id=${MODEL_ID} \
                    --revision=${REVISION} \
                    --mode=${MODE} \
                    --chunk_length=${CHUNK_LENGTH} \
                    --dataset_path=${DATASET_PATH} \
                    --dataset=${DATASET} \
                    --split=test \
                    --device=0 \
                    --batch_size=${BATCH_SIZE} \
                    --pipeline_batch_size=${PIPELINE_BATCH_SIZE} \
                    --max_eval_samples=-1 \
                    --streaming &&
                mkdir -p /results/${MODEL_FOLDER} &&
                cp results/*.jsonl /results/${MODEL_FOLDER}/
            " > /dev/null 2>&1 &    # suppress output and run in background
    done
    if [ -n "$ORG_NAME" ]; then
        echo "For live status see: https://huggingface.co/organizations/${ORG_NAME}/settings/jobs"
    else
        echo "For live status see: https://huggingface.co/settings/jobs"
    fi

    # Wait for all background job submissions to complete
    wait
    echo "All jobs finished."
    sleep 10  # allow time for the last results to be flushed to the bucket

    # Download results and score
    mkdir -p "./results/${MODEL_FOLDER}"

    hf buckets sync \
        "hf://buckets/${RESULTS_BUCKET}/${MODEL_FOLDER}" \
        "./results/${MODEL_FOLDER}" > /dev/null 2>&1

    EXPECTED=${#DATASET_CONFIGS[@]}
    ACTUAL=$(find "./results/${MODEL_FOLDER}" -name "*.jsonl" | wc -l)
    if [[ "$ACTUAL" -lt "$EXPECTED" ]]; then
        echo "WARNING: expected ${EXPECTED} result files but only found ${ACTUAL}. Some jobs may not have finished yet."
    else
        echo "All ${ACTUAL} result files present."
    fi

    # Long-form average as on the leaderboard: mean of earnings21, earnings22 and the CORAAL average
    PYTHONPATH="${REPO_ROOT}" python -c "
from normalizer.eval_utils import score_results
_, results = score_results('$(pwd)/results/${MODEL_FOLDER}', '${MODEL_ID}')
wer = {key.split(' | ')[1]: value['wer'] for key, value in results.items()}
earnings = [w for name, w in wer.items() if 'earnings21' in name or 'earnings22' in name]
coraal = [w for name, w in wer.items() if 'coraal' in name]
if len(earnings) == 2 and coraal:
    print(f'Long-form average WER (earnings21, earnings22, CORAAL avg over {len(coraal)} subsets): '
          f'{(sum(earnings) + sum(coraal) / len(coraal)) / 3:.2f}')
"

done
