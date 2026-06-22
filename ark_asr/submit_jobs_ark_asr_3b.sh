#!/bin/bash
# Submit HF Jobs for AutoArk-AI/ARK-ASR-3B Open ASR Leaderboard evaluation.
# Usage:
#   DRY_RUN=1 RESULTS_BUCKET="<your-bucket>" HF_TOKEN=hf_... bash ark_asr/submit_jobs_ark_asr_3b.sh
#   RESULTS_BUCKET="<your-bucket>" HF_TOKEN=hf_... bash ark_asr/submit_jobs_ark_asr_3b.sh

set -euo pipefail

SPACE="${SPACE:-AutoArk-AI/open-asr-leaderboard-ark-asr-3b}"
RESULTS_BUCKET="${RESULTS_BUCKET:-}"
DATASET_PATH="${DATASET_PATH:-hf-audio/open-asr-leaderboard}"
MODEL_REVISION="${MODEL_REVISION:-}"
FLAVOR="${FLAVOR:-h200}"
ORG_NAME="${ORG_NAME:-}"
TIMEOUT="${TIMEOUT:-8h}"
DRY_RUN="${DRY_RUN:-0}"

MODEL_CONFIGS=(
    "AutoArk-AI/ARK-ASR-3B"
)

DATASET_CONFIGS=(
    "voxpopuli test 64"
    "ami test 64"
    "earnings22 test 64"
    "gigaspeech test 64"
    "librispeech test.clean 64"
    "librispeech test.other 64"
    "spgispeech test 64"
)

if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN is required." >&2
    exit 1
fi

if [ -z "${RESULTS_BUCKET}" ]; then
    echo "RESULTS_BUCKET is required, for example: RESULTS_BUCKET=\"your-org/asr-3b-results\"." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SUBMIT_BACKEND="python"
if command -v hf >/dev/null 2>&1; then
    SUBMIT_BACKEND="hf"
fi

submit_one_job() {
    local model_id="$1"
    local model_folder="$2"
    local dataset="$3"
    local split="$4"
    local batch_size="$5"
    local revision_arg=""
    [ -n "${MODEL_REVISION}" ] && revision_arg="--revision=${MODEL_REVISION}"

    if [ "$DRY_RUN" = "1" ]; then
        echo "DRY_RUN: would submit model=${model_id} revision=${MODEL_REVISION:-main} dataset=${dataset} split=${split} batch_size=${batch_size} space=${SPACE} flavor=${FLAVOR}"
        return
    fi

    if [ "$SUBMIT_BACKEND" = "hf" ]; then
        local namespace_arg=()
        [ -n "${ORG_NAME}" ] && namespace_arg=(--namespace "${ORG_NAME}")

        hf jobs run \
            --flavor "${FLAVOR}" \
            --timeout "${TIMEOUT}" \
            --env HF_TOKEN="${HF_TOKEN}" \
            --env HF_AUDIO_DECODER_BACKEND="soundfile" \
            "${namespace_arg[@]}" \
            --volume "hf://buckets/${RESULTS_BUCKET}:/results" \
            "hf.co/spaces/${SPACE}" \
            bash -c "
                PYTHONPATH=/app python /app/run_eval.py \
                    --model_id=${model_id} \
                    ${revision_arg} \
                    --dataset_path=${DATASET_PATH} \
                    --dataset=${dataset} \
                    --split=${split} \
                    --device=0 \
                    --batch_size=${batch_size} \
                    --max_eval_samples=-1 \
                    --max_new_tokens=256 \
                    --warmup_steps=0 \
                    --dtype=float16 \
                    --attn_impl=sdpa \
                    --audio_input=array \
                    --audio_decode=soundfile \
                    --force_clean_exit &&
                mkdir -p /results/${model_folder} &&
                cp results/*.jsonl /results/${model_folder}/
            " > /dev/null 2>&1
        return
    fi

    HF_TOKEN="${HF_TOKEN}" \
    SPACE="${SPACE}" \
    RESULTS_BUCKET="${RESULTS_BUCKET}" \
    DATASET_PATH="${DATASET_PATH}" \
    MODEL_REVISION="${MODEL_REVISION}" \
    FLAVOR="${FLAVOR}" \
    TIMEOUT="${TIMEOUT}" \
    ORG_NAME="${ORG_NAME}" \
    MODEL_ID="${model_id}" \
    MODEL_FOLDER="${model_folder}" \
    DATASET="${dataset}" \
    SPLIT="${split}" \
    BATCH_SIZE="${batch_size}" \
    python - <<'PY'
import os
import sys
import time

from huggingface_hub import HfApi, Volume

command = f"""
PYTHONPATH=/app python /app/run_eval.py \
    --model_id={os.environ['MODEL_ID']} \
    {f"--revision={os.environ['MODEL_REVISION']} " if os.environ['MODEL_REVISION'] else ""}\
    --dataset_path={os.environ['DATASET_PATH']} \
    --dataset={os.environ['DATASET']} \
    --split={os.environ['SPLIT']} \
    --device=0 \
    --batch_size={os.environ['BATCH_SIZE']} \
    --max_eval_samples=-1 \
    --max_new_tokens=256 \
    --warmup_steps=0 \
    --dtype=float16 \
    --attn_impl=sdpa \
    --audio_input=array \
    --audio_decode=soundfile \
    --force_clean_exit &&
mkdir -p /results/{os.environ['MODEL_FOLDER']} &&
cp results/*.jsonl /results/{os.environ['MODEL_FOLDER']}/
"""

api = HfApi(token=os.environ["HF_TOKEN"])
job = api.run_job(
    image=f"hf.co/spaces/{os.environ['SPACE']}",
    command=["bash", "-c", command],
    env={
        "HF_TOKEN": os.environ["HF_TOKEN"],
        "HF_AUDIO_DECODER_BACKEND": "soundfile",
    },
    flavor=os.environ["FLAVOR"],
    timeout=os.environ["TIMEOUT"],
    volumes=[
        Volume(
            type="bucket",
            source=os.environ["RESULTS_BUCKET"],
            mount_path="/results",
        )
    ],
    namespace=os.environ["ORG_NAME"] or None,
    token=os.environ["HF_TOKEN"],
)

terminal_stages = {"COMPLETED", "ERROR", "CANCELED", "DELETED"}
while True:
    info = api.inspect_job(
        job_id=job.id,
        namespace=os.environ["ORG_NAME"] or None,
        token=os.environ["HF_TOKEN"],
    )
    stage = str(info.status.stage)
    if stage in terminal_stages:
        if stage != "COMPLETED":
            message = info.status.message or ""
            print(f"HF Job failed: id={job.id} stage={stage} message={message}", file=sys.stderr)
            raise SystemExit(1)
        break
    time.sleep(30)
PY
}

sync_results() {
    local model_folder="$1"
    local dest="${SCRIPT_DIR}/results/${model_folder}"

    if [ "$DRY_RUN" = "1" ]; then
        echo "DRY_RUN: would sync hf://buckets/${RESULTS_BUCKET}/${model_folder} -> ${dest}"
        return
    fi

    mkdir -p "${dest}"
    if [ "$SUBMIT_BACKEND" = "hf" ]; then
        hf buckets sync \
            "hf://buckets/${RESULTS_BUCKET}/${model_folder}" \
            "${dest}" > /dev/null 2>&1
        return
    fi

    HF_TOKEN="${HF_TOKEN}" \
    RESULTS_BUCKET="${RESULTS_BUCKET}" \
    MODEL_FOLDER="${model_folder}" \
    DEST="${dest}" \
    python - <<'PY' > /dev/null 2>&1
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
api.sync_bucket(
    f"hf://buckets/{os.environ['RESULTS_BUCKET']}/{os.environ['MODEL_FOLDER']}",
    os.environ["DEST"],
    token=os.environ["HF_TOKEN"],
)
PY
}

for MODEL_ID in "${MODEL_CONFIGS[@]}"; do
    MODEL_FOLDER="${MODEL_ID//\//-}"

    echo "Evaluating: ${MODEL_ID}"
    echo "Submit backend: ${SUBMIT_BACKEND}"
    echo "DRY_RUN=${DRY_RUN}"

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET SPLIT BATCH_SIZE <<< "${cfg}"

        echo "Submitting job: model=${MODEL_ID} dataset=${DATASET} split=${SPLIT} batch_size=${BATCH_SIZE}"
        submit_one_job "${MODEL_ID}" "${MODEL_FOLDER}" "${DATASET}" "${SPLIT}" "${BATCH_SIZE}" &
    done

    if [ -n "${ORG_NAME}" ]; then
        echo "For live status see: https://huggingface.co/organizations/${ORG_NAME}/settings/jobs"
    else
        echo "For live status see: https://huggingface.co/settings/jobs"
    fi

    wait
    if [ "$DRY_RUN" = "1" ]; then
        echo "DRY_RUN=1, skipping result sync and scoring for ${MODEL_ID}."
        continue
    fi
    echo "All jobs finished for ${MODEL_ID}."
    sleep 10

    sync_results "${MODEL_FOLDER}"

    EXPECTED=${#DATASET_CONFIGS[@]}
    ACTUAL=$(find "${SCRIPT_DIR}/results/${MODEL_FOLDER}" -name "*.jsonl" | wc -l)
    if [[ "${ACTUAL}" -lt "${EXPECTED}" ]]; then
        echo "WARNING: expected ${EXPECTED} result files but only found ${ACTUAL}. Some jobs may not have finished yet."
    else
        echo "All ${ACTUAL} result files present."
    fi

    PYTHONPATH="${REPO_ROOT}" python -c "
from normalizer.eval_utils import score_results
score_results('${SCRIPT_DIR}/results/${MODEL_FOLDER}', '${MODEL_ID}')
"
done
