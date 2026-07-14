#!/usr/bin/env bash

set -euo pipefail

# Independent Audio8-ASR HF Jobs submission.
# Usage:
#   RESULTS_BUCKET="AutoArk-AI/audio8-asr-open-asr-results" \
#   HF_TOKEN="hf_..." \
#   ORG_NAME="AutoArk-AI" \
#   bash audio8_asr/submit_jobs.sh

SPACE="${SPACE:-AutoArk-AI/open-asr-leaderboard-audio8-asr}"
RESULTS_BUCKET="${RESULTS_BUCKET:?Set RESULTS_BUCKET to a writable Hugging Face bucket}"
DATASET_PATH="${DATASET_PATH:-hf-audio/open-asr-leaderboard}"
FLAVOR="${FLAVOR:-h200}"
ORG_NAME="${ORG_NAME:-}"
MODEL_ID="${MODEL_ID:-AutoArk-AI/Audio8-ASR-0.1B}"
MODEL_REVISION="${MODEL_REVISION:-b812eff124893ecd76a1dcde74ee58db5adab59c}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_AUDIO_SECONDS="${MAX_AUDIO_SECONDS:-30}"
WARMUP_STEPS="${WARMUP_STEPS:-1}"
FEATURE_WORKERS="${FEATURE_WORKERS:-16}"
TORCH_COMPILE="${TORCH_COMPILE:-}"
JOB_TIMEOUT="${JOB_TIMEOUT:-2h}"
CUDA_ALLOC_CONF="${CUDA_ALLOC_CONF:-expandable_segments:True}"

DATASET_CONFIGS=(
  "ami_cleaned test ${AMI_BATCH_SIZE:-1152}"
  "earnings22 test ${EARNINGS22_BATCH_SIZE:-1024}"
  "gigaspeech_cleaned test ${GIGASPEECH_BATCH_SIZE:-1408}"
  "librispeech test.clean ${LIBRISPEECH_CLEAN_BATCH_SIZE:-1024}"
  "librispeech test.other ${LIBRISPEECH_OTHER_BATCH_SIZE:-1024}"
  "spgispeech test ${SPGISPEECH_BATCH_SIZE:-2048}"
  "voxpopuli_cleaned_aa test ${VOXPOPULI_BATCH_SIZE:-628}"
)

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required" >&2
  exit 1
fi

MODEL_FOLDER="${MODEL_ID//\//-}"
NAMESPACE_ARGS=()
if [[ -n "$ORG_NAME" ]]; then
  NAMESPACE_ARGS=(--namespace "$ORG_NAME")
fi

COMPILE_ARGS=""
if [[ -n "$TORCH_COMPILE" ]]; then
  COMPILE_ARGS="--torch_compile=${TORCH_COMPILE}"
fi

echo "Space: $SPACE"
echo "Model: $MODEL_ID@$MODEL_REVISION"
echo "Results bucket: $RESULTS_BUCKET"
echo "Hardware: $FLAVOR"
echo "Job timeout: $JOB_TIMEOUT"
echo "CUDA allocator: $CUDA_ALLOC_CONF"

pids=()
for config in "${DATASET_CONFIGS[@]}"; do
  read -r dataset split batch_size <<< "$config"
  echo "Submitting dataset=${dataset} split=${split} batch_size=${batch_size}"
  (
    hf jobs run \
      --flavor "$FLAVOR" \
      --timeout "$JOB_TIMEOUT" \
      --secrets HF_TOKEN \
      --env HF_AUDIO_DECODER_BACKEND="soundfile" \
      --env PYTORCH_CUDA_ALLOC_CONF="$CUDA_ALLOC_CONF" \
      "${NAMESPACE_ARGS[@]}" \
      --volume "hf://buckets/${RESULTS_BUCKET}:/results_bucket" \
      "hf.co/spaces/${SPACE}" \
      bash -c "
        set -euo pipefail
        cd /app
        rm -rf /app/results
        PYTHONPATH=/app python /app/run_eval.py \\
          --model_id=${MODEL_ID} \\
          --model_revision=${MODEL_REVISION} \\
          --dataset_path=${DATASET_PATH} \\
          --dataset=${dataset} \\
          --split=${split} \\
          --device=0 \\
          --dtype=bfloat16 \\
          --attn_implementation=eager \\
          --batch_size=${batch_size} \\
          --max_eval_samples=-1 \\
          --max_new_tokens=${MAX_NEW_TOKENS} \\
          --max_audio_seconds=${MAX_AUDIO_SECONDS} \\
          --warmup_steps=${WARMUP_STEPS} \\
          --feature_workers=${FEATURE_WORKERS} \\
          --torch_cpu_threads=1 \\
          ${COMPILE_ARGS}
        mkdir -p /results_bucket/${MODEL_FOLDER}
        cp /app/results/*.jsonl /results_bucket/${MODEL_FOLDER}/
        cp /app/results/*.summary.json /results_bucket/${MODEL_FOLDER}/
      "
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "One or more HF Jobs failed" >&2
  exit 1
fi

if [[ -n "$ORG_NAME" ]]; then
  echo "Jobs page: https://huggingface.co/organizations/${ORG_NAME}/settings/jobs"
else
  echo "Jobs page: https://huggingface.co/settings/jobs"
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
local_results="$repo_root/results/$MODEL_FOLDER"
mkdir -p "$local_results"
hf buckets sync \
  "hf://buckets/${RESULTS_BUCKET}/${MODEL_FOLDER}" \
  "$local_results"

actual="$(find "$local_results" -maxdepth 1 -name '*.jsonl' | wc -l)"
expected="${#DATASET_CONFIGS[@]}"
if [[ "$actual" -ne "$expected" ]]; then
  echo "Expected $expected JSONL manifests, found $actual in $local_results" >&2
  exit 1
fi

PYTHONPATH="$repo_root" python - <<PY
from normalizer.eval_utils import score_results

score_results("$local_results", "$MODEL_ID", families=["public"])
PY

echo "All Audio8-ASR public HF Jobs completed and scored."
