#!/usr/bin/env bash
# VoiceCodeBench for Banafo/Kroko-ASR-XL-EN-Offline: run in Banafo's evaluation Space image, with the local
# run_eval_vcb.py (its run_eval.py plus VoiceCodeBench loading and 30 s chunking) and normalizer/voice_code_bench.py
# injected, then grade CTEM in a GPU job with the shared helpers.
# Sign in with `hf auth login` using an account with access to the gated model.
# Usage: bash kroko/submit_jobs_vcb.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/submit_utils.sh"
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SPACE="Banafo/Kroko-ASR-XL-EN-Offline-Eval"
MODEL_ID="Banafo/Kroko-ASR-XL-EN-Offline"
MODEL_REVISION="8c1fb9b89831c6cd9f63d0295f2ccb2be6da2964"
MODEL_FOLDER="${MODEL_ID//\//-}"
RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_h200}"
DATASET_CONFIGS=("voice_code_bench test besimple-ai/voice-code-bench")
vcb_init

# gzip keeps the command argument under the 128 KiB limit of one argument.
INJECT="echo '$(gzip -c "${SCRIPT_DIR}/run_eval_vcb.py" | base64 -w0)' | base64 -d | gunzip > /app/run_eval.py
echo '$(gzip -c "${SUBMIT_UTILS_REPO_ROOT}/normalizer/voice_code_bench.py" | base64 -w0)' | base64 -d | gunzip > /app/normalizer/voice_code_bench.py"

NAMESPACE_ARGS=()
[[ -n "${ORG_NAME:-}" ]] && NAMESPACE_ARGS=(--namespace "$ORG_NAME")

hf jobs run --flavor h200 --timeout 2h --env PYTHONUNBUFFERED=1 --secrets HF_TOKEN \
    "${NAMESPACE_ARGS[@]}" --volume "hf://buckets/${RESULTS_BUCKET}:/results" \
    -- "hf.co/spaces/${SPACE}" bash -euo pipefail -c "
${INJECT}
mkdir -p /tmp/work && cd /tmp/work
python /app/run_eval.py --model-id ${MODEL_ID} --model-revision ${MODEL_REVISION} \
    --report-name ${MODEL_ID} --model-family zipformer \
    --decoder-type transducer_modified_beam_search --beam 6 --blank-penalty 0.0 \
    --batch-size 256 --workers 3 --precision fp16 --device 0 --optimization-level 5 \
    --min-audio-seconds 0.1 --opt-audio-seconds 10.0 --max-audio-seconds 40.0 --warmup-steps 5 \
    --engine-cache /app/engines --dataset-path besimple-ai/voice-code-bench --dataset default \
    --dataset-revision bef2824f83ef1c796f3e79731a3b0741708730df --split test --max-eval-samples -1
mkdir -p /results/${MODEL_FOLDER}
cp results/*.jsonl results/*.metadata.json /results/${MODEL_FOLDER}/
" > "${SCRIPT_DIR}/vcb_job.log" 2>&1 || echo "WARNING: Kroko job exited non-zero; see ${SCRIPT_DIR}/vcb_job.log" >&2

sleep 10  # allow time for the last results to be flushed to the bucket
vcb_track_model "$MODEL_ID" "$MODEL_FOLDER"
vcb_grade
