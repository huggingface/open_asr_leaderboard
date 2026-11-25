#!/usr/bin/env bash
# Sign in with `hf auth login` using an account with model and Space access.
# Set RESULTS_BUCKET to an owner/bucket you can write to, and optionally ORG_NAME.
# Usage: bash kroko/submit_jobs.sh [--dry-run]
#        bash kroko/submit_jobs.sh --voice-code-bench   (VoiceCodeBench only, graded with the shared helpers)
set -euo pipefail

VCB=0
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --voice-code-bench) VCB=1 ;;
        *) echo 'Usage: bash kroko/submit_jobs.sh [--dry-run] [--voice-code-bench]' >&2; exit 2 ;;
    esac
done

SPACE="hf-audio/Kroko-ASR-XL-EN-Offline-Eval"
MODEL_ID="Banafo/Kroko-ASR-XL-EN-Offline"
MODEL_REVISION="8c1fb9b89831c6cd9f63d0295f2ccb2be6da2964"
if (( VCB )); then
    RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_h200}"
else
    RESULTS_BUCKET="${RESULTS_BUCKET:-Banafo/kroko-asr-xl-en-offline-eval-results}"
fi
RUN_ID="kroko-openasr-$(date -u +%Y%m%dT%H%M%SZ)-$$-${RANDOM}"

options=(--flavor h200 --timeout 2h --name Kroko-ASR-XL-EN-Offline-OpenASR
    --env PYTHONUNBUFFERED=1 --env HF_HUB_DISABLE_TELEMETRY=1
    --secrets HF_TOKEN --volume "hf://buckets/${RESULTS_BUCKET}:/results")
if [[ -n "${ORG_NAME:-}" ]]; then
    options+=(--namespace "$ORG_NAME")
fi
if (( DRY_RUN )); then
    options+=(--dry-run)
fi

if (( VCB )); then
    # VoiceCodeBench: run_eval.py in the Space writes the transcripts (clips longer than 30 s are chunked);
    # the shared helpers then grade CTEM in a GPU job. Output goes to the model's folder of the results bucket.
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/submit_utils.sh"
    MODEL_FOLDER="${MODEL_ID//\//-}"
    DATASET_CONFIGS=("voice_code_bench test besimple-ai/voice-code-bench")
    vcb_init
    hf jobs run "${options[@]}" -- "hf.co/spaces/${SPACE}" bash -euo pipefail -c "
mkdir -p /tmp/work && cd /tmp/work
python /app/run_eval.py --model-id ${MODEL_ID} --model-revision ${MODEL_REVISION} \\
    --report-name ${MODEL_ID} --model-family zipformer \\
    --decoder-type transducer_modified_beam_search --beam 6 --blank-penalty 0.0 \\
    --batch-size 256 --workers 3 --precision fp16 --device 0 --optimization-level 5 \\
    --min-audio-seconds 0.1 --opt-audio-seconds 10.0 --max-audio-seconds 40.0 --warmup-steps 5 \\
    --engine-cache /app/engines --dataset-path besimple-ai/voice-code-bench --dataset default \\
    --dataset-revision bef2824f83ef1c796f3e79731a3b0741708730df --split test --max-eval-samples -1
mkdir -p /results/${MODEL_FOLDER}
cp results/*.jsonl results/*.metadata.json /results/${MODEL_FOLDER}/
"
    (( DRY_RUN )) && exit 0
    sleep 10  # allow time for the last results to be flushed to the bucket
    vcb_track_model "$MODEL_ID" "$MODEL_FOLDER"
    vcb_grade
    exit 0
fi

# HF Jobs uses the latest build of this Space. The reference build used
# revision d787a9ea470a22ab9a520efc42c4ea534060f204 and loads fast-gpu-asr/.
hf jobs run "${options[@]}" -- "hf.co/spaces/${SPACE}" python /app/evaluate.py \
    --model-id "$MODEL_ID" --model-revision "$MODEL_REVISION" \
    --output-dir "/results/runs/${RUN_ID}" --mode full \
    --batch-size 256 --workers 3 --beam 6 --precision fp16 \
    --min-audio-seconds 0.1 --opt-audio-seconds 10.0 --max-audio-seconds 40.0 \
    --warmup-steps 5
