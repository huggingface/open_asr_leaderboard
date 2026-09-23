#!/usr/bin/env bash
set -euo pipefail

# Run one open model, then score its raw transcripts with VoiceCodeBench CTEM.
# Requires the NeMo evaluation environment and OPENAI_API_KEY for the verifier.
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)/..${PYTHONPATH:+:$PYTHONPATH}"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running the CTEM verifier}"

MODEL_ID="nvidia/parakeet-tdt-0.6b-v3"
python run_eval.py \
    --model_id="$MODEL_ID" \
    --dataset_path="besimple-ai/voice-code-bench" \
    --dataset="voice_code_bench" \
    --split="test" \
    --device=0 \
    --batch_size=16 \
    --max_eval_samples=-1

MANIFEST="results/MODEL_nvidia-parakeet-tdt-0.6b-v3_DATASET_besimple-ai-voice-code-bench_voice_code_bench_test.jsonl"
python ../scripts/score_voice_code_bench.py \
    --input "$MANIFEST" \
    --model-id "$MODEL_ID" \
    --output results/parakeet_voice_code_bench_ctem.json
