#!/bin/bash
# Launch the vLLM OpenAI-compatible server for Nemotron-3-Nano-Omni.
# Flags follow the model card on
# https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
#
# Run this in one terminal, then run ./run_nemotron_omni_vllm.sh in another.

MODEL_ID="${MODEL_ID:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-384}"

EXTRA_ARGS=()
# FP8 KV cache only for the FP8 checkpoint per the model card.
case "${MODEL_ID}" in
    *FP8*) EXTRA_ARGS+=(--kv-cache-dtype fp8) ;;
esac
# RTX Pro hardware needs the Triton MoE backend per the model card. Override
# with MOE_BACKEND="" to disable, or to a different backend name to swap.
MOE_BACKEND="${MOE_BACKEND-triton}"
if [[ -n "${MOE_BACKEND}" ]]; then
    EXTRA_ARGS+=(--moe-backend "${MOE_BACKEND}")
fi

vllm serve "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --tensor-parallel-size "${TP}" \
    --trust-remote-code \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --video-pruning-rate 0.5 \
    --allowed-local-media-path / \
    --media-io-kwargs '{"video": {"fps": 2, "num_frames": 256}}' \
    --reasoning-parser nemotron_v3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    "${EXTRA_ARGS[@]}"
