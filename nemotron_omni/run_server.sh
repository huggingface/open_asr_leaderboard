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
SEED="${SEED:-0}"

EXTRA_ARGS=()
# FP8 KV cache for the FP8 and NVFP4 checkpoints per the model card; BF16 omits.
case "${MODEL_ID}" in
    *FP8*|*NVFP4*) EXTRA_ARGS+=(--kv-cache-dtype fp8) ;;
esac
# Default MoE backend per the model card:
#   - BF16/FP8 on RTX Pro: triton
#   - NVFP4: flashinfer_cutlass (triton is not supported for NvFP4 MoE)
# Override with MOE_BACKEND="" to disable, or to a different backend name to swap.
case "${MODEL_ID}" in
    *NVFP4*) MOE_BACKEND="${MOE_BACKEND-flashinfer_cutlass}" ;;
    *)       MOE_BACKEND="${MOE_BACKEND-triton}" ;;
esac
if [[ -n "${MOE_BACKEND}" ]]; then
    EXTRA_ARGS+=(--moe-backend "${MOE_BACKEND}")
fi
# Some environments hit a torch.compile bug in vllm 0.20.0
# (`AlwaysHitShapeEnv has no attribute 'var_to_hint_override'`). Setting
# EAGER=1 skips torch.compile and avoids the broken codepath at the cost of
# ~10-25% decode throughput.
if [[ "${EAGER:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--enforce-eager)
fi

vllm serve "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --tensor-parallel-size "${TP}" \
    --trust-remote-code \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --seed "${SEED}" \
    --video-pruning-rate 0.5 \
    --allowed-local-media-path / \
    --media-io-kwargs '{"video": {"fps": 2, "num_frames": 256}}' \
    --reasoning-parser nemotron_v3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    "${EXTRA_ARGS[@]}"
