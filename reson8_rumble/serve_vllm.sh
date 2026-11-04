#!/usr/bin/env bash
set -euo pipefail

port=${1:?Pass the server port}
num_engines=${2:?Pass the engine count}
memory_per_engine=$(awk -v engines="$num_engines" 'BEGIN { print 0.9 / engines }')

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_THREAD_LIMIT=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
MODEL=Reson8/Rumble-2-Flash
REVISION=e2e60d8dc97b644b2fe3be448482b4ffbaae3541
mkdir -p /tmp/rumble-audio
exec python3 -m vllm.entrypoints.cli.main serve "$MODEL" \
    --revision "$REVISION" --tokenizer-revision "$REVISION" --no-enable-prefix-caching \
    --served-model-name "$MODEL" --host 127.0.0.1 --port "$port" \
    --allowed-local-media-path /tmp/rumble-audio \
    --dtype bfloat16 --max-model-len 4096 --limit-mm-per-prompt '{"audio":1}' \
    --max-num-seqs 1024 --max-num-batched-tokens 32768 \
    --gpu-memory-utilization "$memory_per_engine" --generation-config vllm \
    --api-server-count 2 --mm-processor-cache-gb 0 \
    --disable-uvicorn-access-log
