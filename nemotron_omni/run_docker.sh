#!/bin/bash
# Docker entrypoint: start the vLLM server, wait for /health, run the full
# leaderboard sweep against it, then shut the server down on exit.
#
# Usage (from the repo root, after `docker build -t open-asr-nemotron-omni
# -f nemotron_omni/Dockerfile .`):
#
#     docker run --gpus all \
#         -v $(pwd):/app \
#         -v $HF_HOME:/root/.cache/huggingface \
#         open-asr-nemotron-omni run_docker.sh
#
# Pass EAGER=1 (`-e EAGER=1`) on hardware where vLLM 0.20.0's torch.compile
# path crashes.

set -euo pipefail

cd "$(dirname "$0")"

LOG="${VLLM_LOG:-/tmp/vllm_server.log}"
HEALTH_URL="${HEALTH_URL:-http://localhost:8000/health}"
WAIT_SECONDS="${WAIT_SECONDS:-1800}"

echo "Starting vLLM server (logs at ${LOG})..."
./run_server.sh > "${LOG}" 2>&1 &
SERVER_PID=$!
trap 'echo "Stopping vLLM server (pid=${SERVER_PID})..."; kill -TERM "${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true' EXIT

echo "Waiting for vLLM server (pid=${SERVER_PID}) at ${HEALTH_URL}..."
deadline=$(( $(date +%s) + WAIT_SECONDS ))
until curl -sf "${HEALTH_URL}" > /dev/null; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: vLLM server exited before becoming healthy. Last 100 log lines:"
        tail -n 100 "${LOG}" || true
        exit 1
    fi
    if [ "$(date +%s)" -gt "${deadline}" ]; then
        echo "ERROR: vLLM server did not become healthy within ${WAIT_SECONDS}s."
        tail -n 100 "${LOG}" || true
        exit 1
    fi
    sleep 5
done

echo "vLLM server ready. Launching sweep."
./run_nemotron_omni_vllm.sh "$@"
