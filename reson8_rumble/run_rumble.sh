#!/usr/bin/env bash
# One H200 run: overlap downloads with startup, then reuse the engines.
# Usage: bash reson8_rumble/run_rumble.sh "$DATASET_SPEC" NUM_ENGINES
set -euo pipefail
startup_started=$SECONDS

dataset_spec=${1:?Pass the selected datasets}
num_engines=${2:?Pass the engine count}
if [[ -z "${dataset_spec//[[:space:]]/}" || ! "$num_engines" =~ ^[1-9][0-9]*$ ]]; then
    echo 'Select datasets and a positive integer engine count' >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."
mkdir -p ./results
log_dir="$(pwd)/results"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_THREAD_LIMIT=1
export PYTHONUNBUFFERED=1

uv pip install --system \
    -r "$script_dir/../requirements/requirements_jobs.txt" \
    -r "$script_dir/../requirements/requirements_reson8_rumble.txt"

check_download() {
    if [[ -n "$download_pid" ]] && ! kill -0 "$download_pid" 2>/dev/null; then
        wait "$download_pid"
        download_pid=''
    fi
}

wait_for_server() {
    local pid=$1 port=$2 log=$3 deadline=$((SECONDS + 900))
    while true; do
        check_download
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "vLLM exited during startup; see $log" >&2
            exit 1
        fi
        if curl --fail --silent --max-time 2 --noproxy '*' \
            "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
            return
        fi
        if (( SECONDS >= deadline )); then
            echo "vLLM readiness timed out; see $log" >&2
            exit 1
        fi
        sleep 1
    done
}

# Both passes use the same HF cache. Download failures abort the Job.
python3 -m reson8_rumble.run_eval --datasets "$dataset_spec" \
    --download >"$log_dir/download.log" 2>&1 &
download_pid=$!

# Profiling is sequential; downloads continue while each engine starts/compiles.
for ((engine = 0; engine < num_engines; engine++)); do
    check_download
    port=$((8000 + engine))
    log="$log_dir/server-$engine.log"
    bash "$script_dir/serve_vllm.sh" "$port" "$num_engines" >"$log" 2>&1 &
    wait_for_server "$!" "$port" "$log"
done
echo "vLLM engines ready after $((SECONDS - startup_started))s"
if [[ -n "$download_pid" ]]; then
    wait "$download_pid"
fi
echo "Startup complete after $((SECONDS - startup_started))s (engines and datasets ready)"

while IFS= read -r selection; do
    [[ -n "${selection//[[:space:]]/}" ]] || continue
    python3 -m reson8_rumble.run_eval --datasets "$selection" \
        --num_engines "$num_engines"
done <<< "$dataset_spec"
