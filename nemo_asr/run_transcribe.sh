#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_ID="nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
MODEL_REVISION="e6ed81343993185d8c7617c6029c7378b4116a26"
MODEL_ARCHITECTURE="Nemotron3_5TranscribeForConditionalGeneration"
NEMO_SPEECH_REVISION="7b05f096d54db47078fd1569a54d1f3ba4a4c51f"
NEMO_SPEECH_DIR="${NEMO_SPEECH_DIR:-/opt/nemo-speech}"
VLLM_VERSION="0.32.0"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/tmp/transcribe-model}"
DRAFT_MODEL_ID="nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2"
VLLM_ENV="${VLLM_ENV:-/opt/venv}"
PYTHON_BIN="${PYTHON_BIN:-${VLLM_ENV}/bin/python}"
HF_CLI="${HF_CLI:-hf}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
SERVER_START_TIMEOUT_S="${SERVER_START_TIMEOUT_S:-1800}"
SERVER_HEALTH_INTERVAL_S="${SERVER_HEALTH_INTERVAL_S:-5}"
EVALUATOR_SCRIPT="${EVALUATOR_SCRIPT:-${SCRIPT_DIR}/run_eval_transcribe.py}"
RESULTS_DEST_DIR="${RESULTS_DEST_DIR:-$(pwd)/results}"
WORK_DIR="$(mktemp -d /tmp/transcribe-openasr.XXXXXX)"
SERVER_LOG="${SERVER_LOG:-${WORK_DIR}/vllm-server.log}"
SERVER_PID=""
SPECULATIVE="${SPECULATIVE:-mtp8}"

case "${SPECULATIVE}" in
    off|mtp2|mtp8|dflash2-k7|dflash2-k15) ;;
    *) echo "ERROR: unsupported SPECULATIVE mode: ${SPECULATIVE}" >&2; exit 2 ;;
esac

case "${HOST}" in
    127.0.0.1|localhost) ;;
    *) echo "ERROR: HOST must be loopback-only." >&2; exit 2 ;;
esac

stop_server() {
    [[ -n "${SERVER_PID}" ]] || return
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 30); do
        kill -0 -- "-${SERVER_PID}" 2>/dev/null || break
        sleep 1
    done
    if kill -0 -- "-${SERVER_PID}" 2>/dev/null; then
        kill -KILL -- "-${SERVER_PID}" 2>/dev/null || kill -KILL "${SERVER_PID}" 2>/dev/null || true
    fi
    wait "${SERVER_PID}" 2>/dev/null || true
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    stop_server
    if [[ "${KEEP_WORK_DIR:-0}" != "1" ]]; then
        rm -rf -- "${WORK_DIR}"
    else
        echo "Preserved work directory: ${WORK_DIR}"
    fi
    exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]]; then
    : "${HF_TOKEN:?HF_TOKEN must be set}"
    "${HF_CLI}" download "${MODEL_ID}" \
        --revision "${MODEL_REVISION}" \
        --local-dir "${MODEL_CACHE_DIR}"
    for required in config.json; do
        if [[ ! -f "${MODEL_CACHE_DIR}/${required}" ]]; then
            echo "ERROR: checkpoint download is missing ${required}." >&2
            exit 1
        fi
    done
    "${PYTHON_BIN}" -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as config_file:
    architectures = json.load(config_file).get("architectures", [])
if sys.argv[2] not in architectures:
    raise SystemExit(
        f"checkpoint architecture mismatch: expected {sys.argv[2]!r}, found {architectures!r}"
    )
' "${MODEL_CACHE_DIR}/config.json" "${MODEL_ARCHITECTURE}"
fi

if [[ -z "${MODEL_TOTAL_PARAMETERS:-}" ]]; then
    MODEL_TOTAL_PARAMETERS="$("${PYTHON_BIN}" -c '
import math
import pathlib
import sys
from safetensors import safe_open

total = 0
for path in pathlib.Path(sys.argv[1]).rglob("*.safetensors"):
    with safe_open(path, framework="pt", device="cpu") as handle:
        total += sum(math.prod(handle.get_slice(key).get_shape()) for key in handle.keys())
print(total)
' "${MODEL_CACHE_DIR}")"
fi
if [[ ! "${MODEL_TOTAL_PARAMETERS}" =~ ^[0-9]+$ || "${MODEL_TOTAL_PARAMETERS}" == "0" ]]; then
    echo "ERROR: could not count checkpoint parameters." >&2
    exit 1
fi

if [[ -z "${SERVER_LAUNCHER:-}" ]]; then
    if [[ ! -x "${VLLM_ENV}/bin/vllm" ]]; then
        echo "ERROR: ${VLLM_ENV}/bin/vllm is missing." >&2
        exit 1
    fi
    if [[ ! -d "${NEMO_SPEECH_DIR}/.git" ]]; then
        echo "ERROR: ${NEMO_SPEECH_DIR} is not a NeMo Speech checkout." >&2
        exit 1
    fi
    installed_nemo_revision="$(git -C "${NEMO_SPEECH_DIR}" rev-parse HEAD)"
    if [[ "${installed_nemo_revision}" != "${NEMO_SPEECH_REVISION}" ]]; then
        echo "ERROR: NeMo Speech revision mismatch: expected ${NEMO_SPEECH_REVISION}, found ${installed_nemo_revision}." >&2
        exit 1
    fi
    "${PYTHON_BIN}" -c '
import importlib.metadata
import sys
from pathlib import Path

from nemo.collections.speechlm2.vllm.salm import config as salm_config

installed_vllm = importlib.metadata.version("vllm")
if installed_vllm != sys.argv[1]:
    raise SystemExit(
        f"vLLM version mismatch: expected {sys.argv[1]}, "
        f"found {installed_vllm}"
    )
module_path = Path(salm_config.__file__).resolve()
if not module_path.is_relative_to(Path(sys.argv[3]).resolve()):
    raise SystemExit(
        f"NeMo Speech import mismatch: loaded {module_path}, "
        f"expected a module under {sys.argv[3]}"
    )
if salm_config.TRANSCRIBE_ARCHITECTURE != sys.argv[2]:
    raise SystemExit(
        f"NeMo Speech architecture mismatch: expected {sys.argv[2]!r}, "
        f"found {salm_config.TRANSCRIBE_ARCHITECTURE!r}"
    )
if salm_config.TRANSCRIBE_MTP_ARCHITECTURE != "Nemotron3_5TranscribeMTPModel":
    raise SystemExit(
        "NeMo Speech MTP architecture mismatch: "
        f"{salm_config.TRANSCRIBE_MTP_ARCHITECTURE!r}"
    )
' "${VLLM_VERSION}" "${MODEL_ARCHITECTURE}" "${NEMO_SPEECH_DIR}"
    SERVER_COMMAND=(
        "${VLLM_ENV}/bin/vllm" serve "${MODEL_CACHE_DIR}"
        --served-model-name "${MODEL_ID}"
        --host "${HOST}"
        --port "${PORT}"
        --trust-remote-code
        --dtype bfloat16
        --gpu-memory-utilization 0.94
        --max-model-len 65536
        --max-num-batched-tokens 4096
        --max-num-seqs 192
        --enable-chunked-prefill
        --no-enable-prefix-caching
        --use-replayssm
        --mamba-backend flashinfer
        --mamba-cache-mode none
        --mm-processor-cache-gb 0
        --no-async-scheduling
        --limit-mm-per-prompt '{"audio":{"count":6,"length":120000000}}'
        --reasoning-parser nemotron_v3
        --trust-request-mm-kwargs
    )
    case "${SPECULATIVE}" in
        off)
            export VLLM_USE_V2_MODEL_RUNNER=0
            SERVER_COMMAND+=(
                --middleware nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware
            )
            ;;
        mtp2|mtp8)
            export VLLM_USE_V2_MODEL_RUNNER=0
            speculative_tokens="${SPECULATIVE#mtp}"
            SERVER_COMMAND+=(
                --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${speculative_tokens}}"
                --middleware nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware
            )
            ;;
        dflash2-k7|dflash2-k15)
            unset VLLM_USE_V2_MODEL_RUNNER
            speculative_tokens="${SPECULATIVE#dflash2-k}"
            "${PYTHON_BIN}" - <<'PY'
from pathlib import Path

import vllm

path = Path(vllm.__file__).parent / "model_executor/models/qwen3_dflash2.py"
source = path.read_text()
if "safe_anchor_token_ids" not in source:
    old = "    successors = successor_table[candidate_ids]\n"
    anchor = "            anchor_token_ids[:, None, None].expand(-1, 1, top_k),"
    assert source.count(old) == 1 and source.count(anchor) == 1
    source = source.replace(
        old,
        "    safe_anchor_token_ids = anchor_token_ids.clamp(\n"
        "        min=0, max=predecessor_table.shape[0] - 1\n"
        "    )\n" + old,
    ).replace(
        anchor,
        "            safe_anchor_token_ids[:, None, None].expand(-1, 1, top_k),",
    )
    path.write_text(source)
PY
            SERVER_COMMAND+=(
                --speculative-config "{\"method\":\"dflash\",\"num_speculative_tokens\":${speculative_tokens},\"model\":\"${DRAFT_MODEL_ID}\"}"
                --hf-overrides '{"ctc_timestamps":{"enabled":false}}'
            )
            ;;
    esac
else
    SERVER_COMMAND=(bash "${SERVER_LAUNCHER}" "${MODEL_CACHE_DIR}")
fi

export HOST PORT SPECULATIVE MODEL_TOTAL_PARAMETERS NEMO_SPEECH_REVISION VLLM_VERSION
export VLLM_PLUGINS=nemo_speechlm
export VLLM_MAX_AUDIO_DECODE_DURATION_S=3600
export VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=180
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1
export HF_AUDIO_DECODER_BACKEND=soundfile

health_url="http://${HOST}:${PORT}/health"
models_url="http://${HOST}:${PORT}/v1/models"
if curl --silent --fail --max-time 2 "${health_url}" >/dev/null 2>&1; then
    echo "ERROR: an existing service is already listening at ${health_url}." >&2
    exit 1
fi

echo "Starting the ${SPECULATIVE} vLLM server; log: ${SERVER_LOG}"
setsid "${SERVER_COMMAND[@]}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + SERVER_START_TIMEOUT_S))
while ! curl --silent --show-error --fail "${health_url}" >/dev/null 2>&1; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: vLLM exited before becoming healthy." >&2
        tail -n 200 "${SERVER_LOG}" >&2 || true
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo "ERROR: vLLM did not become healthy within ${SERVER_START_TIMEOUT_S}s." >&2
        tail -n 200 "${SERVER_LOG}" >&2 || true
        exit 1
    fi
    sleep "${SERVER_HEALTH_INTERVAL_S}"
done

if ! SERVED_MODEL_NAME="$(
    curl --silent --show-error --fail "${models_url}" | \
        "${PYTHON_BIN}" -c '
import json
import sys

payload = json.load(sys.stdin)
ids = [
    item.get("id")
    for item in payload.get("data", [])
    if isinstance(item.get("id"), str) and item["id"]
]
if len(ids) != 1:
    raise SystemExit(f"expected exactly one served model, found {ids!r}")
print(ids[0])
'
)"; then
    echo "ERROR: could not resolve the model advertised by vLLM." >&2
    tail -n 200 "${SERVER_LOG}" >&2 || true
    exit 1
fi

cd "${WORK_DIR}"
set +e
PYTHONPATH="${SCRIPT_DIR}:${REPO_ROOT}:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" "${EVALUATOR_SCRIPT}" \
        --model_id "${MODEL_ID}" \
        --model_revision "${MODEL_REVISION}" \
        --served_model_name "${SERVED_MODEL_NAME}" \
        --server_url "http://${HOST}:${PORT}/v1" \
        "$@"
eval_status=$?
set -e

if [[ ${eval_status} -ne 0 ]]; then
    echo "ERROR: evaluator exited with status ${eval_status}." >&2
    tail -n 200 "${SERVER_LOG}" >&2 || true
    exit "${eval_status}"
fi

mapfile -t manifests < <(find "${WORK_DIR}/results" -maxdepth 1 -type f -name '*.jsonl' -print)
if [[ ${#manifests[@]} -ne 1 ]]; then
    echo "ERROR: expected one result manifest, found ${#manifests[@]}." >&2
    exit 1
fi
mkdir -p "${RESULTS_DEST_DIR}"
cp -- "${manifests[0]}" "${RESULTS_DEST_DIR}/"
