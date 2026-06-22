#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_ID="${MODEL_ID:-/data/yumu/model/trained_model/ark_asr_3b_opd}"
MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_ID%/}")}"
MANIFEST_MODEL_ID="${MANIFEST_MODEL_ID:-$MODEL_ID}"
RUN_ID="${RUN_ID:-${MODEL_NAME}_$(date -u +%Y%m%d_%H%M%S)}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/data/tmp}"
if [ ! -w "$SCRATCH_ROOT" ]; then
    SCRATCH_ROOT=/tmp
fi

BASE="${BASE:-$SCRATCH_ROOT/open_asr_leaderboard_$RUN_ID}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs/$RUN_ID}"
FINAL_DIR="${FINAL_DIR:-$SCRIPT_DIR/results.$RUN_ID}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET_PATH="${DATASET_PATH:-hf-audio/open-asr-leaderboard}"
HF_HOME="${HF_HOME:-/data/hf_datasets/.cache/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/hf_datasets}"

BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
DTYPE="${DTYPE:-float16}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
AUDIO_INPUT="${AUDIO_INPUT:-array}"
AUDIO_DECODE="${AUDIO_DECODE:-soundfile}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6}"
CHECK_ONLY="${CHECK_ONLY:-0}"

read -r -a GPUS <<< "$GPU_IDS"
if [ "${#GPUS[@]}" -lt 7 ]; then
    echo "Need at least 7 GPU ids in GPU_IDS; got: $GPU_IDS" >&2
    exit 1
fi

if [[ "$MODEL_ID" == /* && ! -e "$MODEL_ID" ]]; then
    echo "Local MODEL_ID path does not exist: $MODEL_ID" >&2
    exit 1
fi

mkdir -p "$BASE/jobs" "$BASE/parquet" "$LOG_DIR"
if [ "$CHECK_ONLY" != "1" ]; then
    mkdir -p "$FINAL_DIR"
fi
: > "$LOG_DIR/pids.txt"
printf '%s\n' "$RUN_ID" > "$SCRIPT_DIR/logs/current_local_parallel_eval_run.txt"
printf '%s\n' "$RUN_ID" > "$SCRIPT_DIR/logs/current_ark_asr_3b_opd_run.txt"

echo "RUN_ID=$RUN_ID"
echo "MODEL_ID=$MODEL_ID"
echo "MANIFEST_MODEL_ID=$MANIFEST_MODEL_ID"
echo "BASE=$BASE"
echo "LOG_DIR=$LOG_DIR"
echo "FINAL_DIR=$FINAL_DIR"
echo "GPU_IDS=${GPUS[*]}"
echo "CHECK_ONLY=$CHECK_ONLY"

make_split_dir() {
    local name="$1"
    local src="$2"
    local pattern="$3"
    local dst="$BASE/parquet/$name"
    local count

    if [ -e "$dst" ] && [ ! -d "$dst" ]; then
        echo "Parquet target exists but is not a directory: $dst" >&2
        exit 1
    fi
    mkdir -p "$dst"
    find "$dst" -maxdepth 1 -type l -name '*.parquet' -delete
    find "$src" -maxdepth 1 -type f -name "$pattern" -print0 | sort -z |
        while IFS= read -r -d '' file; do
            ln -s "$file" "$dst/$(basename "$file")"
        done

    count=$(find "$dst" -maxdepth 1 -type l -name '*.parquet' | wc -l)
    if [ "$count" -eq 0 ]; then
        echo "No parquet files linked for $name from $src/$pattern" >&2
        exit 1
    fi
    echo "Linked $count parquet files for $name -> $dst"
}

make_split_dir ami /data/hf_datasets/edinburghcstr__ami/ihm 'test-*.parquet'
make_split_dir earnings22 /data/hf_datasets/hf-audio__open-asr-leaderboard__earnings22_official/earnings22 '*.parquet'
make_split_dir gigaspeech /data/hf_datasets/gigaspeech_parquet/parquet-data/test '*.parquet'
make_split_dir librispeech_clean /data/hf_datasets/clean/test '*.parquet'
make_split_dir librispeech_other /data/hf_datasets/other/test '*.parquet'
make_split_dir spgispeech /data/hf_datasets/kensho__spgispeech/test '*.parquet'
make_split_dir voxpopuli /data/hf_datasets/facebook__voxpopuli/en 'test-*.parquet'

CHECK_BASE="$BASE/parquet" PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

import pyarrow.parquet as pq
from normalizer import data_utils

base = Path(os.environ["CHECK_BASE"])
expected = {
    "ami": 11653,
    "earnings22": 2737,
    "gigaspeech": 19898,
    "librispeech_clean": 2620,
    "librispeech_other": 2939,
    "spgispeech": 39341,
    "voxpopuli": 1842,
}

for name, expected_kept in expected.items():
    raw = kept = files = 0
    for path in sorted((base / name).glob("*.parquet")):
        files += 1
        pf = pq.ParquetFile(path)
        for row_group in range(pf.num_row_groups):
            rows = pf.read_row_group(row_group).to_pylist()
            raw += len(rows)
            for row in rows:
                norm = data_utils.normalizer(data_utils.get_text(row))
                kept += int(data_utils.is_target_text_in_range(norm))
    if kept != expected_kept:
        raise SystemExit(f"{name}: expected {expected_kept} filtered rows, got {kept} from {files} files")
    print(f"{name}: files={files} raw={raw} kept={kept}")

print("Preflight dataset counts OK")
PY

if [ "$CHECK_ONLY" = "1" ]; then
    echo "CHECK_ONLY=1, stopping before GPU evaluation."
    exit 0
fi

declare -a JOB_PIDS=()
declare -a JOB_NAMES=()

run_task() {
    local name="$1"
    local gpu="$2"
    local dataset="$3"
    local split="$4"
    local parquet="$5"
    local batch_size="${6:-$BATCH_SIZE}"
    local job_dir="$BASE/jobs/$name"
    local log_file="$LOG_DIR/$name.log"

    mkdir -p "$job_dir"
    (
        cd "$job_dir"
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] START name=$name gpu=$gpu dataset=$dataset split=$split parquet=$parquet"
        HF_HOME="$HF_HOME" \
        HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
        HF_DATASETS_OFFLINE=1 \
        HF_HUB_OFFLINE=1 \
        TOKENIZERS_PARALLELISM=false \
        PYTHONPATH="$REPO_ROOT" \
        CUDA_VISIBLE_DEVICES="$gpu" \
        "$PYTHON_BIN" "$SCRIPT_DIR/run_eval.py" \
            --model_id="$MODEL_ID" \
            --manifest_model_id="$MANIFEST_MODEL_ID" \
            --dataset_path="$DATASET_PATH" \
            --dataset="$dataset" \
            --split="$split" \
            --local_parquet_dir="$parquet" \
            --device=0 \
            --batch_size="$batch_size" \
            --skip_eval_samples=0 \
            --max_eval_samples=-1 \
            --max_new_tokens="$MAX_NEW_TOKENS" \
            --warmup_steps=0 \
            --dtype="$DTYPE" \
            --attn_impl="$ATTN_IMPL" \
            --audio_input="$AUDIO_INPUT" \
            --audio_decode="$AUDIO_DECODE" \
            --force_clean_exit
        status=$?
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] END name=$name status=$status"
        exit "$status"
    ) > "$log_file" 2>&1 &

    local pid="$!"
    JOB_PIDS+=("$pid")
    JOB_NAMES+=("$name")
    echo "$pid $name gpu=$gpu log=$log_file" >> "$LOG_DIR/pids.txt"
    echo "Started $name on gpu=$gpu pid=$pid log=$log_file"
}

run_task spgispeech "${GPUS[0]}" spgispeech test "$BASE/parquet/spgispeech"
run_task gigaspeech "${GPUS[1]}" gigaspeech test "$BASE/parquet/gigaspeech"
run_task ami "${GPUS[2]}" ami test "$BASE/parquet/ami"
run_task librispeech_clean "${GPUS[3]}" librispeech test.clean "$BASE/parquet/librispeech_clean"
run_task librispeech_other "${GPUS[4]}" librispeech test.other "$BASE/parquet/librispeech_other"
run_task voxpopuli "${GPUS[5]}" voxpopuli test "$BASE/parquet/voxpopuli"
run_task earnings22 "${GPUS[6]}" earnings22 test "$BASE/parquet/earnings22"

failed=0
for index in "${!JOB_PIDS[@]}"; do
    pid="${JOB_PIDS[$index]}"
    name="${JOB_NAMES[$index]}"
    if ! wait "$pid"; then
        echo "Job failed: name=$name pid=$pid log=$LOG_DIR/$name.log" >&2
        tail -60 "$LOG_DIR/$name.log" >&2 || true
        failed=1
    fi
done

if [ "$failed" -ne 0 ]; then
    echo "At least one eval job failed. Check logs in $LOG_DIR." >&2
    exit 1
fi

find "$BASE/jobs" -path '*/results/*.jsonl' -type f -exec cp -t "$FINAL_DIR" {} +

RESULT_DIR="$FINAL_DIR" PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

result_dir = Path(os.environ["RESULT_DIR"])
expected = {
    "_ami_test.jsonl": 11653,
    "_earnings22_test.jsonl": 2737,
    "_gigaspeech_test.jsonl": 19898,
    "_librispeech_test.clean.jsonl": 2620,
    "_librispeech_test.other.jsonl": 2939,
    "_spgispeech_test.jsonl": 39341,
    "_voxpopuli_test.jsonl": 1842,
}

files = sorted(result_dir.glob("*.jsonl"))
if len(files) != len(expected):
    raise SystemExit(f"Expected {len(expected)} result files, got {len(files)} in {result_dir}")

for suffix, expected_lines in expected.items():
    matches = [path for path in files if path.name.endswith(suffix)]
    if len(matches) != 1:
        raise SystemExit(f"Expected exactly one file ending with {suffix}, got {len(matches)}")
    actual_lines = sum(1 for _ in matches[0].open("r", encoding="utf-8"))
    if actual_lines != expected_lines:
        raise SystemExit(
            f"Wrong row count for {matches[0].name}: expected {expected_lines}, got {actual_lines}"
        )
    print(f"{matches[0].name}: {actual_lines}")

print(f"All expected result files are present in {result_dir}")
PY

SCORE_MODEL_ID="$MANIFEST_MODEL_ID" RESULT_DIR="$FINAL_DIR" PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" - <<'PY'
import os
from normalizer.eval_utils import score_results

score_results(os.environ["RESULT_DIR"], os.environ["SCORE_MODEL_ID"])
PY

echo "Done."
echo "RUN_ID=$RUN_ID"
echo "LOG_DIR=$LOG_DIR"
echo "FINAL_DIR=$FINAL_DIR"
