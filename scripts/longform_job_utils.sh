#!/usr/bin/env bash

# Shared helpers for the family-specific long-form HF Jobs launchers.
# This file is sourced; callers must define SPACE before calling
# run_longform_model.

set -euo pipefail

RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_h200}"
FLAVOR="${FLAVOR:-h200}"
ORG_NAME="${ORG_NAME:-}"
JOB_TIMEOUT="${JOB_TIMEOUT:-12h}"
POLL_INTERVAL="${POLL_INTERVAL:-20}"
MAX_JOB_ATTEMPTS="${MAX_JOB_ATTEMPTS:-3}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-3}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:--1}"
MODEL_FILTER="${MODEL_FILTER:-}"
DATASET_FILTER="${DATASET_FILTER:-}"
RESUME="${RESUME:-1}"

LONGFORM_DATASET_CONFIGS=(
  "hf-audio/asr-leaderboard-longform earnings21 test d6797370d3189c618e722721ab5b6c9be78c022c /datasets/longform/earnings21/test-*.parquet"
  "hf-audio/asr-leaderboard-longform earnings22 test d6797370d3189c618e722721ab5b6c9be78c022c /datasets/longform/earnings22/test-*.parquet"
  "distil-whisper/tedlium-long-form default test ea3a78479fb5337761f359abcd4e883d2d6e3c5b /datasets/tedlium/data/test-*.parquet"
  "bezzam/coraal ATL test 76417d96197aa8bfa419eb5cc7637dab2f75cba0 /datasets/coraal/ATL/test-*.parquet"
  "bezzam/coraal DCA test 76417d96197aa8bfa419eb5cc7637dab2f75cba0 /datasets/coraal/DCA/test-*.parquet"
  "bezzam/coraal DCB test 76417d96197aa8bfa419eb5cc7637dab2f75cba0 /datasets/coraal/DCB/test-*.parquet"
  "bezzam/coraal DTA test 76417d96197aa8bfa419eb5cc7637dab2f75cba0 /datasets/coraal/DTA/test-*.parquet"
  "bezzam/coraal LES test 76417d96197aa8bfa419eb5cc7637dab2f75cba0 /datasets/coraal/LES/test-*.parquet"
  "bezzam/coraal PRV test 76417d96197aa8bfa419eb5cc7637dab2f75cba0 /datasets/coraal/PRV/test-*.parquet"
  "bezzam/coraal ROC test 76417d96197aa8bfa419eb5cc7637dab2f75cba0 /datasets/coraal/ROC/test-*.parquet"
  "bezzam/coraal VLD test 76417d96197aa8bfa419eb5cc7637dab2f75cba0 /datasets/coraal/VLD/test-*.parquet"
)

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required" >&2
  exit 1
fi
if ! [[ "$MAX_PARALLEL_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PARALLEL_JOBS must be a positive integer" >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_utils_b64="$(base64 < "$repo_root/normalizer/data_utils.py" | tr -d '\n')"

inspect_job_status() {
  local job_id="$1"
  python3 - "$job_id" "$ORG_NAME" <<'PY'
import json
import subprocess
import sys

job_id, namespace = sys.argv[1:]
command = ["hf", "jobs", "inspect"]
if namespace:
    command.extend(["--namespace", namespace])
command.extend([job_id, "--json"])
try:
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
except subprocess.TimeoutExpired:
    print(f"Timed out inspecting Job {job_id}", file=sys.stderr)
    raise SystemExit(124)
except subprocess.CalledProcessError as error:
    print(error.stderr or f"Could not inspect Job {job_id}", file=sys.stderr)
    raise SystemExit(error.returncode)

status = json.loads(result.stdout)[0]["status"]
print(status["stage"] + "\t" + status.get("message", ""))
PY
}

run_longform_model() {
  local model_id="$1"
  local entrypoint="$2"
  local batch_size="$3"
  local extra_args="${4:-}"
  local source_file="$5"
  local evaluator_b64
  evaluator_b64="$(base64 < "$source_file" | tr -d '\n')"

  if [[ -n "$MODEL_FILTER" && "$model_id" != "$MODEL_FILTER" ]]; then
    return 0
  fi

  local model_folder="${model_id//\//-}"
  local model_label="${model_folder//./-}"
  local bucket_prefix="${BUCKET_PREFIX:-}"
  if [[ -z "$bucket_prefix" ]]; then
    bucket_prefix="longform"
    if [[ "$MAX_EVAL_SAMPLES" != "-1" || -n "$DATASET_FILTER" ]]; then
      bucket_prefix="longform-smoke"
    fi
  fi
  local namespace_arg=""
  if [[ -n "$ORG_NAME" ]]; then
    namespace_arg="--namespace $ORG_NAME"
  fi

  echo "Evaluating $model_id on $FLAVOR (batch_size=$batch_size)"
  local pids=()
  local submitted=0
  local skipped=0
  local failed=0
  local existing_results=""
  if [[ "$RESUME" == "1" ]]; then
    local bucket_listing=""
    local list_succeeded=0
    local list_attempt
    for list_attempt in 1 2 3; do
      if bucket_listing="$(
        hf buckets list \
          "${RESULTS_BUCKET}/${bucket_prefix}/${model_folder}" \
          --recursive \
          --format json 2>/dev/null
      )"; then
        list_succeeded=1
        # A missing prefix is a valid empty result set and the CLI may emit
        # either an empty string or an empty JSON list for it.
        if [[ -z "$bucket_listing" ]]; then
          bucket_listing="[]"
        fi
        break
      fi
      echo "Could not list stored results for $model_id (attempt $list_attempt/3)" >&2
      sleep "$POLL_INTERVAL"
    done
    if [[ "$list_succeeded" -ne 1 ]]; then
      echo "Refusing to resume $model_id without a valid bucket listing" >&2
      return 1
    fi
    existing_results="$(
      printf '%s' "$bucket_listing" |
        python3 -c 'import json, os, sys; print("\n".join(os.path.basename(x["path"]) for x in json.load(sys.stdin) if x.get("type") == "file"))'
    )"
  fi

  submit_dataset_job() {
    local dataset_path="$1"
    local dataset="$2"
    local split="$3"
    local dataset_revision="$4"
    local data_files="$5"
    local dataset_volume="$6"
    local attempt=1

    while [[ "$attempt" -le "$MAX_JOB_ATTEMPTS" ]]; do
      echo "Submitting model=$model_id dataset=$dataset split=$split attempt=$attempt"
      local submit_output job_id
      submit_output="$(
        hf jobs run --detach \
          --flavor "$FLAVOR" \
          --timeout "$JOB_TIMEOUT" \
          --secrets HF_TOKEN \
          --env HF_AUDIO_DECODER_BACKEND=soundfile \
          --env TOKENIZERS_PARALLELISM=false \
          --label benchmark=longform \
          --label model="$model_label" \
          --label dataset="${dataset//./-}" \
          --label split="$split" \
          ${namespace_arg} \
          --volume "$dataset_volume" \
          --volume "hf://buckets/${RESULTS_BUCKET}:/results_bucket" \
          "hf.co/spaces/${SPACE}" \
          bash -c "
            set -euo pipefail
            cd /app
            rm -rf /app/results
            echo '${data_utils_b64}' | base64 -d > /app/normalizer/data_utils.py
            echo '${evaluator_b64}' | base64 -d > /app/${entrypoint}
            PYTHONPATH=/app python /app/${entrypoint} \\
              --model_id=${model_id} \\
              --dataset_path=${dataset_path} \\
              --dataset_revision=${dataset_revision} \\
              --data_files='${data_files}' \\
              --dataset=${dataset} \\
              --split=${split} \\
              --device=0 \\
              --batch_size=${batch_size} \\
              --max_eval_samples=${MAX_EVAL_SAMPLES} \\
              --streaming \\
              ${extra_args}
            mkdir -p /results_bucket/${bucket_prefix}/${model_folder}
            cp /app/results/*.jsonl /results_bucket/${bucket_prefix}/${model_folder}/
          "
      )"
      echo "$submit_output"
      job_id="$(printf '%s\n' "$submit_output" | sed -n 's/^id=\([^ ]*\).*/\1/p' | head -1)"
      if [[ -z "$job_id" ]]; then
        echo "Could not parse Job ID for $model_id / $dataset" >&2
        return 1
      fi

      while true; do
        local status_line stage message
        if ! status_line="$(inspect_job_status "$job_id")"; then
          echo "Could not inspect Job $job_id; retrying status check" >&2
          sleep "$POLL_INTERVAL"
          continue
        fi
        stage="${status_line%%$'\t'*}"
        message="${status_line#*$'\t'}"
        case "$stage" in
          COMPLETED)
            echo "Job $job_id completed"
            return 0
            ;;
          ERROR)
            if [[ "$message" == *"Volume mount failed"* && "$attempt" -lt "$MAX_JOB_ATTEMPTS" ]]; then
              echo "Job $job_id hit a transient volume mount failure; retrying" >&2
              break
            fi
            echo "Job $job_id ended with status $stage: $message" >&2
            return 1
            ;;
          CANCELED)
            if [[ "$attempt" -lt "$MAX_JOB_ATTEMPTS" ]]; then
              echo "Job $job_id was canceled before completion; retrying" >&2
              break
            fi
            echo "Job $job_id remained canceled after $attempt attempts" >&2
            return 1
            ;;
          *)
            sleep "$POLL_INTERVAL"
            ;;
        esac
      done
      attempt=$((attempt + 1))
    done
    return 1
  }

  for config in "${LONGFORM_DATASET_CONFIGS[@]}"; do
    local dataset_path dataset split dataset_revision data_files
    read -r dataset_path dataset split dataset_revision data_files <<< "$config"
    if [[ -n "$DATASET_FILTER" && "$dataset $dataset_path" != *"$DATASET_FILTER"* ]]; then
      continue
    fi

    local dataset_path_slug expected_file
    dataset_path_slug="${dataset_path//\//-}"
    expected_file="MODEL_${model_folder}_DATASET_${dataset_path_slug}_${dataset}_${split}.jsonl"
    if [[ "$RESUME" == "1" ]] && printf '%s\n' "$existing_results" | grep -Fqx "$expected_file"; then
      echo "Skipping completed result $expected_file"
      skipped=$((skipped + 1))
      continue
    fi

    local dataset_volume
    case "$dataset_path" in
      hf-audio/asr-leaderboard-longform)
        dataset_volume="hf://datasets/hf-audio/asr-leaderboard-longform@${dataset_revision}:/datasets/longform"
        ;;
      distil-whisper/tedlium-long-form)
        dataset_volume="hf://datasets/distil-whisper/tedlium-long-form@${dataset_revision}:/datasets/tedlium"
        ;;
      bezzam/coraal)
        dataset_volume="hf://datasets/bezzam/coraal@${dataset_revision}:/datasets/coraal"
        ;;
      *)
        echo "No dataset mount configured for $dataset_path" >&2
        return 1
        ;;
    esac

    submit_dataset_job \
      "$dataset_path" \
      "$dataset" \
      "$split" \
      "$dataset_revision" \
      "$data_files" \
      "$dataset_volume" &
    pids+=("$!")
    submitted=$((submitted + 1))

    # Bound H200 concurrency so large splits are not preempted when a whole
    # model's dataset matrix is submitted at once.
    if [[ "${#pids[@]}" -ge "$MAX_PARALLEL_JOBS" ]]; then
      local pid
      for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
          failed=1
        fi
      done
      pids=()
    fi
  done

  if [[ "$submitted" -eq 0 && "$skipped" -eq 0 ]]; then
    echo "No datasets matched DATASET_FILTER=$DATASET_FILTER" >&2
    return 1
  fi

  if [[ "${#pids[@]}" -gt 0 ]]; then
    local pid
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        failed=1
      fi
    done
  fi
  if [[ "$failed" -ne 0 ]]; then
    echo "One or more HF Jobs failed for $model_id" >&2
    return 1
  fi

  # Do not score smoke tests or partial dataset selections as complete runs.
  if [[ "$MAX_EVAL_SAMPLES" != "-1" || -n "$DATASET_FILTER" ]]; then
    echo "Completed partial run for $model_id; skipping sync and scoring."
    return 0
  fi

  sleep 10
  local local_results="$repo_root/results/longform/$model_folder"
  mkdir -p "$local_results"
  hf buckets sync \
    "hf://buckets/${RESULTS_BUCKET}/longform/${model_folder}" \
    "$local_results"

  local actual
  actual="$(find "$local_results" -name '*.jsonl' | wc -l | tr -d ' ')"
  if [[ "$actual" -ne "${#LONGFORM_DATASET_CONFIGS[@]}" ]]; then
    echo "Expected ${#LONGFORM_DATASET_CONFIGS[@]} results for $model_id, found $actual" >&2
    return 1
  fi

  if python -c 'import jiwer, regex' >/dev/null 2>&1; then
    PYTHONPATH="$repo_root" python \
      "$repo_root/scripts/score_longform_results.py" \
      "$local_results" \
      --model-id "$model_id" \
      --current-csv "$repo_root/scripts/data/en_longform.csv"
  else
    PYTHONPATH="$repo_root" uv run \
      --with jiwer \
      --with regex \
      python "$repo_root/scripts/score_longform_results.py" \
      "$local_results" \
      --model-id "$model_id" \
      --current-csv "$repo_root/scripts/data/en_longform.csv"
  fi
}
