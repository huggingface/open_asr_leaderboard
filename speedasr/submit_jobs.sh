#!/usr/bin/env bash
# Launches PAID HF Jobs: one H200, one dataset at a time.
set -euo pipefail

: "${HF_TOKEN:?Set an HF write token in the process environment.}"
SPACE=${SPACE:-SpeedAudio/open-asr-leaderboard-speedasr}
RESULTS_BUCKET=${RESULTS_BUCKET:-SpeedAudio/speedasr-open-asr-results}
MODEL_ID=${MODEL_ID:-SpeedAudio261119/SpeedASR-v1-65M}
MODEL_REVISION=${MODEL_REVISION:-a6f5cacea359929b52ae41b77f6f634441cf0546}
FLAVOR=${FLAVOR:-h200}
TIMEOUT=${TIMEOUT:-1h}
BATCH_SIZE=${BATCH_SIZE:-2048}
BATCH_AUDIO_SECONDS=${BATCH_AUDIO_SECONDS:-5000}
TORCH_THREADS=${TORCH_THREADS:-16}
INCLUDE_URGENT_CLEAN=${INCLUDE_URGENT_CLEAN:-0}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
PYTHON=${PYTHON:-python}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL_FOLDER=${MODEL_ID//\//-}
LOCAL_RESULTS="$SCRIPT_DIR/cloud_runs/$RUN_ID"
export SPACE RESULTS_BUCKET MODEL_ID MODEL_REVISION FLAVOR TIMEOUT BATCH_SIZE
export BATCH_AUDIO_SECONDS TORCH_THREADS INCLUDE_URGENT_CLEAN RUN_ID

command -v hf >/dev/null
"$PYTHON" - <<'PY'
import importlib.metadata
import math
import os
import re

if importlib.metadata.version("huggingface_hub") != "1.18.0":
    raise SystemExit("Install huggingface_hub==1.18.0 in the environment providing python and hf.")
for key in ("SPACE", "RESULTS_BUCKET", "MODEL_ID"):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", os.environ[key]):
        raise SystemExit(f"Invalid namespace/name in {key}")
if not re.fullmatch(r"[0-9a-fA-F]{40}", os.environ["MODEL_REVISION"]):
    raise SystemExit("MODEL_REVISION must be an immutable full model commit SHA.")
if os.getenv("SPACE_REVISION") and not re.fullmatch(r"[0-9a-fA-F]{40}", os.environ["SPACE_REVISION"]):
    raise SystemExit("SPACE_REVISION must be a full Space source commit SHA.")
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", os.environ["RUN_ID"]):
    raise SystemExit("RUN_ID must be a single safe directory name.")
if os.environ["FLAVOR"] != "h200":
    raise SystemExit("Use FLAVOR=h200 for the official English short-form comparison.")
if os.environ["INCLUDE_URGENT_CLEAN"] not in {"0", "1"}:
    raise SystemExit("INCLUDE_URGENT_CLEAN must be 0 or 1.")
for key in ("BATCH_SIZE", "TORCH_THREADS"):
    if not os.environ[key].isdigit() or int(os.environ[key]) < 1:
        raise SystemExit(f"{key} must be a positive integer.")
if (not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", os.environ["BATCH_AUDIO_SECONDS"])
        or not math.isfinite(float(os.environ["BATCH_AUDIO_SECONDS"]))
        or float(os.environ["BATCH_AUDIO_SECONDS"]) <= 0):
    raise SystemExit("BATCH_AUDIO_SECONDS must be a positive finite number.")
if not re.fullmatch(r"[1-9][0-9]*[smh]", os.environ["TIMEOUT"]):
    raise SystemExit("TIMEOUT must be a positive integer followed by s, m or h.")
PY

DATASETS=(
  "ami_cleaned|hf-audio/open-asr-leaderboard|ami_cleaned"
  "gigaspeech_cleaned|hf-audio/open-asr-leaderboard|gigaspeech_cleaned"
  "voxpopuli_cleaned_aa|hf-audio/open-asr-leaderboard|voxpopuli_cleaned_aa"
  "earnings22_cleaned_aa_chunked|ArtificialAnalysis/Earnings22-Cleaned-AA-chunked|"
  "spgispeech|hf-audio/open-asr-leaderboard|spgispeech"
  "urgent2024|hf-audio/open-asr-leaderboard|urgent2024"
  "monsoon_en_in|VoiceArena/Monsoon_en_IN_test|"
)
if [[ "$INCLUDE_URGENT_CLEAN" == 1 ]]; then
  DATASETS+=("urgent2024_clean|hf-audio/open-asr-leaderboard|urgent2024_clean")
fi
read -r -a requested <<< "${ONLY_DATASETS:-}"
for wanted in "${requested[@]}"; do
  found=0
  for cfg in "${DATASETS[@]}"; do
    [[ "${cfg%%|*}" != "$wanted" ]] || found=1
  done
  if (( found == 0 )); then
    printf 'Unknown/unselected dataset: %s\n' "$wanted" >&2
    exit 2
  fi
done
namespace_args=()
if [[ -n ${ORG_NAME:-} ]]; then
  namespace_args=(--namespace "$ORG_NAME")
fi
if [[ -e "$LOCAL_RESULTS/run_config.json" ]]; then
  printf 'RUN_ID already exists locally; choose a new RUN_ID to keep results separate.\n' >&2
  exit 2
fi
mkdir -p "$LOCAL_RESULTS/jobs"
"$PYTHON" - "$LOCAL_RESULTS/run_config.json" <<'PY'
import json
import os
import sys
from huggingface_hub import HfApi

api = HfApi()
space = api.space_info(os.environ["SPACE"])
stage = getattr(space.runtime, "stage", "UNKNOWN")
stage = getattr(stage, "value", stage)
if stage not in {"RUNNING", "SLEEPING"}:
    raise SystemExit(f"Space build is not ready: {stage}")
if os.getenv("SPACE_REVISION") and space.sha != os.environ["SPACE_REVISION"]:
    raise SystemExit("Space source commit differs from SPACE_REVISION.")
if api.model_info(os.environ["MODEL_ID"], revision=os.environ["MODEL_REVISION"]).sha != os.environ["MODEL_REVISION"]:
    raise SystemExit("Model commit did not resolve to the requested immutable revision.")
keys = ("SPACE", "RESULTS_BUCKET", "MODEL_ID", "MODEL_REVISION", "FLAVOR", "TIMEOUT",
        "BATCH_SIZE", "BATCH_AUDIO_SECONDS", "TORCH_THREADS", "INCLUDE_URGENT_CLEAN", "RUN_ID")
data = {key: os.environ[key] for key in keys}
data.update(space_revision=space.sha, space_stage=stage, hub_version="1.18.0",
            only_datasets=os.getenv("ONLY_DATASETS", ""), org_name=os.getenv("ORG_NAME"))
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
print(f"Model: {data['MODEL_ID']}@{data['MODEL_REVISION']}", flush=True)
print(f"Space source: {data['SPACE']}@{space.sha}", flush=True)
PY

JOB_COMMAND=$(cat <<'JOB'
set -euo pipefail
cd /app
OUT="/results/${SPEEDASR_MODEL_FOLDER}/${SPEEDASR_RUN_ID}/${SPEEDASR_DATASET_LABEL}"
mkdir -p "$OUT"
finish() {
  code=$?
  trap - EXIT
  cp results/*.jsonl results/*.meta.json "$OUT/" 2>/dev/null || true
  printf '{"exit_code": %s}\n' "$code" > "$OUT/job_exit.json"
  exit "$code"
}
trap finish EXIT
python leaderboard/run_eval.py \
  --model_id "$SPEEDASR_MODEL_ID" --revision "$SPEEDASR_MODEL_REVISION" \
  --dataset_path "$SPEEDASR_DATASET_PATH" --dataset "$SPEEDASR_DATASET_CONFIG" \
  --split test --device 0 --batch_size "$SPEEDASR_BATCH_SIZE" \
  --batch-audio-seconds "$SPEEDASR_BATCH_AUDIO_SECONDS" \
  --torch-threads "$SPEEDASR_TORCH_THREADS" --warmup-steps 5 \
  --warmup-seconds 10 --max_eval_samples -1 2>&1 | tee "$OUT/job.log"
JOB
)

sync_results() {
  hf buckets sync "hf://buckets/${RESULTS_BUCKET}/${MODEL_FOLDER}/${RUN_ID}" "$LOCAL_RESULTS/results"
}

selected=0
for cfg in "${DATASETS[@]}"; do
  IFS='|' read -r label dataset_path dataset_config <<< "$cfg"
  if [[ -n ${ONLY_DATASETS:-} && " ${ONLY_DATASETS} " != *" ${label} "* ]]; then
    continue
  fi
  selected=$((selected + 1))
  # The Space shorthand has no documented revision argument in Hub 1.18.0.
  # Refuse to mix Space source revisions within a run; save the observed SHA.
  "$PYTHON" - "$LOCAL_RESULTS/run_config.json" <<'PY'
import json
import sys
from huggingface_hub import HfApi
with open(sys.argv[1], encoding="utf-8") as f:
    config = json.load(f)
space = HfApi().space_info(config["SPACE"])
stage = getattr(space.runtime, "stage", "UNKNOWN")
if space.sha != config["space_revision"] or getattr(stage, "value", stage) not in {"RUNNING", "SLEEPING"}:
    raise SystemExit("Space source/build changed during this run; stop before launching another job.")
PY
  printf 'Launching %s on one %s; runtime limit %s\n' "$label" "$FLAVOR" "$TIMEOUT"
  # In 1.18.0, non-detached jobs run streams logs but does not test final status.
  # Capture the ID explicitly and inspect it below. Bare --secrets HF_TOKEN
  # reads the token from the environment without putting its value in argv.
  hf jobs run --detach --format json --flavor "$FLAVOR" --timeout "$TIMEOUT" \
    "${namespace_args[@]}" --secrets HF_TOKEN \
    --label "model=$MODEL_ID" --label "run_id=$RUN_ID" --label "dataset=$label" \
    --env HF_AUDIO_DECODER_BACKEND=soundfile \
    --env "SPEEDASR_MODEL_ID=$MODEL_ID" --env "SPEEDASR_MODEL_REVISION=$MODEL_REVISION" \
    --env "SPEEDASR_MODEL_FOLDER=$MODEL_FOLDER" --env "SPEEDASR_RUN_ID=$RUN_ID" \
    --env "SPEEDASR_DATASET_LABEL=$label" --env "SPEEDASR_DATASET_PATH=$dataset_path" \
    --env "SPEEDASR_DATASET_CONFIG=$dataset_config" --env "SPEEDASR_BATCH_SIZE=$BATCH_SIZE" \
    --env "SPEEDASR_BATCH_AUDIO_SECONDS=$BATCH_AUDIO_SECONDS" \
    --env "SPEEDASR_TORCH_THREADS=$TORCH_THREADS" \
    --volume "hf://buckets/${RESULTS_BUCKET}:/results" \
    "hf.co/spaces/${SPACE}" bash -c "$JOB_COMMAND" > "$LOCAL_RESULTS/jobs/${label}.launch.json"

  if "$PYTHON" - "$LOCAL_RESULTS/jobs/${label}.launch.json" "$LOCAL_RESULTS/jobs/${label}.status.json" "$LOCAL_RESULTS/run_config.json" <<'PY'
import json
import os
import sys
import time
from huggingface_hub import HfApi

with open(sys.argv[1], encoding="utf-8") as f:
    launched = json.load(f)
with open(sys.argv[3], encoding="utf-8") as f:
    run_config = json.load(f)
job_id = launched["id"]
api = HfApi()
namespace = os.getenv("ORG_NAME") or None
last = None
terminal = {"COMPLETED", "CANCELED", "ERROR", "DELETED"}
try:
    while True:
        job = api.inspect_job(job_id=job_id, namespace=namespace)
        stage = getattr(job.status.stage, "value", job.status.stage) if job.status else "UNKNOWN"
        data = {"id": job.id, "url": job.url, "stage": stage,
                "message": getattr(job.status, "message", None), "flavor": job.flavor,
                "model_revision": os.environ["MODEL_REVISION"],
                "space_revision_at_submission": run_config["space_revision"],
                "space_id": job.space_id, "docker_image": job.docker_image}
        with open(sys.argv[2], "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        if stage != last:
            print(f"{job.url}: {stage}", flush=True)
            last = stage
        if stage in terminal:
            if stage != "COMPLETED":
                raise SystemExit(f"Job did not complete successfully: {stage}; {data['message']}")
            break
        time.sleep(15)
except KeyboardInterrupt:
    api.cancel_job(job_id=job_id, namespace=namespace)
    raise SystemExit("Interrupted; cancellation requested for the active paid job.")
PY
  then
    sync_results
  else
    sync_results || true
    exit 1
  fi
done
if (( selected == 0 )); then
  printf 'No datasets selected; check ONLY_DATASETS.\n' >&2
  exit 2
fi
printf 'Completed %s dataset jobs. Results: %s\n' "$selected" "$LOCAL_RESULTS"
