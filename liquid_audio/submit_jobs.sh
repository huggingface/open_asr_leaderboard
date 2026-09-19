#!/usr/bin/env bash
# Dry run by default. RUN_LOCAL=1 evaluates locally; RUN_JOBS=1 submits HF Jobs.
set -euo pipefail

SPACE="${SPACE:-}"
FLAVOR="${FLAVOR:-h200}"
TIMEOUT="${TIMEOUT:-1h}"
RUN_JOBS="${RUN_JOBS:-0}"
RUN_LOCAL="${RUN_LOCAL:-0}"
LOCAL_RESULTS_DIR="${LOCAL_RESULTS_DIR:-./liquid_audio/results}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_BUCKET="${RESULTS_BUCKET:-}"
MODEL_ID="LiquidAI/LFM2.5-Audio-1.5B"
MODEL_REVISION="c362a0625dfe45aa588dce5f0ada28a7e5707628"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
ONLY_DATASETS="${ONLY_DATASETS:-}"

# label, config, split, dataset repository. Effective model batch size is one.
DATASETS=(
  "ami_cleaned ami_cleaned test hf-audio/open-asr-leaderboard"
  "gigaspeech_cleaned gigaspeech_cleaned test hf-audio/open-asr-leaderboard"
  "voxpopuli_cleaned_aa voxpopuli_cleaned_aa test hf-audio/open-asr-leaderboard"
  "earnings22_cleaned_aa_chunked - test ArtificialAnalysis/Earnings22-Cleaned-AA-chunked"
  "librispeech_clean librispeech test.clean hf-audio/open-asr-leaderboard"
  "librispeech_other librispeech test.other hf-audio/open-asr-leaderboard"
  "spgispeech spgispeech test hf-audio/open-asr-leaderboard"
  "monsoon_en_in - test VoiceArena/Monsoon_en_IN_test"
)

if [[ "$RUN_JOBS" == 1 && "$RUN_LOCAL" == 1 ]]; then
  echo "Choose RUN_LOCAL=1 or RUN_JOBS=1, not both." >&2
  exit 1
fi
if [[ "$RUN_JOBS" == 1 || "$RUN_LOCAL" == 1 ]]; then
  : "${HF_TOKEN:?Set HF_TOKEN for dataset access; cloud runs also need HF Jobs and bucket access}"
fi
if [[ "$RUN_JOBS" == 1 ]]; then
  : "${SPACE:?Set SPACE to a published evaluation Space}"
  : "${RESULTS_BUCKET:?Set RESULTS_BUCKET to an existing results bucket}"
fi
SPACE="${SPACE:-YOUR_USERNAME/open-asr-leaderboard-liquid-audio}"
RESULTS_BUCKET="${RESULTS_BUCKET:-YOUR_USERNAME/asr-leaderboard-results}"

selected=0
for entry in "${DATASETS[@]}"; do
  read -r label config split dataset_path <<< "$entry"
  if [[ -n "$ONLY_DATASETS" && " $ONLY_DATASETS " != *" $label "* ]]; then
    continue
  fi
  selected=$((selected + 1))
  [[ "$config" != - ]] || config=""
  if [[ "$RUN_LOCAL" == 1 ]]; then
    PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" "${SCRIPT_DIR}/run_eval.py" \
      --model_id "$MODEL_ID" --revision "$MODEL_REVISION" \
      --dataset_path "$dataset_path" --dataset "$config" --split "$split" \
      --batch_size 1 --max_new_tokens 512 --warmup_steps 3 \
      --output_dir "${LOCAL_RESULTS_DIR}/${label}"
    continue
  fi
  # Arguments are passed separately to bash; no eval or credential interpolation.
  cmd=(hf jobs run --flavor "$FLAVOR" --timeout "$TIMEOUT"
       --secrets HF_TOKEN
       --volume "hf://buckets/${RESULTS_BUCKET}:/results"
       "hf.co/spaces/${SPACE}" bash -c
       'set -euo pipefail; python /app/run_eval.py --model_id "$1" --revision "$2" --dataset_path "$3" --dataset "$4" --split "$5" --batch_size 1 --max_new_tokens 512 --warmup_steps 3 --output_dir "/results/$6/$7"'
       bash "$MODEL_ID" "$MODEL_REVISION" "$dataset_path" "$config" "$split" "$RUN_ID" "$label")
  if [[ "$RUN_JOBS" == 1 ]]; then
    "${cmd[@]}"
  else
    printf '%q ' "${cmd[@]}"
    printf '\n'
  fi
done
if [[ "$selected" == 0 ]]; then
  echo "No dataset matched ONLY_DATASETS=$ONLY_DATASETS" >&2
  exit 1
fi
if [[ "$RUN_JOBS" != 1 && "$RUN_LOCAL" != 1 ]]; then
  echo "Dry run: $selected jobs prepared; no job submitted."
fi
