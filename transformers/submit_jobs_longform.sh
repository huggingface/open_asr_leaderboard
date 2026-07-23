#!/usr/bin/env bash

set -euo pipefail

# Submit every non-API Transformers model currently present on the long-form
# leaderboard. Use MODEL_FILTER or DATASET_FILTER for a partial/smoke run.
# Usage:
#   HF_TOKEN=hf_... ORG_NAME=hf-audio bash transformers/submit_jobs_longform.sh

SPACE="${SPACE:-hf-audio/open-asr-leaderboard-transformers}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/../scripts/longform_job_utils.sh"

# model_id | revision | batch_size | extra generation arguments
# Whisper-family long recordings use batch 32: batch 64 exceeds H200 memory on
# Earnings22 even though it fits the smaller TED-LIUM split.
MODEL_CONFIGS=(
  "openai/whisper-large|4ef9b41f0d4fe232daafdb5f76bb1dd8b23e01d7|32|--longform"
  "openai/whisper-large-v2|ae4642769ce2ad8fc292556ccea8e901f1530655|32|--longform"
  "openai/whisper-large-v3|06f233fe06e710322aca913c1bc4249a0d71fce1|32|--longform"
  "openai/whisper-large-v3-turbo|41f01f3fe87f28c78e2fbf8b568835947dd65ed9|32|--longform"
  "distil-whisper/distil-medium.en|6e61418885eaf4d5cc9f64e508e80ac5b4c052b7|32|--longform"
  "distil-whisper/distil-large-v2|97d2c8f9cae1b0f6c8fc2e173495ee4cedc05843|32|--longform"
  "distil-whisper/distil-large-v3|8031d2e6ce6631b7fc45629dddfc00271116d981|32|--longform"
  "distil-whisper/distil-large-v3.5|728a7691f3ff1d3d971528d3203a6e9559165d41|32|--longform"
  "CohereLabs/cohere-transcribe-03-2026|176856d3145c49b02fbf7182d4a7905ea9232361|1|--max_new_tokens=500 --audio_chunk_length_s=300 --audio_chunk_batch_size=4"
)

for config in "${MODEL_CONFIGS[@]}"; do
  IFS='|' read -r model_id revision batch_size extra_args <<< "$config"
  run_longform_model \
    "$model_id" \
    "run_eval.py" \
    "$batch_size" \
    "--revision=${revision} --warmup_max_new_tokens=1 ${extra_args}" \
    "$script_dir/run_eval.py"
done
