#!/usr/bin/env bash

set -euo pipefail

# Submit every non-API NeMo model currently present on the long-form
# leaderboard. Use MODEL_FILTER or DATASET_FILTER for a partial/smoke run.
# Usage:
#   HF_TOKEN=hf_... ORG_NAME=hf-audio bash nemo_asr/submit_jobs_longform.sh

SPACE="${SPACE:-hf-audio/open-asr-leaderboard-nemo}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/../scripts/longform_job_utils.sh"

# model_id | entrypoint | batch_size | extra arguments
MODEL_CONFIGS=(
  "nvidia/parakeet-tdt-1.1b|run_eval_long.py|1|--longform"
  "nvidia/parakeet-rnnt-1.1b|run_eval_long.py|1|--longform"
  "nvidia/parakeet-rnnt-0.6b|run_eval_long.py|1|--longform"
  "nvidia/stt_en_fastconformer_transducer_large|run_eval_long.py|1|--longform"
  "nvidia/stt_en_conformer_transducer_large|run_eval_long.py|1|--longform"
  "nvidia/stt_en_conformer_transducer_small|run_eval_long.py|1|--longform --model_load_id=stt_en_conformer_transducer_small"
  "nvidia/parakeet-tdt-0.6b-v2|run_eval_long.py|1|--longform"
  "nvidia/parakeet-tdt-0.6b-v3|run_eval_long.py|1|--longform"
  "nvidia/parakeet-ctc-1.1b|run_eval_long.py|1|--longform"
  "nvidia/parakeet-ctc-0.6b|run_eval_long.py|1|--longform"
  "nvidia/stt_en_fastconformer_ctc_large|run_eval_long.py|1|--longform"
  "nvidia/stt_en_conformer_ctc_large|run_eval_long.py|1|--longform"
  "nvidia/stt_en_conformer_ctc_small|run_eval_long.py|1|--longform"
  "nvidia/canary-qwen-2.5b|run_eval_salm.py|192|--longform --chunk_length_s=30"
)

for config in "${MODEL_CONFIGS[@]}"; do
  IFS='|' read -r model_id entrypoint batch_size extra_args <<< "$config"
  run_longform_model \
    "$model_id" \
    "$entrypoint" \
    "$batch_size" \
    "$extra_args" \
    "$script_dir/$entrypoint"
done
