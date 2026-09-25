#!/bin/bash
# Multilingual evaluation settings for submit_jobs_ml.sh: FLEURS, MCV, and MLS
# in the six languages the leaderboard reports.
#   HF_TOKEN=hf_... bash soundsgoodai/submit_jobs_ml.sh
# Decoders and COMMON_ARGS come from config.sh; only the models, the datasets,
# the duration profile, and the bucket differ. run_models.sh does not read this
# file, and submit_jobs.sh (English) does not either.

# Set before sourcing so config.sh's ":-" default does not win; the environment
# still overrides both.
DEFAULT_DATASET_PATH=${DEFAULT_DATASET_PATH:-hf-audio/open-asr-leaderboard-multilingual-datasets}

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"

RESULTS_BUCKET=${RESULTS_BUCKET:-hf-audio/asr_leaderboard_multilingual}
RUN_ID=${RUN_ID:-fast-gpu-asr-ml-$(date -u +%Y%m%dT%H%M%S)}

MODEL_CONFIGS=(
    "nvidia/parakeet-tdt-0.6b-v3 parakeet parakeet-tdt-0.6b-v3.nemo transducer_modified_beam_search 6 128 (fast-gpu-asr)"
)

# Dataset configuration name and split. Configuration names are
# "<dataset>_<language>": submit_jobs_ml.sh takes the language from the suffix,
# for ONLY_LANGUAGES, for --language, and for the per-language summary, and the
# "ml_<language>" scoring families in normalizer/eval_utils.py key on it too.
DATASET_CONFIGS=(
    "fleurs_de test"
    "fleurs_fr test"
    "fleurs_it test"
    "fleurs_es test"
    "fleurs_pt test"
    "fleurs_nl test"
    "mcv_de test"
    "mcv_es test"
    "mcv_fr test"
    "mcv_it test"
    "mcv_nl test"
    "mls_es test"
    "mls_fr test"
    "mls_it test"
    "mls_pt test"
    "mls_nl test"
)

# FLEURS reaches 53 seconds
COMMON_ARGS+=(--max-audio-seconds=60.0)
