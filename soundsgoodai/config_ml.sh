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

# Only multilingual checkpoints belong here. Parakeet TDT V3 covers 25 European
# languages; V2, the Parakeet CTC models, and Zipformer are English-only.
# Same fields as config.sh: repository, family, checkpoint, decoder, beam,
# batch size, reported-name suffix.
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

# FLEURS reaches 53 seconds, past the 40-second profile config.sh exports for
# the English sets, and overlong clips fail rather than being truncated. The
# last value of a repeated argument wins, so this replaces it.
#
# The profile and the batch size are exported together. The feature plugin
# stages its transform in one TensorRT workspace of
#   batch_size * (16000 * max_audio_seconds / 160 + 1) * 514 * 4 bytes,
# plus 16 MiB of cuBLAS scratch, and the export fails outright once that
# exceeds the signed 32-bit limit ("exceeds the feature plugin's signed 32-bit
# TensorRT workspace limit"). The English 256 x 40 s already uses 98.8% of it,
# so 60 s caps the batch at 172; 128 above leaves room. Raising one of the two
# means lowering the other.
COMMON_ARGS+=(--max-audio-seconds=60.0)
