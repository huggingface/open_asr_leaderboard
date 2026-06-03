#!/bin/bash

set -e

export PYTHONPATH="..":${PYTHONPATH:-}

MODEL_IDs=("${MODEL_ID:-AutoArk-AI/ARK-ASR-0.6B}")
BATCH_SIZE=${BATCH_SIZE:-64}
DEVICE_ID=${DEVICE_ID:-0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
WARMUP_STEPS=${WARMUP_STEPS:-10}
DTYPE=${DTYPE:-float16}
ATTN_IMPL=${ATTN_IMPL:-sdpa}
AUDIO_INPUT=${AUDIO_INPUT:-array}
AUDIO_DECODE=${AUDIO_DECODE:-datasets}
DATASET_PATH=${DATASET_PATH:-hf-audio/esb-datasets-test-only-sorted}
# Current dataset main metadata still lists tedlium, but the tedlium parquet was removed in
# a6318480517f735c7627a29adf8725f2a5dd81ef. Pin the last revision with the full
# historical shortform set used by scripts/data/en_shortform.csv. Set DATASET_REVISION=
# explicitly to use the dataset's current default revision.
DATASET_REVISION=${DATASET_REVISION-20a009a3a37d035d965722e5feb890ba7f2d46ac}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:--1}
FORCE_CLEAN_EXIT=${FORCE_CLEAN_EXIT:-false}

if [ -n "${EVAL_DATASETS:-}" ]; then
    read -r -a EVAL_DATASET_ENTRIES <<< "${EVAL_DATASETS}"
else
    declare -a EVAL_DATASET_ENTRIES=(
        "voxpopuli:test"
        "ami:test"
        "earnings22:test"
        "gigaspeech:test"
        "librispeech:test.clean"
        "librispeech:test.other"
        "spgispeech:test"
        "tedlium:test"
    )
fi

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}

    for entry in "${EVAL_DATASET_ENTRIES[@]}"; do
        DATASET="${entry%%:*}"
        SPLIT="${entry##*:}"

        python run_eval.py \
            --model_id="${MODEL_ID}" \
            --dataset_path="${DATASET_PATH}" \
            --dataset_revision="${DATASET_REVISION}" \
            --dataset="${DATASET}" \
            --split="${SPLIT}" \
            --device="${DEVICE_ID}" \
            --batch_size="${BATCH_SIZE}" \
            --max_eval_samples="${MAX_EVAL_SAMPLES}" \
            --max_new_tokens="${MAX_NEW_TOKENS}" \
            --warmup_steps="${WARMUP_STEPS}" \
            --dtype="${DTYPE}" \
            --attn_impl="${ATTN_IMPL}" \
            --audio_input="${AUDIO_INPUT}" \
            --audio_decode="${AUDIO_DECODE}" \
            $(if [ "${FORCE_CLEAN_EXIT}" = "true" ]; then echo "--force_clean_exit"; fi)
    done

    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd "$RUNDIR"

done
