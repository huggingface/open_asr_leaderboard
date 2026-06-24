#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

BATCH_SIZE=32

# ── Models (comment / uncomment to select) ──────────────────────────────────
SOURCES=(
    "speechbrain/asr-wav2vec2-librispeech"
)

# ── Datasets: "name split" (comment / uncomment to select) ──────────────────
DATASET_CONFIGS=(
    "voxpopuli test"
    "ami test"
    "earnings22 test"
    "gigaspeech test"
    "librispeech test.clean"
    "librispeech test.other"
    "spgispeech test"
)

for SOURCE in "${SOURCES[@]}"; do

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET SPLIT <<< "$cfg"

        python run_eval.py \
            --source=${SOURCE} \
            --speechbrain_pretrained_class_name="EncoderASR" \
            --dataset_path="hf-audio/open-asr-leaderboard" \
            --dataset="${DATASET}" \
            --split="${SPLIT}" \
            --device=0 \
            --batch_size=${BATCH_SIZE} \
            --max_eval_samples=-1
    done

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${SOURCE}')" && \
    cd $RUNDIR

done
