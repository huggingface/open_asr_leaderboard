#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

BATCH_SIZE=32
DEVICE_ID=0

# ── Models (comment / uncomment to select) ──────────────────────────────────
SOURCES=(
    "speechbrain/asr-conformer-largescaleasr"
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
    "tedlium test"
)

for SOURCE in "${SOURCES[@]}"; do

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET SPLIT <<< "$cfg"

        python run_eval.py \
            --source=${SOURCE} \
            --speechbrain_pretrained_class_name="EncoderDecoderASR" \
            --dataset_path="hf-audio/open-asr-leaderboard" \
            --dataset="${DATASET}" \
            --split="${SPLIT}" \
            --device=${DEVICE_ID} \
            --batch_size=${BATCH_SIZE} \
            --max_eval_samples=-1 \
            --beam_size=10 \
            --ctc_weight_decode=0
    done

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${SOURCE}')" && \
    cd $RUNDIR

done
