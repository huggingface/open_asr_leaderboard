#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

BATCH_SIZE=256
NUM_BEAMS=2
MAX_NEW_TOKENS=200
CONFIDENCE_THRESHOLD=0.2
CTC_THRESHOLD=0.7

# ── Models (comment / uncomment to select) ──────────────────────────────────
MODEL_IDs=(
    "ibm-granite/granite-4.0-1b-speech"
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

for MODEL_ID in "${MODEL_IDs[@]}"; do

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET SPLIT <<< "$cfg"

        python run_eval_speculative.py \
            --model_id=${MODEL_ID} \
            --dataset_path="hf-audio/open-asr-leaderboard" \
            --dataset="${DATASET}" \
            --split="${SPLIT}" \
            --device=0 \
            --batch_size=${BATCH_SIZE} \
            --num_beams=${NUM_BEAMS} \
            --max_eval_samples=-1 \
            --max_new_tokens=${MAX_NEW_TOKENS} \
            --confidence_threshold=${CONFIDENCE_THRESHOLD} \
            --ctc_threshold=${CTC_THRESHOLD}
    done

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
