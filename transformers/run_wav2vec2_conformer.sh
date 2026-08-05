#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

BATCH_SIZE=24

# ── Models (comment / uncomment to select) ──────────────────────────────────
MODEL_IDs=(
    "facebook/wav2vec2-conformer-rel-pos-large-960h-ft"
    "facebook/wav2vec2-conformer-rope-large-960h-ft"
)

# ── Datasets: "name split" (comment / uncomment to select) ──────────────────
DATASET_CONFIGS=(
    "ami_cleaned test"
    "gigaspeech_cleaned test"
    "voxpopuli_cleaned_aa test"
    "earnings22 test"
    "librispeech test.clean"
    "librispeech test.other"
    "spgispeech test"
)

for MODEL_ID in "${MODEL_IDs[@]}"; do

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET SPLIT <<< "$cfg"

        python run_eval.py \
            --model_id=${MODEL_ID} \
            --dataset_path="hf-audio/open-asr-leaderboard" \
            --dataset="${DATASET}" \
            --split="${SPLIT}" \
            --device=0 \
            --batch_size=${BATCH_SIZE} \
            --max_eval_samples=-1 \
            --attn_implementation=eager
    done

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
