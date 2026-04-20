#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

BATCH_SIZE=64

# ── Models (comment / uncomment to select) ──────────────────────────────────
MODEL_IDs=(
    "openai/whisper-large-v3-turbo"
    # "openai/whisper-tiny.en"
    # "openai/whisper-small.en"
    # "openai/whisper-base.en"
    # "openai/whisper-medium.en"
    # "openai/whisper-large"
    # "openai/whisper-large-v2"
    # "openai/whisper-large-v3"
    # "distil-whisper/distil-medium.en"
    # "distil-whisper/distil-large-v2"
    # "distil-whisper/distil-large-v3"
    # "nyrahealth/CrisperWhisper"
)

# ── Datasets: "name split" (comment / uncomment to select) ──────────────────
DATASET_CONFIGS=(
    # "voxpopuli test"
    # "ami test"
    # "earnings22 test"
    # "gigaspeech test"
    "librispeech test.clean"
    # "librispeech test.other"
    # "spgispeech test"
    # "tedlium test"
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
            --max_eval_samples=-1
    done

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
