#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

BATCH_SIZE=160
NUM_BEAMS=1
MAX_NEW_TOKENS=128
USER_PROMPT="Transcribe the audio clip into text."

# ── Models (comment / uncomment to select) ──────────────────────────────────
MODEL_IDs=(
    "microsoft/Phi-4-multimodal-instruct"
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
            --num_beams=${NUM_BEAMS} \
            --max_eval_samples=-1 \
            --max_new_tokens=${MAX_NEW_TOKENS} \
            --user_prompt="${USER_PROMPT}"
    done

    # Evaluate results
    RUNDIR=$(pwd) && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
