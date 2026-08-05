#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
export HF_AUDIO_DECODER_BACKEND="soundfile"

BATCH_SIZE=192
DEVICE_ID=0

# ── Models (comment / uncomment to select) ──────────────────────────────────
MODEL_IDs=(
    "nvidia/canary-qwen-2.5b"
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

        python run_eval_salm.py \
            --model_id=${MODEL_ID} \
            --dataset_path="hf-audio/open-asr-leaderboard" \
            --dataset="${DATASET}" \
            --split="${SPLIT}" \
            --device=${DEVICE_ID} \
            --batch_size=${BATCH_SIZE} \
            --max_eval_samples=-1
    done

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
