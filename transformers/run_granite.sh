#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

MAX_NEW_TOKENS=200

# ── Models + batch sizes (paired arrays) ────────────────────────────────────
MODEL_IDs=(
    "ibm-granite/granite-speech-3.3-2b"
    "ibm-granite/granite-speech-3.3-8b"
)
BATCH_SIZEs=(
    160
    64
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

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ )); do
    MODEL_ID=${MODEL_IDs[$i]}
    BATCH_SIZE=${BATCH_SIZEs[$i]}

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
            --attn_implementation="eager" \
            --max_new_tokens=${MAX_NEW_TOKENS}
    done

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
