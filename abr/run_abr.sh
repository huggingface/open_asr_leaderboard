#!/bin/bash
set -e

export PYTHONPATH="..":$PYTHONPATH

BATCH_SIZE=256
MAX_EVAL_SAMPLES=-1
WARMUP_STEPS=5
SUBBATCH_SAMPLES=30000000
REVISION="dab6545337495482f2fc05455432a7a05c88d3cc"

# ── Models (comment / uncomment to select) ──────────────────────────────────
MODEL_IDs=(
    "abr-ai/niagara-19m-batch.en"
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

        python run_eval.py \
            --model_id=${MODEL_ID} \
            --revision=${REVISION} \
            --dataset_path="hf-audio/open-asr-leaderboard" \
            --dataset="${DATASET}" \
            --split="${SPLIT}" \
            --batch_size=${BATCH_SIZE} \
            --warmup_steps=${WARMUP_STEPS} \
            --subbatch_samples=${SUBBATCH_SAMPLES} \
            --max_eval_samples=${MAX_EVAL_SAMPLES}
    done

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
