#!/bin/bash
set -e

export PYTHONPATH="..":$PYTHONPATH

BATCH_SIZE=256
MAX_EVAL_SAMPLES=-1
WARMUP_STEPS=5
SUBBATCH_SAMPLES=30000000

# ── Models: "model_id revision" ──────────────────────────────────────────────
MODEL_CONFIGS=(
    "abr-ai/niagara-19m-batch.en dab6545337495482f2fc05455432a7a05c88d3cc"
    "abr-ai/niagara-38m-batch.en 4f3ec18d377b1fd01e94d15dc9b9db0a8cd74bd2"
)

# ── Datasets: "name split" (comment / uncomment to select) ──────────────────
DATASET_CONFIGS=(
    "ami_cleaned test"
    "earnings22 test"
    "gigaspeech_cleaned test"
    "librispeech test.clean"
    "librispeech test.other"
    "spgispeech test"
    "voxpopuli_cleaned_aa test"
)

for model_cfg in "${MODEL_CONFIGS[@]}"; do
    read -r MODEL_ID REVISION <<< "$model_cfg"

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
    RUNDIR=$(pwd)
    PYTHONPATH="${RUNDIR}/..:${PYTHONPATH}" python -c "from normalizer.eval_utils import score_results; score_results('${RUNDIR}/results', '${MODEL_ID}')"

done
