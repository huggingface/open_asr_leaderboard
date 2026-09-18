#!/bin/bash
set -e

export PYTHONPATH="..":$PYTHONPATH

DEFAULT_DATASET_PATH="${DEFAULT_DATASET_PATH:-hf-audio/open-asr-leaderboard}"
BATCH_SIZE=256
MAX_EVAL_SAMPLES=-1
WARMUP_STEPS=5
SUBBATCH_SAMPLES=30000000

# ── Models: "model_id revision" ──────────────────────────────────────────────
MODEL_CONFIGS=(
    "abr-ai/niagara-9m-batch.en 1521edf95a146d06e3c7c1ad18a7209a899bc570"
    "abr-ai/niagara-19m-batch.en d0276b85317389bc679d0206f60d88779cfbd15a"
    "abr-ai/niagara-38m-batch.en 7bfe48fb7fb065484419b1860c6b3d4e4f817c0c"
    "abr-ai/niagara-84m-batch.en ed93390475b146ad5412c668f680609a3ffca1b0"
)

# ── Datasets: "name split [dataset_path]" (comment / uncomment to select) ─────
# An entry that names its own repo passes no config name.
DATASET_CONFIGS=(
    "ami_cleaned test"
    "earnings22_cleaned_aa_chunked test ArtificialAnalysis/Earnings22-Cleaned-AA-chunked"
    "gigaspeech_cleaned test"
    "librispeech test.clean"
    "librispeech test.other"
    "spgispeech test"
    "voxpopuli_cleaned_aa test"
    "monsoon_en_in test VoiceArena/Monsoon_en_IN_test"
)

for model_cfg in "${MODEL_CONFIGS[@]}"; do
    read -r MODEL_ID REVISION <<< "$model_cfg"

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET SPLIT DATASET_PATH <<< "$cfg"
        if [[ -n "$DATASET_PATH" ]]; then
            DATASET_CONFIG=""
        else
            DATASET_PATH="$DEFAULT_DATASET_PATH"
            DATASET_CONFIG="$DATASET"
        fi

        python run_eval.py \
            --model_id=${MODEL_ID} \
            --revision=${REVISION} \
            --dataset_path="${DATASET_PATH}" \
            --dataset="${DATASET_CONFIG}" \
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
