#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

# ── Models (comment / uncomment to select) ──────────────────────────────────
MODELS=(
    "bosonai/higgs-audio-v3-stt"
)

# Default batch size; per-dataset override allowed in DATASET_CONFIGS.
BATCH_SIZE=32

# ── Datasets: "name split [batch_size]" (comment / uncomment to select) ──────
# VoxPopuli has longer audio, so we use a smaller batch size to fit in VRAM.
DATASET_CONFIGS=(
    "voxpopuli test 16"
    "ami test"
    "earnings22 test"
    "gigaspeech test"
    "librispeech test.clean"
    "librispeech test.other"
    "spgispeech test"
    "tedlium test"
)

for MODEL in "${MODELS[@]}"; do
    echo "=== Evaluating $MODEL ==="

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET SPLIT BS <<< "$cfg"
        BS=${BS:-$BATCH_SIZE}

        python run_eval_higgs_audio.py \
            --model_id="${MODEL}" \
            --dataset_path="hf-audio/open-asr-leaderboard" \
            --dataset="${DATASET}" \
            --split="${SPLIT}" \
            --device=0 \
            --batch_size="${BS}" \
            --max_eval_samples=-1
    done

    # Evaluate results
    RUNDIR=$(pwd) && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL}')" && \
    cd "$RUNDIR"
done
