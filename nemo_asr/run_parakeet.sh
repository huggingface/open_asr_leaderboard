#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

BATCH_SIZE=128
DEVICE_ID=0

# ── Models (comment / uncomment to select) ──────────────────────────────────
MODEL_IDs=(
    # RNNT models:
    "nvidia/parakeet-tdt-0.6b-v3"
    "nvidia/parakeet-tdt-0.6b-v2"
    "nvidia/parakeet-tdt-1.1b"
    "nvidia/parakeet-rnnt-1.1b"
    "nvidia/parakeet-rnnt-0.6b"
    "nvidia/stt_en_fastconformer_transducer_large"
    "nvidia/stt_en_conformer_transducer_large"
    "stt_en_conformer_transducer_small"
    # CTC models:
    "nvidia/parakeet-ctc-1.1b"
    "nvidia/parakeet-ctc-0.6b"
    "nvidia/stt_en_fastconformer_ctc_large"
    "nvidia/stt_en_conformer_ctc_large"
    "nvidia/stt_en_conformer_ctc_small"
    # Hybrid
    "nvidia/parakeet-tdt_ctc-110m"
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
