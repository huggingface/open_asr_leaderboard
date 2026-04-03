#!/bin/bash
# Evaluate Higgs Audio v3 8B STT on all ESB datasets

MODELS=(
    "bosonai/higgs-audio-v3-8b-stt"
    "bosonai/higgs-audio-v3-stt"
)

DATASETS=(
    "ami"
    "earnings22"
    "gigaspeech"
    "librispeech_asr:test.clean"
    "librispeech_asr:test.other"
    "spgispeech"
    "tedlium"
    "voxpopuli"
)

export PYTHONPATH="..:$PYTHONPATH"

for MODEL in "${MODELS[@]}"; do
    echo "=== Evaluating $MODEL ==="
    for DS_ENTRY in "${DATASETS[@]}"; do
        IFS=':' read -r DATASET SPLIT <<< "$DS_ENTRY"
        SPLIT=${SPLIT:-test}
        echo "  Dataset: $DATASET ($SPLIT)"
        python run_eval_higgs_audio.py \
            --model_id "$MODEL" \
            --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
            --dataset "$DATASET" \
            --split "$SPLIT" \
            --device 0 \
            --batch_size 1 \
            --max_eval_samples -1
    done
    echo "=== Done: $MODEL ==="
done
