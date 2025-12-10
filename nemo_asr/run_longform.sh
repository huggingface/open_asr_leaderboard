#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

#considering latest model
MODEL_IDs=("nvidia/parakeet-tdt-1.1b" "nvidia/parakeet-rnnt-1.1b" "nvidia/parakeet-rnnt-0.6b" "nvidia/stt_en_fastconformer_transducer_large" "nvidia/stt_en_conformer_transducer_large" "nvidia/stt_en_conformer_transducer_small" "nvidia/parakeet-tdt-0.6b-v2")

# For CTC models:
# MODEL_IDs=("nvidia/parakeet-ctc-1.1b" "nvidia/parakeet-ctc-0.6b" "nvidia/stt_en_fastconformer_ctc_large" "nvidia/stt_en_conformer_ctc_large" "nvidia/stt_en_conformer_ctc_small")

BATCH_SIZE=1
DEVICE_ID=0

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}
    
    python run_eval_long.py \
    --model_id=${MODEL_ID} \
    --dataset_path="hf-audio/asr-leaderboard-longform" \
    --dataset="earnings21" \
    --split="test" \
    --device=0 \
    --batch_size=${BATCH_SIZE} \
    --max_eval_samples=-1 \
    --longform

    python run_eval_long.py \
    --model_id=${MODEL_ID} \
    --dataset_path="hf-audio/asr-leaderboard-longform" \
    --dataset="earnings22" \
    --split="test" \
    --device=0 \
    --batch_size=${BATCH_SIZE} \
    --max_eval_samples=-1 \
    --longform

    python run_eval_long.py \
    --model_id=${MODEL_ID} \
    --dataset_path="hf-audio/asr-leaderboard-longform" \
    --dataset="tedlium" \
    --split="test" \
    --device=0 \
    --batch_size=${BATCH_SIZE} \
    --max_eval_samples=-1 \
    --longform

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
