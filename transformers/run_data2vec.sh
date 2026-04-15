#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

MODEL_IDs=("facebook/data2vec-audio-large-960h" "facebook/data2vec-audio-base-960h")
BATCH_SIZE=8

DATASETS=("voxpopuli" "ami" "earnings22" "gigaspeech" "librispeech" "librispeech" "spgispeech" "tedlium")
SPLITS=(   "test"      "test" "test"      "test"       "test.clean"  "test.other"  "test"       "test")

num_models=${#MODEL_IDs[@]}
num_datasets=${#DATASETS[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}

    for (( j=0; j<${num_datasets}; j++ ));
    do
        python run_eval.py \
            --model_id=${MODEL_ID} \
            --dataset_path="hf-audio/open-asr-leaderboard" \
            --dataset="${DATASETS[$j]}" \
            --split="${SPLITS[$j]}" \
            --device=0 \
            --batch_size=${BATCH_SIZE} \
            --max_eval_samples=-1
    done

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
