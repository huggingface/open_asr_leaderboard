#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

MODEL_IDs=(
    "zai-org/GLM-ASR-Nano-2512"
)

BATCH_SIZE=64
MAX_NEW_TOKENS=500

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
            --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
            --dataset="${DATASETS[$j]}" \
            --split="${SPLITS[$j]}" \
            --device=0 \
            --batch_size=${BATCH_SIZE} \
            --max_eval_samples=-1 \
            --max_new_tokens=${MAX_NEW_TOKENS}
    done

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
