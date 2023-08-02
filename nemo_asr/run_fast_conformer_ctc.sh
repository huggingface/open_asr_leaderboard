#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

#considering FC-XL, FC-XXL CTC models
MODEL_IDs=("nvidia/stt_en_fastconformer_ctc_xlarge" "nvidia/stt_en_fastconformer_ctc_xxlarge")
BATCH_SIZE=8

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="librispeech_asr" \
        --dataset="other" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1


    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="librispeech_asr" \
        --dataset="clean" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
