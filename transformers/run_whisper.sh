#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

MODEL_ID="openai/whisper-tiny"

python run_eval.py \
	--model_id=$MODEL_ID \
	--dataset_path="librispeech_asr" \
	--dataset="other" \
	--split="test" \
	--batch_size=32 \
	--device=0 


# Evaluate results
RUNDIR=`pwd` && \
cd ../normalizer && \
python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
cd $RUNDIR
