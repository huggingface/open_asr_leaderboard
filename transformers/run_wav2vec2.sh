#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

MODEL_ID="facebook/wav2vec2-base-960h"

python run_eval.py \
	--model_id=$MODEL_ID \
	--dataset="librispeech_asr" \
	--split="test" \
	--device=0 


# Evaluate results
RUNDIR=`pwd` && \
cd ../normalizer && \
python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
cd $RUNDIR
