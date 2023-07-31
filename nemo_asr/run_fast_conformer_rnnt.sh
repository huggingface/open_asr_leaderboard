#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

MODEL_ID="nvidia/stt_en_conformer_transducer_large"

python run_eval.py \
	--model_id=$MODEL_ID \
	--dataset_path="librispeech_asr" \
	--dataset="other" \
	--split="test" \
	--device=0 \
	--batch_size=32 \
	--max_eval_samples=-1


python run_eval.py \
	--model_id=$MODEL_ID \
	--dataset_path="librispeech_asr" \
	--dataset="clean" \
	--split="test" \
	--device=0 \
	--batch_size=32 \
	--max_eval_samples=-1


# Evaluate results
RUNDIR=`pwd` && \
cd ../normalizer && \
python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
cd $RUNDIR
