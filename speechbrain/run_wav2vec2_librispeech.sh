#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

SOURCE="speechbrain/asr-wav2vec2-librispeech"

python run_eval.py \
	--source=$SOURCE \
    --speechbrain_pretrained_class_name="EncoderASR" \
	--dataset_path="librispeech_asr" \
	--dataset="clean" \
	--split="test" \
	--device=0 \
	--batch_size=32 \
	--max_eval_samples=-1

python run_eval.py \
	--source=$SOURCE \
    --speechbrain_pretrained_class_name="EncoderASR" \
	--dataset_path="librispeech_asr" \
	--dataset="other" \
	--split="test" \
	--device=0 \
	--batch_size=32 \
	--max_eval_samples=-1

# Evaluate results
RUNDIR=`pwd` && \
cd ../normalizer && \
python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
cd $RUNDIR