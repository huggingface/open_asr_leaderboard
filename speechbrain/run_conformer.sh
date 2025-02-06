#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

SOURCE="speechbrain/asr-conformer-largescaleasr"

# Run with CTC+Attn
python run_eval.py \
  --source=$SOURCE \
  --speechbrain_pretrained_class_name="EncoderDecoderASR" \
  --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
  --dataset="librispeech" \
  --split="test.clean" \
  --device=7 \
  --batch_size=4 \
  --max_eval_samples=-1 \
  --beam_size=40 \

# Run with Attn only
python run_eval.py \
  --source=$SOURCE \
  --speechbrain_pretrained_class_name="EncoderDecoderASR" \
  --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
  --dataset="librispeech" \
  --split="test.clean" \
  --device=7 \
  --batch_size=4 \
  --max_eval_samples=-1 \
  --beam_size=40 \
  --ctc_weight_decode=0

# Evaluate results
RUNDIR=`pwd` && \
cd ../normalizer && \
python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
cd $RUNDIR