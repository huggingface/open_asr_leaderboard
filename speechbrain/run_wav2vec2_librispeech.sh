#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

SOURCE="speechbrain/asr-wav2vec2-librispeech"
BATCH_SIZE=32

python run_eval.py \
  --source=${SOURCE} \
  --speechbrain_pretrained_class_name="EncoderASR" \
  --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
  --dataset="ami" \
  --split="test" \
  --device=0 \
  --batch_size=${BATCH_SIZE} \
  --max_eval_samples=-1

python run_eval.py \
  --source=${SOURCE} \
  --speechbrain_pretrained_class_name="EncoderASR" \
  --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
  --dataset="earnings22" \
  --split="test" \
  --device=0 \
  --batch_size=${BATCH_SIZE} \
  --max_eval_samples=-1

python run_eval.py \
  --source=${SOURCE} \
  --speechbrain_pretrained_class_name="EncoderASR" \
  --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
  --dataset="gigaspeech" \
  --split="test" \
  --device=0 \
  --batch_size=${BATCH_SIZE} \
  --max_eval_samples=-1

python run_eval.py \
  --source=${SOURCE} \
  --speechbrain_pretrained_class_name="EncoderASR" \
  --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
  --dataset="librispeech" \
  --split="test.clean" \
  --device=0 \
  --batch_size=${BATCH_SIZE} \
  --max_eval_samples=-1

python run_eval.py \
  --source=${SOURCE} \
  --speechbrain_pretrained_class_name="EncoderASR" \
  --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
  --dataset="librispeech" \
  --split="test.other" \
  --device=0 \
  --batch_size=${BATCH_SIZE} \
  --max_eval_samples=-1

python run_eval.py \
  --source=${SOURCE} \
  --speechbrain_pretrained_class_name="EncoderASR" \
  --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
  --dataset="spgispeech" \
  --split="test" \
  --device=0 \
  --batch_size=${BATCH_SIZE} \
  --max_eval_samples=-1

python run_eval.py \
  --source=${SOURCE} \
  --speechbrain_pretrained_class_name="EncoderASR" \
  --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
  --dataset="tedlium" \
  --split="test" \
  --device=0 \
  --batch_size=${BATCH_SIZE} \
  --max_eval_samples=-1

python run_eval.py \
  --source=${SOURCE} \
  --speechbrain_pretrained_class_name="EncoderASR" \
  --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
  --dataset="voxpopuli" \
  --split="test" \
  --device=0 \
  --batch_size=${BATCH_SIZE} \
  --max_eval_samples=-1

# Evaluate results
RUNDIR=`pwd` && \
cd ../normalizer && \
python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
cd $RUNDIR