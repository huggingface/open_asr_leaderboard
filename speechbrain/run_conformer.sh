#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

SOURCE="speechbrain/asr-conformer-largescaleasr"
BATCH_SIZE=32
DEVICE_ID=0

# Run with CTC+Attn
python run_eval.py \
  --source=$SOURCE \
  --speechbrain_pretrained_class_name="EncoderDecoderASR" \
  --dataset_path="hf-audio/open-asr-leaderboard" \
  --dataset="librispeech" \
  --split="test.clean" \
  --device=${DEVICE_ID} \
  --batch_size=${BATCH_SIZE} \
  --max_eval_samples=-1 \
  --no-streaming \
  --beam_size=10 \
  --ctc_weight_decode=0

python run_eval.py \
  --source=$SOURCE \
  --speechbrain_pretrained_class_name="EncoderDecoderASR" \
  --dataset_path="hf-audio/open-asr-leaderboard" \
  --dataset="librispeech" \
  --split="test.other" \
  --device=${DEVICE_ID} \
  --batch_size=${BATCH_SIZE} \
  --max_eval_samples=-1 \
  --beam_size=10 \
  --ctc_weight_decode=0

python run_eval.py \
  --source=$SOURCE \
  --speechbrain_pretrained_class_name="EncoderDecoderASR" \
  --dataset_path="hf-audio/open-asr-leaderboard" \
  --dataset="ami" \
  --split="test" \
  --device=${DEVICE_ID} \
  --batch_size=${BATCH_SIZE} \
  --max_eval_samples=-1 \
  --beam_size=10 \

python run_eval.py \
  --source=$SOURCE \
  --speechbrain_pretrained_class_name="EncoderDecoderASR" \
  --dataset_path="hf-audio/open-asr-leaderboard" \
  --dataset="spgispeech" \
  --split="test" \
  --device=${DEVICE_ID} \
  --batch_size=${BATCH_SIZE} \
  --max_eval_samples=-1 \
  --beam_size=10 \

python run_eval.py \
  --source=$SOURCE \
  --speechbrain_pretrained_class_name="EncoderDecoderASR" \
  --dataset_path="hf-audio/open-asr-leaderboard" \
  --dataset="tedlium" \
  --split="test" \
  --device=${DEVICE_ID} \
  --batch_size=${BATCH_SIZE} \
  --max_eval_samples=-1 \
  --beam_size=10 \

python run_eval.py \
  --source=$SOURCE \
  --speechbrain_pretrained_class_name="EncoderDecoderASR" \
  --dataset_path="hf-audio/open-asr-leaderboard" \
  --dataset="earnings22" \
  --split="test" \
  --device=${DEVICE_ID} \
  --batch_size=${BATCH_SIZE} \
  --max_eval_samples=-1 \
  --beam_size=10 \

python run_eval.py \
  --source=$SOURCE \
  --speechbrain_pretrained_class_name="EncoderDecoderASR" \
  --dataset_path="hf-audio/open-asr-leaderboard" \
  --dataset="gigaspeech" \
  --split="test" \
  --device=${DEVICE_ID} \
  --batch_size=${BATCH_SIZE} \
  --max_eval_samples=-1 \
  --beam_size=10 \

python run_eval.py \
  --source=$SOURCE \
  --speechbrain_pretrained_class_name="EncoderDecoderASR" \
  --dataset_path="hf-audio/open-asr-leaderboard" \
  --dataset="voxpopuli" \
  --split="test" \
  --device=${DEVICE_ID} \
  --batch_size=${BATCH_SIZE} \
  --max_eval_samples=-1 \
  --beam_size=10 \

# Evaluate results
RUNDIR=`pwd` && \
cd ../normalizer && \
python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
cd $RUNDIR
