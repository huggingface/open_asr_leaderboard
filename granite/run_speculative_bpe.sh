#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

MODEL_ID="ibm-granite/granite-speech-4.1-2b"
BATCH_SIZE=128
NUM_BEAMS=2
MAX_NEW_TOKENS=200

# Speculative decoding thresholds
CONFIDENCE_THRESHOLD=0.4
CTC_THRESHOLD=0.0

python run_eval_speculative_bpe.py \
    --model_id=${MODEL_ID} \
    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    --dataset="voxpopuli" \
    --split="test" \
    --device=0 \
    --batch_size=${BATCH_SIZE} \
    --num_beams=${NUM_BEAMS} \
    --max_eval_samples=-1 \
    --max_new_tokens=${MAX_NEW_TOKENS} \
    --confidence_threshold=${CONFIDENCE_THRESHOLD} \
    --ctc_threshold=${CTC_THRESHOLD}

python run_eval_speculative_bpe.py \
    --model_id=${MODEL_ID} \
    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    --dataset="ami" \
    --split="test" \
    --device=0 \
    --batch_size=${BATCH_SIZE} \
    --num_beams=${NUM_BEAMS} \
    --max_eval_samples=-1 \
    --max_new_tokens=${MAX_NEW_TOKENS} \
    --confidence_threshold=${CONFIDENCE_THRESHOLD} \
    --ctc_threshold=${CTC_THRESHOLD}

python run_eval_speculative_bpe.py \
    --model_id=${MODEL_ID} \
    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    --dataset="earnings22" \
    --split="test" \
    --device=0 \
    --batch_size=${BATCH_SIZE} \
    --num_beams=${NUM_BEAMS} \
    --max_eval_samples=-1 \
    --max_new_tokens=${MAX_NEW_TOKENS} \
    --confidence_threshold=${CONFIDENCE_THRESHOLD} \
    --ctc_threshold=${CTC_THRESHOLD}

python run_eval_speculative_bpe.py \
    --model_id=${MODEL_ID} \
    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    --dataset="gigaspeech" \
    --split="test" \
    --device=0 \
    --batch_size=${BATCH_SIZE} \
    --num_beams=${NUM_BEAMS} \
    --max_eval_samples=-1 \
    --max_new_tokens=${MAX_NEW_TOKENS} \
    --confidence_threshold=${CONFIDENCE_THRESHOLD} \
    --ctc_threshold=${CTC_THRESHOLD} 

python run_eval_speculative_bpe.py \
    --model_id=${MODEL_ID} \
    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    --dataset="librispeech" \
    --split="test.clean" \
    --device=0 \
    --batch_size=${BATCH_SIZE} \
    --num_beams=${NUM_BEAMS} \
    --max_eval_samples=-1 \
    --max_new_tokens=${MAX_NEW_TOKENS} \
    --confidence_threshold=${CONFIDENCE_THRESHOLD} \
    --ctc_threshold=${CTC_THRESHOLD} 

python run_eval_speculative_bpe.py \
    --model_id=${MODEL_ID} \
    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    --dataset="librispeech" \
    --split="test.other" \
    --device=0 \
    --batch_size=${BATCH_SIZE} \
    --num_beams=${NUM_BEAMS} \
    --max_eval_samples=-1 \
    --max_new_tokens=${MAX_NEW_TOKENS} \
    --confidence_threshold=${CONFIDENCE_THRESHOLD} \
    --ctc_threshold=${CTC_THRESHOLD} 

python run_eval_speculative_bpe.py \
    --model_id=${MODEL_ID} \
    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    --dataset="spgispeech" \
    --split="test" \
    --device=0 \
    --batch_size=${BATCH_SIZE} \
    --num_beams=${NUM_BEAMS} \
    --max_eval_samples=-1 \
    --max_new_tokens=${MAX_NEW_TOKENS} \
    --confidence_threshold=${CONFIDENCE_THRESHOLD} \
    --ctc_threshold=${CTC_THRESHOLD} 

python run_eval_speculative_bpe.py \
    --model_id=${MODEL_ID} \
    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    --dataset="tedlium" \
    --split="test" \
    --device=0 \
    --batch_size=${BATCH_SIZE} \
    --num_beams=${NUM_BEAMS} \
    --max_eval_samples=-1 \
    --max_new_tokens=${MAX_NEW_TOKENS} \
    --confidence_threshold=${CONFIDENCE_THRESHOLD} \
    --ctc_threshold=${CTC_THRESHOLD}

# Evaluate results
RUNDIR=`pwd` && \
cd ../normalizer && \
python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
cd $RUNDIR
