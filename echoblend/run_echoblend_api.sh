#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

API_URL="http://localhost:8000/transcribe"
BATCH_SIZE=64

# Check if API is running
if ! curl -s --fail "$API_URL" > /dev/null 2>&1; then
    echo "Error: API server is not running at $API_URL"
    echo "Please start the API server first"
    exit 1
fi

echo "Running evaluation against API at $API_URL"

python run_eval_api.py \
    --api_url="$API_URL" \
    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    --dataset="voxpopuli" \
    --split="test" \
    --batch_size=${BATCH_SIZE} \
    --max_eval_samples=-1

python run_eval_api.py \
    --api_url="$API_URL" \
    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    --dataset="ami" \
    --split="test" \
    --batch_size=${BATCH_SIZE} \
    --max_eval_samples=-1

python run_eval_api.py \
    --api_url="$API_URL" \
    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    --dataset="earnings22" \
    --split="test" \
    --batch_size=${BATCH_SIZE} \
    --max_eval_samples=-1

python run_eval_api.py \
    --api_url="$API_URL" \
    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    --dataset="gigaspeech" \
    --split="test" \
    --batch_size=${BATCH_SIZE} \
    --max_eval_samples=-1

python run_eval_api.py \
    --api_url="$API_URL" \
    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    --dataset="librispeech" \
    --split="test.clean" \
    --batch_size=${BATCH_SIZE} \
    --max_eval_samples=-1

python run_eval_api.py \
    --api_url="$API_URL" \
    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    --dataset="librispeech" \
    --split="test.other" \
    --batch_size=${BATCH_SIZE} \
    --max_eval_samples=-1

python run_eval_api.py \
    --api_url="$API_URL" \
    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    --dataset="spgispeech" \
    --split="test" \
    --batch_size=${BATCH_SIZE} \
    --max_eval_samples=-1

python run_eval_api.py \
    --api_url="$API_URL" \
    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    --dataset="tedlium" \
    --split="test" \
    --batch_size=${BATCH_SIZE} \
    --max_eval_samples=-1

# Evaluate results
RUNDIR=`pwd` && \
cd ../normalizer && \
python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', 'echoblend-api')" && \
cd $RUNDIR