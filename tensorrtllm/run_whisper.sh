#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

download_model() {
    local MODEL_ID=$1
    local MODEL_TRT_LLM=${MODEL_ID}_tllm_checkpoint
    echo "Downloading $MODEL_ID from Hugging Face"
    mkdir -p $MODEL_TRT_LLM
    # Choose download source based on model ID prefix
    if [[ $MODEL_ID == distil* ]]; then
        huggingface-cli download --local-dir whisper-${MODEL_ID}-trt-llm-checkpoint Steveeeeeeen/${MODEL_ID}-trt-llm-checkpoint
    else
        huggingface-cli download --local-dir whisper-${MODEL_ID}-trt-llm-checkpoint yuekai/whisper-${MODEL_ID}-trt-llm-checkpoint
    fi
    wget -nc --directory-prefix=assets "$URL"
    wget -nc --directory-prefix=assets assets/mel_filters.npz https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz
    wget -nc --directory-prefix=assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken
    wget -nc --directory-prefix=assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/gpt2.tiktoken
}

build_model() {
    local model_id=$1
    local checkpoint_dir="whisper-${model_id}-trt-llm-checkpoint"
    local output_dir="whisper_${model_id}"

    local INFERENCE_PRECISION=float16
    local MAX_BEAM_WIDTH=4
    local MAX_BATCH_SIZE=256

    echo "Building encoder for model: $model_id"
    trtllm-build --checkpoint_dir "${checkpoint_dir}/encoder" \
                  --output_dir "${output_dir}/encoder" \
                  --moe_plugin disable \
                  --enable_xqa disable \
                  --max_batch_size "$MAX_BATCH_SIZE" \
                  --gemm_plugin disable \
                  --bert_attention_plugin "$INFERENCE_PRECISION" \
                  --max_input_len 3000 --max_seq_len 3000

    echo "Building decoder for model: $model_id"
    trtllm-build --checkpoint_dir "${checkpoint_dir}/decoder" \
                  --output_dir "${output_dir}/decoder" \
                  --moe_plugin disable \
                  --enable_xqa disable \
                  --max_beam_width "$MAX_BEAM_WIDTH" \
                  --max_batch_size "$MAX_BATCH_SIZE" \
                  --max_seq_len 114 \
                  --max_input_len 14 \
                  --max_encoder_input_len 3000 \
                  --gemm_plugin "$INFERENCE_PRECISION" \
                  --bert_attention_plugin "$INFERENCE_PRECISION" \
                  --gpt_attention_plugin "$INFERENCE_PRECISION"
}

MODEL_IDs=("large-v3-turbo" "large-v3" "large-v2" "large-v1" "medium" "base" "small" "tiny" "medium.en" "base.en" "small.en" "tiny.en" "distil-large-v3" "distil-large-v2" "distil-medium.en" "distil-small.en")
DEVICE_INDEX=0
BATCH_SIZE=64

num_models=${#MODEL_IDs[@]}

# pip install -r ../requirements/requirements_trtllm.txt

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}
    download_model $MODEL_ID
    build_model $MODEL_ID

    python3 run_eval.py \
        --model_id=whisper_${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="ami" \
        --split="test" \
        --device=${DEVICE_INDEX} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1

    python3 run_eval.py \
        --model_id=whisper_${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="earnings22" \
        --split="test" \
        --device=${DEVICE_INDEX} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1

    python3 run_eval.py \
        --model_id=whisper_${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="gigaspeech" \
        --split="test" \
        --device=${DEVICE_INDEX} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1

    python3 run_eval.py \
        --model_id=whisper_${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="librispeech" \
        --split="test.clean" \
        --device=${DEVICE_INDEX} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1

    python3 run_eval.py \
        --model_id=whisper_${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="librispeech" \
        --split="test.other" \
        --device=${DEVICE_INDEX} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1

    python3 run_eval.py \
        --model_id=whisper_${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="spgispeech" \
        --split="test" \
        --device=${DEVICE_INDEX} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1

    python3 run_eval.py \
        --model_id=whisper_${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="tedlium" \
        --split="test" \
        --device=${DEVICE_INDEX} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1

    python3 run_eval.py \
        --model_id=whisper_${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="voxpopuli" \
        --split="test" \
        --device=${DEVICE_INDEX} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python3 -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" > log_${MODEL_ID}.txt && \
    cd $RUNDIR

done
