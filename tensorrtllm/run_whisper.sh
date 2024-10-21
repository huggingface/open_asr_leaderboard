#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH
# Download Models https://github.com/openai/whisper/blob/main/whisper/__init__.py#L17
declare -A MODELS=(
    ["large-v3"]="https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt"
    ["large-v3-turbo"]="https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt"
)

download_model() {
    local MODEL_ID=$1
    local URL=${MODELS[$MODEL_ID]}
    
    echo "Downloading $MODEL_ID from $URL..."
    wget -nc --directory-prefix=assets "$URL"
    wget -nc --directory-prefix=assets assets/mel_filters.npz https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz
    wget -nc --directory-prefix=assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken

}

build_model() {
    local model_id=$1
    local checkpoint_dir="${model_id}_tllm_checkpoint"
    local output_dir="whisper_${model_id}"
    echo "Converting checkpoint for model: $model_id"
    python3 convert_checkpoint.py \
        --output_dir "$checkpoint_dir" \
        --model_name "$model_id"

    local INFERENCE_PRECISION=float16
    local MAX_BEAM_WIDTH=4
    local MAX_BATCH_SIZE=64
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

MODEL_IDs=("large-v3-turbo" "large-v3")
DEVICE_INDEX=0

num_models=${#MODEL_IDs[@]}

pip install -r ../requirements/requirements_trtllm.txt

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}
    download_model $MODEL_ID
    build_model $MODEL_ID

    python run_eval.py \
        --model_id=whisper_${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="ami" \
        --split="test" \
        --device=${DEVICE_INDEX} \
        --max_eval_samples=-1

    python run_eval.py \
        --model_id=whisper_${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="earnings22" \
        --split="test" \
        --device=${DEVICE_INDEX} \
        --max_eval_samples=-1

    python run_eval.py \
        --model_id=whisper_${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="gigaspeech" \
        --split="test" \
        --device=${DEVICE_INDEX} \
        --max_eval_samples=-1

    python3 run_eval.py \
        --model_id=whisper_${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="librispeech" \
        --split="test.clean" \
        --device=${DEVICE_INDEX} \
        --max_eval_samples=-1

    python run_eval.py \
        --model_id=whisper_${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="librispeech" \
        --split="test.other" \
        --device=${DEVICE_INDEX} \
        --max_eval_samples=-1

    python run_eval.py \
        --model_id=whisper_${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="spgispeech" \
        --split="test" \
        --device=${DEVICE_INDEX} \
        --max_eval_samples=-1

    python3 run_eval.py \
        --model_id=whisper_${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="tedlium" \
        --split="test" \
        --device=${DEVICE_INDEX} \
        --max_eval_samples=-1

    python run_eval.py \
        --model_id=whisper_${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="voxpopuli" \
        --split="test" \
        --device=${DEVICE_INDEX} \
        --max_eval_samples=-1

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
