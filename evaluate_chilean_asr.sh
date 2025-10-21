#!/bin/bash

# Batch evaluation script for Chilean Spanish ASR models
# Dataset: astroza/es-cl-asr-test-only
# Metrics: WER (Word Error Rate) and RTFx (Real-Time Factor)

set -e

DATASET_PATH="astroza"
DATASET="es-cl-asr-test-only"
SPLIT="test"
DEVICE=${DEVICE:-0}  # GPU device, default to 0

echo "=================================="
echo "Chilean Spanish ASR Batch Evaluation"
echo "Dataset: ${DATASET_PATH}/${DATASET}"
echo "Split: ${SPLIT}"
echo "Device: ${DEVICE}"
echo "=================================="
echo ""

# Create results directory
mkdir -p results

# Function to run Whisper models (transformers framework)
run_whisper_model() {
    local model_id=$1
    local batch_size=${2:-16}
    echo ">>> Evaluating Whisper model: ${model_id}"
    python transformers/run_eval.py \
        --model_id "${model_id}" \
        --dataset_path "${DATASET_PATH}" \
        --dataset "${DATASET}" \
        --split "${SPLIT}" \
        --device ${DEVICE} \
        --batch_size ${batch_size} \
        --no-streaming \
        --max_new_tokens 128 \
        --warmup_steps 5
    echo ""
}

# Function to run NeMo models
run_nemo_model() {
    local model_id=$1
    local batch_size=${2:-32}
    echo ">>> Evaluating NeMo model: ${model_id}"
    python nemo_asr/run_eval.py \
        --model_id "${model_id}" \
        --dataset_path "${DATASET_PATH}" \
        --dataset "${DATASET}" \
        --split "${SPLIT}" \
        --device ${DEVICE} \
        --batch_size ${batch_size} \
        --no-streaming
    echo ""
}

# Function to run Phi-4 model
run_phi4_model() {
    local model_id=$1
    local batch_size=${2:-4}
    echo ">>> Evaluating Phi-4 model: ${model_id}"
    python phi/run_eval.py \
        --model_id "${model_id}" \
        --dataset_path "${DATASET_PATH}" \
        --dataset "${DATASET}" \
        --split "${SPLIT}" \
        --device ${DEVICE} \
        --batch_size ${batch_size} \
        --no-streaming \
        --max_new_tokens 128 \
        --warmup_steps 2 \
        --user_prompt "Transcribe el audio a texto en español."
    echo ""
}

# Function to run API-based models
run_api_model() {
    local model_name=$1
    local max_workers=${2:-50}
    echo ">>> Evaluating API model: ${model_name}"
    python api/run_eval.py \
        --dataset_path "${DATASET_PATH}" \
        --dataset "${DATASET}" \
        --split "${SPLIT}" \
        --model_name "${model_name}" \
        --max_workers ${max_workers}
    echo ""
}

echo "Starting evaluation of 7 ASR models..."
echo ""

# 1. OpenAI Whisper Large V3
run_whisper_model "openai/whisper-large-v3" 16

# 2. OpenAI Whisper Large V3 Turbo
run_whisper_model "openai/whisper-large-v3-turbo" 16

# 3. NVIDIA Canary 1B V2
run_nemo_model "nvidia/canary-1b-v2" 32

# 4. NVIDIA Parakeet TDT 0.6B V3
run_nemo_model "nvidia/parakeet-tdt-0.6b-v3" 32

# 5. Microsoft Phi-4 Multimodal Instruct
run_phi4_model "microsoft/Phi-4-multimodal-instruct" 4

# 6. Mistral Voxtral Mini 3B
# Note: Voxtral might need special handling. Check if it works with transformers run_eval
echo ">>> Evaluating Voxtral model: mistralai/Voxtral-Mini-3B-2507"
echo "NOTE: If this model requires special handling, please check the model documentation"
python transformers/run_eval.py \
    --model_id "mistralai/Voxtral-Mini-3B-2507" \
    --dataset_path "${DATASET_PATH}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --device ${DEVICE} \
    --batch_size 8 \
    --no-streaming \
    --max_new_tokens 128 \
    --warmup_steps 5 || echo "WARNING: Voxtral evaluation failed. May need custom implementation."
echo ""

# 7. ElevenLabs Scribe V1 (API-based)
# Note: Requires ELEVENLABS_API_KEY in .env file
if [ -z "${ELEVENLABS_API_KEY}" ] && [ ! -f .env ]; then
    echo "WARNING: ELEVENLABS_API_KEY not found. Skipping ElevenLabs Scribe V1"
    echo "To evaluate this model, create a .env file with your API key:"
    echo "ELEVENLABS_API_KEY=your_key_here"
else
    run_api_model "elevenlabs/scribe_v1" 50
fi

echo ""
echo "=================================="
echo "Evaluation complete!"
echo "Results saved in: ./results/"
echo "=================================="
echo ""
echo "To view WER and RTFx metrics for all models:"
echo "ls -lh results/*.jsonl"
