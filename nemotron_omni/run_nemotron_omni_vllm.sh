#!/bin/bash
# Drive the full Open ASR Leaderboard sweep against a running vLLM server
# (start it first with ./run_server.sh).

export PYTHONPATH="..":$PYTHONPATH
export TOKENIZERS_PARALLELISM=false

MODEL_ID="${MODEL_ID:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
API_KEY="${API_KEY:-EMPTY}"
BATCH_SIZE="${BATCH_SIZE:-256}"
CONCURRENCY="${CONCURRENCY:-256}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
USER_PROMPT="${USER_PROMPT:-Transcribe the audio clip into text. Return only the transcription.}"

COMMON_ARGS=(
    --model_id="${MODEL_ID}"
    --base_url="${BASE_URL}"
    --api_key="${API_KEY}"
    --dataset_path="hf-audio/esb-datasets-test-only-sorted"
    --max_eval_samples=-1
    --max_new_tokens=${MAX_NEW_TOKENS}
    --warmup_steps=${WARMUP_STEPS}
    --concurrency=${CONCURRENCY}
    --user_prompt="${USER_PROMPT}"
)

python run_eval_vllm.py "${COMMON_ARGS[@]}" --batch_size=${BATCH_SIZE} --dataset="voxpopuli"   --split="test"
python run_eval_vllm.py "${COMMON_ARGS[@]}" --batch_size=${BATCH_SIZE} --dataset="ami"         --split="test"
python run_eval_vllm.py "${COMMON_ARGS[@]}" --batch_size=${BATCH_SIZE} --dataset="earnings22"  --split="test"
python run_eval_vllm.py "${COMMON_ARGS[@]}" --batch_size=${BATCH_SIZE} --dataset="gigaspeech"  --split="test"
python run_eval_vllm.py "${COMMON_ARGS[@]}" --batch_size=${BATCH_SIZE} --dataset="librispeech" --split="test.clean"
python run_eval_vllm.py "${COMMON_ARGS[@]}" --batch_size=${BATCH_SIZE} --dataset="librispeech" --split="test.other"
python run_eval_vllm.py "${COMMON_ARGS[@]}" --batch_size=${BATCH_SIZE} --dataset="spgispeech"  --split="test"
python run_eval_vllm.py "${COMMON_ARGS[@]}" --batch_size=${BATCH_SIZE} --dataset="tedlium"     --split="test"

RUNDIR=$(pwd) && \
cd ../normalizer && \
python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
cd "${RUNDIR}"
