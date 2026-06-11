#!/bin/bash
# Runs all submit_jobs_*.sh scripts sequentially.
# Usage: HF_TOKEN=hf_... bash submit_all_jobs.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

scripts=(
    "submit_jobs_whisper.sh"
    "submit_jobs_crisper_whisper.sh"
    "submit_jobs_wav2vec2.sh"
    "submit_jobs_wav2vec2_conformer.sh"
    "submit_jobs_hubert.sh"
    "submit_jobs_data2vec.sh"
    "submit_jobs_mms.sh"
    "submit_jobs_moonshine.sh"
    "submit_jobs_voxtral.sh"
    "submit_jobs_voxtral_24b.sh"
    "submit_jobs_voxtral_realtime.sh"
    "submit_jobs_vibevoice.sh"
    "submit_jobs_glm_asr.sh"
    "submit_jobs_granite.sh"
    "submit_jobs_cohere.sh"
)

for script in "${scripts[@]}"; do
    echo ""
    echo "▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶"
    echo "  Running: ${script}"
    echo "▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶"
    bash "${SCRIPT_DIR}/${script}" &
done

wait

echo ""
echo "▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶"
echo "  All scripts completed."
echo "▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶"
