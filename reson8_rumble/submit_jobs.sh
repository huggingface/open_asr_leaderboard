#!/usr/bin/env bash
# One H200 job for Rumble; all selected datasets share the vLLM engines.
# Usage: HF_TOKEN=hf_... RESULTS_BUCKET=your/bucket bash reson8_rumble/submit_jobs.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/submit_utils.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE=vllm/vllm-openai:v0.30.0@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90
MODEL_ID=Reson8/Rumble-2-Flash
MODEL_FOLDER="${MODEL_ID//\//-}"
RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_h200}"
ORG_NAME="${ORG_NAME:-}"
: "${HF_TOKEN:?Set HF_TOKEN to a token with model/dataset, Jobs and results-bucket access}"
NUM_ENGINES="${NUM_ENGINES:-4}"

# Datasets: "name split [dataset_path [config [language]]]"
# Without a path, name selects a config in hf-audio/open-asr-leaderboard.
# With a path, config defaults to the repository default; "-" also selects it.
# Language defaults to en. Config and language are explicit for multilingual sets.
DATASET_CONFIGS=(
    "ami_cleaned test"
    "gigaspeech_cleaned test"
    "voxpopuli_cleaned_aa test"
    "earnings22_cleaned_aa_chunked test ArtificialAnalysis/Earnings22-Cleaned-AA-chunked"
    "spgispeech test"
    "urgent2024 test"
    "urgent2024_clean test"
    "monsoon_en_in test VoiceArena/Monsoon_en_IN_test"

    # Multilingual leaderboard datasets.
    "fleurs_de test hf-audio/open-asr-leaderboard-multilingual-datasets fleurs_de de"
    "fleurs_fr test hf-audio/open-asr-leaderboard-multilingual-datasets fleurs_fr fr"
    "fleurs_it test hf-audio/open-asr-leaderboard-multilingual-datasets fleurs_it it"
    "fleurs_es test hf-audio/open-asr-leaderboard-multilingual-datasets fleurs_es es"
    "fleurs_pt test hf-audio/open-asr-leaderboard-multilingual-datasets fleurs_pt pt"
    "fleurs_nl test hf-audio/open-asr-leaderboard-multilingual-datasets fleurs_nl nl"
    "mcv_de test hf-audio/open-asr-leaderboard-multilingual-datasets mcv_de de"
    "mcv_es test hf-audio/open-asr-leaderboard-multilingual-datasets mcv_es es"
    "mcv_fr test hf-audio/open-asr-leaderboard-multilingual-datasets mcv_fr fr"
    "mcv_it test hf-audio/open-asr-leaderboard-multilingual-datasets mcv_it it"
    "mcv_nl test hf-audio/open-asr-leaderboard-multilingual-datasets mcv_nl nl"
    "mls_es test hf-audio/open-asr-leaderboard-multilingual-datasets mls_es es"
    "mls_fr test hf-audio/open-asr-leaderboard-multilingual-datasets mls_fr fr"
    "mls_it test hf-audio/open-asr-leaderboard-multilingual-datasets mls_it it"
    "mls_pt test hf-audio/open-asr-leaderboard-multilingual-datasets mls_pt pt"
    "mls_nl test hf-audio/open-asr-leaderboard-multilingual-datasets mls_nl nl"
    "monsoon_hi test VoiceArena/Monsoon_hi_test - hi"
)
# Optional: restrict this run to specific datasets, matched against the first
# field of each DATASET_CONFIGS entry, e.g.:
#   ONLY_DATASETS="monsoon_en_in" bash <this script>
#   ONLY_DATASETS="ami_cleaned spgispeech" bash <this script>
filter_only_datasets 1 || exit 1


# The upstream image has no evaluator, so always send the local runtime files.
LOCAL_SCRIPT_INJECT=$(USE_LOCAL_SCRIPT=1 local_script_inject "$SCRIPT_DIR/.." \
    reson8_rumble/run_eval.py \
    reson8_rumble/run_rumble.sh \
    reson8_rumble/serve_vllm.sh \
    requirements/requirements_jobs.txt \
    requirements/requirements_reson8_rumble.txt)
LOCAL_NORMALIZER_INJECT=$(USE_LOCAL_NORMALIZER=1 local_normalizer_inject)
DATASET_SPEC=$(printf '%s\n' "${DATASET_CONFIGS[@]}")

namespace_args=()
if [[ -n "$ORG_NAME" ]]; then
    namespace_args=(--namespace "$ORG_NAME")
fi
echo "Submitting one H200 job for ${#DATASET_CONFIGS[@]} datasets with $NUM_ENGINES engines"
hf jobs run \
    --flavor h200 --timeout 8h --secrets HF_TOKEN ${namespace_args[@]+"${namespace_args[@]}"} \
    --volume "hf://buckets/${RESULTS_BUCKET}/${MODEL_FOLDER}:/app/results:rw" \
    "$IMAGE" -- bash -c "
        set -euo pipefail
        mkdir -p /app/reson8_rumble /app/requirements
        cd /app
        ${LOCAL_SCRIPT_INJECT}
        ${LOCAL_NORMALIZER_INJECT}
        exec bash reson8_rumble/run_rumble.sh \"\$@\"
    " bash "$DATASET_SPEC" "$NUM_ENGINES"

# A non-detached hf jobs run waits and fails if the remote job fails.
sleep 10  # allow the final bucket writes to become visible
mkdir -p "./results/${MODEL_FOLDER}"
python "$FETCH_RUN_RESULTS" \
    --bucket "$RESULTS_BUCKET" --model-folder "$MODEL_FOLDER" \
    --local-dir "./results/${MODEL_FOLDER}" --since "$RUN_START" \
    --expected "${#DATASET_CONFIGS[@]}"
