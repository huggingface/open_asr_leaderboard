#!/bin/bash
# HF Jobs submission for Phonon-2 (pattern of this repo's submit_jobs scripts): one job per dataset on an H200,
# results synced to a bucket, CSV summary printed. SPACE = a duplicate of hf-audio/open-asr-leaderboard-transformers
# with this folder's Dockerfile + run_eval.py. Set the variables below before running.
set -euo pipefail
SPACE=${SPACE:-FermionResearch/open-asr-leaderboard-phonon2}
BUCKET=${BUCKET:-hf://datasets/FermionResearch/open-asr-leaderboard-phonon2-results}
MODEL_ID=${MODEL_ID:-FermionResearch/Phonon-2}
FLAVOR=${FLAVOR:-h200}
DATASETS=("ami_cleaned test" "earnings22_cleaned_aa_chunked test ArtificialAnalysis/Earnings22-Cleaned-AA-chunked" "gigaspeech_cleaned test" "librispeech test.clean" "librispeech test.other" "spgispeech test" "voxpopuli_cleaned_aa test" " test VoiceArena/Monsoon_en_IN_test")
for cfg in "${DATASETS[@]}"; do
  read -r DS SPLIT DSPATH <<< "$cfg"
  hf jobs run --flavor "$FLAVOR" --secrets HF_TOKEN --detach "$SPACE" \
    bash -c "cd /app/phonon2 && PYTHONPATH=/app python run_eval.py --model_id=$MODEL_ID --dataset=$DS --split=$SPLIT --dataset_path=${DSPATH:-hf-audio/open-asr-leaderboard} --device=0 --batch_size=64 --dtype=bfloat16 --warmup_steps=5 --max_eval_samples=-1 && hf upload $BUCKET results results/"
done
echo "submitted ${#DATASETS[@]} jobs; sync with: hf download $BUCKET --repo-type dataset --local-dir results_synced && python -c \"from normalizer.eval_utils import score_results; score_results('results_synced/results', '$MODEL_ID')\""
