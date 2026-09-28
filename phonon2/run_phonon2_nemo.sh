#!/bin/bash
# SUBMITTED PATH. Open ASR Leaderboard, English short-form: Phonon-2 through NVIDIA NeMo, exactly as the board runs the stock
# Parakeet rows (nemo_asr/run_eval.py: greedy_batch TDT = label-looping decoder + CUDA graphs, bf16, batch 128, timestamps off).
# The container is first expanded EXACTLY to a dense .nemo of the base architecture (nemo_export.py); the WER is that of the shipped
# bytes, the RTFx that of NeMo's decoder on the dense expansion. requirements: ../requirements/requirements_nemo.txt (+ zstd for the archive).
set -euo pipefail; export PYTHONPATH="..":${PYTHONPATH:-}
MODEL_ID=${MODEL_ID:-FermionResearch/Phonon-2}; BATCH_SIZE=${BATCH_SIZE:-128}; DEVICE_ID=${DEVICE_ID:-0}
NEMO=${NEMO:-export/phonon2_dense.nemo}; mkdir -p export
if [ ! -f "$NEMO" ]; then
  CONT=$(python -c "import sys; sys.path.insert(0,'.'); from run_eval import resolve_container; print(resolve_container('$MODEL_ID', None))")
  python nemo_export.py "$CONT" "$NEMO" nvidia/parakeet-tdt-0.6b-v3
fi
run(){ python ../nemo_asr/run_eval.py --model_id="$PWD/$NEMO" --dataset="$1" --split="$2" --dataset_path="${3:-hf-audio/open-asr-leaderboard}" --device=${DEVICE_ID} --batch_size=${BATCH_SIZE} --max_eval_samples=-1; }
run ami_cleaned test; run earnings22_cleaned_aa_chunked test ArtificialAnalysis/Earnings22-Cleaned-AA-chunked; run gigaspeech_cleaned test
run librispeech test.clean; run librispeech test.other; run spgispeech test; run voxpopuli_cleaned_aa test; run "" test VoiceArena/Monsoon_en_IN_test
python -c "import sys; sys.path.insert(0,'..'); from normalizer.eval_utils import score_results; score_results('results', '$PWD/$NEMO')"
