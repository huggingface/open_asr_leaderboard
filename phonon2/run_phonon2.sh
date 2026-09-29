#!/bin/bash
# Open ASR Leaderboard, English short-form: Phonon-2 via the dense transformers reference path.
# Same decoding hyper-parameters on every dataset (greedy TDT, bf16 activations, batch 64).
export PYTHONPATH="..":$PYTHONPATH
MODEL_ID=${MODEL_ID:-FermionResearch/Phonon-2}
BATCH_SIZE=${BATCH_SIZE:-64}
DEVICE_ID=${DEVICE_ID:-0}
run() {  # dataset split [dataset_path]
  python run_eval.py --model_id="${MODEL_ID}" --dataset_path="${3:-hf-audio/open-asr-leaderboard}" --dataset="$1" --split="$2" \
    --device=${DEVICE_ID} --batch_size=${BATCH_SIZE} --dtype=bfloat16 --warmup_steps=5 --max_eval_samples=-1
}
run ami_cleaned test
run earnings22_cleaned_aa_chunked test ArtificialAnalysis/Earnings22-Cleaned-AA-chunked
run gigaspeech_cleaned test
run librispeech test.clean
run librispeech test.other
run spgispeech test
run voxpopuli_cleaned_aa test
run "" test VoiceArena/Monsoon_en_IN_test
# Evaluate results
RUNDIR=$(dirname $0)
python -c "import sys; sys.path.insert(0, '${RUNDIR}/..'); from normalizer import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')"
