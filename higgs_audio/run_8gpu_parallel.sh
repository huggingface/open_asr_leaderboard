#!/bin/bash
# Launch 8-way parallel Open-ASR-Leaderboard eval of bosonai/higgs-audio-v3-8b-stt.
# Each GPU processes a specific (dataset, shard) assignment so the whole run
# tops out at the slowest GPU.  Per-sample checkpoints live under
# results/hf_leaderboard_5p30/ so an unexpected shutdown loses at most ~2 min.
#
# Expected per-dataset WER targets (from data.md, idea-0524ba):
#   AMI 6.23  E22 11.33  GS 9.34  LS-C 1.24  LS-O 2.34
#   SPG 3.14  TED 3.14   VP 5.63   →  avg 5.30

set -u
cd "$(dirname "$0")"

export PYTHONPATH=/ceph/workspace/peter/projects/boson-multimodal-ref:$PWD/..
: "${HF_TOKEN:?set HF_TOKEN to a valid HuggingFace token}"
export HF_HOME=/ceph/workspace/erik/auto-research/hf_hub_cache
export TRANSFORMERS_OFFLINE=0

MODEL_ID=bosonai/higgs-audio-v3-8b-stt
OUT=/ceph/workspace/erik/auto-research/results/hf_leaderboard_5p30
LOG=$OUT/logs
mkdir -p "$OUT" "$LOG"

run_one() {
    local gpu=$1 ds=$2 split=$3 shard=$4 nshards=$5
    local tag="${ds}_${split}_s${shard}of${nshards}"
    local log="$LOG/gpu${gpu}__${tag}.log"
    echo "[gpu $gpu] $tag -> $log"
    CUDA_VISIBLE_DEVICES=$gpu python run_eval.py \
        --model_id $MODEL_ID \
        --dataset $ds --split $split --device 0 \
        --shard_idx $shard --num_shards $nshards \
        --output_dir $OUT \
        >> "$log" 2>&1
}

# Each GPU runs a sequence of jobs; slots roughly balanced by total samples.
(
  run_one 0 ami test 0 1
) &
(
  run_one 1 earnings22 test 0 1
  run_one 1 voxpopuli test 0 1
) &
(
  run_one 2 gigaspeech test 0 2
) &
(
  run_one 3 gigaspeech test 1 2
) &
(
  run_one 4 librispeech test.clean 0 1
  run_one 4 librispeech test.other 0 1
) &
(
  run_one 5 spgispeech test 0 2
) &
(
  run_one 6 spgispeech test 1 2
) &
(
  run_one 7 tedlium test 0 1
) &

wait
echo "ALL GPUS DONE"
