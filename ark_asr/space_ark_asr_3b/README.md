---
title: Open ASR Leaderboard ARK-ASR-3B
emoji: 🎙️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: apache-2.0
short_description: HF Jobs runner for ARK-ASR-3B
---

# Open ASR Leaderboard ARK-ASR-3B

This Space is duplicated from `AutoArk-AI/open-asr-leaderboard-ark-asr` and
adapted as the HF Jobs runner for evaluating `AutoArk-AI/ARK-ASR-3B` on
`hf-audio/open-asr-leaderboard`.

HF Jobs should select hardware with `hf jobs run --flavor h200`; the Space
default runtime hardware is not used for leaderboard scoring.

## HF Jobs Example

```bash
PYTHONPATH=/app python /app/run_eval.py \
  --model_id=AutoArk-AI/ARK-ASR-3B \
  --dataset_path=hf-audio/open-asr-leaderboard \
  --dataset=ami \
  --split=test \
  --device=0 \
  --batch_size=64 \
  --max_eval_samples=-1 \
  --max_new_tokens=256 \
  --warmup_steps=0 \
  --dtype=float16 \
  --attn_impl=sdpa \
  --audio_input=array \
  --audio_decode=soundfile \
  --force_clean_exit
```

Use `ark_asr/submit_jobs_ark_asr_3b.sh` from the repo to submit all seven public
English splits and sync the raw JSONL manifests to a results bucket.
