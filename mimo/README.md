# Xiaomi MiMo-V2.5-ASR

Runner for [XiaomiMiMo/MiMo-V2.5-ASR](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-ASR)
on the Open ASR Leaderboard benchmark.

MiMo-V2.5-ASR is an end-to-end ASR model jointly optimized for Mandarin
Chinese and English, with strong support for Chinese dialects, code-switching,
and noisy / multi-speaker audio.

## Setup

MiMo's Python package is **not** pip-installable. It must be cloned from its
GitHub repo and added to `PYTHONPATH`.

```bash
# 1. Clone MiMo's source code (sibling to this repo is fine)
git clone https://github.com/XiaomiMiMo/MiMo-V2.5-ASR.git ../MiMo-V2.5-ASR
cd ../MiMo-V2.5-ASR
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1   # optional; if unavailable see Notes
cd -

# 2. Open ASR Leaderboard's own deps
pip install datasets evaluate jiwer librosa num2words soundfile torch torchaudio torchcodec transformers==4.49.0

# 3. Download model weights + audio tokenizer
hf download XiaomiMiMo/MiMo-V2.5-ASR \
    --local-dir ./models/MiMo-V2.5-ASR
hf download XiaomiMiMo/MiMo-Audio-Tokenizer \
    --local-dir ./models/MiMo-Audio-Tokenizer

# 4. Tell run_eval.py where to find MiMo's source tree
export MIMO_REPO_PATH=$(realpath ../MiMo-V2.5-ASR)
```

## Run

```bash
bash run_mimo.sh
```

Or invoke `run_eval.py` directly:

```bash
python run_eval.py \
    --model_id="XiaomiMiMo/MiMo-V2.5-ASR" \
    --model_path="./models/MiMo-V2.5-ASR" \
    --tokenizer_path="./models/MiMo-Audio-Tokenizer" \
    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    --dataset="earnings22" \
    --split="test" \
    --audio_tag="<english>" \
    --device=0
```

## Notes

- **No batching.** `MimoAudio.asr_sft()` accepts only a single audio
  sample per call, so we iterate sequentially. Expect lower RTFx than
  models with native batched inference.
- **Resumable.** Each completed sample is appended to the result manifest
  immediately; rerunning the same command picks up after the last
  written row, so timeouts/crashes are recoverable.
- **Hyperparameters held identical across all datasets** per ESB rules.
  This runner uses MiMo's default generation config; only `audio_tag`
  changes for non-English benchmarks.
- **Hardware**: benchmark in this submission run on NVIDIA RTX 4090
  (24 GB VRAM). MiMo's bf16 weights (~16 GB) fit comfortably.
  A100-80GB (the leaderboard's reference HW) is fully supported.
- **`flash_attn_shim.py`**: MiMo's audio tokenizer hard-imports
  `flash_attn`. On hosts where `flash-attn` cannot be installed
  (Windows native, exotic torch+CUDA combos), this shim provides a
  PyTorch SDPA-based fallback for `flash_attn_varlen_func`. It only
  supports full attention (encoder/decoder paths) — sliding-window
  attention (vocoder path, unused for ASR) raises clearly. The shim
  is auto-injected only when the real `flash_attn` import fails, so
  Linux + flash-attn hosts use the canonical path.
