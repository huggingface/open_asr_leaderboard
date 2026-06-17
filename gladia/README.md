# Gladia Solaria

[Gladia Solaria](https://www.gladia.io/) is a cloud speech-to-text API. This folder evaluates the production **Solaria-3** model on the [Open ASR Leaderboard](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard) using the official [gladiaio-sdk](https://pypi.org/project/gladiaio-sdk/).

Gladia is a closed API model, so benchmark numbers are reported here rather than on the Hugging Face Hub. Results below come from Gladia's [open benchmark methodology](https://www.gladia.io/competitors/benchmarks) (74 hours of audio across 7 datasets, Whisper text normalizer, default API settings).

## Benchmark results (WER %, lower is better)

| Dataset | WER |
| --- | --- |
| Switchboard (conversational speech) | 35.8 |
| Common Voice 24 (clean, multilingual) | 6.7 |
| VoxPopuli (formal discourse) | 2.2 |
| Earnings22 (financial calls, long-form) | 11.8 |
| Multilingual LibriSpeech (5-language average) | 5.8 |
| Pipecat STT Benchmark (real-time streaming) | 2.7 |
| Speaker diarization (DIHARD III weighted avg DER) | 16.6 |

Full methodology and reproducible evaluation code: [github.com/gladiaio/normalization](https://github.com/gladiaio/normalization).

## Setup

```bash
pip install -r ../requirements/requirements.txt
pip install -r ../requirements/requirements_gladia.txt
export GLADIA_API_KEY="your_api_key"
export HF_TOKEN="hf_your_key"  # optional, avoids HF rate limits
```

## Run evaluation

Single dataset:

```bash
export PYTHONPATH="..":$PYTHONPATH

python run_eval.py \
    --dataset="librispeech" \
    --split="test.clean" \
    --max_workers=20
```

All Open ASR Leaderboard subsets:

```bash
bash run_gladia.sh
```

## Notes

- Gladia runs on the cloud, not on a local GPU. Concurrency is controlled with `--max_workers` instead of `--batch_size`.
- `torch.compile` does not apply to API inference; the script performs a warmup request before timed evaluation.
- Tune `--max_workers` to stay within your Gladia rate limits.
