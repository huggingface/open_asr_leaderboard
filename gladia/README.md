# Gladia Solaria

[Gladia Solaria](https://www.gladia.io/) is a cloud speech-to-text API. We evaluate here the **Solaria-3** model on the [Open ASR Leaderboard](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard) using the official [gladiaio-sdk](https://pypi.org/project/gladiaio-sdk/).

Results below come from Gladia's [open benchmark methodology](https://www.gladia.io/competitors/benchmarks)

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

This is the normalization repo of Gladia: [github.com/gladiaio/normalization](https://github.com/gladiaio/normalization).

## Setup

```bash
cd  gladia
pip install -r ../requirements/requirements.txt
pip install -r ../requirements/requirements_gladia.txt
export GLADIA_API_KEY="your_api_key"
export HF_TOKEN="hf_your_key" 
```

## Run evaluation

Single dataset:

```bash
python run_eval.py \
    --dataset="librispeech" \
    --split="test.clean" \
    --max_eval_samples=64 \
    --max_workers=5
```

By default the script downloads the dataset locally (`--no-streaming`) and decodes audio with `soundfile`.

All Open ASR Leaderboard subsets:

```bash
bash run_gladia.sh
```
