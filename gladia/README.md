# Gladia Solaria

[Gladia Solaria](https://www.gladia.io/) is a cloud speech-to-text API. 

Results below are for **Solaria-3**, from Gladia's [open benchmark methodology](https://www.gladia.io/competitors/benchmarks).

## Solaria-3 benchmark results (WER %, lower is better)

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
cd gladia
pip install -r ../requirements/requirements.txt
pip install -r ../requirements/requirements-api.txt
export GLADIA_API_KEY="your_api_key"
export HF_TOKEN="hf_your_key"
```

## Run evaluation

Single dataset (from the `gladia/` directory):

```bash
python ../api/run_eval.py \
    --dataset_path="hf-audio/open-asr-leaderboard" \
    --dataset="librispeech" \
    --split="test.clean" \
    --model_name="gladia/solaria-3" \
    --max_samples=64 \
    --max_workers=5
```

All Open ASR Leaderboard subsets:

```bash
bash run_gladia.sh
```

The Gladia provider lives in [`api/providers/gladia_provider.py`](../api/providers/gladia_provider.py).

### Adding a new model variant

1. Register the variant in `api/providers/gladia_provider.py` (`MODEL_ALIASES`) if the leaderboard id differs from the Gladia API model name.
2. Add `gladia/<variant>` to the `MODEL_IDs` array in `run_gladia.sh`.
3. Run `bash run_gladia.sh` (or pass `--model_name="gladia/<variant>"` to `../api/run_eval.py` for a single dataset).
