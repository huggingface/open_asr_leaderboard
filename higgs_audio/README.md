# Higgs-Audio v3 — Open-ASR-Leaderboard reproduction

This folder follows the leaderboard's
[Add-a-new-library template](https://github.com/huggingface/open_asr_leaderboard#add-a-new-library).
`run_eval.py` reproduces the numbers reported on the
[`bosonai/higgs-audio-v3-8b-stt`](https://huggingface.co/bosonai/higgs-audio-v3-8b-stt)
and
[`bosonai/higgs-audio-v3-stt`](https://huggingface.co/bosonai/higgs-audio-v3-stt) (1.7 B)
model cards.

## Target numbers (ESB test, all samples)

| Dataset             | 8B WER | 1.7B WER |
|---------------------|-------:|---------:|
| AMI                 |   6.23 |    6.34  |
| Earnings-22         |  11.33 |   11.98  |
| GigaSpeech          |   9.34 |    9.53  |
| LibriSpeech clean   |   1.24 |    1.48  |
| LibriSpeech other   |   2.34 |    2.90  |
| SPGI Speech         |   3.14 |    3.68  |
| TED-LIUM            |   3.14 |    3.29  |
| VoxPopuli           |   5.63 |    6.11  |
| **Macro avg**       | **5.30** | **5.67** |

## Why our own `run_eval.py`

The merged HF weights reproduce 5.30 % only when three behaviours are
applied at decode/score time:

1. **Prompt**: the Qwen-ChatML template from
   `prepare_chatml_sample_qwen` — real `<|audio_bos|>=151669`,
   `<|audio_eos|>=151670`, `<|AUDIO|>` placeholder, `enable_thinking=True`.
2. **Post-proc**: `_fix_repetitions(max_repeat=3)` on the decoded string
   (caps runaway repetitions on AMI / Earnings-22).
3. **Normalizer**: `whisper_normalizer.english.EnglishTextNormalizer` on
   *both* hyp and ref (the leaderboard's in-tree normalizer keeps
   `'cause` / `dunno` etc. where pip expands them; applying the richer
   pip normalizer symmetrically on both sides removes the bias).

The 1.24 pp gap between a naive leaderboard run (≈6.54 %, reproduced
independently by the HF maintainer in PR #135) and 5.30 % comes from
these three pieces combined.

## Usage

Single dataset, single GPU:

```bash
export HF_TOKEN=hf_...
python open_asr_leaderboard/higgs_audio/run_eval.py \
    --model_id bosonai/higgs-audio-v3-8b-stt \
    --dataset_path hf-audio/esb-datasets-test-only-sorted \
    --dataset librispeech \
    --split "test.clean" \
    --device 0 \
    --batch_size 1 \
    --output_dir open_asr_leaderboard/higgs_audio/results
```

Full 8-way parallel sweep over the ESB suite (what the authors run):

```bash
bash open_asr_leaderboard/higgs_audio/run_8gpu_parallel.sh \
     bosonai/higgs-audio-v3-8b-stt \
     results/hf_leaderboard_5p30
python open_asr_leaderboard/higgs_audio/aggregate.py \
     --dir results/hf_leaderboard_5p30
```

The launcher checkpoints partial results every ~2 min as
`CHK_<dataset>_<split>_s<i>of<n>.json`, so the aggregator works even
mid-run.

## Environment

* Python ≥ 3.10
* `transformers==4.51.0` (newer versions break the model's custom
  `generate` path)
* `peft`, `whisper_normalizer`, `evaluate`, `jiwer`, `librosa`
* `boson_multimodal` on `PYTHONPATH` (collator / `ChatMLDatasetSample`)

Tested on 8×H100; a single H100 takes ~18 GB VRAM per shard.
