# Higgs Audio v3 Evaluation

## Models
- `bosonai/higgs-audio-v3-stt` (2.68B params, 5.67% avg WER)
- `bosonai/higgs-audio-v3-8b-stt` (8.91B params, 5.30% avg WER)

## Pre-computed Results

| Model | AMI | E22 | GS | LS-C | LS-O | SPG | TED | VP | Avg |
|-------|-----|-----|-----|------|------|-----|-----|-----|-----|
| higgs-audio-v3-stt | 10.49 | 11.48 | 7.71 | 1.07 | 2.03 | 2.81 | 2.81 | 6.93 | 5.67 |
| higgs-audio-v3-8b-stt | 6.23 | 11.33 | 9.34 | 1.24 | 2.34 | 3.14 | 3.14 | 5.63 | 5.30 |

Full-scale evaluation on all samples with Whisper `EnglishTextNormalizer`.

## Setup

These models use a custom architecture. Install dependencies:

```bash
pip install torch transformers>=4.51.0 whisper-normalizer jiwer
pip install boson-multimodal@git+https://github.com/boson-ai/boson-multimodal.git
```

## Run

```bash
bash run_higgs_audio.sh
```

Or evaluate a single dataset:

```bash
python run_eval_higgs_audio.py \
    --model_id bosonai/higgs-audio-v3-8b-stt \
    --dataset_path hf-audio/esb-datasets-test-only-sorted \
    --dataset ami --split test --device 0
```

## Notes
- Models require `trust_remote_code=True`
- Audio input: 16kHz mono
- Uses thinking mode (chain-of-thought before transcription)
- Batch size 1 (model processes one sample at a time with audio chunking)
