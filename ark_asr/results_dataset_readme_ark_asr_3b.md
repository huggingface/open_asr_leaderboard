---
pretty_name: ARK-ASR-3B Open ASR Leaderboard Results
license: apache-2.0
task_categories:
- automatic-speech-recognition
tags:
- open-asr-leaderboard
- ark-asr
- automatic-speech-recognition
---

# ARK-ASR-3B Open ASR Leaderboard Results

Raw JSONL manifests for `AutoArk-AI/ARK-ASR-3B` on the public English
short-form `hf-audio/open-asr-leaderboard` splits.

These manifests were generated on a local 8x RTX 4090 machine and scored with
the shared Open ASR Leaderboard scorer:

```bash
PYTHONPATH=. python - <<'PY'
from normalizer.eval_utils import score_results
score_results(
    'ark_asr/results.AutoArk-AI-ARK-ASR-3B_20260622_official',
    'AutoArk-AI/ARK-ASR-3B',
)
PY
```

Important: `earnings22` uses the official `hf-audio/open-asr-leaderboard`
leaderboard data, not the old chunked `distil-whisper/earnings22` cache.

## Results

| Split | Samples | WER | RTFx |
| --- | ---: | ---: | ---: |
| ami/test | 11653 | 8.91 | 272.25 |
| earnings22/test | 2737 | 8.25 | 243.20 |
| gigaspeech/test | 19898 | 7.30 | 140.23 |
| librispeech/test.clean | 2620 | 1.09 | 293.56 |
| librispeech/test.other | 2939 | 2.41 | 284.61 |
| spgispeech/test | 39341 | 2.49 | 208.87 |
| voxpopuli/test | 1842 | 5.48 | 293.41 |

Composite:

- Average WER: `5.13`
- Overall RTFx: `197.14`

## Files

```text
MODEL_AutoArk-AI-ARK-ASR-3B_DATASET_hf-audio-open-asr-leaderboard_ami_test.jsonl
MODEL_AutoArk-AI-ARK-ASR-3B_DATASET_hf-audio-open-asr-leaderboard_earnings22_test.jsonl
MODEL_AutoArk-AI-ARK-ASR-3B_DATASET_hf-audio-open-asr-leaderboard_gigaspeech_test.jsonl
MODEL_AutoArk-AI-ARK-ASR-3B_DATASET_hf-audio-open-asr-leaderboard_librispeech_test.clean.jsonl
MODEL_AutoArk-AI-ARK-ASR-3B_DATASET_hf-audio-open-asr-leaderboard_librispeech_test.other.jsonl
MODEL_AutoArk-AI-ARK-ASR-3B_DATASET_hf-audio-open-asr-leaderboard_spgispeech_test.jsonl
MODEL_AutoArk-AI-ARK-ASR-3B_DATASET_hf-audio-open-asr-leaderboard_voxpopuli_test.jsonl
```
