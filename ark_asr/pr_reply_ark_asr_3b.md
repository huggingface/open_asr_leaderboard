Thanks, I prepared the `AutoArk-AI/ARK-ASR-3B` leaderboard submission with the
current standardized Open ASR Leaderboard scoring path.

What is included:

- New HF Jobs Space prepared and public for this submission:
  https://huggingface.co/spaces/AutoArk-AI/open-asr-leaderboard-ark-asr-3b
- 3B-specific HF Jobs submit script: `ark_asr/submit_jobs_ark_asr_3b.sh`
- Model repo eval file prepared at `.eval_results/open_asr_leaderboard.yaml`
- Raw JSONL manifests for all seven public English short-form splits:
  https://huggingface.co/datasets/AutoArk-AI/ark-asr-3b-open-asr-leaderboard-results
- Local reproducibility wrapper: `ark_asr/run_local_parallel_eval.sh`
- Machine-specific runbook: `ark_asr/RUNBOOK.md`
- Re-scored results using `normalizer.eval_utils.score_results(...)`

One important correction: `earnings22` is evaluated with the official
`hf-audio/open-asr-leaderboard` cache, not the old chunked
`distil-whisper/earnings22` cache. The official `earnings22` split used here has
2737 filtered samples.

Final 3B results:

| Split | WER | RTFx |
| --- | ---: | ---: |
| ami/test | 8.91 | 272.25 |
| earnings22/test | 8.25 | 243.20 |
| gigaspeech/test | 7.30 | 140.23 |
| librispeech/test.clean | 1.09 | 293.56 |
| librispeech/test.other | 2.41 | 284.61 |
| spgispeech/test | 2.49 | 208.87 |
| voxpopuli/test | 5.48 | 293.41 |

Composite:

- Average WER: `5.13`
- Overall RTFx: `197.14`

Transparency note: these manifests were generated on our local 8x RTX 4090
machine. They use the shared leaderboard normalizer/scorer and all raw JSONL
manifests are available for maintainer re-scoring or rerun on reference
hardware if required.

HF Jobs have not been submitted in this update; the Space and submit script are
ready for reproduction if maintainers want an H200 rerun.
