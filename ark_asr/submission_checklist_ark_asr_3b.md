# ARK-ASR-3B Leaderboard Submission Checklist

Target model repo:

```text
AutoArk-AI/ARK-ASR-3B
```

Required model repo file:

```text
.eval_results/open_asr_leaderboard.yaml
```

Prepared local YAML copy:

```text
ark_asr/open_asr_leaderboard_ark_asr_3b.yaml
```

Prepared local JSONL manifests:

```text
ark_asr/results.AutoArk-AI-ARK-ASR-3B_20260622_official/
```

Prepared HF Jobs Space:

```text
AutoArk-AI/open-asr-leaderboard-ark-asr-3b
ark_asr/space_ark_asr_3b/
```

Prepared HF Jobs submit script:

```text
ark_asr/submit_jobs_ark_asr_3b.sh
```

Prepared dataset README and upload commands:

```text
ark_asr/results_dataset_readme_ark_asr_3b.md
ark_asr/hf_upload_commands_ark_asr_3b.md
```

Online verification links:

```text
https://huggingface.co/AutoArk-AI/ARK-ASR-3B/blob/main/.eval_results/open_asr_leaderboard.yaml
https://huggingface.co/spaces/AutoArk-AI/open-asr-leaderboard-ark-asr-3b
https://huggingface.co/datasets/AutoArk-AI/ark-asr-3b-open-asr-leaderboard-results
```

Files to upload for maintainer verification, ideally to a HF dataset repo:

```text
MODEL_AutoArk-AI-ARK-ASR-3B_DATASET_hf-audio-open-asr-leaderboard_ami_test.jsonl
MODEL_AutoArk-AI-ARK-ASR-3B_DATASET_hf-audio-open-asr-leaderboard_earnings22_test.jsonl
MODEL_AutoArk-AI-ARK-ASR-3B_DATASET_hf-audio-open-asr-leaderboard_gigaspeech_test.jsonl
MODEL_AutoArk-AI-ARK-ASR-3B_DATASET_hf-audio-open-asr-leaderboard_librispeech_test.clean.jsonl
MODEL_AutoArk-AI-ARK-ASR-3B_DATASET_hf-audio-open-asr-leaderboard_librispeech_test.other.jsonl
MODEL_AutoArk-AI-ARK-ASR-3B_DATASET_hf-audio-open-asr-leaderboard_spgispeech_test.jsonl
MODEL_AutoArk-AI-ARK-ASR-3B_DATASET_hf-audio-open-asr-leaderboard_voxpopuli_test.jsonl
```

Expected row counts:

```text
ami                 11653
earnings22           2737
gigaspeech          19898
librispeech clean    2620
librispeech other    2939
spgispeech          39341
voxpopuli            1842
total               81030
```

Final scoring command:

```bash
cd /workspace/yiming/open_asr_leaderboard.worktrees.add-ark-asr
PYTHONPATH=. python - <<'PY'
from normalizer.eval_utils import score_results
score_results(
    'ark_asr/results.AutoArk-AI-ARK-ASR-3B_20260622_official',
    'AutoArk-AI/ARK-ASR-3B',
)
PY
```

Expected scoring output:

```text
AMI WER          8.91
Earnings22 WER   8.25
Gigaspeech WER   7.30
LS Clean WER     1.09
LS Other WER     2.41
SPGISpeech WER   2.49
Voxpopuli WER    5.48
Composite WER    5.13
Composite RTFx 197.14
```

Before opening/updating PR:

- [x] Create/upload Space `AutoArk-AI/open-asr-leaderboard-ark-asr-3b` from `ark_asr/space_ark_asr_3b/`.
- [x] Run `bash -n ark_asr/submit_jobs_ark_asr_3b.sh`.
- [x] Dry-run the HF Jobs submission plan with `DRY_RUN=1`.
- [x] Upload `.eval_results/open_asr_leaderboard.yaml` to `AutoArk-AI/ARK-ASR-3B`.
- [x] Upload raw JSONL manifests to the verification dataset repo.
- [x] Replace/add the results dataset URL in `pr_body_ark_asr_3b.md`.
- [x] Confirm the PR mentions official `earnings22`, not chunked `distil-whisper/earnings22`.
- [ ] Paste `ark_asr/pr_body_ark_asr_3b.md` as the PR body.
- [ ] Use `ark_asr/pr_reply_ark_asr_3b.md` as the maintainer-facing update comment if needed.

HF Jobs were intentionally not submitted in this preparation pass. The Space and
`submit_jobs_ark_asr_3b.sh` are prepared for reviewer reproduction.
