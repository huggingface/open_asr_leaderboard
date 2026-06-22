# HF Upload Commands For ARK-ASR-3B

These commands are prepared for an environment with `hf` or
`huggingface-cli` installed and authenticated. The Space, model eval YAML, and
raw result dataset are already uploaded; use these commands only for a rerun or
repair upload.

Use the local proxy on this machine:

```bash
export HTTP_PROXY="http://127.0.0.1:17890"
export HTTPS_PROXY="http://127.0.0.1:17890"
```

## Create And Upload The HF Jobs Space

Target Space:

```text
AutoArk-AI/open-asr-leaderboard-ark-asr-3b
```

Create it first if needed:

```bash
hf repo create AutoArk-AI/open-asr-leaderboard-ark-asr-3b --type space --space-sdk docker
```

Upload the prepared Space directory:

```bash
hf upload AutoArk-AI/open-asr-leaderboard-ark-asr-3b \
  ark_asr/space_ark_asr_3b \
  . \
  --repo-type space
```

The Space source directory is:

```text
ark_asr/space_ark_asr_3b/
```

## Upload Eval YAML To Model Repo

Target:

```text
AutoArk-AI/ARK-ASR-3B/.eval_results/open_asr_leaderboard.yaml
```

With `hf` CLI:

```bash
hf upload AutoArk-AI/ARK-ASR-3B \
  ark_asr/open_asr_leaderboard_ark_asr_3b.yaml \
  .eval_results/open_asr_leaderboard.yaml \
  --repo-type model
```

With `huggingface-cli`:

```bash
huggingface-cli upload AutoArk-AI/ARK-ASR-3B \
  ark_asr/open_asr_leaderboard_ark_asr_3b.yaml \
  .eval_results/open_asr_leaderboard.yaml \
  --repo-type model
```

## Upload Raw JSONL Manifests To Dataset Repo

Suggested dataset repo:

```text
AutoArk-AI/ark-asr-3b-open-asr-leaderboard-results
```

Create it first if needed:

```bash
hf repo create AutoArk-AI/ark-asr-3b-open-asr-leaderboard-results --type dataset
```

Upload README:

```bash
hf upload AutoArk-AI/ark-asr-3b-open-asr-leaderboard-results \
  ark_asr/results_dataset_readme_ark_asr_3b.md \
  README.md \
  --repo-type dataset
```

Upload result manifests:

```bash
hf upload AutoArk-AI/ark-asr-3b-open-asr-leaderboard-results \
  ark_asr/results.AutoArk-AI-ARK-ASR-3B_20260622_official \
  . \
  --repo-type dataset
```

The PR body should reference this public dataset:

```text
https://huggingface.co/datasets/AutoArk-AI/ark-asr-3b-open-asr-leaderboard-results
```
