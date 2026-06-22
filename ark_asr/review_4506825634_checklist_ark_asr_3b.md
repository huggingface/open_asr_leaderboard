# Review 4506825634 Checklist For ARK-ASR-3B

Review URL:

```text
https://github.com/huggingface/open_asr_leaderboard/pull/163#pullrequestreview-4506825634
```

Review text was fetched through the GitHub API. The review asks for:

- Try the new H200/HF Jobs setup from `ebezzam/standardize_transformers`.
- Follow the new-model checklist for a model not in Transformers.
- Share a bucket/dataset with raw results, as in PR #164.
- Run the submit job script from the `standardize_transformers` branch/fork so the new normalizer is used.

## Link Checks

| Link | Status | Notes |
| --- | --- | --- |
| `https://github.com/huggingface/open_asr_leaderboard/pull/163#pullrequestreview-4506825634` | Checked | Review body fetched via GitHub API; reviewer is `ebezzam`, submitted 2026-06-16. |
| `https://github.com/huggingface/open_asr_leaderboard/pull/142` | Checked | PR title is `Switch to HF jobs + normalizer improvements`; review explicitly references this transition. |
| `https://github.com/ebezzam/open_asr_leaderboard/tree/standardize_transformers#evaluate-a-model-as-of-10-june-2026` | Checked | Branch exists at `4ad1aaff5cebf142eb554bb9248223abb9922f0a`; README says English short-form evals use HF Jobs on H200, one Space per model family, results bucket, and `submit_jobs` scripts. |
| `https://github.com/ebezzam/open_asr_leaderboard/blob/standardize_transformers/.github/PULL_REQUEST_TEMPLATE.md#my-model-is-not-in-transformers-yet-` | Checked | Non-Transformers checklist requires duplicating a closest Space, adapting `run_eval.py`, and creating `submit_jobs_<model>.sh` pointing to the new Space. |
| `https://github.com/huggingface/open_asr_leaderboard/pull/164#issuecomment-4698052192` | Checked | Example shares full per-dataset results, says results are scored locally with repository scorer, lists 7 sets, official data on H200, and provides a public raw JSONL dataset link. |
| `https://github.com/ebezzam/open_asr_leaderboard/tree/standardize_transformers` | Checked | `git ls-remote` confirms branch exists at `4ad1aaff5cebf142eb554bb9248223abb9922f0a`. |
| `https://huggingface.co/collections/hf-audio/open-asr-leaderboard-eval-configurations` | Checked | Collection URL opens and is the PR-template source for duplicating a closest Space. |
| `https://huggingface.co/spaces/AutoArk-AI/open-asr-leaderboard-ark-asr/commits/main` | Checked | Old ARK-ASR Space history checked; latest commit is `cc984fbff4425c8fd0b1f3802b0c02ba89e7748b` from 2026-06-17. |

Hugging Face links were checked through:

```bash
HTTP_PROXY="http://127.0.0.1:17890" HTTPS_PROXY="http://127.0.0.1:17890"
```

Current online status:

- `AutoArk-AI/ARK-ASR-3B`: exists, public, API returns 200.
- `AutoArk-AI/ARK-ASR-3B/.eval_results/open_asr_leaderboard.yaml`: exists and
  contains the 5.13 / per-split WER values.
- Existing old Space `AutoArk-AI/open-asr-leaderboard-ark-asr`: exists, public,
  API returns 200.
- New 3B Space `AutoArk-AI/open-asr-leaderboard-ark-asr-3b`: exists, public,
  API/page returns 200; uploaded at commit `c1337d608c2c645a1abb008fc90f902757541685`.
- Result dataset `AutoArk-AI/ark-asr-3b-open-asr-leaderboard-results`: exists,
  public, API/page returns 200; raw JSONL upload is at commit
  `b98599d8b525251956c36fd237b0c30af18fea90`.

## Required Artifacts

| Requirement | Prepared Artifact | Status |
| --- | --- | --- |
| New Space duplicated/adapted for model family | `ark_asr/space_ark_asr_3b/` targeting `AutoArk-AI/open-asr-leaderboard-ark-asr-3b` | Prepared locally and uploaded; public Space verified |
| Dockerfile with dependencies | `ark_asr/space_ark_asr_3b/Dockerfile` | Prepared |
| Adapted batched `run_eval.py` | `ark_asr/space_ark_asr_3b/run_eval.py` and `ark_asr/run_eval.py` | Prepared |
| Submit Jobs script pointing to new Space | `ark_asr/submit_jobs_ark_asr_3b.sh` | Prepared, `bash -n` and `DRY_RUN=1` passed |
| Result bucket/dataset with raw JSONL | Target dataset: `AutoArk-AI/ark-asr-3b-open-asr-leaderboard-results`; local README and upload commands prepared | Public dataset verified; all 7 JSONL files present |
| Model repo eval YAML | `ark_asr/open_asr_leaderboard_ark_asr_3b.yaml`; matches local and online model repo `.eval_results/open_asr_leaderboard.yaml` | Prepared and online |
| Official 7 public English splits | `ark_asr/results.AutoArk-AI-ARK-ASR-3B_20260622_official/` | Prepared locally |
| Official `earnings22`, not old chunked data | Official `earnings22` has 2737 filtered rows | Verified locally |

## Old Space Change Audit

Compared old Space `AutoArk-AI/open-asr-leaderboard-ark-asr` at latest commit
`cc984fbff4425c8fd0b1f3802b0c02ba89e7748b` with 3B Space
`AutoArk-AI/open-asr-leaderboard-ark-asr-3b` at
`c1337d608c2c645a1abb008fc90f902757541685`.

Findings:

- `normalizer/data_utils.py` and `normalizer/eval_utils.py` are identical
  between old Space and 3B Space.
- Dockerfile differences are comment punctuation only; installed runtime
  dependencies are equivalent for ARK-ASR English evaluation.
- 3B `run_eval.py` is based on the old Space ARK-ASR `run_eval.py` and adds
  local parquet `bytes/path` row support, `manifest_model_id`, official
  Transformers-Space-style model `--revision`, and 3B streaming model offline
  decoding support.
- Old Space contains `run_eval_ml.py`; 3B Space does not. This is not required
  for the current English short-form Open ASR Leaderboard submission. Add it
  only if a future multilingual ARK-ASR evaluation is requested.

Conclusion: no old-Space fix is missing from the current 3B English submission.

## Verification Already Run

```bash
bash -n ark_asr/submit_jobs_ark_asr_3b.sh
DRY_RUN=1 RESULTS_BUCKET="AutoArk-AI/ark-asr-3b-open-asr-leaderboard-results" HF_TOKEN=dummy bash ark_asr/submit_jobs_ark_asr_3b.sh
DRY_RUN=1 MODEL_REVISION="refs/pr/11" RESULTS_BUCKET="AutoArk-AI/ark-asr-3b-open-asr-leaderboard-results" HF_TOKEN=dummy bash ark_asr/submit_jobs_ark_asr_3b.sh
python -m py_compile ark_asr/space_ark_asr_3b/run_eval.py ark_asr/space_ark_asr_3b/normalizer/data_utils.py ark_asr/space_ark_asr_3b/normalizer/eval_utils.py ark_asr/space_ark_asr_3b/app.py
PYTHONPATH=. python - <<'PY'
from normalizer.eval_utils import score_results
score_results(
    'ark_asr/results.AutoArk-AI-ARK-ASR-3B_20260622_official',
    'AutoArk-AI/ARK-ASR-3B',
)
PY
```

Expected score:

```text
Composite WER  = 5.13
Composite RTFx = 197.14
```

## Remaining Before Claiming Fully Ready

- [x] Upload/create Space `AutoArk-AI/open-asr-leaderboard-ark-asr-3b`.
- [x] Verify the Space URL opens.
- [x] Clearly state that current numbers are local 8x RTX 4090 raw-manifest numbers and HF Jobs were not submitted.
- [x] Upload raw JSONL manifests to `AutoArk-AI/ark-asr-3b-open-asr-leaderboard-results`.
- [x] Verify the result dataset URL opens.
- [x] Verify `AutoArk-AI/ARK-ASR-3B/.eval_results/open_asr_leaderboard.yaml` opens on the Hub.
- [ ] Rebase/merge against `ebezzam/standardize_transformers` (`4ad1aaff...`) before final PR update if maintainers require branch ancestry, since current local HEAD is not a descendant of that branch even though the standardized scorer/data-utils changes are present.

HF Jobs are not submitted in this pass per instruction. The new Space, raw
dataset, and dry-run-checked submit script are ready for maintainer reproduction.
