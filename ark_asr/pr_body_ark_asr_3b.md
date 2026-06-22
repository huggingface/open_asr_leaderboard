## Description

Adds Open ASR Leaderboard results for `AutoArk-AI/ARK-ASR-3B`.

Model repo result file:

```text
https://huggingface.co/AutoArk-AI/ARK-ASR-3B/blob/main/.eval_results/open_asr_leaderboard.yaml
```

Raw JSONL manifests for maintainer verification:

```text
https://huggingface.co/datasets/AutoArk-AI/ark-asr-3b-open-asr-leaderboard-results
```

HF Jobs Space for reproduction:

```text
https://huggingface.co/spaces/AutoArk-AI/open-asr-leaderboard-ark-asr-3b
```

How to dry-run the HF Jobs submission plan without creating jobs:

```bash
DRY_RUN=1 RESULTS_BUCKET="<your-bucket>" HF_TOKEN=hf_... bash ark_asr/submit_jobs_ark_asr_3b.sh
```

Optional model revision pinning follows the official Transformers Space
`--revision` behavior:

```bash
MODEL_REVISION="<commit-or-ref>" DRY_RUN=1 RESULTS_BUCKET="<your-bucket>" HF_TOKEN=hf_... bash ark_asr/submit_jobs_ark_asr_3b.sh
```

How to rescore the uploaded local manifests:

```bash
PYTHONPATH=. python - <<'PY'
from normalizer.eval_utils import score_results
score_results(
    'ark_asr/results.AutoArk-AI-ARK-ASR-3B_20260622_official',
    'AutoArk-AI/ARK-ASR-3B',
)
PY
```

Results were generated locally on 8x RTX 4090 and scored with the repository
`normalizer.eval_utils.score_results(...)`. HF Jobs were not submitted for this
update; the Space and submit script are prepared for maintainer reproduction.

`earnings22` was evaluated with the official `hf-audio/open-asr-leaderboard`
data, not the old chunked `distil-whisper/earnings22` cache. The official local
split used here has 2741 raw rows and 2737 filtered rows.

Evaluation configuration:

| Split | Batch size | Decoding |
| --- | ---: | --- |
| ami/test | 64 | `max_new_tokens=256`, `dtype=float16`, `attn_impl=sdpa` |
| earnings22/test | 64 | `max_new_tokens=256`, `dtype=float16`, `attn_impl=sdpa` |
| gigaspeech/test | 64 | `max_new_tokens=256`, `dtype=float16`, `attn_impl=sdpa` |
| librispeech/test.clean | 64 | `max_new_tokens=256`, `dtype=float16`, `attn_impl=sdpa` |
| librispeech/test.other | 64 | `max_new_tokens=256`, `dtype=float16`, `attn_impl=sdpa` |
| spgispeech/test | 64 | `max_new_tokens=256`, `dtype=float16`, `attn_impl=sdpa` |
| voxpopuli/test | 64 | `max_new_tokens=256`, `dtype=float16`, `attn_impl=sdpa` |

Results:

| Split | Samples | WER | RTFx |
| --- | ---: | ---: | ---: |
| ami/test | 11653 | 8.91 | 272.25 |
| earnings22/test | 2737 | 8.25 | 243.20 |
| gigaspeech/test | 19898 | 7.30 | 140.23 |
| librispeech/test.clean | 2620 | 1.09 | 293.56 |
| librispeech/test.other | 2939 | 2.41 | 284.61 |
| spgispeech/test | 39341 | 2.49 | 208.87 |
| voxpopuli/test | 1842 | 5.48 | 293.41 |

Average WER: `5.13`

Overall RTFx: `197.14`

## Type of change
- [x] New model
- [ ] New dataset
- [ ] Bug fix
- [ ] New feature
- [ ] Other

## New Model Checklist

Results are reported on the HF Hub via `.eval_results/open_asr_leaderboard.yaml`
in the model repo:

```text
https://huggingface.co/AutoArk-AI/ARK-ASR-3B/blob/main/.eval_results/open_asr_leaderboard.yaml
```

### HF Jobs setup (recommended)

Using HF Jobs makes it straightforward for maintainers to reproduce and verify your results. There are configurations for multiple model libraries available [here](https://huggingface.co/collections/hf-audio/open-asr-leaderboard-eval-configurations).

- [x] (If a custom configuration is needed) duplicate one of the Space above (click the ⋮ menu → **Duplicate this Space**) to create your own copy, e.g. `your-username/open-asr-leaderboard-mymodel`.
- [x] Modify the `Dockerfile` to install your model's dependencies.
- [x] Adapt `run_eval.py` for your model — use the [Transformers one](https://huggingface.co/spaces/hf-audio/open-asr-leaderboard-transformers/blob/main/run_eval.py) as a starting point.
- [x] In this repo, create a folder for your model library and a `submit_jobs_<your_model>.sh` script (use any existing one in this repo as a template) pointing to your Space and a results bucket. Include it in your PR.

### Key guidelines (all submissions)
- [x] Use the **same decoding hyper-parameters** across all datasets for a given model.
- [ ] Run with the **maximum possible batch size** (can differ per dataset) on an H200 GPU.
- [x] If you're not using HF Jobs, provide a `Dockerfile` in your PR for reproducible evaluation.
- [x] `run_eval.py` must support batch processing and use `normalizer/data_utils.py` for data loading, normalization, and manifest writing.

### My model is in Transformers 🤗
- [ ] (If necessary) adapt [`run_eval.py`](https://huggingface.co/spaces/hf-audio/open-asr-leaderboard-transformers/blob/main/run_eval.py) to use your checkpoint (using the `AutoModelForXXX` API).
- [ ] If your model requires dependencies, add them to the [`Dockerfile`](https://huggingface.co/spaces/hf-audio/open-asr-leaderboard-transformers/blob/main/Dockerfile).

### My model is not in Transformers (yet 🙃)
- [x] Duplicate an [existing Space](https://huggingface.co/collections/hf-audio/open-asr-leaderboard-eval-configurations) that is closest to your model's framework, then modify the `Dockerfile` to install the required dependencies.
- [x] Adapt `run_eval.py` for your model's inference API (supports batch processing, uses `torch.compile` and/or relevant optimizations including warmup).
- [x] Create a `submit_jobs_<your_model>.sh` script pointing to your new Space, and include it in your PR.


## New Dataset Checklist
- [ ] The dataset is hosted on the HF Hub with just the test set.
- [ ] Create a new Bash script with one of the model suites. For example, adapting the [Whisper](https://github.com/huggingface/open_asr_leaderboard/blob/main/transformers/run_whisper.sh) or [Voxtral](https://github.com/huggingface/open_asr_leaderboard/blob/main/voxtral/run_voxtral.sh) script to run the eval on that new dataset (adding a call like [this](https://github.com/huggingface/open_asr_leaderboard/blob/main/transformers/run_whisper.sh#L14-L21)).

## Related issues
Closes #
