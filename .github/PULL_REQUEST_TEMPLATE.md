## Description
Please include a summary of your pull request and how to run/use it.

## Type of change
- [ ] New model
- [ ] New dataset
- [ ] Bug fix
- [ ] New feature
- [ ] Other

## New Model Checklist

Please report your results (WER on each split, average WER, and RTFx) on the HF Hub by adding a `.eval_results/open_asr_leaderboard.yaml` file like [this](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026/blob/main/.eval_results/open_asr_leaderboard.yaml) in the model repo. Closed models can report their results in the PR text.

### HF Jobs setup (recommended)

Using HF Jobs makes it straightforward for maintainers to reproduce and verify your results. There are configurations for multiple model libraries available [here](https://huggingface.co/collections/hf-audio/open-asr-leaderboard-eval-configurations).

- [ ] (If a custom configuration is needed) duplicate one of the Space above (click the ⋮ menu → **Duplicate this Space**) to create your own copy, e.g. `your-username/open-asr-leaderboard-mymodel`.
- [ ] Modify the `Dockerfile` to install your model's dependencies.
- [ ] Adapt `run_eval.py` for your model — use the [Transformers one](https://huggingface.co/spaces/hf-audio/open-asr-leaderboard-transformers/blob/main/run_eval.py) as a starting point.
- [ ] In this repo, create a folder for your model library and a `submit_jobs_<your_model>.sh` script (use any existing one in this repo as a template) pointing to your Space and a results bucket. Include it in your PR.

### Key guidelines (all submissions)
- [ ] Use the **same decoding hyper-parameters** across all datasets for a given model.
- [ ] Run with the **maximum possible batch size** (can differ per dataset) on an A100-SXM4-80GB GPU.
- [ ] If you're not using HF Jobs, provide a `Dockerfile` in your PR for reproducible evaluation.
- [ ] `run_eval.py` must support batch processing and use `normalizer/data_utils.py` for data loading, normalization, and manifest writing.

### My model is in Transformers 🤗
- [ ] (If necessary) adapt [`run_eval.py`](https://huggingface.co/spaces/hf-audio/open-asr-leaderboard-transformers/blob/main/run_eval.py) to use your checkpoint (using the `AutoModelForXXX` API).
- [ ] If your model requires dependencies, add them to the [`Dockerfile`](https://huggingface.co/spaces/hf-audio/open-asr-leaderboard-transformers/blob/main/Dockerfile).

### My model is not in Transformers (yet 🙃)
- [ ] Duplicate an [existing Space](https://huggingface.co/collections/hf-audio/open-asr-leaderboard-eval-configurations) that is closest to your model's framework, then modify the `Dockerfile` to install the required dependencies.
- [ ] Adapt `run_eval.py` for your model's inference API (supports batch processing, uses `torch.compile` and/or relevant optimizations including warmup).
- [ ] Create a `submit_jobs_<your_model>.sh` script pointing to your new Space, and include it in your PR.


## New Dataset Checklist
- [ ] The dataset is hosted on the HF Hub with just the test set.
- [ ] Create a new Bash script with one of the model suites. For example, adapting the [Whisper](https://github.com/huggingface/open_asr_leaderboard/blob/main/transformers/run_whisper.sh) or [Voxtral](https://github.com/huggingface/open_asr_leaderboard/blob/main/voxtral/run_voxtral.sh) script to run the eval on that new dataset (adding a call like [this](https://github.com/huggingface/open_asr_leaderboard/blob/main/transformers/run_whisper.sh#L14-L21)).

## Related issues
Closes #