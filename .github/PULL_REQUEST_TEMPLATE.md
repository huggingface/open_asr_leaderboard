## Description

Please include a summary of your pull request and how to run/use it.

## Type of change
- [ ] New model
- [ ] New dataset
- [ ] Bug fix
- [ ] New feature
- [ ] Other

## New Model Checklist

- [ ] If your model is hosted on the Hugging Face Hub, please report your results (WER on each split, average WER, and RTFx) on the HF Hub by adding a `.eval_results/open_asr_leaderboard.yaml` file like [this](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026/blob/main/.eval_results/open_asr_leaderboard.yaml) in the model repo.

### HF Jobs evaluation (open-source models)

Using [HF Jobs](https://huggingface.co/docs/hub/en/jobs-overview) makes it straightforward for maintainers to reproduce and verify your results. There are configurations for multiple model libraries available [here](https://huggingface.co/collections/hf-audio/open-asr-leaderboard-eval-configurations).

- [ ] (If a custom configuration is needed) duplicate one of the Space above (click the ⋮ menu → **Duplicate this Space**) to create your own copy, e.g. `your-username/open-asr-leaderboard-mymodel`.
    - [ ] Modify the `Dockerfile` to install your model's dependencies, e.g. installing from a specific version/fork of Transformers.
    - [ ] Adapt `run_eval.py` for your model — use the [Transformers one](https://huggingface.co/spaces/hf-audio/open-asr-leaderboard-transformers/blob/main/run_eval.py) as a starting point. There is no need to modify/update the normalizer files, as the `run_eval.py` script should save the raw (un-normalized) transcripts and model outptus, and the normalizer from the repo is locally score the model outputs.
    - [ ] For models that use `trust_remote_code=True`, please default to a `revision` tag and specify it in your bash script. 
- [ ] In this repo, create a folder for your model library and a `submit_jobs.sh` script (use any existing one in this repo as a template) pointing to your Space and a results bucket.

A similar checklist applies for multilingual evaluation (with `run_eval_ml.py` and `submit_jobs_ml.sh`).

####  Key guidelines
- [ ] Use the **same decoding hyper-parameters** across all datasets for a given model.
- [ ] (If writing a new) `run_eval.py`, it must support **batch processing** and use `normalizer/data_utils.py` for data loading, normalization, and manifest writing.
- [ ] Use the **maximum possible batch size** (can differ per dataset) on an H200 GPU.
- [ ] Use `torch.compile` and/or relevant optimizations including warmup to maximize RTFx.
- [ ] Even if you're not using HF Jobs, prepare an HF space like the [existing models](https://huggingface.co/collections/hf-audio/open-asr-leaderboard-eval-configurations), such that the maintainers can reproduce your results on HF Jobs.
- [ ] Please report your results on the relevant public sets, as well as the RTFx.
- [ ] Be sure to count the **total number** of parameters. You can get the exact number by adding the following line in your `run_eval.py` script:
```python
print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
```
- [ ] Please provide the following model metadata (see [here](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard-results/blob/main/english_short_latest.csv) for existing models).

License | Size (B) | # Languages | Encoder | Decoder | (Recommended) Link to training data disclosure on model card ([example](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3#training-dataset))
-- | -- | -- | -- | -- | --
 x | x | x | x | x | x

- [ ] Does the license appear on your model card? 

### API models

- [ ] Please contact the maintainers (Eric Bezzam or Steven Zheng) to provide an API key.
- [ ] Add an `<API>_provider.py` file [here](https://github.com/huggingface/open_asr_leaderboard/tree/main/api/providers).
- [ ] Import your `<API>_provider.py` file [here](https://github.com/huggingface/open_asr_leaderboard/blob/4e4880b9fb203d60830ed2920c521e067026f269/api/providers/__init__.py#L48).
- [ ] In [api/run_api.sh](https://github.com/huggingface/open_asr_leaderboard/blob/main/api/run_api.sh) add:
    - a line in `MODEL_CONFIGS`
    - a line when calling `docker run` to pass the API key 
- [ ] Provide a link to your model's documentation that we can link on the leaderboard.
- [ ] Please provide the following information. Note that the cost should represent the "entry-level" costs, e.g. when one first signs up to your platform.

Cost ($/hour) |  # Languages | Link to model announcement or API
 -- | -- | -- 
 x | x | x 

See [here](https://github.com/huggingface/open_asr_leaderboard/blob/main/api/README.md) for tips on evaluating API models.


## New Dataset Checklist
- [ ] The dataset is hosted on the HF Hub with **just** the test set.
- [ ] Create a new Bash script with one of the existing models. For example, adapting the `submit_jobs.sh` script for Parakeet or Whisper to add a line for your dataset. 

## Related issues
Closes #
