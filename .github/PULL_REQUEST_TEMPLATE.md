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

### HF Jobs evaluation (recommended)

Using [HF Jobs](https://huggingface.co/docs/hub/en/jobs-overview) makes it straightforward for maintainers to reproduce and verify your results. There are configurations for multiple model libraries available [here](https://huggingface.co/collections/hf-audio/open-asr-leaderboard-eval-configurations).

- [ ] (If a custom configuration is needed) duplicate one of the Space above (click the ⋮ menu → **Duplicate this Space**) to create your own copy, e.g. `your-username/open-asr-leaderboard-mymodel`.
    - [ ] Modify the `Dockerfile` to install your model's dependencies, e.g. installing from a specific version/fork of Transformers.
    - [ ] Adapt `run_eval.py` for your model — use the [Transformers one](https://huggingface.co/spaces/hf-audio/open-asr-leaderboard-transformers/blob/main/run_eval.py) as a starting point. There is no need to modify/update the normalizer files, as the `run_eval.py` script should save the raw (un-normalized) transcripts and model outptus, and the normalizer from the repo is locally score the model outputs.
    - [ ] For models that use `trust_remote_code=True`, please default to a `revision` tag and specify it in your bash script. 
- [ ] In this repo, create a folder for your model library and a `submit_jobs.sh` script (use any existing one in this repo as a template) pointing to your Space and a results bucket.

### Key guidelines (all submissions)
- [ ] Use the **same decoding hyper-parameters** across all datasets for a given model.
- [ ] `run_eval.py` must support **batch processing** and use `normalizer/data_utils.py` for data loading, normalization, and manifest writing.
- [ ] Use the **maximum possible batch size** (can differ per dataset) on an H200 GPU.
- [ ] Use `torch.compile` and/or relevant optimizations including warmup to maximize RTFx.
- [ ] Even if you're not using HF Jobs, prepare an HF space like the [existing models](https://huggingface.co/collections/hf-audio/open-asr-leaderboard-eval-configurations), such that the maintainers can reproduce your results on HF Jobs.
- [ ] Please provide the following model metadata (see [here](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard-results/blob/main/english_short_latest.csv) for existing models).

License | Size (B) | # Languages | Encoder | Decoder
-- | -- | -- | -- | -- 
 x | x | x | x | x 

For LLM-based models, be sure to count the **total number** of parameters. You can get the exact number by adding the following line in your `run_eval.py` script:
```python
print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
```

## New Dataset Checklist
- [ ] The dataset is hosted on the HF Hub with **just** the test set.
- [ ] Create a new Bash script with one of the existing models. For example, adapting the `submit_jobs.sh` script for Parakeet or Whisper to add a line for your dataset. 

## Related issues
Closes #