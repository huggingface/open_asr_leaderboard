# Running API models

Instead of using HF jobs, as no GPU is required, you can run a local Docker container.

Hugging Face setup for writing results to HF bucket:
1. Create an account at https://huggingface.co/ and add credits for HF Jobs: https://huggingface.co/settings/billing
2. Create a [WRITE token](https://huggingface.co/settings/tokens/new?tokenType=write) and copy it.
3. Create a Storage Bucket to store results: https://huggingface.co/new-bucket


Below is example usage for a script that will both build the Docker container and run the evaluation.
- Set the relevant model in `MODEL_CONFIGS` of `api/run_api.sh` with the model ID and the number max workers that the API allows, or pass `MODEL` via env var (see examples below).
- Make sure your API key is set in the environment.

Smoke test (single model, single dataset):
```bash
ZOOM_API_KEY=<YOUR_KEY> MODEL="zoom/scribe_v1 32" \
DATASETS="librispeech:test.clean" \
bash api/run_api.sh
```

Full run with bucket upload:
```bash
ZOOM_API_KEY=<YOUR_KEY> MODEL="zoom/scribe_v1 32" \
HF_TOKEN=<YOUR_TOKEN> \
RESULTS_BUCKET=<YOUR_BUCKET> \
bash api/run_api.sh
```

Multilingual Soniox Async evaluation (FLEURS, Common Voice, and MLS):
```bash
cd api
SONIOX_API_KEY=<YOUR_KEY> \
MODEL_ID=soniox/stt-async-v5 \
MAX_WORKERS=20 \
bash run_api_ml.sh
```

Add `MAX_SAMPLES=1` for a smoke test. If the Hugging Face Dataset Viewer is
available for the benchmark dataset, add `USE_URL=1` to let Soniox fetch the
public audio URLs directly instead of downloading the Parquet audio locally.
The multilingual runner checkpoints completed samples by default and resumes
after interruption; set `RESUME=0` to disable this. Soniox requests are capped
at 240 file-management and 240 transcription requests per minute by default.
Override `SONIOX_FILE_REQUESTS_PER_MINUTE` or
`SONIOX_TRANSCRIPTION_REQUESTS_PER_MINUTE` only when the corresponding Soniox
project limits are known to be different.

Soniox model documentation: https://soniox.com/docs/stt/models

The full multilingual run can also be executed without local compute through
the `Soniox multilingual benchmark` GitHub Actions workflow. Configure
`SONIOX_API_KEY` as a repository Actions secret, then dispatch the workflow.
Set `max_samples` to `1` for a 13-config smoke test or leave it empty for the
full benchmark. Dataset configs run serially to share one Soniox project rate
limit safely; result manifests and the final CSV row are uploaded as workflow
artifacts.
