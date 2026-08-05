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

## Multilingual

Same idea, but using `api/run_api_ml.sh`, which evaluates on FLEURS, MCV (Mozilla Common Voice), and MLS (Multilingual LibriSpeech).
- Set the relevant model in `MODEL_CONFIGS` of `api/run_api_ml.sh` with the model ID and the number max workers that the API allows, or pass `MODEL` via env var.
- Set the relevant dataset/language pairs in `DATASET_CONFIGS` of `api/run_api_ml.sh`, or pass `"dataset:language"` pairs via `DATASETS` (see examples below).

Smoke test (single model, single dataset/language):
```bash
AZURE_API_KEY=<YOUR_KEY> MODEL="microsoft/azure-speech 4" \
DATASETS="fleurs:de" \
bash api/run_api_ml.sh
```

To pass multiple dataset/language pairs via `DATASETS`, separate them with a space:
```bash
AZURE_API_KEY=<YOUR_KEY> MODEL="microsoft/azure-speech 4" \
DATASETS="fleurs:de mcv:de mls:it" \
bash api/run_api_ml.sh
```

Full run with bucket upload:
```bash
AZURE_API_KEY=<YOUR_KEY> MODEL="microsoft/azure-speech 4" \
HF_TOKEN=<YOUR_TOKEN> \
RESULTS_BUCKET=<YOUR_BUCKET> \
bash api/run_api_ml.sh
```