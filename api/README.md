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

## Multilingual benchmarks on GitHub Actions

The `Multilingual API benchmark` workflow runs every registered API provider
against the 13 standard multilingual configurations. It is manual/reusable by
design: paid credentials are never made available to pull-request code.

Before the first run:

1. Add the provider adapter under `api/providers/` and register it in
   `api/providers/__init__.py`.
2. Add any required SDK to `api/action-requirements.txt`.
3. Store the API key as a GitHub Actions repository or environment secret.
4. Open **Actions → Multilingual API benchmark → Run workflow**.

Supply the registered `provider/model` ID, the secret name, and the environment
variable expected by the adapter. Start with `max_samples=1`. A full paid run
requires the explicit value `max_samples=0`; this avoids an accidental full run
when accepting the defaults. `max_workers × max_parallel` is the effective
account-level concurrency, so `max_parallel=1` is the safe default.

Each dataset produces an immutable result artifact. Successful samples are
checkpointed, and failures upload the partial checkpoint. Set `reuse_run_id` to
a previous smoke/interrupted run to avoid paying for those samples again. Set
`result_run_id` to a complete run to regenerate its scores without API calls.
The final summary artifact contains a two-line CSV ready for review; the
workflow deliberately does not push leaderboard changes by itself.

The same workflow can be called from another workflow while passing the secret
explicitly:

```yaml
jobs:
  benchmark:
    uses: ./.github/workflows/multilingual-api-benchmark.yml
    with:
      model_id: assembly/universal-3-pro
      api_key_env: ASSEMBLYAI_API_KEY
      max_samples: 1
    secrets:
      provider_api_key: ${{ secrets.ASSEMBLYAI_API_KEY }}
      hf_token: ${{ secrets.HF_TOKEN }}
```

Fork contributors must run the workflow in their own fork with their own API
secret. Do not add a `pull_request_target` trigger or otherwise expose upstream
secrets to code from forks. For maintainers, a protected GitHub Environment
with required approval is recommended for full paid runs.
