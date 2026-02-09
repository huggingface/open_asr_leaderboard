# Gemini ASR Evaluation

This folder contains a self-contained flow to evaluate Google Gemini models on the Open ASR Leaderboard datasets. It aligns with the repository’s template: consistent dataset loading, normalization, manifest writing, WER/RTFx computation, and scripts to run and score.

## Quick Start

1) Install dependencies (from repo root or ensure they’re available):

```bash
# From repo root
pip install -r ../requirements/requirements.txt

# From gemini/ (adds Gemini client)
pip install -r requirements_gemini.txt
```

2) Provide your Gemini API key:

```bash
# Option A (recommended): Put it in gemini/.env
echo "GOOGLE_API_KEY=your_api_key_here" > .env

# Option B: Export it in your shell for this session
export GOOGLE_API_KEY=your_api_key_here   # bash/zsh
# or
$env:GOOGLE_API_KEY = "your_api_key_here"  # PowerShell
```

3) Ensure Python can import the repo’s modules. Scripts set `PYTHONPATH` automatically. For manual runs from gemini/:

```bash
export PYTHONPATH="$(pwd)/.."          # bash/zsh
# or
$env:PYTHONPATH = ".."                 # PowerShell
```

## How It Works

- Data loading: For English, audio is accessed without automatic decoding to avoid `torchcodec`; audio bytes/paths are read via `soundfile` and cached under `gemini/audio_cache/<dataset>/<split>`. Multilingual follows the shared pattern.
- Normalization: References and predictions are normalized (English: `EnglishTextNormalizer`; multilingual: `BasicMultilingualTextNormalizer`).
- Transcription: Each audio file is uploaded to the Gemini API, transcribed with retries + exponential backoff, then cleaned up.
- Outputs: Each run writes a JSONL manifest under `gemini/results/` and prints WER and RTFx.
- Scoring: `normalizer/eval_utils.score_results` aggregates across results and prints per-dataset and composite metrics.

## Run Individual Evaluations

English (run_eval.py):

```bash
python run_eval.py \
  --model_id "gemini/gemini-2.5-pro" \
  --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
  --dataset "ami" \
  --split "test" \
  --max_eval_samples 2
```

Multilingual (run_eval_ml.py):

```bash
python run_eval_ml.py \
  --model_id "gemini/gemini-2.5-pro" \
  --dataset "nithinraok/asr-leaderboard-datasets" \
  --config_name "fleurs_en" \
  --language "en" \
  --split "test" \
  --max_eval_samples 2
```

Notes:
- `--model_id` must start with `gemini/` (e.g., `gemini/gemini-2.5-pro`, `gemini/gemini-2.5-flash`).
- English script loads audio offline via bytes/path—no `torchcodec` required.

## Run Full Benchmark Suite

Both scripts resolve Python from PATH automatically; you can override with `PYTHON_CMD`.

- Bash (Linux/macOS):
```bash
chmod +x run_gemini.sh
./run_gemini.sh
```

- PowerShell (Windows):
```powershell
./run_gemini.ps1
```

Behavior:
- Auto-loads `gemini/.env` if present (so you don’t need to export `GOOGLE_API_KEY` manually).
- Sets `PYTHONPATH` to the repo root automatically.
- Runs a short smoke test first, then loops through core English datasets (and multilingual configs) with a small sample size for validation. Adjust sample sizes and datasets in the scripts as needed.

## Scoring Results

Score all manifests under `gemini/results/` for a given model id:

```bash
python -c "import normalizer.eval_utils as e; e.score_results('gemini/results', 'gemini/gemini-2.5-pro')"
```

This prints per-dataset WER and RTFx and a composite WER/RTFx by model.

## Environment Variables

- `GOOGLE_API_KEY` (required): Gemini API key. Set via `.env` or your shell.
- `PYTHONPATH`: Path to the repo root. Scripts set this automatically; for manual runs set it to `..` from inside `gemini/`.
- `PYTHON_CMD` (optional): Override which Python to use in the scripts (e.g., `PYTHON_CMD=/path/to/python`).
- `HF_TOKEN` (optional): Hugging Face token (only needed for private datasets).

## Troubleshooting

- Missing packages: Install both the repo requirements and `requirements_gemini.txt`.
- API key errors: Ensure `GOOGLE_API_KEY` is set. Scripts read `.env` automatically.
- Exec permissions (Linux/macOS): `chmod +x run_gemini.sh`.
- Torchcodec errors: English script reads audio from bytes/paths with `soundfile` and does not require `torchcodec`.

## Files

- `run_eval.py`: English evaluation script (Gemini transcription + WER/RTFx + manifest writing).
- `run_eval_ml.py`: Multilingual evaluation script.
- `run_gemini.sh`/`run_gemini.ps1`: Batch runners (auto-load `.env`, resolve Python, set `PYTHONPATH`).
- `requirements_gemini.txt`: Gemini client dependency.
- `audio_cache/`, `results/`: Local outputs (cached audio and manifests).
