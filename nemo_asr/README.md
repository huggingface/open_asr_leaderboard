# NeMo ASR Evaluation

This folder contains evaluation scripts for ASR models supported by NVIDIA NeMo.

## Supported Models

| Script | Models | Eval Script |
|--------|--------|-------------|
| `run_parakeet.sh` | Parakeet TDT/RNNT/CTC, FastConformer, Conformer | `run_eval.py` |
| `run_canary.sh` | Canary 1B, Canary 1B Flash, Canary 180M Flash | `run_eval.py` |
| `run_salm.sh` | Canary-Qwen 2.5B (SALM) | `run_eval_salm.py` |

### Multilingual

| Script | Models |
|--------|--------|
| `run_nemo_ml.sh` | Parakeet TDT, Canary 1B v2 |

Multilingual scripts evaluate on FLEURS, MCV (Mozilla Common Voice), and MLS (Multilingual LibriSpeech) for German, French, Italian, Spanish, and Portuguese. They use `run_eval_ml.py` which applies language-specific normalization.

### Longform

| Script | Models |
|--------|--------|
| `run_longform.sh` | Parakeet TDT/RNNT/CTC, FastConformer, Conformer |

Longform scripts evaluate on Earnings21, Earnings22, TED-LIUM, and CORAAL subsets using `run_eval_long.py` with `--longform` flag and batch size 1.

## Docker Usage (recommended)

From the **repository root**, build the Docker image:

```bash
docker build -t open-asr-nemo -f nemo_asr/Dockerfile .
```

### Run a specific script directly

From the **repository root**, you can run a script without entering the container. The command below uses `--gpus` to expose all GPUs, mounts the local repo so scripts reflect latest changes, and mounts the HuggingFace cache for model downloads:

```bash
docker run --gpus all \
    -v $(pwd):/app \
    -v $HF_HOME:/root/.cache/huggingface \
    open-asr-nemo run_parakeet.sh
```

Results are written to `nemo_asr/results/` and are automatically persisted on the host since the repo is mounted.

To select a specific GPU (e.g. GPU 1):

```bash
docker run --gpus '"device=1"' \
    -v $(pwd):/app \
    -v $HF_HOME:/root/.cache/huggingface \
    open-asr-nemo run_canary.sh
```

### Run interactively

From the **repository root**, you can also enter the container to run interactively:

```bash
docker run --gpus all -it \
    -v $(pwd):/app \
    -v $HF_HOME:/root/.cache/huggingface \
    open-asr-nemo -i
```

The `-i` at the end overrides the default HTTP server command and drops you into an interactive bash shell inside `/app/nemo_asr`. From there, run any evaluation script:

```bash
# Evaluate all Parakeet models
bash run_parakeet.sh

# Evaluate Canary models
bash run_canary.sh

# Evaluate SALM (Canary-Qwen)
bash run_salm.sh

# Evaluate a single model/dataset manually
python run_eval.py \
    --model_id=nvidia/parakeet-tdt-0.6b-v3 \
    --dataset_path="hf-audio/open-asr-leaderboard" \
    --dataset="librispeech" \
    --split="test.clean" \
    --device=0 \
    --batch_size=128 \
    --max_eval_samples=-1
```

### Docker cheat sheet

- Exit and stop a container, type `exit` or press `Ctrl+D`.
- Detach from a container (without stopping): `Ctrl+P` then `Ctrl+Q`.
- List running containers: `docker ps -a`.
- Attach to a container: `docker attach <container_id>`
- Delete a container: `docker rm <container_id>`

## Local Setup (without Docker)

From the repository root:

```bash
pip install -r requirements/requirements.txt
pip install "nemo_toolkit[asr]==2.7.2"
pip install lhotse sentencepiece   # only needed for SALM
pip install "cuda-python>=12.4"    # for fast TDT/RNN-T decoding
cd nemo_asr
bash run_parakeet.sh
```

## Notes

- Audio files are cached locally in `audio_cache/` to avoid re-downloading between runs. Use `--data_cache_root` to change the cache location.
- The `HF_AUDIO_DECODER_BACKEND=soundfile` environment variable is set in the Dockerfile and SALM scripts to avoid torchcodec/FFmpeg compatibility issues.
