# Transformers-library ASR Evaluation

This folder contains evaluation scripts for ASR models supported by the 🤗 Transformers library.

## Supported Models

| Script | Models |
|--------|--------|
| `run_whisper.sh` | OpenAI Whisper, Distil-Whisper, CrisperWhisper |
| `run_wav2vec2.sh` | Wav2Vec2 |
| `run_wav2vec2_conformer.sh` | Wav2Vec2 Conformer |
| `run_hubert.sh` | HuBERT |
| `run_data2vec.sh` | Data2Vec |
| `run_mms.sh` | MMS |
| `run_moonshine.sh` | Moonshine, Moonshine Streaming |
| `run_voxtral.sh` | Voxtral Mini, Voxtral Small |
| `run_voxtral_realtime.sh` | Voxtral Realtime |
| `run_vibevoice.sh` | VibeVoice |
| `run_glm_asr.sh` | GLM-ASR |
| `run_granite.sh` | Granite Speech |

### Multilingual

| Script | Models |
|--------|--------|
| `run_whisper_ml.sh` | OpenAI Whisper (multilingual) |
| `run_voxtral_ml.sh` | Voxtral Mini, Voxtral Small |
| `run_voxtral_realtime_ml.sh` | Voxtral Realtime |

Multilingual scripts evaluate on FLEURS, MCV (Mozilla Common Voice), and MLS (Multilingual LibriSpeech) for German, French, Italian, Spanish, and Portuguese. They use `run_eval_ml.py` which applies language-specific normalization. By default, models auto-detect the language during inference as per the leaderboard convention. The argument `--language` can be used to force a specific language.

## Docker usage (recommended)

From the **repository root**, build the Docker image:

```bash
docker build -t open-asr-transformers -f transformers/Dockerfile .
```

### Run a specific script directly

From the **repository root**, you can run a script without entering the container. The command below uses `--gpus` to expose all GPUs, mounts the local repo so scripts reflect latest changes, and mounts the HuggingFace cache for model downloads:

```bash
docker run --gpus all \
    -v $(pwd):/app \
    -v $HF_HOME:/root/.cache/huggingface \
    open-asr-transformers run_whisper.sh
```

Results are written to `transformers/results/` and are automatically persisted on the host since the repo is mounted.

To select a specific GPU (e.g. GPU 1):

```bash
docker run --gpus '"device=1"' \
    -v $(pwd):/app \
    -v $HF_HOME:/root/.cache/huggingface \
    open-asr-transformers run_whisper.sh
```

### Run interactively

From the **repository root**, you can also enter the container to run interactively:

```bash
docker run --gpus all -it \
    -v $(pwd):/app \
    -v $HF_HOME:/root/.cache/huggingface \
    open-asr-transformers -i
```

This drops you into a bash shell inside `/app/transformers`. From there, run any evaluation script:

```bash
# Evaluate all Whisper models
bash run_whisper.sh

# Evaluate Granite models
bash run_granite.sh

# Evaluate a single model/dataset manually
python run_eval.py \
    --model_id=openai/whisper-large-v3-turbo \
    --dataset_path="hf-audio/open-asr-leaderboard" \
    --dataset="librispeech" \
    --split="test.clean" \
    --device=0 \
    --batch_size=64 \
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
pip install "mistral-common[audio]>=1.9.0"   # only needed for Voxtral
pip install peft    # for Granite
cd transformers
bash run_whisper.sh
```
