# Nemotron-3-Nano-Omni ASR Evaluation

This folder contains evaluation scripts for [`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16) on the Open ASR Leaderboard.

## Supported Runners

| Script | Path | Notes |
|--------|------|-------|
| `run_nemotron_omni_vllm.sh` | vLLM OpenAI-compatible server | Production runner. Talks to `vllm serve` over HTTP. Verified on H100 and newer GPUs. |
| `run_nemotron_omni.sh` | HF transformers `model.generate()` | Reference runner. Loads the checkpoint in-process via `AutoModelForCausalLM`. Verified on A100 and newer GPUs. |

The vLLM path follows the model card's deployment guide (vllm 0.20.0, `--reasoning-parser nemotron_v3`, `--tool-call-parser qwen3_coder`, audio extras). The HF path drives the same checkpoint through `transformers.AutoModelForCausalLM` with `trust_remote_code=True`. Both share the Dockerfile.

## Docker usage (recommended)

From the **repository root**, build the Docker image:

```bash
docker build -t open-asr-nemotron-omni -f nemotron_omni/Dockerfile .
```

The image is layered on top of `vllm/vllm-openai:v0.20.0` and adds `vllm[audio]`, the leaderboard runtime deps, the OpenAI client, `num2words` for the shared normalizer, and the HF transformers extras (`accelerate`, `causal-conv1d`, `mamba-ssm`, `open-clip-torch`, `timm`, …) so the same image can drive either runner.

### Run the full sweep (vLLM path, default)

`run_docker.sh` starts the vLLM server, waits for `/health`, runs the 8-dataset leaderboard sweep, and shuts the server down on exit:

```bash
docker run --gpus all \
    -v $(pwd):/app \
    -v $HF_HOME:/root/.cache/huggingface \
    open-asr-nemotron-omni run_docker.sh
```

Results are written to `nemotron_omni/results/` and persist on the host since the repo is mounted.

### Run the full sweep (HF transformers path)

`run_docker_hf.sh` drives the 8-dataset sweep through `run_eval.py` (HF `model.generate()`) — no server is started; the model is loaded in-process by each invocation:

```bash
docker run --gpus all \
    -v $(pwd):/app \
    -v $HF_HOME:/root/.cache/huggingface \
    open-asr-nemotron-omni run_docker_hf.sh
```

Tunables go on the same env-var path as the vLLM runner (`MODEL_ID`, `BATCH_SIZE`, `BATCH_SIZE_VOXPOPULI`, `MAX_NEW_TOKENS`, `WARMUP_STEPS`, `DEVICE_ID`). Default `BATCH_SIZE=128` fits a single 96 GB GPU; voxpopuli has the longest clips and falls back to `BATCH_SIZE_VOXPOPULI=64`.

To select a specific GPU (e.g. GPU 1):

```bash
docker run --gpus '"device=1"' \
    -v $(pwd):/app \
    -v $HF_HOME:/root/.cache/huggingface \
    open-asr-nemotron-omni run_docker.sh
```

In environments where torch.compile codepath hits the `AlwaysHitShapeEnv` bug, pass `EAGER=1` to fall back to eager execution at the cost of ~10–25% decode throughput:

```bash
docker run --gpus all -e EAGER=1 \
    -v $(pwd):/app \
    -v $HF_HOME:/root/.cache/huggingface \
    open-asr-nemotron-omni run_docker.sh
```

### Run interactively

```bash
docker run --gpus all -it \
    -v $(pwd):/app \
    -v $HF_HOME:/root/.cache/huggingface \
    open-asr-nemotron-omni
```

This drops you into a bash shell inside `/app/nemotron_omni`. From there you can drive any of the runners directly. For the vLLM path you can split server and client across two terminals to avoid paying the ~1-minute server warm-up each time:

```bash
# Terminal A (or `&` in the background):
./run_server.sh

# Terminal B, once the server logs "Application startup complete":
./run_nemotron_omni_vllm.sh

# Or a single dataset / smoke test:
python run_eval_vllm.py \
    --model_id=nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
    --base_url=http://localhost:8000/v1 \
    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    --dataset="librispeech" --split="test.clean" \
    --batch_size=256 --concurrency=256 \
    --max_eval_samples=-1
```

For the HF transformers path:

```bash
# Full sweep:
./run_nemotron_omni.sh

# Single-dataset smoke test:
python run_eval.py \
    --model_id=nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    --dataset="librispeech" --split="test.clean" \
    --device=0 --batch_size=128 --max_eval_samples=-1
```

### Docker cheat sheet

- Exit and stop a container: type `exit` or press `Ctrl+D`.
- Detach without stopping: `Ctrl+P` then `Ctrl+Q`.
- List running containers: `docker ps -a`.
- Attach to a container: `docker attach <container_id>`.
- Delete a container: `docker rm <container_id>`.

## Local setup (without Docker)

From the repository root:

```bash
pip install -r requirements/requirements.txt
pip install -r requirements/requirements_nemotron_omni.txt
cd nemotron_omni

# Terminal A:
./run_server.sh
# Terminal B, once the server is ready:
./run_nemotron_omni_vllm.sh
```

`requirements_nemotron_omni.txt` pins the load-bearing versions: `vllm[audio]==0.20.0` (model-card requirement) and `torchcodec==0.11.1` (older torchcodec is ABI-incompatible with the torch 2.11 that vllm 0.20 pulls in). The Docker image sidesteps the torchcodec pin by setting `HF_AUDIO_DECODER_BACKEND=soundfile` instead.

## Notes

- The vLLM server is reused across all 8 datasets; the sweep script just drives concurrent OpenAI-style chat completions against it. Each request writes a short-lived WAV under `/tmp` and references it via a `file://` URL (the server is launched with `--allowed-local-media-path /`).
- ASR sampling matches the model card: `temperature=0.2`, `top_k=1`, `enable_thinking=False`.
- Override at runtime via env vars: `MODEL_ID`, `BASE_URL`, `BATCH_SIZE`, `CONCURRENCY`, `MAX_NEW_TOKENS`, `MOE_BACKEND`, `EAGER`.
