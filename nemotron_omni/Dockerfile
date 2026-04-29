FROM vllm/vllm-openai:v0.20.0

ENV DEBIAN_FRONTEND=noninteractive

# git for pip+git installs (HF normalizer wheels), libsndfile/ffmpeg for audio I/O.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# vllm/vllm-openai ships only `python3`; symlink `python` for the run scripts.
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Allow pip to install packages system-wide in the container (PEP 668)
ENV PIP_BREAK_SYSTEM_PACKAGES=1

WORKDIR /app

# Audio extras for vLLM (mistral-common[audio], soundfile, librosa, …) per the
# Nemotron Omni model card. The base image already has vllm 0.20.0, so this
# only fills in the audio-only deps.
RUN pip install --no-cache-dir "vllm[audio]==0.20.0"

# Common leaderboard requirements (jiwer, evaluate, datasets, num2words, …)
COPY requirements/requirements.txt /app/requirements/requirements.txt
RUN pip install --no-cache-dir -r requirements/requirements.txt

# Nemotron Omni vLLM-path extras: the OpenAI client used by run_eval_vllm.py,
# num2words for the shared normalizer, and torchcodec for HF datasets `Audio`
# decode. The base image ships torch 2.11; torchcodec 0.11.1 is the matching
# version (older releases hit an ABI mismatch and refuse to load).
RUN pip install --no-cache-dir openai num2words torchcodec==0.11.1

# Copy the full repo (the .dockerignore at the root keeps results/oldbkp out).
COPY . /app

WORKDIR /app/nemotron_omni

# vllm/vllm-openai sets ENTRYPOINT ["vllm", "serve"]; reset to bash so the
# orchestrator (run_docker.sh) can start the server itself and run the sweep.
ENTRYPOINT ["bash"]

# 8000 = vLLM server, 7860 = HF Spaces keep-alive. HF Jobs / `docker run`
# override the CMD with `run_docker.sh` (or any other script).
EXPOSE 7860 8000
CMD ["-c", "python3 -m http.server 7860"]
