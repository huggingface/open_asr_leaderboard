FROM nvidia/cuda:12.9.0-runtime-ubuntu24.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
# Ubuntu 24.04 ships FFmpeg 6.1 (torchcodec requires FFmpeg 5+, <8)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set Python alias (Ubuntu 24.04 ships Python 3.12)
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Allow pip to install packages system-wide in the container (PEP 668)
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Set working directory
WORKDIR /app

# Install PyTorch ecosystem (cu128 wheels for CUDA 12.8+/12.9 compat)
RUN pip install --no-cache-dir \
    torch==2.8.0 \
    torchaudio==2.8.0 \
    torchcodec==0.6.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Copy and install common requirements (torch already installed above, pip will skip it)
COPY requirements/requirements.txt /app/requirements/requirements.txt
RUN pip install --no-cache-dir -r requirements/requirements.txt

# Install NeMo toolkit with ASR extras
RUN pip install --no-cache-dir \
    "nemo_toolkit[asr]==2.7.2"

# Install SALM dependencies (lhotse for audio loading, transformers for GenerationConfig)
RUN pip install --no-cache-dir \
    lhotse \
    sentencepiece

# CUDA-accelerated TDT and RNN-T decoding
RUN pip install --no-cache-dir \
    "cuda-python>=12.4"

# Force soundfile backend for datasets audio decoding (avoids torchcodec/FFmpeg issues)
ENV HF_AUDIO_DECODER_BACKEND=soundfile

# Copy the full repository
COPY . /app

# Default working directory for running nemo scripts
WORKDIR /app/nemo_asr

# Default entrypoint
ENTRYPOINT ["bash"]

# Keep-alive CMD so the Space runtime stays healthy. HF Jobs and `docker run`
# override this with their own command (e.g. `run_parakeet.sh`).
EXPOSE 7860
CMD ["-c", "python3 -m http.server 7860"]
