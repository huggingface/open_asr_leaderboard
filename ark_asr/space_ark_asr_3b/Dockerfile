FROM nvidia/cuda:12.9.0-runtime-ubuntu24.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies.
# Ubuntu 24.04 ships FFmpeg 6.1 (torchcodec requires FFmpeg 5+, <8).
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set Python alias (Ubuntu 24.04 ships Python 3.12).
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Allow pip to install packages system-wide in the container (PEP 668).
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Set working directory.
WORKDIR /app

# Install PyTorch ecosystem (cu128 wheels for CUDA 12.8+/12.9 compat).
RUN pip install --no-cache-dir \
    torch==2.8.0 \
    torchaudio==2.8.0 \
    torchcodec==0.6.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Install common Open ASR Leaderboard requirements and ARK-ASR dependencies.
RUN pip install --no-cache-dir \
    transformers \
    accelerate \
    evaluate \
    datasets \
    librosa \
    jiwer \
    num2words \
    numpy \
    tqdm \
    pyarrow \
    soundfile \
    kaldialign==0.11.0

# Copy the Space files.
COPY . /app

# Default entrypoint.
ENTRYPOINT ["bash"]

# Keep-alive CMD so the Space runtime stays healthy. HF Jobs and `docker run`
# override this with their own command.
EXPOSE 7860
CMD ["-c", "python3 -m http.server 7860"]
