FROM python:3.12-slim

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for audio I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Allow pip to install packages system-wide in the container (PEP 668)
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Set working directory
WORKDIR /app

# ── Core eval dependencies ────────────────────────────────────────────────────
# pin datasets to 2.19.0 to avoid torchcodec dependency
RUN pip install --no-cache-dir \
    transformers \
    evaluate \
    datasets==2.19.0 \
    jiwer \
    num2words \
    regex \
    soundfile \
    librosa \
    requests \
    tqdm \
    python-dotenv \
    kaldialign==0.12.0

# ── Provider-specific SDKs ────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    openai \
    assemblyai \
    elevenlabs \
    rev_ai \
    speechmatics-python \
    httpx \
    requests-toolbelt \
    gladiaio-sdk

# Force soundfile backend for datasets audio decoding
ENV HF_AUDIO_DECODER_BACKEND=soundfile

# Copy the full repository
# Pass --build-arg CACHEBUST=$(date +%s) to force re-copy of repo files
# without re-running the expensive pip install layers above.
ARG CACHEBUST=1
COPY . /app

# Default working directory for running api scripts
WORKDIR /app/api

# Default entrypoint
ENTRYPOINT ["bash"]

EXPOSE 7860
CMD ["-c", "python3 -m http.server 7860"]
