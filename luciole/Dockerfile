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
    build-essential \
    cmake \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set Python alias (Ubuntu 24.04 ships Python 3.12)
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Allow pip to install packages system-wide in the container (PEP 668)
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Set working directory
WORKDIR /app

# uv installs these in about a minute, where pip's resolver backtracks for 20+ min
RUN pip install --no-cache-dir uv

# Install the LINAGORA NeMo fork, whose `nemo_speechlm` vLLM plugin serves Luciole-8B-Audio,
# then vLLM, in separate steps as in the model card: the two cannot be resolved together
# (vllm==0.28.0 pins quack-kernels==0.6.4, NeMo's nemo-automodel pins 0.6.1), so vLLM's
# pins take precedence. The fork is pinned to a commit for reproducibility.
RUN uv pip install --system --break-system-packages --no-cache \
    "nemo-toolkit[speechlm2,tts] @ git+https://github.com/linagora-labs/NeMo.git@be29ca83d0ff22577f0dff62c178b711ee760d16"
RUN uv pip install --system --break-system-packages --no-cache \
    vllm==0.28.0 \
    transformers==5.12.1 \
    "numpy<=2.4"

# Install evaluation requirements
RUN uv pip install --system --break-system-packages --no-cache \
    evaluate \
    datasets \
    librosa \
    soundfile \
    jiwer \
    num2words \
    "kaldialign>=0.12.0" \
    && python -c "from kaldialign import batch_error_rate; print('kaldialign OK')"

# Force soundfile backend for datasets audio decoding (avoids torchcodec/FFmpeg issues)
ENV HF_AUDIO_DECODER_BACKEND=soundfile

# vLLM's FlashInfer sampler JIT-compiles its kernels with nvcc, which the CUDA runtime
# image lacks; use vLLM's PyTorch sampler instead (decoding is greedy anyway)
ENV VLLM_USE_FLASHINFER_SAMPLER=0

# Copy the full repository
COPY . /app

# Default entrypoint
ENTRYPOINT ["bash"]

# Keep-alive CMD so the Space runtime stays healthy. HF Jobs and `docker run`
# override this with their own command (e.g. `python run_eval.py ...`).
EXPOSE 7860
CMD ["-c", "python3 -m http.server 7860"]
