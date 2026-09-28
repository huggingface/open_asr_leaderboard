# Duplicate of hf-audio/open-asr-leaderboard-transformers with the Phonon-2 container reader; zstd for the .bps archive.
# NeMo path (submitted): add `pip install -r requirements/requirements_nemo.txt` -- its `texterrors` wheel needs g++ and Python headers (both below).
FROM nvidia/cuda:12.9.0-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PIP_BREAK_SYSTEM_PACKAGES=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-dev build-essential g++ git libsndfile1 ffmpeg sox zstd && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3 /usr/bin/python
WORKDIR /app
RUN pip install --no-cache-dir torch==2.8.0 torchaudio==2.8.0 torchcodec==0.6.0 --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir git+https://github.com/huggingface/transformers.git evaluate datasets==3.6.0 librosa jiwer num2words kaldialign voi_oiwer soundfile
COPY . /app
ENTRYPOINT ["bash"]
EXPOSE 7860
CMD ["-c", "python3 -m http.server 7860"]
