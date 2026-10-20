FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir -r /app/requirements.txt
RUN git clone https://github.com/huggingface/open_asr_leaderboard.git /tmp/leaderboard \
    && git -C /tmp/leaderboard checkout 48e6f0d75ad7dad68c962112356fec44776f3bbb \
    && cp -r /tmp/leaderboard/normalizer /app/normalizer && rm -rf /tmp/leaderboard
COPY run_eval.py /app/run_eval.py
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
# HF Jobs overrides this command with the evaluator invocation.
CMD ["python", "-m", "http.server", "7860"]
