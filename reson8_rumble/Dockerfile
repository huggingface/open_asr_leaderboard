FROM vllm/vllm-openai:v0.30.0@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl mawk \
    && rm -rf /var/lib/apt/lists/*

ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_THREAD_LIMIT=1 \
    PYTHONUNBUFFERED=1

COPY requirements/requirements_reson8_rumble.txt /app/requirements/requirements_reson8_rumble.txt
RUN uv pip install --system -r requirements/requirements_reson8_rumble.txt

COPY normalizer/ /app/normalizer/
COPY reson8_rumble/*.py reson8_rumble/serve_vllm.sh reson8_rumble/run_rumble.sh /app/reson8_rumble/
COPY LICENSE /app/LICENSE
