FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

COPY pyproject.toml README.md ./

RUN pip install --upgrade pip && \
    pip install .

COPY . .

RUN chmod +x /app/entrypoint.sh

EXPOSE 8888

ENTRYPOINT ["/app/entrypoint.sh"]

CMD ["python", "examples/web_demo.py", "--model_path", "/app/ckpt", "--port", "8888"]
