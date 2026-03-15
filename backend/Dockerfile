FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    curl \
    ca-certificates \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
COPY requirements.server.txt /tmp/requirements.server.txt

RUN python3 - <<'PY'
from pathlib import Path
source = Path('/tmp/requirements.server.txt')
filtered = []
for line in source.read_text(encoding='utf-8').splitlines():
    if line.strip().startswith('torch'):
        continue
    filtered.append(line)
Path('/tmp/requirements.docker.txt').write_text('\n'.join(filtered) + '\n', encoding='utf-8')
PY

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install -r /tmp/requirements.docker.txt && \
    python3 -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio

COPY backend /app/backend
COPY data /app/data
COPY models /app/models
COPY .env.example /app/.env.example

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
