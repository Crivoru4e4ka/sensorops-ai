# ─── SensorOps AI: Enterprise PoC ─────────────────────────────────────────────
# Python 3.10+ (в контейнере), ffmpeg нужен для декодирования аудио faster-whisper.
# База закреплена на Debian bookworm (trixie-репозитории нестабильны на некоторых сетях).
#
# INSTALL_TRANSFORMERS: по умолчанию false — образ собирается без torch
# (зеркало download.pytorch.org на многих сетях недоступно/медленное).
# При true ставится CPU-torch + sentence-transformers; при сбое сборка
# продолжается, и приложение автоматически использует TF-IDF fallback.
#   docker compose build --build-arg INSTALL_TRANSFORMERS=true
FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_RETRIES=5 \
    PIP_TIMEOUT=120

WORKDIR /app

RUN apt-get update -o Acquire::Retries=5 && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-embeddings.txt ./
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

ARG INSTALL_TRANSFORMERS=false
RUN if [ "$INSTALL_TRANSFORMERS" = "true" ]; then \
        (pip install --no-cache-dir --prefer-binary torch \
            --index-url https://download.pytorch.org/whl/cpu \
            --extra-index-url https://pypi.org/simple \
         && pip install --no-cache-dir --prefer-binary -r requirements-embeddings.txt) \
        || echo "WARN: torch/sentence-transformers не установлены — приложение перейдёт на TF-IDF fallback"; \
    fi

COPY . .

ENV QDRANT_URL=http://qdrant:6333 \
    VECTOR_BACKEND=qdrant

EXPOSE 8000 8501

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
