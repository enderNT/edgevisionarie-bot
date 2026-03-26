FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY app ./app
COPY config ./config
COPY datasets ./datasets
COPY scripts ./scripts

RUN pip install --upgrade pip setuptools wheel \
    && pip install .

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"APP_PORT\", \"8000\")}/health', timeout=3)" || exit 1

CMD ["sh", "-c", "uvicorn app.main:create_app --factory --host ${APP_HOST:-0.0.0.0} --port ${APP_PORT:-8000}"]
