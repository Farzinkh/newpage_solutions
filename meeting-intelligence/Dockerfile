# Backend image. The UI runs separately (see docker-compose).
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install deps first for layer caching.
COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install "fastapi[standard]" uvicorn \
    "pydantic>=2" "pydantic-settings>=2" "chromadb>=0.5" "sentence-transformers>=2.7"

COPY app ./app
COPY data ./data
COPY eval ./eval
COPY ui ./ui
COPY scripts ./scripts

EXPOSE 8000
# Non-root for a slightly better security posture.
RUN useradd -m appuser && chown -R appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
