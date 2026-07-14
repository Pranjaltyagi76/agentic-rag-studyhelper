# Multi-stage image for the Agentic RAG Study Helper.
# Serves on port 7860 (Hugging Face Spaces convention). See deployment.md.

# --- builder: install dependencies into an isolated prefix ---
FROM python:3.13-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- runtime ---
FROM python:3.13-slim

WORKDIR /app

# Bring in the installed packages from the builder.
COPY --from=builder /install /usr/local

COPY app/ app/

ENV PORT=7860 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    VECTOR_BACKEND=pgvector

# Pre-download the fastembed ONNX model into the image so the first request is fast
# and startup does not depend on model-host availability.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
