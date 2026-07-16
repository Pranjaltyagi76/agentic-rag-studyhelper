# Multi-stage image for the Agentic RAG Study Helper.
# Binds to $PORT when the host sets one (Render, Cloud Run, ...), else 7860.
# See deployment.md.

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
# The frontend is served at "/" by app.main, so it must be in the image.
COPY StudySpace.html ./

ENV PORT=7860 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    VECTOR_BACKEND=pgvector \
    EMBED_CACHE_DIR=/app/.fastembed

# Bake the fastembed ONNX model into the image so the first request is fast and startup
# doesn't depend on the model host. It MUST go to EMBED_CACHE_DIR, not fastembed's
# default (/tmp/fastembed_cache): platforms mount a fresh tmpfs over /tmp at runtime,
# which would hide the baked model and trigger an ~83 MB re-download on first use.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2', cache_dir='/app/.fastembed')" \
    && test -d /app/.fastembed

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://localhost:%s/health' % os.environ.get('PORT','7860'))" || exit 1

# Shell form so ${PORT} expands: hosts like Render inject their own port.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
