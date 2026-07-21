"""Shared fixtures + a hermetic test environment.

The suite is deliberately hermetic: it exercises routing, config parsing, the error
envelopes, the structured-output salvage path, the session-isolation retrieval filter,
and the repository — WITHOUT ever calling Groq / Google / Tavily. No real API keys are
required, so it runs the same locally and in CI.

Two things are set up here, before any ``app`` module is imported (pytest imports
conftest first), so importing the app can never touch the developer's real data or a
live provider:

1. Every stateful backend (SQLite, checkpointer, Chroma) is redirected to a throwaway
   temp directory.
2. Dummy provider keys are injected so client objects (ChatGroq, google-genai,
   TavilySearch) *construct* offline. The tests never invoke them, so the dummy values
   are never validated against a real service.

``setdefault`` is used so an explicit value already in the environment still wins;
``app.config`` calls ``load_dotenv()`` with ``override=False``, so it won't clobber
what we set here either.
"""

import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="studyhelper-tests-")
_fwd = _TMP.replace("\\", "/")  # forward slashes for the sqlite:/// URL on Windows

# --- Isolate all local state to the temp dir ---
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_fwd}/app.db")
os.environ.setdefault("CHECKPOINT_DB", f"{_fwd}/checkpoints.sqlite")
os.environ.setdefault("CHROMA_DIR", f"{_fwd}/chroma")
os.environ.setdefault("CHROMA_COLLECTION", "test")
os.environ.setdefault("UPLOAD_FOLDER", f"{_fwd}/uploads")  # keep test uploads out of ./uploads
os.environ.setdefault("LANGCHAIN_API_KEY", "")  # keep tracing off during tests

# --- Dummy keys so provider clients construct offline (never actually called) ---
os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")
os.environ.setdefault("GOOGLE_API_KEY", "test-key-not-used")
os.environ.setdefault("TAVILY_API_KEY", "test-key-not-used")

# Stub the embedding model BEFORE importing the app. Importing app.persistence.vectorstore
# eagerly constructs FastEmbedEmbeddings, which downloads the ~83 MB ONNX model from the
# Hugging Face Hub. No test actually embeds (they all mock the vector store), and prod
# bakes the model into the Docker image — but a fresh CI runner making UNAUTHENTICATED HF
# requests gets rate-limited, so the download (and thus every test) fails at import. The
# stub removes that network dependency entirely; the suite stays hermetic and fast.
import fastembed


class _StubTextEmbedding:
    """No-op stand-in for fastembed.TextEmbedding — constructs without any download."""

    def __init__(self, *args, **kwargs):
        pass

    def _zeros(self, texts):
        for _ in texts:
            yield [0.0] * 384  # all-MiniLM-L6-v2 dimensionality

    def embed(self, texts, *args, **kwargs):
        return self._zeros(texts)

    def passage_embed(self, texts, *args, **kwargs):
        return self._zeros(texts)

    def query_embed(self, query, *args, **kwargs):
        return iter([[0.0] * 384])


fastembed.TextEmbedding = _StubTextEmbedding

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    # The context manager runs the lifespan hook (init_db against the temp SQLite).
    with TestClient(app) as c:
        yield c
