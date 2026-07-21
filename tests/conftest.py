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

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    # The context manager runs the lifespan hook (init_db against the temp SQLite).
    with TestClient(app) as c:
        yield c
