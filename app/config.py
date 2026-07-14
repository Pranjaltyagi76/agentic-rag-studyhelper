"""Centralized, env-driven configuration (requirements NFR-5).

All tunables live here so nothing is hard-coded in the modules. Defaults preserve
the exact values used by the original scripts, so Phase 1 stays behavior-preserving.
"""

import os

from dotenv import load_dotenv

# Load .env once, at import time, before anything reads os.getenv.
load_dotenv()


class Settings:
    # --- LLM (reasoning) ---
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # --- Embeddings (canonical: HF all-MiniLM-L6-v2, both ingest + query; decision 2026-07-14) ---
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # --- Vector store ---
    # pgvector is the canonical deploy backend (Phase 9); Chroma is local-dev only.
    VECTOR_BACKEND: str = os.getenv("VECTOR_BACKEND", "chroma")
    CHROMA_DIR: str = os.getenv("CHROMA_DIR", "Chromadb")
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "Stuff")

    # --- Relational store (sessions, documents) ---
    # Local dev/test defaults to SQLite (zero setup); deploy sets DATABASE_URL to
    # Neon Postgres. Same code, config-only swap — mirrors the Chroma/pgvector split.
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./studyhelper.db")

    # --- Ingestion / OCR ---
    UPLOAD_FOLDER: str = os.getenv("UPLOAD_FOLDER", "uploads")
    OCR_MODEL: str = os.getenv("OCR_MODEL", "gemini-2.5-flash")
    GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")

    # --- Web research ---
    TAVILY_MAX_RESULTS: int = int(os.getenv("TAVILY_MAX_RESULTS", "5"))

    # --- Self-correcting retrieval (Phase 3) ---
    # Max retrieve->grade->rewrite loops before giving up and falling back to web /
    # general knowledge. Guarantees termination (design.md section 3).
    RETRIEVAL_MAX_ATTEMPTS: int = int(os.getenv("RETRIEVAL_MAX_ATTEMPTS", "2"))


settings = Settings()
