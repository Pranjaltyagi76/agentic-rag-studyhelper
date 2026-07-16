"""Centralized, env-driven configuration (requirements NFR-5).

All tunables live here so nothing is hard-coded in the modules. Defaults preserve
the exact values used by the original scripts, so Phase 1 stays behavior-preserving.
"""

import os

from dotenv import load_dotenv

# Load .env once, at import time, before anything reads os.getenv.
load_dotenv()


class Settings:
    # --- App / security ---
    APP_ENV: str = os.getenv("APP_ENV", "development")
    # Comma-separated allowed origins for CORS. "*" (default) = any origin (dev only).
    CORS_ALLOW_ORIGINS: str = os.getenv("CORS_ALLOW_ORIGINS", "*")
    # Optional API key gate: when set, every endpoint (except /health) requires it.
    APP_API_KEY: str | None = os.getenv("APP_API_KEY") or None

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.CORS_ALLOW_ORIGINS.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.APP_ENV.lower() == "production"

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

    # --- LangGraph checkpointer (Phase 5) ---
    # Separate SQLite file for durable graph state locally; on Postgres (Neon) the
    # checkpointer shares DATABASE_URL. Backend is chosen from DATABASE_URL's scheme.
    CHECKPOINT_DB: str = os.getenv("CHECKPOINT_DB", "checkpoints.sqlite")

    # --- Ingestion / OCR ---
    UPLOAD_FOLDER: str = os.getenv("UPLOAD_FOLDER", "uploads")
    # Chunks embedded per batch at upload. Small on purpose: embedding a whole document
    # at once peaks memory and OOM-kills small instances (Render free = 512 MB).
    EMBED_BATCH_SIZE: int = int(os.getenv("EMBED_BATCH_SIZE", "8"))
    OCR_MODEL: str = os.getenv("OCR_MODEL", "gemini-2.5-flash")
    GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")

    # --- Web research ---
    TAVILY_MAX_RESULTS: int = int(os.getenv("TAVILY_MAX_RESULTS", "5"))

    # --- Self-correcting retrieval (Phase 3) ---
    # Max retrieve->grade->rewrite loops before giving up and falling back to web /
    # general knowledge. Guarantees termination (design.md section 3).
    RETRIEVAL_MAX_ATTEMPTS: int = int(os.getenv("RETRIEVAL_MAX_ATTEMPTS", "2"))

    # --- Generate + verify loop / adaptive planner (Phase 4) ---
    # Max teacher (generate -> groundedness check -> regenerate) attempts.
    GENERATION_MAX_ATTEMPTS: int = int(os.getenv("GENERATION_MAX_ATTEMPTS", "2"))
    # Max adaptive re-plans on failure before the planner stops intervening.
    REPLAN_MAX: int = int(os.getenv("REPLAN_MAX", "3"))

    # --- Structured-output robustness (A16) ---
    # Retries for structured LLM calls before salvaging / defaulting.
    STRUCTURED_MAX_RETRIES: int = int(os.getenv("STRUCTURED_MAX_RETRIES", "2"))

    # --- Observability (Phase 8) ---
    # LangSmith tracing turns on automatically once LANGCHAIN_API_KEY is set; empty = off.
    LANGCHAIN_API_KEY: str | None = os.getenv("LANGCHAIN_API_KEY") or None
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "studyhelper")
    LANGCHAIN_ENDPOINT: str = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    # OpenTelemetry: FastAPI is always instrumented; spans are exported only if set.
    OTEL_EXPORTER_OTLP_ENDPOINT: str | None = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or None

    # --- Evaluation (Phase 8.5) — MLflow, local SQLite backend by default ---
    # (recent MLflow deprecated the ./mlruns file store; SQLite is the local backend.)
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "studyhelper-rag-eval")


settings = Settings()
