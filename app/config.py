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
    # Model for the chunk-grading step. Tunable for model cascading, but defaults to the
    # reasoning model on purpose: the ablation showed llama-3.1-8b-instant too lenient —
    # it kept irrelevant chunks on unanswerable questions, which broke abstention and
    # sent hallucination_rate 0.00 -> 1.00. Cost must not come at the core metric's
    # expense, so grading stays on the capable model. (The token win below comes from
    # truncation + the k-cap, not from a cheaper grader.)
    GRADER_MODEL: str = os.getenv("GRADER_MODEL", GROQ_MODEL)

    # --- Embeddings (canonical: HF all-MiniLM-L6-v2, both ingest + query; decision 2026-07-14) ---
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    # Where fastembed keeps the ONNX model. Unset locally (its default is fine). In the
    # container this MUST be set to a non-/tmp path: the image bakes the model in at
    # build time, but platforms mount a fresh tmpfs over /tmp at runtime, which hides it
    # and forces an ~83 MB re-download on the first request.
    EMBED_CACHE_DIR: str | None = os.getenv("EMBED_CACHE_DIR") or None

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
    # Max accepted upload size (MB). The file is streamed to disk in chunks and rejected
    # once it crosses this, so a large or malicious upload can't exhaust memory on a
    # small (512 MB) instance before ingestion even starts.
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "20"))
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
    # Hard cap on chunks retrieved per attempt. The planner's own guidance can request
    # up to 30 ("whole chapter"), which on a real textbook makes one request cost ~20k
    # tokens (a fifth of the free daily budget). Capping ~halves cost.
    RETRIEVAL_K_CAP: int = int(os.getenv("RETRIEVAL_K_CAP", "15"))
    # Only the first N chars of each chunk go into the GRADE prompt — enough to judge
    # relevance, ~70% fewer tokens than the full chunk. Full chunks still reach the
    # generator; this trims only the grader's input.
    GRADE_CHUNK_CHARS: int = int(os.getenv("GRADE_CHUNK_CHARS", "320"))

    # --- Generate + verify loop / adaptive planner (Phase 4) ---
    # Max teacher (generate -> groundedness check -> regenerate) attempts.
    GENERATION_MAX_ATTEMPTS: int = int(os.getenv("GENERATION_MAX_ATTEMPTS", "2"))
    # Max adaptive re-plans on failure before the planner stops intervening.
    REPLAN_MAX: int = int(os.getenv("REPLAN_MAX", "3"))

    # --- Structured-output robustness (A16) ---
    # Retries for structured LLM calls before salvaging / defaulting.
    STRUCTURED_MAX_RETRIES: int = int(os.getenv("STRUCTURED_MAX_RETRIES", "2"))

    # --- Groq rate-limit (429) backoff ---
    # We disable the Groq SDK's opaque internal retry and own one explicit, logged
    # backoff layer (app/agent/llm.py:invoke_with_backoff). On a 429 it waits the
    # server's Retry-After when given, else capped exponential backoff with jitter.
    RATE_LIMIT_MAX_RETRIES: int = int(os.getenv("RATE_LIMIT_MAX_RETRIES", "5"))
    RATE_LIMIT_BASE_DELAY: float = float(os.getenv("RATE_LIMIT_BASE_DELAY", "1.0"))
    RATE_LIMIT_MAX_DELAY: float = float(os.getenv("RATE_LIMIT_MAX_DELAY", "30.0"))

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
