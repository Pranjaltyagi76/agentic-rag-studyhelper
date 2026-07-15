"""FastAPI application entry point.

Run locally with:  uvicorn app.main:app --reload --port 8000
On Hugging Face Spaces the container serves this on port 7860 (see deployment.md).

Phase 1: split the original app.py into routers + added /health.
Phase 2: session-scoped endpoints (session_id) + relational store; tables are created
on startup via the lifespan hook.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import settings
from app.persistence.db import init_db
from app.observability.tracing import setup_langsmith, setup_otel
from app.api import upload, chat, evaluate

logger = logging.getLogger("studyhelper.main")

# Enable LangSmith tracing as early as possible (no-op until LANGCHAIN_API_KEY is set).
setup_langsmith()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail fast if deploying on SQLite in production (#3) — it can't handle concurrency.
    if settings.is_production and settings.DATABASE_URL.startswith("sqlite"):
        raise RuntimeError(
            "APP_ENV=production but DATABASE_URL is SQLite. Set DATABASE_URL to a "
            "Postgres/Neon URL (SQLite is not safe for concurrent production use)."
        )
    # Create tables if they do not exist (SQLite locally / Neon Postgres in prod).
    init_db()
    yield


app = FastAPI(title="Agentic RAG Study Helper", lifespan=lifespan)

# Instrument the FastAPI layer with OpenTelemetry (Phase 8).
setup_otel(app)

# --- CORS (#1): configurable origins; credentials only allowed with explicit origins ---
_origins = settings.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=("*" not in _origins),  # credentialed wildcard is invalid/unsafe
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Optional API-key gate (#2): when APP_API_KEY is set, require it on every call ---
@app.middleware("http")
async def _api_key_gate(request: Request, call_next):
    if settings.APP_API_KEY and request.method != "OPTIONS" and request.url.path != "/health":
        provided = request.headers.get("x-api-key") or request.headers.get(
            "authorization", ""
        ).removeprefix("Bearer ").strip()
        if provided != settings.APP_API_KEY:
            return JSONResponse(
                status_code=401,
                content={"error": {"code": 401, "message": "Unauthorized"}},
            )
    return await call_next(request)


app.include_router(upload.router)
app.include_router(chat.router)
app.include_router(evaluate.router)


# --- Consistent error envelope (Phase 6): { "error": { code, message, detail? } } ---
@app.exception_handler(StarletteHTTPException)
async def _http_exc(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.status_code, "message": exc.detail}},
    )


@app.exception_handler(RequestValidationError)
async def _validation_exc(request: Request, exc: RequestValidationError):
    detail = [{"loc": list(e.get("loc", [])), "msg": e.get("msg")} for e in exc.errors()]
    return JSONResponse(
        status_code=422,
        content={"error": {"code": 422, "message": "Invalid request", "detail": detail}},
    )


@app.exception_handler(Exception)
async def _unhandled_exc(request: Request, exc: Exception):
    # Log the full error server-side; only expose details outside production (#4).
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    content = {"error": {"code": 500, "message": "Internal server error"}}
    if not settings.is_production:
        content["error"]["detail"] = str(exc)
    return JSONResponse(status_code=500, content=content)


@app.get("/health")
def health():
    return {"status": "ok"}


# Serve the frontend at "/" so the deployed URL IS the app (not a 404). Same origin as
# the API, so the browser needs no CORS and the page can use relative paths.
_FRONTEND = Path(__file__).resolve().parent.parent / "StudySpace.html"


@app.get("/", include_in_schema=False)
def index():
    if _FRONTEND.exists():
        return FileResponse(_FRONTEND)
    return JSONResponse({"service": "Agentic RAG Study Helper", "docs": "/docs"})
