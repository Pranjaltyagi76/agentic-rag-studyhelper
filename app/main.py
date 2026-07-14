"""FastAPI application entry point.

Run locally with:  uvicorn app.main:app --reload --port 8000
On Hugging Face Spaces the container serves this on port 7860 (see deployment.md).

Phase 1: split the original app.py into routers + added /health.
Phase 2: session-scoped endpoints (session_id) + relational store; tables are created
on startup via the lifespan hook.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.persistence.db import init_db
from app.api import upload, chat, evaluate


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables if they do not exist (SQLite locally / Neon Postgres in prod).
    init_db()
    yield


app = FastAPI(title="Agentic RAG Study Helper", lifespan=lifespan)

# NOTE (audit A9): wildcard CORS is carried over for behavior parity; tighten before
# deploy (Phase 7).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    return JSONResponse(
        status_code=500,
        content={"error": {"code": 500, "message": "Internal server error", "detail": str(exc)}},
    )


@app.get("/health")
def health():
    return {"status": "ok"}
