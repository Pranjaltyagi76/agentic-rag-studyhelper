"""FastAPI application entry point.

Run locally with:  uvicorn app.main:app --reload --port 8000
On Hugging Face Spaces the container serves this on port 7860 (see deployment.md).

Phase 1: split the original app.py into routers + added /health.
Phase 2: session-scoped endpoints (session_id) + relational store; tables are created
on startup via the lifespan hook.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


@app.get("/health")
def health():
    return {"status": "ok"}
