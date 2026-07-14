"""FastAPI application entry point.

Run locally with:  uvicorn app.main:app --reload
On Hugging Face Spaces the container serves this on port 7860 (see deployment.md).

Phase 1: assembles the same three endpoints as the original app.py, now split into
routers. ``/health`` is a small addition used by Docker/HF healthchecks (deployment
.md section 7); the streaming variant of ``/chat`` arrives in Phase 6.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import upload, chat, evaluate

app = FastAPI(title="Agentic RAG Study Helper")

# NOTE (audit A9): wildcard CORS is carried over from the original app.py for
# behavior parity; tighten before deploy (Phase 7).
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
