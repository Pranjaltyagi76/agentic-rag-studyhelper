"""/upload — accept a PDF for a session, ingest it, store its (session-tagged) chunks.

Input validation guards the endpoint before any expensive work runs:
- session_id must be a safe token (no path traversal into other sessions' folders).
- filename is reduced to its basename and must be a ``.pdf``.
- the body is streamed to disk in bounded chunks and rejected once it exceeds
  ``settings.MAX_UPLOAD_MB`` — so a huge (or non-PDF) upload can't exhaust memory or
  fill the disk before ingestion begins.
All rejections raise ``HTTPException`` and therefore come back in the standard
``{ "error": { code, message } }`` envelope.
"""

import os
import re

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import settings
from app.ingest import ingest
from app.persistence.vectorstore import vectordb
from app.persistence.db import get_db
from app.persistence.repository import add_document

router = APIRouter()

# Session ids are app-generated UUID-like tokens; constrain them so a crafted value
# (e.g. "../../etc") can never be used as a directory name to escape UPLOAD_FOLDER.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

# Read the upload in 1 MB slices so peak memory stays flat regardless of file size.
_CHUNK = 1024 * 1024


@router.post("/upload")
async def upload(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    # --- Validate the session id (also a filesystem path component below) ---
    if not _SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id.")

    # --- Validate the filename: strip any path parts, require a PDF ---
    filename = os.path.basename(file.filename or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")

    # Store files under a per-session folder so identical file names across sessions
    # never collide on disk.
    session_dir = os.path.join(settings.UPLOAD_FOLDER, session_id)
    os.makedirs(session_dir, exist_ok=True)
    file_path = os.path.join(session_dir, filename)

    # --- Stream to disk in bounded chunks, enforcing the size cap ---
    size_cap = settings.MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    try:
        with open(file_path, "wb") as f:
            while chunk := await file.read(_CHUNK):
                written += len(chunk)
                if written > size_cap:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {settings.MAX_UPLOAD_MB} MB limit.",
                    )
                f.write(chunk)
        if written == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    except HTTPException:
        # Don't leave a partial/rejected file lying around on disk.
        if os.path.exists(file_path):
            os.remove(file_path)
        raise

    chunks = ingest(file_path, filename, session_id)

    # Embed in batches. Passing every chunk at once made fastembed hold all the
    # embeddings in memory simultaneously, which OOM-killed the worker on a small
    # (512 MB) instance: a 20-page PDF -> ~55 chunks -> 502. Batching keeps peak
    # memory flat regardless of document size.
    for i in range(0, len(chunks), settings.EMBED_BATCH_SIZE):
        vectordb.add_documents(chunks[i : i + settings.EMBED_BATCH_SIZE])

    add_document(db, session_id, filename, len(chunks))

    return {
        "filename": filename,
        "chunk_count": len(chunks),
        "session_id": session_id,
    }
