"""/upload — accept a PDF for a session, ingest it, store its (session-tagged) chunks."""

import os

from fastapi import APIRouter, UploadFile, File, Form, Depends
from sqlalchemy.orm import Session

from app.config import settings
from app.ingest import ingest
from app.persistence.vectorstore import vectordb
from app.persistence.db import get_db
from app.persistence.repository import add_document

router = APIRouter()


@router.post("/upload")
async def upload(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    # Store files under a per-session folder so identical file names across sessions
    # never collide on disk.
    session_dir = os.path.join(settings.UPLOAD_FOLDER, session_id)
    os.makedirs(session_dir, exist_ok=True)
    file_path = os.path.join(session_dir, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    chunks = ingest(file_path, file.filename, session_id)
    vectordb.add_documents(chunks)

    add_document(db, session_id, file.filename, len(chunks))

    return {
        "filename": file.filename,
        "chunk_count": len(chunks),
        "session_id": session_id,
    }
