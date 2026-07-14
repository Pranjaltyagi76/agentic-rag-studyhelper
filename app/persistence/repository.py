"""Data-access helpers for sessions and documents.

Thin repository layer so routers never touch ORM queries directly.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.persistence.models import SessionRow, DocumentRow


def get_or_create_session(db: Session, session_id: str) -> SessionRow:
    row = db.get(SessionRow, session_id)
    if row is None:
        row = SessionRow(session_id=session_id)
        db.add(row)
        db.commit()
    return row


def add_document(
    db: Session, session_id: str, file_name: str, chunk_count: int
) -> DocumentRow:
    get_or_create_session(db, session_id)
    doc = DocumentRow(
        session_id=session_id,
        file_name=file_name,
        chunk_count=chunk_count,
    )
    db.add(doc)
    db.commit()
    return doc


def list_filenames(db: Session, session_id: str) -> list[str]:
    """Distinct file names for a session, preserving first-seen order."""
    rows = (
        db.execute(
            select(DocumentRow.file_name).where(DocumentRow.session_id == session_id)
        )
        .scalars()
        .all()
    )
    seen: list[str] = []
    for f in rows:
        if f not in seen:
            seen.append(f)
    return seen
