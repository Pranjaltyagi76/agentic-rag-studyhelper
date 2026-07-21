"""Repository layer (app/persistence/repository.py) against the temp SQLite DB.

list_filenames must de-duplicate repeated uploads and never leak another session's
files — the relational half of the isolation guarantee.
"""

from app.persistence.db import SessionLocal, init_db
from app.persistence.repository import add_document, list_filenames


def test_list_filenames_dedups_and_scopes_by_session():
    init_db()  # idempotent create_all on the temp DB from conftest
    db = SessionLocal()
    try:
        add_document(db, "repo-sess", "a.pdf", 3)
        add_document(db, "repo-sess", "a.pdf", 2)  # same file re-uploaded
        add_document(db, "repo-sess", "b.pdf", 1)
        add_document(db, "other-sess", "c.pdf", 1)  # different session

        files = list_filenames(db, "repo-sess")
        assert files == ["a.pdf", "b.pdf"]  # de-duped, first-seen order, session-scoped

        assert list_filenames(db, "other-sess") == ["c.pdf"]
        assert list_filenames(db, "empty-sess") == []
    finally:
        db.close()
