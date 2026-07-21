"""/upload input validation.

The rejection paths fire before any ingestion/embedding, so they're hermetic on their
own. The happy path mocks ``ingest`` (no PyPDFLoader/Gemini) and the embedding call
(no fastembed), but still exercises the real streaming-to-disk + repository write.
"""

import io

import app.api.upload as upload_mod
from langchain_core.documents import Document


def _pdf(name="notes.pdf", content=b"%PDF-1.4 minimal", mime="application/pdf"):
    return {"file": (name, io.BytesIO(content), mime)}


def test_rejects_non_pdf(client):
    r = client.post(
        "/upload",
        data={"session_id": "sess1"},
        files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert r.status_code == 415
    assert r.json()["error"]["code"] == 415


def test_rejects_bad_session_id(client):
    # A path-traversal session id must never become a directory name.
    r = client.post("/upload", data={"session_id": "../../etc"}, files=_pdf())
    assert r.status_code == 400


def test_rejects_empty_file(client):
    r = client.post("/upload", data={"session_id": "sess1"}, files=_pdf(content=b""))
    assert r.status_code == 400


def test_rejects_oversize(client, monkeypatch):
    # Shrink the cap to a few bytes, then send a body comfortably past it.
    monkeypatch.setattr(upload_mod.settings, "MAX_UPLOAD_MB", 1 / (1024 * 1024))  # ~1 byte
    r = client.post("/upload", data={"session_id": "sess1"}, files=_pdf(content=b"x" * 500))
    assert r.status_code == 413
    assert r.json()["error"]["code"] == 413


def test_accepts_valid_pdf(client, monkeypatch):
    fake_chunks = [
        Document(page_content="chunk", metadata={"session_id": "okpath", "file_name": "notes.pdf"})
    ]
    monkeypatch.setattr(upload_mod, "ingest", lambda path, name, sid: fake_chunks)
    monkeypatch.setattr(upload_mod.vectordb, "add_documents", lambda docs: None)

    r = client.post("/upload", data={"session_id": "okpath"}, files=_pdf())
    assert r.status_code == 200
    body = r.json()
    assert body == {"filename": "notes.pdf", "chunk_count": 1, "session_id": "okpath"}


def test_sanitizes_traversal_in_filename(client, monkeypatch):
    seen = {}

    def _capture_name(path, name, sid):
        seen["name"] = name
        return []

    monkeypatch.setattr(upload_mod, "ingest", _capture_name)
    monkeypatch.setattr(upload_mod.vectordb, "add_documents", lambda docs: None)

    r = client.post("/upload", data={"session_id": "okpath2"}, files=_pdf(name="../../evil.pdf"))
    assert r.status_code == 200
    # The stored/reported name is the basename only — no path components survive.
    assert r.json()["filename"] == "evil.pdf"
    assert seen["name"] == "evil.pdf"
