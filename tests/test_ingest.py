"""Real-PDF ingestion — exercises the actual PyPDFLoader + chunker path.

Unlike test_upload.py (which mocks ``ingest``), this feeds a genuine PDF through
``ingest`` and the /upload endpoint. Only the embedding call is mocked, so no model
download / network is needed; the text is kept >100 chars so ``load_pdf`` never falls
back to the Gemini OCR path.
"""

import io

import app.api.upload as upload_mod
from app.ingest import ingest

_TEXT = (
    "Photosynthesis is the process by which green plants use sunlight to synthesize "
    "food from carbon dioxide and water, releasing oxygen as a byproduct."
)


def _make_pdf(text: str) -> bytes:
    """A minimal single-page PDF with extractable text (computes its own xref)."""
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
    ]
    stream = b"BT /F1 18 Tf 72 700 Td (" + text.encode("latin-1") + b") Tj ET"
    objs.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    out += b"xref\n0 " + str(len(objs) + 1).encode() + b"\n0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += b"trailer\n<< /Size " + str(len(objs) + 1).encode() + b" /Root 1 0 R >>\n"
    out += b"startxref\n" + str(xref_pos).encode() + b"\n%%EOF"
    return out


def test_ingest_produces_session_tagged_chunks(tmp_path):
    pdf = tmp_path / "bio.pdf"
    pdf.write_bytes(_make_pdf(_TEXT))

    chunks = ingest(str(pdf), "bio.pdf", "sessX")

    assert len(chunks) >= 1
    assert "Photosynthesis" in chunks[0].page_content
    # Every chunk carries the isolation + provenance metadata.
    for c in chunks:
        assert c.metadata["session_id"] == "sessX"
        assert c.metadata["file_name"] == "bio.pdf"


def test_upload_endpoint_ingests_real_pdf(client, monkeypatch):
    # Mock only the embedding step; real PDF parsing + chunking + DB write run.
    added = []
    monkeypatch.setattr(upload_mod.vectordb, "add_documents", lambda docs: added.extend(docs))

    r = client.post(
        "/upload",
        data={"session_id": "realpdf"},
        files={"file": ("bio.pdf", io.BytesIO(_make_pdf(_TEXT)), "application/pdf")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["filename"] == "bio.pdf"
    assert body["chunk_count"] >= 1
    # The chunks handed to the vector store are session-tagged.
    assert added and all(d.metadata["session_id"] == "realpdf" for d in added)
