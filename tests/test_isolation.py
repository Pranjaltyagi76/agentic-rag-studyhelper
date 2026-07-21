"""Multi-user isolation (NFR-1, audit A3) — the security-critical retrieval filter.

RAG_Tool MUST always constrain retrieval to the caller's session_id so one session can
never read another's uploaded notes. We assert the exact ``where`` filter it builds,
using portable ``$eq``/``$and`` operators (understood by both Chroma and pgvector),
without needing a live vector store.
"""

import app.agent.retrieval as retrieval_mod
from app.agent.retrieval import RAG_Tool


class _Retriever:
    def invoke(self, query):
        return []


class _RecordingVDB:
    """Captures the search_kwargs RAG_Tool passes to as_retriever()."""

    def __init__(self):
        self.search_kwargs = None

    def as_retriever(self, search_kwargs):
        self.search_kwargs = search_kwargs
        return _Retriever()


def test_session_scope_always_applied(monkeypatch):
    vdb = _RecordingVDB()
    monkeypatch.setattr(retrieval_mod, "vectordb", vdb)

    RAG_Tool(query="anything", filename=None, k=5, session_id="SESSION-A")

    assert vdb.search_kwargs["k"] == 5
    assert vdb.search_kwargs["filter"] == {"session_id": {"$eq": "SESSION-A"}}


def test_filename_narrows_within_session(monkeypatch):
    vdb = _RecordingVDB()
    monkeypatch.setattr(retrieval_mod, "vectordb", vdb)

    RAG_Tool(query="anything", filename="notes.pdf", k=3, session_id="S1")

    # Session scope is ANDed with the file filter — the session_id never drops out.
    assert vdb.search_kwargs["filter"] == {
        "$and": [
            {"session_id": {"$eq": "S1"}},
            {"file_name": {"$eq": "notes.pdf"}},
        ]
    }
