"""LangGraph checkpointer (Phase 5) — durable, resumable graph state per session.

Backend follows ``DATABASE_URL``, mirroring the rest of persistence:
  - sqlite  -> SqliteSaver against ``CHECKPOINT_DB`` (local dev/test)
  - postgres -> PostgresSaver against ``DATABASE_URL`` (Neon, deploy — Phase 9)

Keyed by ``thread_id == session_id`` at invoke time, so every session's full agent
state (including the ``messages`` history and any generated quiz) survives a process
restart and can be resumed. Built once and cached.
"""

import sqlite3

from app.config import settings

_checkpointer = None


def build_checkpointer():
    """Return the process-wide checkpointer, constructing it on first call."""
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer

    url = settings.DATABASE_URL

    if url.startswith("sqlite"):
        from langgraph.checkpoint.sqlite import SqliteSaver

        conn = sqlite3.connect(settings.CHECKPOINT_DB, check_same_thread=False)
        cp = SqliteSaver(conn)
        cp.setup()
        _checkpointer = cp
    else:
        # Postgres / Neon. Finalized (connection pooling) in Phase 9; this keeps a
        # single long-lived saver alive for the process.
        from langgraph.checkpoint.postgres import PostgresSaver

        cp = PostgresSaver.from_conn_string(url)
        if hasattr(cp, "__enter__"):  # some versions return a context manager
            cp = cp.__enter__()
        cp.setup()
        _checkpointer = cp

    return _checkpointer
