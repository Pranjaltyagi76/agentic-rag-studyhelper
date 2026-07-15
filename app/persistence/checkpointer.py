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
        # Postgres / Neon (Phase 9). Use a long-lived connection pool: `from_conn_string`
        # returns a context manager whose connection closes as soon as it exits, which
        # would break every checkpoint read/write.
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool
        from langgraph.checkpoint.postgres import PostgresSaver

        pool = ConnectionPool(
            conninfo=url,
            min_size=1,
            max_size=5,
            # PostgresSaver requires autocommit + dict rows. prepare_threshold=0 keeps
            # us safe if the URL is ever pointed at a PgBouncer/pooled endpoint.
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
            open=True,
        )
        cp = PostgresSaver(pool)
        cp.setup()
        _checkpointer = cp

    return _checkpointer
