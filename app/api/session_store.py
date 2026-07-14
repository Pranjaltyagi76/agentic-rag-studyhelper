"""Per-session transient agent state (Phase 2 — resolves audit A1).

Replaces the old process-global ``current_agent_state``. State is now keyed by
``session_id`` so concurrent users never overwrite each other. This is the working
state carried from ``/chat`` to ``/evaluate`` (mainly the generated quiz).

Still in-memory, so it does NOT survive a restart — durable, resumable memory is
Phase 5 (LangGraph PostgresSaver checkpointer). Durable data that must persist now
(which files belong to a session) lives in the DB, not here.
"""

from threading import Lock


class SessionStateStore:
    def __init__(self):
        self._states: dict[str, dict] = {}
        self._lock = Lock()

    def get(self, session_id: str) -> dict | None:
        with self._lock:
            return self._states.get(session_id)

    def set(self, session_id: str, state: dict) -> None:
        with self._lock:
            self._states[session_id] = state


session_states = SessionStateStore()
