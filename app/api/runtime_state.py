"""Process-global runtime state shared across routers.

This preserves the original app.py behavior EXACTLY: a single in-memory list of
uploaded files and a single "current" agent state, shared for the whole process.

WARNING (audit A1, roadmap Phase 2): this is single-user-only and not safe for
concurrent users. Phase 2 replaces it with per-``session_id`` state in Postgres. It
lives in its own module so the whole app shares one instance (module import is a
singleton), which is what the original module-level globals did.
"""


class RuntimeState:
    def __init__(self):
        self.uploaded_files: list[str] = []
        self.current_agent_state = None


runtime = RuntimeState()
