"""Teacher generate -> verify-groundedness -> regenerate loop (Phase 4).

The README notes this path "is wired and capped but rarely observed firing" because it's
hard to force with a live model. Here we force it deterministically: notes are present
(so grounding IS verified) and the groundedness verdicts are canned, exercising both the
regenerate-then-succeed path and the give-up-and-flag path.
"""

from langchain_core.messages import HumanMessage, AIMessage

import app.agent.teacher as teacher_mod
from app.agent.teacher import teacher_node, Groundedness


def _patch(monkeypatch, verdicts):
    """Notes present + a fake model (counts drafts) + canned groundedness verdicts."""
    monkeypatch.setattr(
        teacher_mod, "run_retrieval",
        lambda **kw: {"notes_text": "Mitochondria produce ATP via respiration.", "web_material": "", "use_rag": True},
    )

    drafts = {"n": 0}

    class _Model:
        def invoke(self, messages):
            drafts["n"] += 1
            return AIMessage(content=f"lesson draft {drafts['n']}")

    monkeypatch.setattr(teacher_mod, "model", _Model())

    queue = list(verdicts)

    def _fake_structured(schema, messages, *, default=None, on_fallback=None, llm=None):
        assert schema is Groundedness
        return queue.pop(0)

    monkeypatch.setattr(teacher_mod, "structured_invoke", _fake_structured)
    return drafts


def _state():
    return {
        "session_id": "t1",
        "query": "Teach me from my notes.",
        "uploaded_files": ["notes.pdf"],
        "messages": [HumanMessage(content="Teach me from my notes.")],
        "current_task_index": 0,
    }


def test_regenerates_when_first_draft_ungrounded(monkeypatch):
    drafts = _patch(monkeypatch, [
        Groundedness(grounded=False, unsupported_claims=["overreach"]),
        Groundedness(grounded=True),
    ])

    out = teacher_node(_state())

    assert drafts["n"] == 2          # regenerated once after the ungrounded verdict
    assert out["grounded"] is True
    assert out["current_task_index"] == 1


def test_flags_lesson_when_ungrounded_after_cap(monkeypatch):
    # GENERATION_MAX_ATTEMPTS defaults to 2; both drafts come back ungrounded.
    drafts = _patch(monkeypatch, [
        Groundedness(grounded=False, unsupported_claims=["claim A"]),
        Groundedness(grounded=False, unsupported_claims=["claim A"]),
    ])

    out = teacher_node(_state())

    assert drafts["n"] == 2          # tried, regenerated, then stopped at the cap
    assert out["grounded"] is False
    # The lesson is still returned, but with an explicit unsupported-claims caveat.
    assert "may not be fully supported" in out["lesson"]
    assert "claim A" in out["lesson"]
