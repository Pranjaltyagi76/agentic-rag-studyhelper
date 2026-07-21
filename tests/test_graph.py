"""Full-graph orchestration with a mocked LLM — deterministic, no API keys.

Runs a real end-to-end teach turn through the compiled agent (planner -> executor
routing -> retrieval subgraph -> teacher -> checkpointer) with the Groq calls replaced
by canned, schema-aware responses. Verifies the WIRING, not the model: plan routing,
lesson flow, message accumulation, and checkpoint persistence.
"""

from langchain_core.messages import HumanMessage, AIMessage

_LESSON = "Photosynthesis converts sunlight into chemical energy stored as glucose."


def _canned(schema):
    """Return a minimal valid instance for each structured schema the teach path hits."""
    from app.agent.state import PlannerState, task
    from app.agent.retrieval import RetrievalPlan, WebPlan

    if schema is PlannerState:
        return PlannerState(
            tasks=[task(task_id=1, tool="teacher", task="explain photosynthesis", topic="photosynthesis")]
        )
    if schema is RetrievalPlan:
        return RetrievalPlan(use_rag=False)          # no uploaded files -> general knowledge
    if schema is WebPlan:
        return WebPlan(use_web=False)                # notes/general knowledge suffice
    raise AssertionError(f"unexpected schema requested from the fake LLM: {schema}")


class _FakeStructured:
    def __init__(self, schema):
        self.schema = schema

    def invoke(self, messages):
        return _canned(self.schema)


class _FakeModel:
    def with_structured_output(self, schema):
        return _FakeStructured(schema)

    def invoke(self, messages):                      # the teacher's direct lesson call
        return AIMessage(content=_LESSON)


def test_teach_turn_runs_end_to_end(monkeypatch):
    import app.agent.structured as structured_mod
    import app.agent.teacher as teacher_mod

    fake = _FakeModel()
    monkeypatch.setattr(structured_mod, "model", fake)   # planner / retrieval / web planning
    monkeypatch.setattr(teacher_mod, "model", fake)      # lesson generation

    from app.agent.graph import agent

    sid = "graph-teach-1"
    turn = {
        "session_id": sid,
        "query": "Teach me photosynthesis.",
        "uploaded_files": [],
        "messages": [HumanMessage(content="Teach me photosynthesis.")],
        "current_task_index": 0,
        "plan": None,
        "finished": False,
        "grounded": None,
        "replans": 0,
        "lesson": None,
        "quiz": None,
        "quiz_evaluation": None,
    }
    config = {"configurable": {"thread_id": sid}}
    result = agent.invoke(turn, config=config)

    # Planner decomposed into a single teacher task, and the executor ran it.
    assert [t.tool for t in result["plan"].tasks] == ["teacher"]
    assert result["lesson"] == _LESSON
    assert result["current_task_index"] == 1

    # The AI lesson was appended to the conversation history.
    assert any(isinstance(m, AIMessage) and m.content == _LESSON for m in result["messages"])

    # And the checkpointer durably persisted it for this thread.
    persisted = agent.get_state(config).values
    assert persisted["lesson"] == _LESSON
