"""The top-level executor router (app/agent/graph.py).

Pure routing logic — no LLM. Guards that the graph ends when the goal is met or the
plan is exhausted, and otherwise dispatches to the current task's tool.
"""

from langgraph.graph import END

from app.agent.graph import executor
from app.agent.state import PlannerState, task


def _state(**overrides):
    base = {
        "plan": PlannerState(
            tasks=[task(task_id=1, tool="teacher", task="explain X", topic="X")]
        ),
        "current_task_index": 0,
        "finished": False,
    }
    base.update(overrides)
    return base


def test_routes_to_current_task_tool():
    assert executor(_state()) == "teacher"


def test_ends_when_finished_flag_set():
    # Phase 4: adaptive planner can end early once the goal is met.
    assert executor(_state(finished=True)) == END


def test_ends_when_index_past_plan():
    assert executor(_state(current_task_index=1)) == END


def test_routes_quiz_tool():
    st = _state(
        plan=PlannerState(
            tasks=[task(task_id=1, tool="quiz_generator", task="quiz me", topic="X")]
        )
    )
    assert executor(st) == "quiz_generator"
