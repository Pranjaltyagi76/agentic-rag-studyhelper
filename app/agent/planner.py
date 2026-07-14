"""Planner node — decomposes a query into an ordered plan, and adapts it on failure.

Phase 4: the planner is entered once to build the initial plan, and re-entered after
each task. On re-entry it stays cheap — it only invokes the LLM when there is a
failure signal to react to (e.g. the teacher produced an ungrounded lesson), then it
may insert a corrective task or finish early. Bounded by ``settings.REPLAN_MAX`` so it
always terminates (design.md section 5).
"""

from typing_extensions import Literal

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.agent.llm import model
from app.agent.state import AgentState, PlannerState, task


class AdaptiveDecision(BaseModel):
    action: Literal["continue", "add_task", "finish"] = Field(
        description="continue = proceed with the remaining plan; add_task = insert a corrective task now; finish = end early because the goal is met or cannot be improved."
    )
    reason: str = Field(description="Brief justification.")
    new_task: task | None = Field(
        default=None, description="The corrective task to run next (required only if action=add_task)."
    )


adaptive_system = """You are the supervising planner of an AI Study Assistant.

A task just finished and there was a FAILURE SIGNAL (e.g. a lesson could not be fully
grounded in the available sources). Given the user's original request, the current
plan, progress, and a summary of results, decide how to recover:

- "continue": the issue is minor / already flagged to the user; proceed with the plan.
- "add_task": insert ONE corrective task now (e.g. another `teacher` pass to fill a
  specific gap). Only if it would plausibly help. Provide new_task.
- "finish": stop early — the request is satisfied, or further attempts won't help.

Prefer "continue" or "finish" unless a corrective task clearly helps. Return ONLY the
AdaptiveDecision schema."""


def _initial_plan(state: AgentState):
    Planner_system = f"""You are the Planning Agent of an AI Study Assistant.

Your ONLY responsibility is to analyze the user's request and generate an ordered execution plan.

You have access to the following execution nodes:

1. teacher
Description:
Teaches and explains topics using either:
- the user's uploaded notes (if requested), or
- general knowledge.

Use this node whenever the user asks to:
- explain
- teach
- summarize
- describe
- answer questions
- clarify concepts

--------------------------------------------------

2. quiz_generator
Description:
Generates a quiz about a topic using either:
- the user's uploaded notes (if requested), or
- general knowledge.

Use this node whenever the user asks to:
- quiz
- test
- assess
- generate questions
- create MCQs
- create True/False questions
- create short-answer questions

--------------------------------------------------

3. quiz_eval
Description:
Evaluates a user's answer to an existing quiz question.

Use this node ONLY when the user is asking to:
- evaluate an answer
- check an answer
- grade an answer
- review a quiz response
- score a response

This node MUST NOT generate a new quiz.
This node MUST NOT teach.
Its ONLY responsibility is evaluation.

--------------------------------------------------

Planning Rules

1. Break the user's request into atomic tasks.

2. Assign each task:
- task_id
- tool
- task
- topic

3. task_id must begin at 1 and increase sequentially.

4. tool MUST be EXACTLY one of:

teacher
quiz_generator
quiz_eval

Do NOT invent tool names.

5. Preserve the order requested by the user.

Example:

User:
"Teach me Transformers and then quiz me."

Plan:

Task 1
tool = teacher

Task 2
tool = quiz_generator

--------------------------------------------

User:
"Quiz me on Linear Regression."

Plan:

Task 1
tool = quiz_generator

--------------------------------------------

User:
"Evaluate my answer."

Plan:

Task 1
tool = quiz_eval

--------------------------------------------

User:
"Teach me Trees, then quiz me, then evaluate my answer."

Plan:

Task 1
tool = teacher

Task 2
tool = quiz_generator

Task 3
tool = quiz_eval

--------------------------------------------

6. Do NOT answer the user's question.

7. Do NOT explain your reasoning.

8. Return ONLY the PlannerState schema.
"""
    query = state['query']
    planner = model.with_structured_output(schema=PlannerState)
    plan = planner.invoke([
        SystemMessage(content=Planner_system),
        HumanMessage(content=f""" Query: {query}
                                        """),
    ])
    return {"plan": plan, "replans": 0}


def _results_summary(state: AgentState) -> str:
    plan = state.get("plan")
    total = len(plan.tasks) if plan else 0
    done = state.get("current_task_index", 0)
    parts = [f"Completed {done}/{total} planned tasks."]
    if state.get("lesson"):
        parts.append(f"A lesson was produced (grounded={state.get('grounded')}).")
    if state.get("quiz"):
        parts.append("A quiz was generated.")
    if state.get("quiz_evaluation"):
        parts.append("An answer was evaluated.")
    return " ".join(parts)


def PlannerNode(state: AgentState):
    """Build the initial plan, or (on re-entry) adapt it when a task failed.

    First entry (no plan yet) → LLM decomposes the request into an ordered plan.
    Re-entry → cheap by default: only when there is a failure signal (an ungrounded
    lesson) and we are under the replan cap do we ask the LLM how to recover, and
    possibly insert a corrective task or finish early.
    """
    if state.get("plan") is None:
        return _initial_plan(state)

    replans = state.get("replans") or 0
    idx = state.get("current_task_index", 0)
    failure = state.get("grounded") is False

    if not failure or replans >= settings.REPLAN_MAX:
        return {}  # nothing to adapt; executor advances or ends

    plan = state["plan"]
    decision = model.with_structured_output(AdaptiveDecision).invoke([
        SystemMessage(content=adaptive_system),
        HumanMessage(content=(
            f"Original request:\n{state['query']}\n\n"
            f"Planned tools: {[t.tool for t in plan.tasks]}\n"
            f"Progress: {_results_summary(state)}\n"
            f"Failure signal: the most recent lesson was not fully grounded in its sources."
        )),
    ])

    updates: dict = {"replans": replans + 1}
    if decision.action == "finish":
        updates["finished"] = True
    elif decision.action == "add_task" and decision.new_task:
        tasks = list(plan.tasks)
        tasks.insert(idx, decision.new_task)  # run the corrective task next
        plan.tasks = tasks
        updates["plan"] = plan
    return updates
