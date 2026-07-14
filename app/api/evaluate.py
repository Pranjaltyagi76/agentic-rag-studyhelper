"""/evaluate — evaluate the user's answer to a quiz question."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.agent.graph import agent
from app.api.runtime_state import runtime

router = APIRouter()


class QuizEvaluationInput(BaseModel):
    question_id: int
    user_answer: str


@router.post("/evaluate")
def evaluate(input: QuizEvaluationInput):
    if runtime.current_agent_state is None:
        raise HTTPException(status_code=400, detail="No active session.")

    if runtime.current_agent_state.get("quiz") is None:
        raise HTTPException(status_code=400, detail="Generate a quiz first.")

    runtime.current_agent_state["query"] = "Evaluate the user's answer."
    runtime.current_agent_state["current_question_id"] = input.question_id
    runtime.current_agent_state["user_answer"] = input.user_answer
    runtime.current_agent_state["current_task_index"] = 0

    runtime.current_agent_state = agent.invoke(runtime.current_agent_state)

    return {"quiz_evaluation": runtime.current_agent_state["quiz_evaluation"]}
