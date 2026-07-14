"""/evaluate — evaluate a session's answer to a quiz question."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.agent.graph import agent
from app.api.session_store import session_states

router = APIRouter()


class QuizEvaluationInput(BaseModel):
    session_id: str
    question_id: int
    user_answer: str


@router.post("/evaluate")
def evaluate(input: QuizEvaluationInput):
    state = session_states.get(input.session_id)

    if state is None:
        raise HTTPException(status_code=400, detail="No active session.")

    if state.get("quiz") is None:
        raise HTTPException(status_code=400, detail="Generate a quiz first.")

    state["query"] = "Evaluate the user's answer."
    state["current_question_id"] = input.question_id
    state["user_answer"] = input.user_answer
    state["current_task_index"] = 0

    result = agent.invoke(state)
    session_states.set(input.session_id, result)

    return {"quiz_evaluation": result["quiz_evaluation"]}
