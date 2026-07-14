"""/evaluate — evaluate a session's answer, resuming durable state from the checkpoint."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.agent.graph import agent

router = APIRouter()


class QuizEvaluationInput(BaseModel):
    session_id: str
    question_id: int
    user_answer: str


@router.post("/evaluate")
def evaluate(input: QuizEvaluationInput):
    config = {"configurable": {"thread_id": input.session_id}}

    # Load durable state from the checkpoint (survives restarts — Phase 5).
    snapshot = agent.get_state(config)
    values = snapshot.values if snapshot else {}

    if not values:
        raise HTTPException(status_code=400, detail="No active session.")
    if values.get("quiz") is None:
        raise HTTPException(status_code=400, detail="Generate a quiz first.")

    turn = {
        "query": "Evaluate the user's answer.",
        "current_question_id": input.question_id,
        "user_answer": input.user_answer,
        "current_task_index": 0,
        "plan": None,
        "finished": False,
    }

    result = agent.invoke(turn, config=config)
    return {"quiz_evaluation": result["quiz_evaluation"]}
