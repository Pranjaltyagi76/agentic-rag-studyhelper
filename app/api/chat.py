"""/chat — run the agent for a session's query, with durable per-session memory."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from langchain_core.messages import HumanMessage

from app.agent.graph import agent
from app.persistence.db import get_db
from app.persistence.repository import list_filenames

router = APIRouter()


class TextInput(BaseModel):
    session_id: str
    query: str


@router.post("/chat")
def chat(input: TextInput, db: Session = Depends(get_db)):
    uploaded_files = list_filenames(db, input.session_id)

    # thread_id ties this turn to the session's durable checkpoint (Phase 5).
    config = {"configurable": {"thread_id": input.session_id}}

    # Per-turn inputs: reset the per-turn working fields (a new request is a fresh
    # task sequence) but let `messages` accumulate via its add_messages reducer, so
    # conversation memory persists across turns and restarts.
    turn = {
        "session_id": input.session_id,
        "query": input.query,
        "uploaded_files": uploaded_files,
        "messages": [HumanMessage(content=input.query)],
        "current_task_index": 0,
        "plan": None,
        "finished": False,
        "grounded": None,
        "replans": 0,
        "lesson": None,
        "quiz": None,
        "quiz_evaluation": None,
    }

    result = agent.invoke(turn, config=config)

    # Drop the raw message objects from the API response (frontend needs lesson/quiz).
    clean = {k: v for k, v in result.items() if k != "messages"}
    return {"state": clean}
