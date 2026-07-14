"""/chat — run the agent for a session's query."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.agent.graph import agent
from app.persistence.db import get_db
from app.persistence.repository import list_filenames
from app.api.session_store import session_states

router = APIRouter()


class TextInput(BaseModel):
    session_id: str
    query: str


@router.post("/chat")
def chat(input: TextInput, db: Session = Depends(get_db)):
    # Uploaded files come from this session's durable document records.
    uploaded_files = list_filenames(db, input.session_id)

    state = {
        "session_id": input.session_id,
        "query": input.query,
        "uploaded_files": uploaded_files,
        "messages": [],
        "current_task_index": 0,
    }

    result = agent.invoke(state)
    session_states.set(input.session_id, result)

    return {"state": result}
