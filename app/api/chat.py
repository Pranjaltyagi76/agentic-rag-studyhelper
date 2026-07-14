"""/chat — run the agent for a user query."""

from fastapi import APIRouter
from pydantic import BaseModel

from app.agent.graph import agent
from app.api.runtime_state import runtime

router = APIRouter()


class TextInput(BaseModel):
    query: str


@router.post("/chat")
def chat(input: TextInput):
    state = {
        "query": input.query,
        "uploaded_files": runtime.uploaded_files,
        "messages": [],
        "current_task_index": 0,
    }

    runtime.current_agent_state = agent.invoke(state)

    return {"state": runtime.current_agent_state}
