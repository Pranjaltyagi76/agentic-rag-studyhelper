"""/chat — run the agent for a session's query.

Two endpoints:
- POST /chat         : non-streaming, returns the final state (kept for compatibility).
- POST /chat/stream  : SSE, emits node-by-node progress then the final state (Phase 6).
"""

import json

from fastapi import APIRouter, Depends
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
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


# Friendly progress labels per graph node (Phase 6 streaming).
FRIENDLY = {
    "planner": "Planning your request",
    "teacher": "Researching and writing your lesson",
    "quiz_generator": "Building your quiz",
    "quiz_eval": "Evaluating your answer",
}


def _build_turn(session_id: str, query: str, uploaded_files: list[str]) -> dict:
    """Per-turn input: reset working fields, accumulate messages (see Phase 5)."""
    return {
        "session_id": session_id,
        "query": query,
        "uploaded_files": uploaded_files,
        "messages": [HumanMessage(content=query)],
        "current_task_index": 0,
        "plan": None,
        "finished": False,
        "grounded": None,
        "replans": 0,
        "lesson": None,
        "quiz": None,
        "quiz_evaluation": None,
    }


def _clean(state: dict) -> dict:
    return {k: v for k, v in state.items() if k != "messages"}


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(jsonable_encoder(payload))}\n\n"


@router.post("/chat")
def chat(input: TextInput, db: Session = Depends(get_db)):
    uploaded_files = list_filenames(db, input.session_id)
    config = {"configurable": {"thread_id": input.session_id}}
    result = agent.invoke(_build_turn(input.session_id, input.query, uploaded_files), config=config)
    return {"state": _clean(result)}


@router.post("/chat/stream")
def chat_stream(input: TextInput, db: Session = Depends(get_db)):
    # Fetch files before streaming starts (the db session closes when this returns).
    uploaded_files = list_filenames(db, input.session_id)
    config = {"configurable": {"thread_id": input.session_id}}
    turn = _build_turn(input.session_id, input.query, uploaded_files)

    def gen():
        try:
            yield _sse({"type": "progress", "node": "start", "message": "Starting"})
            for update in agent.stream(turn, config=config, stream_mode="updates"):
                for node in update:
                    msg = FRIENDLY.get(node)
                    if msg:
                        yield _sse({"type": "progress", "node": node, "message": msg})
            final = agent.get_state(config).values
            yield _sse({"type": "final", "state": _clean(final)})
        except Exception as e:  # stream a structured error instead of dropping the connection
            yield _sse({"type": "error", "error": {"code": "agent_error", "message": str(e)}})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
