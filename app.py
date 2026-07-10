from fastapi import FastAPI,WebSocket,HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel
import os
from ingest import ingest
from database import vectordb

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class Textinput(BaseModel):
    query:str

from Agent import agent
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace

llm = HuggingFaceEndpoint(
    repo_id= "deepseek-ai/DeepSeek-V4-Flash",
    task = "text-generation"
)
model = ChatHuggingFace(llm = llm)
vectordb = vectordb

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
uploaded_files =[]
current_agent_state = None

@app.post('/upload')
async def Upload(file: UploadFile = File(...)):
    file_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )
    uploaded_files.append(file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    chunks = ingest(file_path,file.filename)

    vectordb.add_documents(chunks)

    return {
        "filename": file.filename
    }
@app.post('/chat')
def chat(input:Textinput):
    global current_agent_state

    state = {
        "query": input.query,
        "uploaded_files": uploaded_files,
        "messages": [],
        "current_task_index":0
    }

    current_agent_state = agent.invoke(state)

    return {
        "state": current_agent_state
    }
class QuizEvaluationInput(BaseModel):

    question_id: int

    user_answer: str
@app.post("/evaluate")
def evaluate(input: QuizEvaluationInput):

    global current_agent_state

    if current_agent_state is None:
        raise HTTPException(
            status_code=400,
            detail="No active session."
        )

    if current_agent_state.get("quiz") is None:
        raise HTTPException(
            status_code=400,
            detail="Generate a quiz first."
        )

    current_agent_state["query"] = (
        "Evaluate the user's answer."
    )

    current_agent_state["current_question_id"] = (
        input.question_id
    )

    current_agent_state["user_answer"] = (
        input.user_answer
    )
    current_agent_state["current_task_index"] = 0

    current_agent_state = agent.invoke(
        current_agent_state
    )

    return {
        "quiz_evaluation":
            current_agent_state["quiz_evaluation"]
    }