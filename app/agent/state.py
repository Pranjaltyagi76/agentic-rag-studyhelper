"""Agent state and shared schemas.

Carried over verbatim from the original Agent.py (behavior-preserving). Node-local
planning schemas (TeacherPlan, QuizPlan) live with their nodes; the shared
graph-level types live here.
"""

from typing_extensions import TypedDict, Annotated, Literal, Optional

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class task(BaseModel):
    task_id: int = Field(description="Give task their task id in order of their occurence")
    tool: Literal["teacher", "quiz_generator", "quiz_eval"] = Field(
        description="Node to execute this task"
    )
    task: str = Field(description="what the task is and what tools need to be called")
    topic: str = Field(description="The topic being discussed in the query")


class PlannerState(BaseModel):
    tasks: list[task]


class QuizQuestion(BaseModel):
    id: int
    question: str
    system_answer: str
    type: Literal["mcq", "true_false", "short_answer"]
    explanation: str
    options: list[str] | None = None


class Quiz(BaseModel):
    topic: str
    questions: list[QuizQuestion]


class QuizEval(BaseModel):
    explanation: str
    eval: Literal["Correct", "Wrong"] | None
    rating: Optional[int] = Field(
        default=None,
        ge=0,
        le=5,
        description="Understanding rating from 0 to 5.",
    )


class AgentState(TypedDict):
    session_id: str
    query: str
    plan: PlannerState | None
    uploaded_files: list[str]
    lesson: str | None
    quiz: Quiz | None
    current_question_id: int | None
    user_answer: str | None
    quiz_evaluation: QuizEval | None
    current_task_index: int | None
    messages: Annotated[list[BaseMessage], add_messages]
