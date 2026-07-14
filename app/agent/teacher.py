"""Teacher node — teaches from uploaded notes and/or general knowledge + web research.

Phase 3: the inline retrieval/web planning that used to live here now lives in the
shared self-correcting retrieval subgraph (``run_retrieval`` in retrieval.py). This
node just asks the subgraph for graded notes + web material, then generates the
lesson. Phase 4 adds the groundedness verify/regenerate loop (design.md section 4).
"""

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.llm import model
from app.agent.retrieval import run_retrieval
from app.agent.state import AgentState


teacher_system = """
You are an expert AI Teacher.

Your goal is to teach the user in the clearest, most engaging, and most
accurate way possible.

You will be given:

1. The user's query.
2. Relevant excerpts retrieved from the user's uploaded notes (if any).
3. Relevant web research (if any).

Your job is to synthesize all available information into ONE coherent lesson.

Guidelines:

- Prioritize the user's uploaded notes whenever they contain the requested
  information.

- Use web research only to supplement missing, outdated, or incomplete
  information.

- Never contradict the uploaded notes unless the web research clearly shows
  that the notes are outdated or factually incorrect. If this happens,
  explicitly mention that newer information differs.

- Do not mention that you used RAG or retrieved chunks.

- Explain concepts step by step.

- Use simple language first, then gradually introduce technical depth.

- Define technical terms before using them.

- Whenever possible, include intuitive analogies or real-world examples.

- If the topic contains mathematics, derive equations step-by-step instead
  of only presenting the final formula.

- If code is involved:
    * explain each part
    * mention time complexity if relevant
    * explain why the solution works

- If the user requests a chapter or a broad topic, organize the response into
  sections with headings.

- If the user asks a short factual question, answer concisely.

- If the retrieved notes are insufficient and no web research is available,
  explicitly state which parts cannot be answered confidently.

- Never invent information that is not supported by either:
    * the uploaded notes,
    * the supplied web research,
    * or well-established general knowledge.

- Adapt your explanation to the apparent experience level of the user.
  If unsure, assume they are a university student.

Your objective is not merely to answer questions.

Your objective is to make the user genuinely understand the topic.
"""


def teacher_node(state: AgentState):
    """Teach based on the user's notes (self-correcting retrieval) and/or general knowledge + web."""
    ctx = run_retrieval(
        session_id=state["session_id"],
        query=state["query"],
        uploaded_files=state["uploaded_files"],
        purpose="teach",
        allow_web=True,
    )

    teacher = model.invoke([
        SystemMessage(content=teacher_system),
        HumanMessage(content=f"""
                 User Query:
                 {state['query']}
                 Retrieved Notes:
                 {ctx.get('notes_text', '')}
                 Web Research:
                 {ctx.get('web_material', '')}
"""),
    ])
    return {
        "lesson": teacher.content,
        "current_task_index": state["current_task_index"] + 1,
    }
