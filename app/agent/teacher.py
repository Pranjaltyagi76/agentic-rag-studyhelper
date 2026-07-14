"""Teacher node — teaches from uploaded notes and/or general knowledge + web research.

Phase 3: the inline retrieval/web planning that used to live here now lives in the
shared self-correcting retrieval subgraph (``run_retrieval`` in retrieval.py). This
node just asks the subgraph for graded notes + web material, then generates the
lesson. Phase 4 adds the groundedness verify/regenerate loop (design.md section 4).
"""

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.agent.llm import model
from app.agent.retrieval import run_retrieval
from app.agent.state import AgentState


class Groundedness(BaseModel):
    grounded: bool = Field(
        description="True if EVERY substantive claim in the lesson is supported by the sources."
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims in the lesson not supported by the sources (empty if grounded).",
    )


groundedness_system = """You are a strict fact-checker for an AI Teacher.

You are given SOURCES (the user's notes and/or web research) and a LESSON generated
from them. Decide whether the lesson is grounded: every substantive factual claim
must be supported by the sources.

- grounded = true only if all substantive claims are supported.
- unsupported_claims = list any claims that go beyond or contradict the sources.

Ignore generic pedagogical phrasing, analogies, and well-established common knowledge.
Return ONLY the Groundedness schema."""


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


def _generate_lesson(query: str, notes: str, web: str, feedback: str) -> str:
    """One lesson draft. `feedback` (if any) steers a regeneration toward grounding."""
    human = f"""
                 User Query:
                 {query}
                 Retrieved Notes:
                 {notes}
                 Web Research:
                 {web}
"""
    if feedback:
        human += f"\n\nIMPORTANT — your previous draft had unsupported claims: {feedback}\nRewrite the lesson using ONLY the provided sources; drop or hedge anything not supported."
    return model.invoke([
        SystemMessage(content=teacher_system),
        HumanMessage(content=human),
    ]).content


def teacher_node(state: AgentState):
    """Teach with self-correcting retrieval + a generate/verify/regenerate loop (Self-RAG).

    Phase 4: after generating a lesson we verify it is grounded in its sources. If not,
    we regenerate (feeding back the unsupported claims), capped by
    GENERATION_MAX_ATTEMPTS. When there are no sources (pure general-knowledge answer)
    there is nothing to verify against, so the check is skipped.
    """
    ctx = run_retrieval(
        session_id=state["session_id"],
        query=state["query"],
        uploaded_files=state["uploaded_files"],
        purpose="teach",
        allow_web=True,
    )
    notes = ctx.get("notes_text", "") or ""
    web = ctx.get("web_material", "") or ""
    sources = (notes + "\n" + web).strip()

    grader = model.with_structured_output(Groundedness)
    feedback = ""
    grounded: bool | None = None
    lesson = ""

    for attempt in range(1, settings.GENERATION_MAX_ATTEMPTS + 1):
        lesson = _generate_lesson(state["query"], notes, web, feedback)

        if not sources:
            # No retrieved/web context to verify against — general-knowledge answer.
            grounded = None
            break

        verdict = grader.invoke([
            SystemMessage(content=groundedness_system),
            HumanMessage(content=f"SOURCES:\n{sources}\n\nLESSON:\n{lesson}"),
        ])
        grounded = verdict.grounded

        if verdict.grounded or attempt >= settings.GENERATION_MAX_ATTEMPTS:
            if not verdict.grounded and verdict.unsupported_claims:
                claims = "; ".join(verdict.unsupported_claims)
                lesson += (
                    "\n\n---\n*Note: the following points may not be fully supported by "
                    f"your uploaded notes and should be verified: {claims}.*"
                )
            break

        feedback = "; ".join(verdict.unsupported_claims) or "unsupported claims present"

    return {
        "lesson": lesson,
        "grounded": grounded,
        "current_task_index": state["current_task_index"] + 1,
    }
