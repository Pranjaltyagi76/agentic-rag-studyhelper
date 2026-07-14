"""Teacher node — teaches from uploaded notes and/or general knowledge + web research.

Behavior-preserving copy from the original Agent.py. Phase 3 wraps the retrieval
here in the shared self-correcting subgraph; Phase 4 adds the groundedness
verify/regenerate loop (design.md section 4).
"""

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.llm import model
from app.agent.retrieval import RAG_Tool, research_tool
from app.agent.state import AgentState


class TeacherPlan(BaseModel):
    use_rag: bool = Field(
        description="True if information is required from the user's  uploaded documents based on the query given"
    )
    number_chunks: int | None = Field(
        default=None, description="Number of relevant documents to be fetched based on the query"
    )
    filename: str | None = Field(
        default=None,
        description="Select one filename from the uploaded_files list that contains the most relevant information.",
    )
    rag_query: str | None = Field(
        default=None,
        description="The query sent to the retriever to fetch the relevant documents to the query",
    )
    use_web: bool | None = Field(
        default=None, description="Do we need to research for the given query or not"
    )
    web_queries: list[str] | None = Field(
        default=None,
        description="The exact query that we need to send to the web to get the desired content",
    )


web_research_system = """
You are the research planning component of an AI Teacher.

Your job is to determine whether additional information from the web is
required before teaching the user.

You will receive:

1. The user's query.
2. The retrieved documents from the user's uploaded notes if provided.

Your responsibilities are:

1. Carefully examine the retrieved notes if given.

2. Determine whether the notes alone are sufficient to answer the user's
request accurately and completely.

3. If the notes are sufficient:
   - Set use_web=False.
   - Leave web_query empty.

4. If the notes are incomplete, outdated, or missing important
information needed to answer the user's request:
   - Set use_web=True.
   - Generate a list of concise and specific web search queries.No. of queries can depend on the amount of info required.

5. If the knowledge base of your own is insufficient to answer the question :
    - Set use_web = True
    - Generate a list of concise and specific web search queries.No. of queries can depend on the amount of info required.

Use web research only when it genuinely improves the quality or accuracy
of the final response.

Do NOT use web research simply because the topic exists online.

Do NOT answer the user's question.

Do NOT summarize the notes.

Do NOT teach the topic.

Your ONLY job is to fill the TeacherPlan schema.
"""


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


TeachingPlanner = model.with_structured_output(TeacherPlan)


def teacher_node(state: AgentState):
    """Teaches people based on their uploaded notes if asked in the query or teaches based on the tool's own knowledge base. This tool handles all the questions asked and give answers"""

    teacher_planner_system = f"""
You are the resource planning component of an AI Teacher.

Your ONLY responsibility is to determine whether information from the user's
uploaded notes is required before teaching based on the given query of the user.

The user currently has access to the following uploaded files:

{state['uploaded_files']}

Follow these rules:

1. If the user explicitly asks:
   - "from my notes"
   - "from the uploaded pdf"
   - "from the slides"
   - "according to my notes"

   then use_rag MUST be True.

2. If the user's question can be answered accurately using general knowledge
and they do not request their uploaded notes, set use_rag=False.

3. If use_rag=True:
   - Choose the single most relevant filename from the uploaded files.
   - Generate a retrieval query that would retrieve the best chunks.
   - Decide how many chunks should be retrieved.

Chunk guide:
- Small definition/question → k = 5
- Explain a concept → k = 10
- Teach a topic → k = 20
- Large chapter/topic → k = 30
-Summarize -> Whole document's summary

Do NOT answer the user's question.

Do NOT explain the topic.

Your ONLY job is to fill the TeacherPlan schema.
"""
    rag_plan = TeachingPlanner.invoke([
        SystemMessage(content=teacher_planner_system),
        HumanMessage(content=state['query']),
    ])
    docs = ""
    if rag_plan.use_rag:
        docs = RAG_Tool(
            query=rag_plan.rag_query,
            filename=rag_plan.filename,
            k=rag_plan.number_chunks,
            session_id=state["session_id"],
        )
    research_plan = TeachingPlanner.invoke([
        SystemMessage(content=web_research_system),
        HumanMessage(
            content=f"""
User Query:
{state['query']}

Retrieved Documents:
{docs}
"""
        ),
    ])
    research_material = ""
    if research_plan.use_web:
        research_material = research_tool(queries=research_plan.web_queries)
    teacher = model.invoke([
        SystemMessage(content=teacher_system),
        HumanMessage(content=f"""
                 User Query:
                 {state['query']}
                 Retrieved Notes:
                 {docs}
                 Web Research:
                 {research_material}
"""),
    ])
    return {
        "lesson": teacher.content,
        "current_task_index": state["current_task_index"] + 1,
    }
