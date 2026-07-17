"""Retrieval tools + the self-correcting retrieval subgraph (Phase 3).

Phase 2 gave us session-scoped tools. Phase 3 wraps them in a small LangGraph
subgraph that grades what it retrieves and retries with a rewritten query before
falling back to the web — the CRAG-style loop from design.md section 3:

    plan_retrieval -> retrieve -> grade -> (rewrite -> retrieve)*  -> plan_web -> web_fetch
                          ^__________________________|   (<= RETRIEVAL_MAX_ATTEMPTS)

The subgraph is shared by the teacher and quiz nodes via ``run_retrieval``. It
returns the graded notes text and any web material; the caller then generates.

Isolation (A3): ``RAG_Tool`` always filters by ``session_id``.
Termination: the rewrite loop is capped by ``settings.RETRIEVAL_MAX_ATTEMPTS``.
"""

from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END

from app.config import settings
from app.agent.llm import grader_model
from app.agent.structured import structured_invoke
from app.observability.metrics import log_retrieval_grade, log_grade_degraded
from app.persistence.vectorstore import vectordb

search = TavilySearch(max_results=settings.TAVILY_MAX_RESULTS)


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
def RAG_Tool(query: str, filename: str | None, k: int, session_id: str):
    """Fetch relevant documents for the query from THIS session's uploaded documents.

    The Chroma ``where`` filter always constrains to ``session_id`` (multi-user
    isolation, NFR-1) and additionally to ``file_name`` when the planner selected one.
    """
    # Portable operators ($eq/$and) understood by both Chroma and PGVector.
    conditions: list[dict] = [{"session_id": {"$eq": session_id}}]
    if filename:
        conditions.append({"file_name": {"$eq": filename}})
    where = conditions[0] if len(conditions) == 1 else {"$and": conditions}

    retriever = vectordb.as_retriever(search_kwargs={"k": k, "filter": where})
    return retriever.invoke(query)


def research_tool(queries: list[str]):
    """Search the web for up-to-date information (fallback when notes are weak)."""
    results = []
    for query in queries:
        results += search.invoke(query)
    return results


# --------------------------------------------------------------------------- #
# Schemas (structured LLM outputs)
# --------------------------------------------------------------------------- #
class RetrievalPlan(BaseModel):
    use_rag: bool = Field(description="Whether to retrieve from the user's uploaded notes.")
    filename: str | None = Field(default=None, description="Single most relevant uploaded filename.")
    rag_query: str | None = Field(default=None, description="Semantic query to send to the retriever.")
    number_chunks: int | None = Field(default=None, description="How many chunks to fetch.")


class DocsGrade(BaseModel):
    relevant_ids: list[int] = Field(
        default_factory=list, description="Indices of the chunks that are relevant to the query."
    )
    sufficient: bool = Field(
        description="True if the relevant chunks together are enough to answer the query."
    )
    missing: str | None = Field(
        default=None, description="What information is still missing, if not sufficient."
    )


class QueryRewrite(BaseModel):
    rag_query: str = Field(description="An improved retrieval query targeting the missing information.")


class WebPlan(BaseModel):
    use_web: bool = Field(description="Whether a web search is needed to answer well.")
    web_queries: list[str] | None = Field(default=None, description="Concise web search queries.")


# --------------------------------------------------------------------------- #
# Subgraph state
# --------------------------------------------------------------------------- #
class RetrievalState(TypedDict, total=False):
    # inputs
    session_id: str
    query: str
    uploaded_files: list[str]
    purpose: str          # "teach" | "quiz" — tunes planning
    allow_web: bool

    # retrieval plan
    use_rag: bool
    filename: str | None
    rag_query: str | None
    k: int

    # loop state
    attempts: int
    raw_docs: list
    relevant_docs: list
    sufficient: bool
    missing: str | None
    grade_degraded: bool   # grader unavailable -> chunks kept ungraded (not a quality signal)

    # web
    use_web: bool
    web_queries: list[str]

    # outputs
    notes_text: str
    web_material: str


# --------------------------------------------------------------------------- #
# Node prompts
# --------------------------------------------------------------------------- #
_GRADE_SYSTEM = """You grade retrieved note chunks for relevance to a user's query.

You are given the query and a numbered list of chunks. Decide which chunks are
relevant, and whether the relevant ones together are SUFFICIENT to answer the query.

Rules:
- relevant_ids: the indices ([0], [1], ...) of chunks that genuinely help answer the query. Drop off-topic chunks.
- sufficient: true only if the relevant chunks cover what's needed to answer well.
- missing: if not sufficient, briefly say what is still missing.

Do NOT answer the query. Return ONLY the DocsGrade schema."""

_REWRITE_SYSTEM = """You improve a retrieval query that returned insufficient results.

Given the original user query, the query that was tried, and what was missing,
write ONE better semantic retrieval query that targets the missing information.
Return ONLY the QueryRewrite schema."""

_WEB_SYSTEM = """You decide whether web research is needed before answering.

You are given the user's query, the notes retrieved so far, and whether those notes
are sufficient. If the notes fully answer the query, set use_web=false. If they are
missing, outdated, or insufficient, set use_web=true and provide concise web_queries.
Do NOT answer the query. Return ONLY the WebPlan schema."""


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def _plan_retrieval(state: RetrievalState) -> dict:
    system = f"""You are the retrieval-planning component of an AI Study Assistant.

Purpose of this retrieval: {state.get('purpose', 'teach')}
Files the user has uploaded: {state.get('uploaded_files', [])}

Decide whether to retrieve from the user's uploaded notes, and if so, plan it.

Rules:
1. If the user explicitly refers to their notes ("from my notes", "the pdf",
   "the slides", "according to my notes"), use_rag MUST be True.
2. If the uploaded files list is empty, use_rag MUST be False.
3. If the user names NO topic at all (e.g. just "generate quiz", "quiz me",
   "teach me") and they HAVE uploaded files, they mean their uploaded material:
   use_rag MUST be True, and the rag_query must broadly cover the main concepts
   of the most relevant file.
4. If the user names a specific topic answerable from general knowledge and does
   not request their notes, use_rag may be False.
5. If use_rag is True:
   - Pick the single most relevant filename from the uploaded files.
   - Write a focused semantic rag_query for the concepts to retrieve. For a quiz,
     base it ONLY on the quiz topic; ignore teaching/summarizing instructions.
   - Choose number_chunks by scope: small question 5, single concept 8,
     whole topic 12, whole chapter {settings.RETRIEVAL_K_CAP} (the maximum).

Do NOT answer or teach. Return ONLY the RetrievalPlan schema."""
    plan = structured_invoke(
        RetrievalPlan,
        [SystemMessage(content=system), HumanMessage(content=state["query"])],
        default=RetrievalPlan(use_rag=False),  # safe: fall back to general knowledge / web
    )
    # Hard cap regardless of what the planner asked for — bounds token cost per request.
    k = min(plan.number_chunks or 10, settings.RETRIEVAL_K_CAP)
    return {
        "use_rag": bool(plan.use_rag),
        "filename": plan.filename,
        "rag_query": plan.rag_query or state["query"],
        "k": k,
    }


def _retrieve(state: RetrievalState) -> dict:
    docs = RAG_Tool(
        query=state.get("rag_query") or state["query"],
        filename=state.get("filename"),
        k=state.get("k", 10),
        session_id=state["session_id"],
    )
    return {"raw_docs": docs, "attempts": state.get("attempts", 0) + 1}


def _grade(state: RetrievalState) -> dict:
    raw = state.get("raw_docs") or []
    if not raw:
        return {"sufficient": False, "missing": "No documents were retrieved."}

    # Grading only needs enough of each chunk to judge relevance — feed the opening,
    # not the whole thing (~70% fewer tokens). Full chunks still reach the generator.
    cap = settings.GRADE_CHUNK_CHARS
    numbered = "\n\n".join(f"[{i}] {d.page_content[:cap]}" for i, d in enumerate(raw))
    degraded: dict = {"hit": False}
    grade = structured_invoke(
        DocsGrade,
        [
            SystemMessage(content=_GRADE_SYSTEM),
            HumanMessage(content=f"Query:\n{state['query']}\n\nChunks:\n{numbered}"),
        ],
        # Grading is classification, not reasoning — use the cheap model (cascading),
        # which also draws from a separate quota bucket.
        llm=grader_model,
        # If grading fails, keep all retrieved chunks and treat them as sufficient —
        # but record it, so a rate limit/outage is never mistaken for "graded well".
        default=DocsGrade(relevant_ids=list(range(len(raw))), sufficient=True),
        on_fallback=lambda e: (degraded.__setitem__("hit", True), log_grade_degraded(e)),
    )

    picked = [raw[i] for i in grade.relevant_ids if 0 <= i < len(raw)]
    log_retrieval_grade(len(picked), len(raw), state.get("attempts", 0))

    # Accumulate relevant docs across retries, de-duplicated by content.
    merged = list(state.get("relevant_docs") or [])
    seen = {d.page_content for d in merged}
    for d in picked:
        if d.page_content not in seen:
            merged.append(d)
            seen.add(d.page_content)

    notes_text = "\n\n".join(d.page_content for d in merged)
    return {
        "relevant_docs": merged,
        "notes_text": notes_text,
        "sufficient": bool(grade.sufficient and picked),
        "missing": grade.missing,
        # True when the grader was unavailable and we kept everything ungraded.
        "grade_degraded": bool(degraded["hit"]),
    }


def _rewrite(state: RetrievalState) -> dict:
    rewrite = structured_invoke(
        QueryRewrite,
        [
            SystemMessage(content=_REWRITE_SYSTEM),
            HumanMessage(
                content=(
                    f"Original user query:\n{state['query']}\n\n"
                    f"Query that was tried:\n{state.get('rag_query')}\n\n"
                    f"What was missing:\n{state.get('missing')}"
                )
            ),
        ],
        default=QueryRewrite(rag_query=state["query"]),  # fall back to the original query
    )
    return {"rag_query": rewrite.rag_query}


def _plan_web(state: RetrievalState) -> dict:
    if not state.get("allow_web"):
        return {"use_web": False, "web_queries": []}

    plan = structured_invoke(
        WebPlan,
        [
            SystemMessage(content=_WEB_SYSTEM),
            HumanMessage(
                content=(
                    f"Query:\n{state['query']}\n\n"
                    f"Retrieved notes:\n{state.get('notes_text', '') or '(none)'}\n\n"
                    f"Notes sufficient: {state.get('sufficient', False)}"
                )
            ),
        ],
        default=WebPlan(use_web=False),  # safe: skip web if planning fails
    )
    return {"use_web": bool(plan.use_web), "web_queries": plan.web_queries or []}


def _web_fetch(state: RetrievalState) -> dict:
    results = research_tool(state.get("web_queries") or [])
    # str([]) is "[]" — a non-empty string that downstream treats as real material and
    # then narrates as "the web research contains nothing". Empty means empty.
    return {"web_material": str(results) if results else ""}


# --------------------------------------------------------------------------- #
# Routers
# --------------------------------------------------------------------------- #
def _after_plan(state: RetrievalState) -> str:
    return "retrieve" if state.get("use_rag") else "plan_web"


def _after_grade(state: RetrievalState) -> str:
    if state.get("sufficient"):
        return "plan_web"
    if state.get("attempts", 0) < settings.RETRIEVAL_MAX_ATTEMPTS:
        return "rewrite"
    # Retries exhausted: proceed; the web step (if allowed) can compensate.
    return "plan_web"


def _after_web(state: RetrievalState):
    return "web_fetch" if state.get("use_web") else END


# --------------------------------------------------------------------------- #
# Subgraph assembly
# --------------------------------------------------------------------------- #
_g = StateGraph(RetrievalState)
_g.add_node("plan_retrieval", _plan_retrieval)
_g.add_node("retrieve", _retrieve)
_g.add_node("grade", _grade)
_g.add_node("rewrite", _rewrite)
_g.add_node("plan_web", _plan_web)
_g.add_node("web_fetch", _web_fetch)

_g.add_edge(START, "plan_retrieval")
_g.add_conditional_edges("plan_retrieval", _after_plan, {"retrieve": "retrieve", "plan_web": "plan_web"})
_g.add_edge("retrieve", "grade")
_g.add_conditional_edges("grade", _after_grade, {"rewrite": "rewrite", "plan_web": "plan_web"})
_g.add_edge("rewrite", "retrieve")
_g.add_conditional_edges("plan_web", _after_web, {"web_fetch": "web_fetch", END: END})
_g.add_edge("web_fetch", END)

retrieval_subgraph = _g.compile()


# --------------------------------------------------------------------------- #
# Public helper
# --------------------------------------------------------------------------- #
def run_retrieval(
    session_id: str,
    query: str,
    uploaded_files: list[str],
    purpose: str = "teach",
    allow_web: bool = True,
) -> dict:
    """Run the self-correcting retrieval subgraph and return its final state.

    Returned keys of interest: ``notes_text`` (graded relevant notes), ``web_material``
    (web results, if any), ``use_rag``, ``use_web``, ``attempts``, ``sufficient``.
    """
    initial: RetrievalState = {
        "session_id": session_id,
        "query": query,
        "uploaded_files": uploaded_files,
        "purpose": purpose,
        "allow_web": allow_web,
        "attempts": 0,
        "relevant_docs": [],
        "notes_text": "",
        "web_material": "",
        "sufficient": False,
    }
    return retrieval_subgraph.invoke(initial)
