"""Retrieval + web-research tools.

Phase 1: the two tools (`RAG_Tool`, `research_tool`) lifted from the original
Agent.py, unchanged in behavior. Phase 3 turns this into the self-correcting
retrieval subgraph (rewrite -> retrieve -> grade -> retry -> web fallback) described
in design.md section 3.

Phase 2 (audit A3 resolved): `RAG_Tool` now filters by `session_id` (always) plus
`file_name` (when known), so a session can only ever retrieve its own documents.
"""

from langchain_tavily import TavilySearch

from app.config import settings
from app.persistence.vectorstore import vectordb

search = TavilySearch(max_results=settings.TAVILY_MAX_RESULTS)


def RAG_Tool(query: str, filename: str | None, k: int, session_id: str):
    """Fetch relevant documents for the query from THIS session's uploaded documents.

    The Chroma ``where`` filter always constrains to ``session_id`` (multi-user
    isolation, NFR-1) and additionally to ``file_name`` when the planner selected one.
    """
    conditions: list[dict] = [{"session_id": session_id}]
    if filename:
        conditions.append({"file_name": filename})
    where = conditions[0] if len(conditions) == 1 else {"$and": conditions}

    retriever = vectordb.as_retriever(
        search_kwargs={
            "k": k,
            "filter": where,
        }
    )
    docs = retriever.invoke(query)
    return docs


def research_tool(queries: list[str]):
    """
    Searches the web for up-to-date information.

    Use this tool only when the user's uploaded notes are
    insufficient or outdated.

    Args:
        queries: Concise web search queries.

    Returns:
        Relevant web search results.
    """
    results = []
    for query in queries:
        results += search.invoke(query)
    return results
