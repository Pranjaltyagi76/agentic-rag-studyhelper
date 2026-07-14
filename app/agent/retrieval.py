"""Retrieval + web-research tools.

Phase 1: the two tools (`RAG_Tool`, `research_tool`) lifted from the original
Agent.py, unchanged in behavior. Phase 3 turns this into the self-correcting
retrieval subgraph (rewrite -> retrieve -> grade -> retry -> web fallback) described
in design.md section 3.

Note (Phase 2, audit A3): `RAG_Tool` filters by `file_name` only; a `session_id`
filter must be added for multi-user isolation.
"""

from langchain_tavily import TavilySearch

from app.config import settings
from app.persistence.vectorstore import vectordb

search = TavilySearch(max_results=settings.TAVILY_MAX_RESULTS)


def RAG_Tool(query: str, filename: str, k: int):
    """Fetch relevant documents for the query from the user's uploaded documents."""
    retriever = vectordb.as_retriever(
        search_kwargs={
            "k": k,
            "filter": {
                "file_name": filename,
            },
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
