"""Vector store wiring.

Embeddings are ``all-MiniLM-L6-v2`` (canonical — both ingest and query) running on
fastembed (ONNX, torch-free); ``onnxruntime`` is already a chromadb dependency.

Backend follows ``settings.VECTOR_BACKEND``:
  - "chroma"   -> Chroma on a local persistent directory (bare-metal local dev)
  - "pgvector" -> PGVector in Postgres/Neon (docker-compose + deploy, Phase 7/9)

Both are addressed through the same LangChain vector-store interface, so the rest of
the app is backend-agnostic. Retrieval filters (session_id + file_name) use portable
``$eq``/``$and`` operators understood by both backends.
"""

from langchain_community.embeddings import FastEmbedEmbeddings

from app.config import settings

_embed_kwargs = {"model_name": settings.EMBEDDING_MODEL}
if settings.EMBED_CACHE_DIR:
    # Use the model baked into the image rather than re-downloading it at runtime.
    _embed_kwargs["cache_dir"] = settings.EMBED_CACHE_DIR

embeddings = FastEmbedEmbeddings(**_embed_kwargs)


def _build_vectorstore():
    if settings.VECTOR_BACKEND == "pgvector":
        from langchain_postgres import PGVector

        # PGVector uses psycopg v3; map a plain postgresql:// URL to the +psycopg driver.
        conn = settings.DATABASE_URL
        if conn.startswith("postgresql://"):
            conn = "postgresql+psycopg://" + conn[len("postgresql://"):]
        return PGVector(
            embeddings=embeddings,
            collection_name=settings.CHROMA_COLLECTION,
            connection=conn,
            use_jsonb=True,
        )

    from langchain_chroma import Chroma

    return Chroma(
        collection_name=settings.CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=settings.CHROMA_DIR,
    )


vectordb = _build_vectorstore()
