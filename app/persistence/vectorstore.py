"""Vector store wiring.

Embeddings are HF ``all-MiniLM-L6-v2`` (canonical — used for BOTH ingestion and
query; decision 2026-07-14). This resolves audit item A2: the original ingest.py
defined an unused Google embeddings object while the real pipeline already embedded
with this HF model.

Backend is Chroma for local dev; pgvector on Neon becomes canonical at deploy
(Phase 9). Kept behind ``settings.VECTOR_BACKEND`` so the swap is config-only.
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings

embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

vectordb = Chroma(
    collection_name=settings.CHROMA_COLLECTION,
    embedding_function=embeddings,
    persist_directory=settings.CHROMA_DIR,
)
