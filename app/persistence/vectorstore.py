"""Vector store wiring.

Embeddings are HF ``all-MiniLM-L6-v2`` (canonical — used for BOTH ingestion and
query; decision 2026-07-14). This resolves audit item A2: the original ingest.py
defined an unused Google embeddings object while the real pipeline already embedded
with this HF model.

Embeddings run on fastembed (ONNX) rather than sentence-transformers (PyTorch):
identical model, but ~15 MB instead of ~2.5 GB and no torch — far lighter for the
free HF Spaces deploy. ``onnxruntime`` is already a chromadb dependency.

Backend is Chroma for local dev; pgvector on Neon becomes canonical at deploy
(Phase 9). Kept behind ``settings.VECTOR_BACKEND`` so the swap is config-only.
"""

from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

from app.config import settings

embeddings = FastEmbedEmbeddings(model_name=settings.EMBEDDING_MODEL)

vectordb = Chroma(
    collection_name=settings.CHROMA_COLLECTION,
    embedding_function=embeddings,
    persist_directory=settings.CHROMA_DIR,
)
