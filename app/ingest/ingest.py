"""PDF ingestion pipeline: load text (OCR fallback via Gemini) then chunk.

Behavior-preserving copy of the original ingest.py, with one fix (audit A2): the
unused ``GoogleGenerativeAIEmbeddings`` object has been removed. Embedding is done by
the vector store (HF ``all-MiniLM-L6-v2``) when chunks are added, not here — so this
module only produces chunks. The Gemini client is still used for OCR of scanned PDFs.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai

from app.config import settings

client = genai.Client(api_key=settings.GOOGLE_API_KEY)

PROMPT = """
Extract all text from this PDF.

Return Markdown.

Preserve:

- headings
- equations
- bullet points
- numbering
- tables

If diagrams exist,
describe them.

Do not summarize.
"""


def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path=file_path)
    docs = loader.load()
    text = "\n\n".join(doc.page_content for doc in docs)
    if len(text.strip()) < 100:
        uploaded_pdf = client.files.upload(file=file_path)
        response = client.models.generate_content(
            model=settings.OCR_MODEL,
            contents=[
                uploaded_pdf,
                PROMPT,
            ],
        )
        return response.text.strip()
    else:
        return text


def create_chunks(text, filename, session_id):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.create_documents(
        texts=[text],
        metadatas=[
            {
                "file_name": filename,
                # Phase 2: tag every vector with its owning session for isolation.
                "session_id": session_id,
            }
        ],
    )
    return chunks


def ingest(file_path, filename, session_id):
    text = load_pdf(file_path)
    chunks = create_chunks(text, filename, session_id)
    return chunks
