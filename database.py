from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectordb = Chroma(
    collection_name='Stuff',
    embedding_function=embeddings,
    persist_directory="Chromadb"
)