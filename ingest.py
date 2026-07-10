from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
from google import genai
import os
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
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
def load_pdf(file_path:str):
    loader = PyPDFLoader(file_path=file_path)
    docs = loader.load()
    text = "\n\n".join(doc.page_content for doc in docs)
    if len(text.strip()) < 100:
        uploaded_pdf = client.files.upload(file=file_path)
        response = client.models.generate_content(

        model="gemini-2.5-flash",

        contents=[
            uploaded_pdf,
            PROMPT
        ]
    )
        return response.text.strip()
    else:
        return text
        
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(text,filename):
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
    chunks = splitter.create_documents(
        texts=[text],
        metadatas=[
            {
                "file_name": filename
            }
        ]
    )
    return chunks
    
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)
    
def ingest(file_path,filename):

    text = load_pdf(file_path)

    chunks = create_chunks(text,filename)

    return chunks

