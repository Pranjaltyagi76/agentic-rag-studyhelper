"""/upload — accept a PDF, ingest it, add its chunks to the vector store."""

import os

from fastapi import APIRouter, UploadFile, File

from app.config import settings
from app.ingest import ingest
from app.persistence.vectorstore import vectordb
from app.api.runtime_state import runtime

router = APIRouter()

os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_path = os.path.join(settings.UPLOAD_FOLDER, file.filename)

    runtime.uploaded_files.append(file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    chunks = ingest(file_path, file.filename)
    vectordb.add_documents(chunks)

    return {"filename": file.filename}
