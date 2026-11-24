"""FastAPI service exposing ingestion and question-answering endpoints."""
import shutil
from functools import lru_cache

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.chain import RAGChain, get_llm
from app.config import settings
from app.ingest import chunk_documents, load_pdf
from app.vectorstore import VectorStore, get_embeddings

app = FastAPI(title="RAG Document Q&A", version="1.0.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int | None = Field(default=None, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


@lru_cache
def get_store() -> VectorStore:
    return VectorStore(get_embeddings())


def get_chain(store: VectorStore = Depends(get_store)) -> RAGChain:
    return RAGChain(store, get_llm())


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(file: UploadFile = File(...), store: VectorStore = Depends(get_store)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.upload_dir / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    pages = load_pdf(dest)
    if not pages:
        raise HTTPException(422, "No extractable text found in the PDF.")
    chunks = chunk_documents(pages)
    store.add_documents(chunks)
    return {"file": file.filename, "pages": len(pages), "chunks": len(chunks)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, chain: RAGChain = Depends(get_chain)):
    if not chain.store.ready:
        raise HTTPException(409, "No documents ingested yet. POST a PDF to /ingest first.")
    return chain.ask(req.question, req.top_k)
