"""FAISS vector store with on-disk persistence."""
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import settings


def get_embeddings() -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)


class VectorStore:
    def __init__(self, embeddings: Embeddings, index_dir: Path | None = None):
        self.embeddings = embeddings
        self.index_dir = Path(index_dir or settings.index_dir)
        self._store: FAISS | None = None
        if (self.index_dir / "index.faiss").exists():
            self._store = FAISS.load_local(
                str(self.index_dir), embeddings, allow_dangerous_deserialization=True
            )

    @property
    def ready(self) -> bool:
        return self._store is not None

    def add_documents(self, docs: list[Document]) -> int:
        if not docs:
            return 0
        if self._store is None:
            self._store = FAISS.from_documents(docs, self.embeddings)
        else:
            self._store.add_documents(docs)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(self.index_dir))
        return len(docs)

    def retriever(self, k: int | None = None):
        if self._store is None:
            raise RuntimeError("No documents indexed yet.")
        return self._store.as_retriever(search_kwargs={"k": k or settings.top_k})
