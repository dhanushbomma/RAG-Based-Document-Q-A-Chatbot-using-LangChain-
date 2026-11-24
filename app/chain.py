"""RAG chain: retrieve -> format context -> prompt -> LLM -> answer + sources."""
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from app.prompts import QA_PROMPT
from app.vectorstore import VectorStore


def get_llm() -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    from app.config import settings

    return ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key, temperature=0)


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[source: {d.metadata.get('source')}, page {d.metadata.get('page')}]\n{d.page_content}"
        for d in docs
    )


class RAGChain:
    def __init__(self, store: VectorStore, llm: BaseChatModel):
        self.store = store
        self.llm = llm
        self.chain = QA_PROMPT | llm | StrOutputParser()

    def ask(self, question: str, k: int | None = None) -> dict:
        docs = self.store.retriever(k).invoke(question)
        answer = self.chain.invoke({"context": format_docs(docs), "question": question})
        sources = sorted({(d.metadata.get("source"), d.metadata.get("page")) for d in docs})
        return {
            "answer": answer,
            "sources": [{"source": s, "page": p} for s, p in sources],
        }
