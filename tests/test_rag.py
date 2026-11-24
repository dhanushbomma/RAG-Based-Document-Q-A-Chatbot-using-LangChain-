from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app import main
from app.chain import RAGChain, format_docs
from app.ingest import chunk_documents, load_pdf
from app.prompts import NO_ANSWER, QA_PROMPT
from app.vectorstore import VectorStore


def test_load_pdf_keeps_page_metadata(sample_pdf):
    docs = load_pdf(sample_pdf)
    assert len(docs) == 2
    assert docs[0].metadata == {"source": "handbook.pdf", "page": 1}


def test_chunking_respects_size(sample_pdf):
    chunks = chunk_documents(load_pdf(sample_pdf), chunk_size=30, chunk_overlap=0)
    assert all(len(c.page_content) <= 30 for c in chunks)
    assert len(chunks) > 2


def test_prompt_contains_grounding_rules():
    text = QA_PROMPT.format_messages(context="ctx", question="q?")[0].content
    assert NO_ANSWER in text and "ONLY" in text and "ctx" in text


def test_vectorstore_persists(tmp_path, embeddings, sample_pdf):
    chunks = chunk_documents(load_pdf(sample_pdf))
    store = VectorStore(embeddings, tmp_path / "idx")
    assert not store.ready
    store.add_documents(chunks)
    reloaded = VectorStore(embeddings, tmp_path / "idx")
    assert reloaded.ready


def test_chain_returns_answer_and_sources(tmp_path, embeddings, sample_pdf):
    store = VectorStore(embeddings, tmp_path / "idx")
    store.add_documents(chunk_documents(load_pdf(sample_pdf)))
    chain = RAGChain(store, FakeListChatModel(responses=["25 days."]))
    out = chain.ask("How many vacation days?", k=2)
    assert out["answer"] == "25 days."
    assert {s["source"] for s in out["sources"]} == {"handbook.pdf"}


def test_format_docs_includes_citation(sample_pdf):
    assert "[source: handbook.pdf, page 1]" in format_docs(load_pdf(sample_pdf))


def test_api_end_to_end(tmp_path, embeddings, sample_pdf, monkeypatch):
    store = VectorStore(embeddings, tmp_path / "idx")
    llm = FakeListChatModel(responses=["25 days."])
    main.app.dependency_overrides[main.get_store] = lambda: store
    main.app.dependency_overrides[main.get_chain] = lambda: RAGChain(store, llm)
    monkeypatch.setattr(main.settings, "upload_dir", tmp_path / "up")
    client = TestClient(main.app)
    try:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.post("/ask", json={"question": "vacation days?"}).status_code == 409
        with sample_pdf.open("rb") as f:
            r = client.post("/ingest", files={"file": ("handbook.pdf", f, "application/pdf")})
        assert r.status_code == 200 and r.json()["pages"] == 2
        r = client.post("/ask", json={"question": "vacation days?"})
        assert r.status_code == 200 and r.json()["answer"] == "25 days."
        bad = client.post("/ingest", files={"file": ("x.txt", b"hi", "text/plain")})
        assert bad.status_code == 400
    finally:
        main.app.dependency_overrides.clear()
