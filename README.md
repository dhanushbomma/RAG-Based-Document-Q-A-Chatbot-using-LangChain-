# 📄 RAG Document Q&A Chatbot



![Python](https://img.shields.io/badge/python-3.11-blue) ![LangChain](https://img.shields.io/badge/LangChain-RAG-green) ![FastAPI](https://img.shields.io/badge/FastAPI-REST-009688) ![Docker](https://img.shields.io/badge/docker-ready-2496ED)

## Why I built this

Anyone who has pasted a long document into a chatbot knows the problem: the answer sounds confident, but you can't tell whether it came from your document or from the model's imagination.

I wanted to build the fix myself. This project is a small Retrieval-Augmented Generation (RAG) service. It only lets the model answer from passages it has actually retrieved from your PDFs, it admits when the answer isn't there, and it tells you which page to check.

## What it does

- Ingests PDFs and splits them into overlapping chunks
- Turns each chunk into an embedding and stores it in a FAISS index that is saved to disk, so you don't re-embed after a restart
- Retrieves the most relevant chunks for each question
- Answers using only those chunks, with `[source: file, page n]` citations
- Replies with a fixed "I don't know based on the provided documents." when the context doesn't contain the answer
- Exposes everything through a FastAPI REST API, packaged in Docker, with tests running in GitHub Actions

## How it works

```
        INGEST                                   ASK
  PDF ──► pages ──► chunks                 question
                      │                        │
                      ▼                        ▼
               embeddings ──► FAISS ◄──── top-k retrieval
                                              │
                                              ▼
                                   grounded prompt + context
                                              │
                                              ▼
                                      LLM ──► answer + sources
```

| File | Responsibility |
|---|---|
| `app/ingest.py` | Read PDFs page by page (keeping `source` and `page` metadata), split into chunks |
| `app/vectorstore.py` | Build, save and reload the FAISS index |
| `app/prompts.py` | The anti-hallucination prompt |
| `app/chain.py` | Retrieve → format context → prompt → LLM → answer and sources |
| `app/main.py` | FastAPI endpoints |
| `app/config.py` | Settings loaded from environment variables |

### The part I spent the most time on: the prompt

The prompt in `app/prompts.py` is deliberately strict:

1. Use **only** the text inside `<context>`
2. If the answer isn't there, reply with one exact sentence instead of guessing
3. Keep answers short and faithful to the source
4. Cite the source file and page

Using a fixed refusal sentence has a practical benefit: it is easy to detect in code and in tests, so "the model refused" is something you can check rather than something you hope for.

## Quick start

**1. Install and configure**

```bash
git clone https://github.com/YOUR_USERNAME/rag-doc-qa.git
cd rag-doc-qa
cp .env.example .env          # then put your OPENAI_API_KEY in .env
pip install -r requirements.txt
```

**2. Run the API**

```bash
uvicorn app.main:app --reload
```

**3. Try it**

```bash
# upload a PDF
curl -F "file=@handbook.pdf" http://localhost:8000/ingest
# {"file":"handbook.pdf","pages":12,"chunks":31}

# ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many vacation days do employees get?"}'
```

The response looks like:

```json
{
  "answer": "Employees receive 25 days of paid vacation per year. [source: handbook.pdf, page 1]",
  "sources": [{"source": "handbook.pdf", "page": 1}]
}
```

Interactive API docs are at http://localhost:8000/docs.

### With Docker

```bash
docker compose up --build
```

The `data/` folder is mounted as a volume, so your index survives container restarts.

## API reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/ingest` | Upload a PDF (multipart form field `file`) |
| `POST` | `/ask` | Body: `{"question": "...", "top_k": 4}` (`top_k` optional) |

Errors you might see: `400` for non-PDF uploads, `422` for PDFs with no extractable text, `409` if you ask before ingesting anything.

## Configuration

Everything can be set in `.env`:

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | none | Your API key |
| `LLM_MODEL` | `gpt-4o-mini` | Chat model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |

Chunk size (1000), overlap (150) and `top_k` (4) are defaults in `app/config.py`. Smaller chunks give more precise retrieval but less context per chunk, so this is worth tuning for your own documents.

## Testing

```bash
pytest -q
```

The tests use a fake embedding model and a fake chat model, so they run offline, cost nothing, and work in CI. They cover PDF loading and metadata, chunk sizes, the grounding rules in the prompt, index persistence, the full chain, and the API end to end (including error cases).

## CI

Every push runs two GitHub Actions jobs: the test suite, then a Docker image build.

## Known limitations

I'd rather be upfront about these than have you find them:

- **Text PDFs only.** Scanned PDFs need OCR, which isn't included
- **One shared index.** All uploaded documents go into the same store, with no per-user separation
- **No authentication** on the API
- **Basic retrieval.** Plain similarity search, with no re-ranking or hybrid keyword search
- The FAISS index is loaded with `allow_dangerous_deserialization=True`. This is fine for an index you created yourself, but never load an index file from a source you don't trust

## Ideas for next steps

- [ ] ChromaDB option (the store is isolated in `vectorstore.py`, so the swap is small)
- [ ] Re-ranking for better retrieval
- [ ] Streaming responses
- [ ] Per-document filtering ("only search `handbook.pdf`")
- [ ] A simple web UI
- [ ] Retrieval evaluation (is the right chunk in the top k?)

## Tech stack

Python · LangChain · OpenAI · FAISS · FastAPI · Pydantic · Docker · GitHub Actions · Pytest
