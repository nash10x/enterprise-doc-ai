# Project Guidelines

## Overview

RAG system for querying OpenText Access Manager documentation. Users ask natural language questions; the system retrieves relevant chunks from a ChromaDB vector store and generates answers via an internal LLM (Model Broker). Tavily web search provides fallback against the official docs site.

## Architecture

```
ingest.py  →  All PDFs in docs/ → chunks → Model Broker embeddings → ChromaDB server
query.py   →  ChromaDB retriever → LCEL chain → CLI answers (+ Tavily fallback)
app.py     →  ChromaDB retriever → LCEL chain → Streamlit UI (+ Tavily fallback)
web_search.py → Tavily client, restricted to docs.microfocus.com
config.py  →  Central config, all values from env vars with defaults
```

- **No shared state** between modules — each file independently creates its own embeddings/LLM/vectorstore clients
- Prompt templates are duplicated between `query.py` and `app.py`
- ChromaDB runs as a standalone server (container), connected via `chromadb.HttpClient`

## Build and Run

```bash
# Install dependencies
pip install -r requirements.txt

# Docker (preferred)
docker compose up -d                        # Start ChromaDB + Streamlit app
docker compose run app python ingest.py     # Ingest all PDFs from docs/

# Local development (requires ChromaDB running on localhost:8000)
python ingest.py                            # Ingest PDFs
python query.py                             # Interactive CLI
python -m streamlit run app.py              # Web UI at localhost:8501
```

No test suite exists. Validation is manual via `test_search()` in `ingest.py`.

## Key Configuration

All config lives in `config.py`, read from environment variables (`.env` file). Defaults are set for the internal Model Broker.

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_BASE_URL` | Model Broker `/v1` endpoint | OpenAI-compatible API base |
| `LLM_API_KEY` | *(required)* | Virtual key for Model Broker |
| `LLM_MODEL` | `llama-3.3-70b` | Chat completion model |
| `EMBEDDING_MODEL` | `snowflake-arctic-embed-l-v2.0` | Embedding model |
| `CHROMA_HOST` / `CHROMA_PORT` | `localhost` / `8000` | ChromaDB server address |
| `CHROMA_COLLECTION` | `enterprise_docs` | Collection name in ChromaDB |
| `TAVILY_API_KEY` | *(optional)* | Enables web search fallback |
| `DOCS_DIR` | `docs` | Directory scanned for PDFs |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `200` | Text splitting parameters |
| `RETRIEVER_K` | `5` | Number of chunks retrieved per query |

When changing models or parameters, update `.env` — `config.py` will pick them up. No code changes needed.

## Conventions

- Python 3.10+, no type hints
- LangChain LCEL pattern: `prompt | llm | StrOutputParser()`
- All LLM/embedding calls use `langchain-openai` (`ChatOpenAI`, `OpenAIEmbeddings`) pointed at Model Broker
- `print()` for output (no logging framework)
- Rate limit retry: detect "429" in exception, exponential backoff
- Tavily web search is **locked to `docs.microfocus.com`** — never search the broader web

## Pitfalls

- ChromaDB must be running before `query.py` or `app.py` (container or standalone)
- `ingest.py` must run before querying (populates the ChromaDB collection)
- `ingest.py` **deletes and recreates** the collection on each run — not incremental
- Tavily fallback only triggers when the vectorstore returns **zero** documents
- Docker Compose overrides `CHROMA_HOST=chromadb` — don't hardcode `localhost` in `.env` if deploying via containers
