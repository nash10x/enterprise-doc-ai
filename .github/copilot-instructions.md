# Project Guidelines

## Overview

RAG system for querying OpenText Access Manager documentation. Users ask natural language questions; the system retrieves relevant chunks from a ChromaDB vector store and generates answers via an internal LLM (Model Broker). Tavily web search provides fallback against the official docs site.

## Architecture

```
ingest.py      →  PDFs → parent chunks → child chunks → Model Broker embeddings → ChromaDB
                  Parents stored as JSON in docstore/
retriever.py   →  HybridParentChildRetriever: dense (ChromaDB) + sparse (BM25) → RRF → parents
query.py       →  retriever → LCEL chain → CLI answers (+ Tavily fallback)
app.py         →  retriever → LCEL chain → Streamlit UI (+ Tavily fallback)
web_search.py  →  Tavily client, restricted to docs.microfocus.com
config.py      →  Central config, all values from env vars with defaults
```

- `retriever.py` is the **shared retriever module** — both `query.py` and `app.py` use `create_retriever()`
- **Hybrid retrieval**: dense semantic search (ChromaDB embeddings) + sparse keyword search (BM25) fused via Reciprocal Rank Fusion (RRF), aggregated by parent
- Parent-child splitting: large parent chunks (~1500 chars) stored as JSON; small child chunks (~400 chars) embedded in ChromaDB for similarity search; child corpus saved as JSON for BM25 index; retrieval returns the parent chunk
- Optional semantic chunking (embedding-based split points) via `USE_SEMANTIC_CHUNKING=true`
- Prompt templates are duplicated between `query.py` and `app.py`
- ChromaDB runs as a standalone server (container), connected via `chromadb.HttpClient`
- ChromaDB Admin UI available at `localhost:3000` (Docker only)

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
| `EMBEDDING_MODEL` | `snowflake-arctic` | Embedding model |
| `CHROMA_HOST` / `CHROMA_PORT` | `localhost` / `8000` | ChromaDB server address |
| `CHROMA_COLLECTION` | `enterprise_docs` | Collection name in ChromaDB |
| `TAVILY_API_KEY` | *(optional)* | Enables web search fallback |
| `DOCS_DIR` | `docs` | Directory scanned for PDFs |
| `PARENT_CHUNK_SIZE` / `PARENT_CHUNK_OVERLAP` | `1500` / `200` | Parent chunk splitting |
| `CHILD_CHUNK_SIZE` / `CHILD_CHUNK_OVERLAP` | `400` / `100` | Child chunk splitting |
| `CHUNK_MIN_SIZE` | `50` | Minimum chunk size (filters noise) |
| `USE_SEMANTIC_CHUNKING` | `false` | Use embedding-based split points for children |
| `DOCSTORE_DIR` | `docstore` | Directory for parent document JSON files |
| `RETRIEVER_K` | `5` | Number of child chunks searched per query |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant |

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
- `ingest.py` must run before querying (populates ChromaDB collection + `docstore/`)
- `ingest.py` **deletes and recreates** the collection and docstore on each run — not incremental
- Tavily fallback only triggers when the vectorstore returns **zero** documents
- Docker Compose overrides `CHROMA_HOST=chromadb` — don't hardcode `localhost` in `.env` if deploying via containers
- Rebuild Docker image (`docker compose up -d --build`) after any code changes — otherwise containers use cached image
