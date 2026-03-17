# Enterprise Doc AI

A Retrieval-Augmented Generation (RAG) system that lets you ask natural language questions about OpenText Access Manager documentation and get accurate, context-grounded answers powered by an internal LLM via Model Broker.

## How It Works

1. **Ingest** — All PDF files in the `docs/` directory are loaded, split into chunks, embedded using Snowflake Arctic Embed via Model Broker, and stored in a ChromaDB server.
2. **Query** — User questions are matched against the vector store to retrieve the most relevant chunks, which are then passed as context to an LLM (default: `llama-3.3-70b`) to generate an answer with source references.
3. **Web Search Fallback** — When the vector store returns no results, Tavily searches the [official documentation site](https://docs.microfocus.com/doc/76/25.2/home) for additional context.
4. **Web UI** — A Streamlit-based web interface provides an interactive way to query the documentation from your browser.

## Architecture

```
PDF Documents (docs/) → PyPDFLoader → Text Chunks → Model Broker Embeddings → ChromaDB Server
                                                                                     ↓
User Question → Retriever → Relevant Chunks + Question → Model Broker LLM → Answer + Sources
                    ↓ (if no results)
              Tavily Web Search (docs.microfocus.com) ──────────────────────────↗
```

## Prerequisites

- Python 3.10+
- Docker & Docker Compose (for containerized deployment)
- An `LLM_API_KEY` (virtual key for the internal Model Broker)
- *(Optional)* A [Tavily API key](https://tavily.com/) for web search fallback

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/enterprise-doc-ai.git
   cd enterprise-doc-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   Create a `.env` file in the project root:
   ```
   LLM_API_KEY=your-virtual-key-here
   TAVILY_API_KEY=your-tavily-key-here   # optional, enables web search fallback
   ```

4. **Add documentation**

   Place your PDF files in the `docs/` directory. All PDFs are automatically discovered during ingestion.

## Usage

### Docker (Recommended)

```bash
docker compose up -d                        # Start ChromaDB + Streamlit app
docker compose run app python ingest.py     # Ingest all PDFs from docs/
```

The web UI will be available at `http://localhost:8501`.

### Local Development

ChromaDB must be running as a standalone server before using `query.py` or `app.py`:

```bash
chroma run --host localhost --port 8000     # Start ChromaDB server
```

#### Step 1: Ingest documents

```bash
python ingest.py
```

This scans all PDFs in `docs/`, splits them into chunks, generates embeddings via Model Broker, and stores them in the ChromaDB server. Includes automatic retry with exponential backoff for rate limits.

> **Note:** Ingestion deletes and recreates the collection each run — it is not incremental.

#### Step 2: Query the documentation

**Option A — Command line (interactive):**

```bash
python query.py
```

This starts an interactive terminal session where you can ask questions. Type `quit` to exit.

**Option B — Web UI (Streamlit):**

```bash
python -m streamlit run app.py
```

This launches a web-based interface at `http://localhost:8501` with a chat-style UI for querying the documentation.

**Example:**
```
Question: How do I configure OAuth authentication?

Answer: To configure OAuth authentication, you need to...

Sources:
  1. docs/Applications_configuration_guide.pdf (page 42)
  2. docs/Applications_configuration_guide.pdf (page 43)
```

## Configuration

All configuration is centralized in `config.py` and read from environment variables. Defaults are set for the internal Model Broker.

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_BASE_URL` | Model Broker `/v1` endpoint | OpenAI-compatible API base |
| `LLM_API_KEY` | *(required)* | Virtual key for Model Broker |
| `LLM_MODEL` | `llama-3.3-70b` | Chat completion model |
| `LLM_TEMPERATURE` | `0.3` | LLM temperature |
| `EMBEDDING_MODEL` | `snowflake-arctic` | Embedding model |
| `CHROMA_HOST` | `localhost` | ChromaDB server host |
| `CHROMA_PORT` | `8000` | ChromaDB server port |
| `CHROMA_COLLECTION` | `enterprise_docs` | ChromaDB collection name |
| `DOCS_DIR` | `docs` | Directory scanned for PDFs |
| `PARENT_CHUNK_SIZE` / `PARENT_CHUNK_OVERLAP` | `1500` / `200` | Parent chunk splitting |
| `CHILD_CHUNK_SIZE` / `CHILD_CHUNK_OVERLAP` | `400` / `100` | Child chunk splitting |
| `CHUNK_MIN_SIZE` | `50` | Minimum chunk size (filters noise) |
| `USE_SEMANTIC_CHUNKING` | `false` | Use embedding-based split points for children |
| `DOCSTORE_DIR` | `docstore` | Directory for parent document JSON files |
| `RETRIEVER_K` | `5` | Number of child chunks searched per query |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant |
| `TAVILY_API_KEY` | *(optional)* | Enables web search fallback |

## Project Structure

```
enterprise-doc-ai/
├── config.py          # Centralized configuration (env vars with defaults)
├── ingest.py          # Document ingestion — PDFs → parent/child chunks → ChromaDB + docstore
├── retriever.py       # Parent-child retriever (shared by query.py and app.py)
├── query.py           # Interactive CLI query interface
├── app.py             # Streamlit web UI for querying documentation
├── web_search.py      # Tavily web search fallback (docs.microfocus.com only)
├── requirements.txt   # Python dependencies
├── Dockerfile         # Container image for the Streamlit app
├── docker-compose.yml # ChromaDB + chromadb-admin + app orchestration
├── .env               # API key configuration (not tracked in git)
├── docs/              # PDF documentation (not tracked in git)
└── docstore/          # Parent document JSON files (not tracked in git)
```

## Key Technologies

- **LangChain (LCEL)** — LangChain Expression Language for composable LLM pipelines
- **Model Broker** — Internal OpenAI-compatible API for LLM (`llama-3.3-70b`) and embeddings (`snowflake-arctic`)
- **ChromaDB** — Vector database for similarity search (runs as standalone server, admin UI at `localhost:3000`)
- **Tavily** — Web search fallback, restricted to official documentation at `docs.microfocus.com`
- **Streamlit** — Web UI framework for the interactive assistant
- **Docker Compose** — Container orchestration for ChromaDB and the app
- **PyPDF** — PDF document loading
