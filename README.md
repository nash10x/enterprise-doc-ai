# Enterprise Doc AI

A Retrieval-Augmented Generation (RAG) system that lets you ask natural language questions about enterprise product documentation and get accurate, context-grounded answers powered by Google Gemini.

## How It Works

1. **Ingest** — PDF documentation is loaded, split into chunks, embedded using Google's Gemini Embedding model, and stored in a local ChromaDB vector database.
2. **Query** — User questions are matched against the vector store to retrieve the most relevant chunks, which are then passed as context to Gemini 2.5 Flash to generate an answer with source references.
3. **Web UI** — A Streamlit-based web interface provides an interactive way to query the documentation from your browser.

## Architecture

```
PDF Documents → PyPDFLoader → Text Chunks → Gemini Embeddings → ChromaDB
                                                                     ↓
User Question → Retriever → Relevant Chunks + Question → Gemini LLM → Answer + Sources
```

## Prerequisites

- Python 3.10+
- A [Google AI API key](https://ai.google.dev/gemini-api/docs/api-key)

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

3. **Configure your API key**

   Create a `.env` file in the project root:
   ```
   GOOGLE_API_KEY=your-api-key-here
   ```

4. **Add documentation**

   Place your PDF files in the `docs/` directory and update the file path in `ingest.py` if needed.

## Usage

### Step 1: Ingest documents

```bash
python ingest.py
```

This loads the PDF, splits it into chunks, generates embeddings, and stores them in the `vectorstore/` directory. Includes automatic retry with exponential backoff for API rate limits.

### Step 2: Query the documentation

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

## Project Structure

```
enterprise-doc-ai/
├── ingest.py          # Document ingestion and vector store creation
├── query.py           # Interactive CLI query interface
├── app.py             # Streamlit web UI for querying documentation
├── requirements.txt   # Python dependencies
├── .env               # API key configuration (not tracked in git)
├── docs/              # PDF documentation (not tracked in git)
└── vectorstore/       # ChromaDB vector database (not tracked in git)
```

## Key Technologies

- **LangChain (LCEL)** — LangChain Expression Language for composable LLM pipelines
- **Google Gemini** — Embedding model (`gemini-embedding-001`) and LLM (`gemini-2.5-flash`)
- **ChromaDB** — Local vector database for similarity search
- **Streamlit** — Web UI framework for the interactive assistant
- **PyPDF** — PDF document loading
