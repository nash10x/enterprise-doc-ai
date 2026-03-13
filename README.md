# Enterprise Doc AI

A Retrieval-Augmented Generation (RAG) system that lets you ask natural language questions about enterprise product documentation and get accurate, context-grounded answers powered by Google Gemini.

## How It Works

1. **Ingest** — PDF documentation is loaded, split into chunks, embedded using Google's Gemini Embedding model, and stored in a local ChromaDB vector database.
2. **Query** — User questions are matched against the vector store to retrieve the most relevant chunks, which are then passed as context to Gemini 2.0 Flash to generate an answer with source references.

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

This loads the PDF, splits it into chunks, generates embeddings, and stores them in the `vectorstore/` directory.

### Step 2: Query the documentation

```bash
python query.py
```

This starts an interactive session where you can ask questions. Type `quit` to exit.

**Example:**
```
Question: How do I configure OAuth authentication?

Answer: To configure OAuth authentication, you need to...

Sources:
  1. docs/Product_overview.pdf (page 42)
  2. docs/Product_overview.pdf (page 43)
```

## Project Structure

```
enterprise-doc-ai/
├── ingest.py          # Document ingestion and vector store creation
├── query.py           # Interactive RAG query interface
├── requirements.txt   # Python dependencies
├── docs/              # PDF documentation (not tracked in git)
└── vectorstore/       # ChromaDB vector database (not tracked in git)
```

## Key Technologies

- **LangChain** — Orchestration framework for LLM applications
- **Google Gemini** — Embedding model (`gemini-embedding-001`) and LLM (`gemini-2.0-flash`)
- **ChromaDB** — Local vector database for similarity search
- **PyPDF** — PDF document loading
