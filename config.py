import os
from dotenv import load_dotenv

load_dotenv()

# Model Broker (OpenAI-compatible)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://model-broker.aviator-model.bp.anthos.otxlab.net/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-4-17b-maverick")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "snowflake-arctic")

# Document ingestion
DOCS_DIR = os.getenv("DOCS_DIR", "docs")
CHUNK_SEPARATORS = ["\n\n\n", "\n\n", "\n", ". ", " ", ""]
CHUNK_MIN_SIZE = int(os.getenv("CHUNK_MIN_SIZE", "50"))

# Parent-child chunking
PARENT_CHUNK_SIZE = int(os.getenv("PARENT_CHUNK_SIZE", "1500"))
PARENT_CHUNK_OVERLAP = int(os.getenv("PARENT_CHUNK_OVERLAP", "200"))
CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", "400"))
CHILD_CHUNK_OVERLAP = int(os.getenv("CHILD_CHUNK_OVERLAP", "100"))
USE_SEMANTIC_CHUNKING = os.getenv("USE_SEMANTIC_CHUNKING", "false").lower() == "true"

# Parent document store
DOCSTORE_DIR = os.getenv("DOCSTORE_DIR", "docstore")

# ChromaDB (standalone server)
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "enterprise_docs")

# Retrieval
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "5"))
RRF_K = int(os.getenv("RRF_K", "60"))

# Tavily (web search fallback)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_SEARCH_DOMAIN = "docs.microfocus.com"
