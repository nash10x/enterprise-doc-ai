import os
from dotenv import load_dotenv

load_dotenv()

# Model Broker (OpenAI-compatible)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://model-broker.aviator-model.bp.anthos.otxlab.net/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "snowflake-arctic-embed-l-v2.0")

# Document ingestion
DOCS_DIR = os.getenv("DOCS_DIR", "docs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ChromaDB (standalone server)
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "enterprise_docs")

# Retrieval
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "5"))

# Tavily (web search fallback)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_SEARCH_DOMAIN = "docs.microfocus.com"
