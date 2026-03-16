import os
import glob
from dotenv import load_dotenv
import time
import random

# LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb

from config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, LLM_BASE_URL, LLM_API_KEY, CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION


def load_documents():
    """
    Load all PDF files from the docs directory
    """
    print(f"Scanning {DOCS_DIR}/ for PDF files...")

    pdf_files = sorted(glob.glob(os.path.join(DOCS_DIR, "*.pdf")))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {DOCS_DIR}/")

    all_documents = []
    for pdf_path in pdf_files:
        print(f"  Loading {os.path.basename(pdf_path)}...")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        all_documents.extend(documents)
        print(f"    → {len(documents)} pages")

    print(f"Loaded {len(all_documents)} pages from {len(pdf_files)} PDF(s)")

    return all_documents


def split_documents(documents):
    """
    Split documents into chunks
    """

    print("Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    return chunks


def create_vectorstore(chunks):
    """
    Create embeddings and store in ChromaDB server
    """

    print("Creating embeddings...")

    embeddings = OpenAIEmbeddings(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=EMBEDDING_MODEL
    )

    print(f"Connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}...")

    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    # Delete existing collection to re-ingest cleanly
    try:
        chroma_client.delete_collection(CHROMA_COLLECTION)
        print(f"  Cleared existing collection '{CHROMA_COLLECTION}'")
    except Exception:
        pass

    print("Building vector database...")

    max_retries = 5
    for attempt in range(max_retries):
        try:
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                client=chroma_client,
                collection_name=CHROMA_COLLECTION
            )
            break
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limited. Retrying in {wait:.1f}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                raise

    print("Vector database saved successfully")

    return vectorstore


def test_search(vectorstore):
    """
    Test similarity search
    """

    print("\nTesting retrieval...")

    query = "How do I configure OAuth authentication?"

    results = vectorstore.similarity_search(query, k=3)

    for i, doc in enumerate(results):
        print(f"\nResult {i+1}")
        print("-" * 50)
        print(doc.page_content[:500])


def main():

    load_dotenv()

    if not LLM_API_KEY:
        raise ValueError("LLM_API_KEY not found in environment variables")

    documents = load_documents()

    chunks = split_documents(documents)

    vectorstore = create_vectorstore(chunks)

    test_search(vectorstore)


if __name__ == "__main__":
    main()