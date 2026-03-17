import os
import glob
import json
import re
import uuid
from dotenv import load_dotenv
import time
import random

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import chromadb

from config import (
    DOCS_DIR, CHUNK_SEPARATORS, CHUNK_MIN_SIZE,
    PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP,
    USE_SEMANTIC_CHUNKING, DOCSTORE_DIR,
    EMBEDDING_MODEL, LLM_BASE_URL, LLM_API_KEY,
    CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION
)

if USE_SEMANTIC_CHUNKING:
    from langchain_experimental.text_splitter import SemanticChunker


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


def enrich_metadata(chunks):
    """Extract numbered section headings and add to chunk metadata."""
    heading_pattern = re.compile(r"^(\d+(?:\.\d+)+\s+\S.{2,})$", re.MULTILINE)
    enriched = 0
    for chunk in chunks:
        headings = heading_pattern.findall(chunk.page_content)
        if headings:
            chunk.metadata["section"] = headings[0].strip()
            enriched += 1
    print(f"  Enriched {enriched}/{len(chunks)} chunks with section headings")
    return chunks


def create_parent_chunks(documents):
    """Split documents into large parent chunks with structural separators."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
        strip_whitespace=True,
    )
    parents = splitter.split_documents(documents)
    for p in parents:
        p.metadata["doc_id"] = str(uuid.uuid4())
    return parents


def create_child_chunks(parent_chunks, embeddings):
    """Split parent chunks into smaller child chunks for precise retrieval."""
    if USE_SEMANTIC_CHUNKING:
        print("  Using semantic chunking (embedding-based split points)")
        child_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
        )
    else:
        print("  Using recursive character splitting")
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHILD_CHUNK_SIZE,
            chunk_overlap=CHILD_CHUNK_OVERLAP,
            separators=CHUNK_SEPARATORS,
            strip_whitespace=True,
        )

    all_children = []
    for i, parent in enumerate(parent_chunks):
        # Skip semantic splitting for parents already smaller than child size
        if USE_SEMANTIC_CHUNKING and len(parent.page_content.strip()) <= CHILD_CHUNK_SIZE:
            all_children.append(Document(
                page_content=parent.page_content,
                metadata=dict(parent.metadata),
            ))
        else:
            children = child_splitter.split_documents([parent])
            for child in children:
                child.metadata["doc_id"] = parent.metadata["doc_id"]
            all_children.extend(children)

        if (i + 1) % 100 == 0:
            print(f"    Split {i + 1}/{len(parent_chunks)} parents...")

    # Filter noise (headers, footers, ToC fragments)
    before = len(all_children)
    all_children = [c for c in all_children if len(c.page_content.strip()) >= CHUNK_MIN_SIZE]
    filtered = before - len(all_children)
    if filtered:
        print(f"  Filtered {filtered} chunks below {CHUNK_MIN_SIZE} chars")

    return all_children


def store_parents(parent_chunks):
    """Store parent documents as JSON files for retrieval."""
    os.makedirs(DOCSTORE_DIR, exist_ok=True)
    for parent in parent_chunks:
        doc_id = parent.metadata["doc_id"]
        path = os.path.join(DOCSTORE_DIR, f"{doc_id}.json")
        with open(path, "w") as f:
            json.dump({
                "page_content": parent.page_content,
                "metadata": parent.metadata,
            }, f)


def store_children_corpus(child_chunks):
    """Save child chunk texts and metadata for BM25 index at query time."""
    corpus = []
    for child in child_chunks:
        corpus.append({
            "page_content": child.page_content,
            "doc_id": child.metadata.get("doc_id"),
        })
    path = os.path.join(DOCSTORE_DIR, "children_corpus.json")
    with open(path, "w") as f:
        json.dump(corpus, f)
    print(f"  Saved {len(corpus)} child texts to {path}")


def create_vectorstore(child_chunks, embeddings):
    """Embed child chunks and store in ChromaDB."""

    print(f"Connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}...")

    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    try:
        chroma_client.delete_collection(CHROMA_COLLECTION)
        print(f"  Cleared existing collection '{CHROMA_COLLECTION}'")
    except Exception:
        pass

    print("Embedding and storing child chunks...")

    max_retries = 5
    for attempt in range(max_retries):
        try:
            vectorstore = Chroma.from_documents(
                documents=child_chunks,
                embedding=embeddings,
                client=chroma_client,
                collection_name=CHROMA_COLLECTION,
            )
            break
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limited. Retrying in {wait:.1f}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                raise

    print("Child chunks stored in ChromaDB")

    return vectorstore


""" def test_search():
    
    from retriever import create_retriever

    print("\nTesting retrieval...")

    retriever = create_retriever()
    query = "How do I configure OAuth authentication?"
    results = retriever.invoke(query)

    for i, doc in enumerate(results):
        print(f"\nResult {i + 1}")
        print("-" * 50)
        section = doc.metadata.get("section", "")
        if section:
            print(f"Section: {section}")
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        print(f"Source: {source} (page {page})")
        print(doc.page_content[:500]) """


def main():

    load_dotenv()

    if not LLM_API_KEY:
        raise ValueError("LLM_API_KEY not found in environment variables")

    embeddings = OpenAIEmbeddings(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=EMBEDDING_MODEL,
    )

    documents = load_documents()

    # Parent chunks
    print("\nCreating parent chunks...")
    parent_chunks = create_parent_chunks(documents)
    parent_chunks = enrich_metadata(parent_chunks)
    print(f"  {len(parent_chunks)} parent chunks")

    # Child chunks
    print("\nCreating child chunks...")
    child_chunks = create_child_chunks(parent_chunks, embeddings)
    print(f"  {len(child_chunks)} child chunks")

    # Store parents as JSON for retrieval
    print("\nStoring parent documents...")
    if os.path.exists(DOCSTORE_DIR):
        for f in os.listdir(DOCSTORE_DIR):
            os.remove(os.path.join(DOCSTORE_DIR, f))
    store_parents(parent_chunks)
    store_children_corpus(child_chunks)
    print(f"  Saved {len(parent_chunks)} parents to {DOCSTORE_DIR}/")

    # Store children in ChromaDB
    print("\nBuilding vector database...")
    create_vectorstore(child_chunks, embeddings)

    print(f"\nIngestion complete: {len(parent_chunks)} parents, {len(child_chunks)} children")


if __name__ == "__main__":
    main()