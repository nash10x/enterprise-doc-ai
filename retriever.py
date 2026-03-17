import os
import json

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb

from config import (
    EMBEDDING_MODEL, LLM_BASE_URL, LLM_API_KEY,
    CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION,
    DOCSTORE_DIR, RETRIEVER_K
)


class ParentChildRetriever:
    """Searches child chunks in ChromaDB, returns their larger parent documents."""

    def __init__(self, vectorstore, docstore_dir, k):
        self.vectorstore = vectorstore
        self.docstore_dir = docstore_dir
        self.k = k

    def invoke(self, query):
        children = self.vectorstore.similarity_search(query, k=self.k)
        print(f"[RETRIEVER] {len(children)} child chunks matched")

        seen_ids = set()
        parent_docs = []
        for child in children:
            parent_id = child.metadata.get("doc_id")
            if parent_id and parent_id not in seen_ids:
                seen_ids.add(parent_id)
                parent = self._load_parent(parent_id)
                if parent:
                    parent_docs.append(parent)

        print(f"[RETRIEVER] {len(parent_docs)} parent documents returned")
        return parent_docs

    def _load_parent(self, doc_id):
        path = os.path.join(self.docstore_dir, f"{doc_id}.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        return Document(
            page_content=data["page_content"],
            metadata=data["metadata"],
        )


def create_retriever():
    """
    Build a parent-child retriever backed by ChromaDB and a local JSON docstore.
    Child chunks are searched for similarity; their larger parent chunks are returned as context.
    """
    embeddings = OpenAIEmbeddings(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=EMBEDDING_MODEL,
    )

    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    vectorstore = Chroma(
        client=chroma_client,
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
    )

    return ParentChildRetriever(vectorstore, DOCSTORE_DIR, RETRIEVER_K)
