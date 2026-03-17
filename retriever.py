import os
import json
import re

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi
import chromadb

from config import (
    EMBEDDING_MODEL, LLM_BASE_URL, LLM_API_KEY,
    CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION,
    DOCSTORE_DIR, RETRIEVER_K, RRF_K
)

TOKENIZE_PATTERN = re.compile(r"\w+")


def tokenize(text):
    return TOKENIZE_PATTERN.findall(text.lower())


class HybridParentChildRetriever:
    """
    Hybrid retriever combining dense (ChromaDB) and sparse (BM25) search
    with Reciprocal Rank Fusion, then maps child results to parent documents.
    """

    def __init__(self, vectorstore, bm25_index, bm25_corpus, docstore_dir, k, rrf_k):
        self.vectorstore = vectorstore
        self.bm25 = bm25_index
        self.bm25_corpus = bm25_corpus  # list of {"page_content": ..., "doc_id": ...}
        self.docstore_dir = docstore_dir
        self.k = k
        self.rrf_k = rrf_k

    def invoke(self, query):
        # Dense search (semantic)
        dense_results = self.vectorstore.similarity_search(query, k=self.k)
        print(f"[DENSE] {len(dense_results)} child chunks matched")

        # Sparse search (BM25 keyword)
        tokenized_query = tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:self.k]
        sparse_results = [self.bm25_corpus[i] for i in top_indices if bm25_scores[i] > 0]
        print(f"[SPARSE] {len(sparse_results)} child chunks matched via BM25")

        # Reciprocal Rank Fusion
        parent_scores = {}

        # Score dense results by rank
        for rank, child in enumerate(dense_results):
            doc_id = child.metadata.get("doc_id")
            if doc_id:
                rrf_score = 1.0 / (self.rrf_k + rank + 1)
                parent_scores[doc_id] = parent_scores.get(doc_id, 0) + rrf_score

        # Score sparse results by rank
        for rank, child in enumerate(sparse_results):
            doc_id = child.get("doc_id")
            if doc_id:
                rrf_score = 1.0 / (self.rrf_k + rank + 1)
                parent_scores[doc_id] = parent_scores.get(doc_id, 0) + rrf_score

        # Rank parents by fused score
        ranked_parents = sorted(parent_scores.items(), key=lambda x: x[1], reverse=True)

        parent_docs = []
        for doc_id, score in ranked_parents:
            parent = self._load_parent(doc_id)
            if parent:
                parent_docs.append(parent)
                print(f"  parent {len(parent_docs)}: {doc_id[:8]}... (RRF score: {score:.4f})")
            if len(parent_docs) >= self.k:
                break

        print(f"[RETRIEVER] {len(parent_docs)} parent documents returned (hybrid RRF)")
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
    Build a hybrid parent-child retriever:
    - Dense: ChromaDB similarity search on child embeddings
    - Sparse: BM25 keyword search on child texts
    - Fusion: Reciprocal Rank Fusion aggregated by parent
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

    # Build BM25 index from saved child corpus
    corpus_path = os.path.join(DOCSTORE_DIR, "children_corpus.json")
    with open(corpus_path) as f:
        bm25_corpus = json.load(f)

    tokenized_docs = [tokenize(child["page_content"]) for child in bm25_corpus]
    bm25_index = BM25Okapi(tokenized_docs)
    print(f"[BM25] Index built with {len(bm25_corpus)} child chunks")

    return HybridParentChildRetriever(
        vectorstore, bm25_index, bm25_corpus,
        DOCSTORE_DIR, RETRIEVER_K, RRF_K,
    )
