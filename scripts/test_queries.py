"""
test_queries.py
Quick administrative query verification script for testing retrieval & LLM generation.
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from src.indexing.embeddings import get_embeddings_model
from src.indexing.vector_store import load_vector_store, load_bm25_store
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.reranker import DocumentReranker
from src.llm.client import LLMClient


def run_test_queries():
    emb = get_embeddings_model()
    vs = load_vector_store(os.path.join(PROJECT_ROOT, "output", "vector_store"), emb)
    bm, corp = load_bm25_store(os.path.join(PROJECT_ROOT, "output", "bm25_store"))
    retriever = HybridRetriever(vs, bm, corp)
    reranker = DocumentReranker()
    llm = LLMClient()

    queries = [
        "order for Release of Child Education Allowance for the 2025-26",
        "Child Education Allowance 2025-26",
        "What are the leave rules for faculty?",
        "Anti-Ragging Committee details"
    ]

    for q in queries:
        cands = retriever.retrieve(q, top_k=10)
        reranked = reranker.rerank(q, cands)
        print(f"\n=== QUERY: '{q}' ===")
        for i, (d, s) in enumerate(reranked[:3]):
            src = d.metadata.get("source")
            pg = d.metadata.get("page")
            print(f"  {i+1}. {src} (Page {pg}) - Rerank: {s:.3f}")
        top_docs = [d for d, _ in reranked]
        ans = llm.generate_answer(q, top_docs)
        print("Answer summary:", ans["answer"][:150].replace("\n", " "))


if __name__ == "__main__":
    run_test_queries()
