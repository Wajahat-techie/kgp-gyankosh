import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
load_dotenv()
from src.indexing.embeddings import get_embeddings_model
from src.indexing.vector_store import load_vector_store, load_bm25_store
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.reranker import DocumentReranker
from src.llm.client import LLMClient

emb = get_embeddings_model()
vs = load_vector_store('output/vector_store', emb)
bm, corp = load_bm25_store('output/bm25_store')
retriever = HybridRetriever(vs, bm, corp)
reranker = DocumentReranker()
llm = LLMClient()

queries = [
    'order for Release of Child Education Allowance for the 2025-26',
    'Release of Child Education Allowance for the 2025-26',
    'Child Education Allowance 2025-26',
    'order for Release of Child Education Allowance',
    'Child Education Allowance',
    'CEA 2025-26'
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
    print("Answer summary:", ans["answer"][:120].replace('\n', ' '))
