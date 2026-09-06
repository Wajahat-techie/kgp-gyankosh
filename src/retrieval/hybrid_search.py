"""
Hybrid Search Retriever for KGP Gyankosh
Combines dense semantic search (FAISS) with sparse lexical search (BM25Okapi)
and merges candidates using Reciprocal Rank Fusion (RRF).
"""

import os
import logging
from typing import List, Tuple, Dict, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

from src.indexing.vector_store import tokenize_text

logger = logging.getLogger("kgp_gyankosh.hybrid_search")


def reciprocal_rank_fusion(
    vector_results: List[Tuple[Document, float]],
    bm25_results: List[Tuple[Document, float]],
    alpha: float = 0.5,
    k_constant: int = 60
) -> List[Tuple[Document, float]]:
    """
    Combines ranked lists from vector search and BM25 using Reciprocal Rank Fusion (RRF).
    
    Formula:
      RRF_score(d) = alpha * (1 / (k + rank_vector(d))) + (1 - alpha) * (1 / (k + rank_bm25(d)))
      
    Args:
        vector_results: List of (Document, score) ranked by semantic similarity.
        bm25_results: List of (Document, score) ranked by BM25 keyword score.
        alpha: Weight balancing dense (alpha) vs sparse (1 - alpha). Default 0.5.
        k_constant: Rank smoothing constant (default 60).
        
    Returns:
        Combined list of (Document, fused_rrf_score) sorted descending by score.
    """
    scores: Dict[str, float] = {}
    doc_lookup: Dict[str, Document] = {}

    # 1. Process Vector Search rankings
    for rank, (doc, _) in enumerate(vector_results, start=1):
        doc_id = doc.metadata.get("chunk_id", doc.page_content[:50])
        doc_lookup[doc_id] = doc
        dense_score = alpha * (1.0 / (k_constant + rank))
        scores[doc_id] = scores.get(doc_id, 0.0) + dense_score

    # 2. Process BM25 rankings
    for rank, (doc, _) in enumerate(bm25_results, start=1):
        doc_id = doc.metadata.get("chunk_id", doc.page_content[:50])
        doc_lookup[doc_id] = doc
        sparse_score = (1.0 - alpha) * (1.0 / (k_constant + rank))
        scores[doc_id] = scores.get(doc_id, 0.0) + sparse_score

    # 3. Sort fused candidates
    sorted_candidates = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    fused_results = [(doc_lookup[doc_id], score) for doc_id, score in sorted_candidates]

    logger.debug(
        f"RRF Fusion: {len(vector_results)} vector + {len(bm25_results)} BM25 -> {len(fused_results)} unified chunks"
    )
    return fused_results


class HybridRetriever:
    """
    Orchestrates hybrid retrieval against a pre-loaded FAISS vector store
    and pre-loaded BM25Okapi index.
    """

    def __init__(
        self,
        vector_store: FAISS,
        bm25_model: BM25Okapi,
        bm25_corpus: List[Document],
        alpha: float = None,
        top_k: int = None
    ):
        self.vector_store = vector_store
        self.bm25_model = bm25_model
        self.bm25_corpus = bm25_corpus
        
        self.alpha = float(os.getenv("HYBRID_ALPHA", "0.5")) if alpha is None else alpha
        self.top_k = int(os.getenv("TOP_K_RETRIEVAL", "10")) if top_k is None else top_k
        self.k_constant = int(os.getenv("RRF_K_CONSTANT", "60"))

    def search_vector(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Executes similarity search on FAISS vector store."""
        try:
            # similarity_search_with_score returns (Document, L2_distance)
            return self.vector_store.similarity_search_with_score(query, k=k)
        except Exception as exc:
            logger.error(f"Vector search failed for query '{query}': {exc}", exc_info=True)
            return []

    def search_bm25(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Executes keyword search on BM25Okapi index."""
        if not self.bm25_model or not self.bm25_corpus:
            return []

        tokens = tokenize_text(query)
        if not tokens:
            return []

        try:
            bm25_scores = self.bm25_model.get_scores(tokens)
            # Pair scores with corresponding documents
            ranked_pairs = sorted(
                zip(self.bm25_corpus, bm25_scores),
                key=lambda pair: pair[1],
                reverse=True
            )
            # Filter out entries with zero score
            positive_pairs = [(doc, score) for doc, score in ranked_pairs if score > 0]
            return positive_pairs[:k]
        except Exception as exc:
            logger.error(f"BM25 search failed for query '{query}': {exc}", exc_info=True)
            return []

    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[Document, float]]:
        """
        Executes hybrid retrieval by running both vector and BM25 searches in parallel,
        then combining them via Reciprocal Rank Fusion.
        
        Args:
            query: User's search or reformulated question.
            top_k: Number of combined candidates to return.
            
        Returns:
            List of (Document, rrf_score) tuples.
        """
        limit = top_k or self.top_k
        fetch_limit = limit * 2  # Retrieve more from each individual branch before fusion

        vector_results = self.search_vector(query, k=fetch_limit)
        bm25_results = self.search_bm25(query, k=fetch_limit)

        fused = reciprocal_rank_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
            alpha=self.alpha,
            k_constant=self.k_constant
        )

        return fused[:limit]
