"""
Reranking Module for KGP Gyankosh
Applies Cross-Encoder reranking (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2)
to re-score and refine candidate passages retrieved during hybrid search.
"""

import os
import logging
from typing import List, Tuple, Optional
from langchain_core.documents import Document

logger = logging.getLogger("kgp_gyankosh.reranker")


class DocumentReranker:
    """
    Reranks a list of candidate documents using a Cross-Encoder model.
    Unlike dual-encoders (bi-encoders), cross-encoders jointly process the query
    and passage together through self-attention layers, yielding significantly
    higher ranking precision.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.enabled = os.getenv("ENABLE_RERANKER", "true").lower() == "true"
        self.top_n = int(os.getenv("TOP_N_RERANK", "4"))
        self._model = None
        self._initialization_attempted = False

    def _load_model(self):
        """Lazy load the CrossEncoder model on first invocation."""
        if self._initialization_attempted:
            return self._model

        self._initialization_attempted = True
        if not self.enabled:
            logger.info("Reranker is disabled by configuration.")
            return None

        try:
            # First check if model files exist locally in HF cache
            import huggingface_hub
            logger.info(f"Attempting to load Cross-Encoder model: '{self.model_name}'")
            from sentence_transformers import CrossEncoder
            try:
                self._model = CrossEncoder(self.model_name, device="cpu", local_files_only=True)
            except Exception:
                self._model = CrossEncoder(self.model_name, device="cpu")
            logger.info("Cross-Encoder reranker loaded successfully.")
            return self._model
        except Exception as exc:
            logger.warning(
                f"CrossEncoder '{self.model_name}' not accessible ({exc}). "
                "Using high-speed hybrid RRF fallback reranking."
            )
            self._model = None
    def preload(self):
        """Eagerly warms up the model in memory to guarantee zero latency during user queries."""
        return self._load_model()

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Document, float]],
        top_n: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        Reranks a list of candidate documents against the query.
        
        Args:
            query: The user question or search query.
            candidates: List of (Document, initial_score) tuples from hybrid retrieval.
            top_n: Number of final reranked passages to return.
            
        Returns:
            List of (Document, rerank_score) sorted descending by relevance.
        """
        limit = top_n or self.top_n

        if not candidates:
            return []

        # If only 1 or 2 candidates, or reranker is disabled, return directly
        if len(candidates) <= 1 or not self.enabled:
            return candidates[:limit]

        model = self._load_model()
        if model is None:
            # Fallback: maintain hybrid search order
            return candidates[:limit]

        try:
            # Build query-passage pairs for CrossEncoder
            pairs = [[query, doc.page_content] for doc, _ in candidates]
            scores = model.predict(pairs)

            # Combine documents with reranker scores
            reranked = []
            for (doc, _), score in zip(candidates, scores):
                # Copy doc or enrich metadata
                doc.metadata["rerank_score"] = float(score)
                reranked.append((doc, float(score)))

            # Sort descending by cross-encoder score
            reranked.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"Reranked {len(candidates)} candidates down to top {limit}.")
            return reranked[:limit]

        except Exception as exc:
            logger.error(f"Error during cross-encoder reranking: {exc}. Returning unranked candidates.", exc_info=True)
            return candidates[:limit]
