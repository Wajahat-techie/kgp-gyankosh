"""
Unit Tests for Retrieval and Reranker Module
Tests BM25 tokenization, Reciprocal Rank Fusion (RRF) scoring, and Reranker fallback mechanisms.
"""

import pytest
from langchain_core.documents import Document
from src.indexing.vector_store import tokenize_text
from src.retrieval.hybrid_search import reciprocal_rank_fusion
from src.retrieval.reranker import DocumentReranker


class TestTokenization:
    """Test text tokenization used by BM25."""

    def test_tokenize_clean_text(self):
        tokens = tokenize_text("Kashmir Govt Polytechnic Srinagar Leave Rules 2026")
        assert "kashmir" in tokens
        assert "govt" in tokens
        assert "polytechnic" in tokens
        assert "srinagar" in tokens
        assert "2026" in tokens

    def test_tokenize_punctuation_and_casing(self):
        tokens = tokenize_text("Order No. KGP/EST/2026-99: Granted!")
        assert "order" in tokens
        assert "no" in tokens
        assert "kgp" in tokens
        assert "est" in tokens
        assert "granted" in tokens

    def test_tokenize_empty(self):
        assert tokenize_text("") == []
        assert tokenize_text("   --- !!! ") == []


class TestReciprocalRankFusion:
    """Test Reciprocal Rank Fusion (RRF) algorithm."""

    def test_rrf_empty_inputs(self):
        assert reciprocal_rank_fusion([], []) == []

    def test_rrf_overlap_boost(self):
        # Document present in both lists should rank highest
        doc_shared = Document(page_content="Shared institutional notice", metadata={"chunk_id": "shared#p1_c1"})
        doc_dense_only = Document(page_content="Vector only candidate", metadata={"chunk_id": "dense#p1_c1"})
        doc_bm25_only = Document(page_content="BM25 only candidate", metadata={"chunk_id": "bm25#p1_c1"})

        vector_results = [
            (doc_dense_only, 0.2),
            (doc_shared, 0.4)
        ]
        bm25_results = [
            (doc_shared, 8.5),
            (doc_bm25_only, 5.0)
        ]

        fused = reciprocal_rank_fusion(vector_results, bm25_results, alpha=0.5, k_constant=60)
        
        assert len(fused) == 3
        top_doc, top_score = fused[0]
        # doc_shared got points from both rank 2 in vector and rank 1 in bm25:
        # dense: 0.5 * (1/62) = 0.00806
        # bm25: 0.5 * (1/61) = 0.00819
        # total shared: ~0.01625
        # dense_only: 0.5 * (1/61) = 0.00819
        # Therefore, doc_shared MUST be top ranked
        assert top_doc.metadata["chunk_id"] == "shared#p1_c1"

    def test_rrf_alpha_weighting(self):
        doc_a = Document(page_content="Doc A", metadata={"chunk_id": "A"})
        doc_b = Document(page_content="Doc B", metadata={"chunk_id": "B"})

        vector_results = [(doc_a, 0.1)]
        bm25_results = [(doc_b, 10.0)]

        # Alpha = 1.0 (pure dense) -> Doc A must rank #1
        fused_dense = reciprocal_rank_fusion(vector_results, bm25_results, alpha=1.0)
        assert fused_dense[0][0].metadata["chunk_id"] == "A"

        # Alpha = 0.0 (pure sparse) -> Doc B must rank #1
        fused_sparse = reciprocal_rank_fusion(vector_results, bm25_results, alpha=0.0)
        assert fused_sparse[0][0].metadata["chunk_id"] == "B"


class TestDocumentReranker:
    """Test DocumentReranker behaviors and fallback."""

    def test_reranker_disabled(self, monkeypatch):
        monkeypatch.setenv("ENABLE_RERANKER", "false")
        reranker = DocumentReranker()
        assert not reranker.enabled

        doc1 = Document(page_content="Notice 1", metadata={"chunk_id": "1"})
        doc2 = Document(page_content="Notice 2", metadata={"chunk_id": "2"})
        candidates = [(doc1, 0.9), (doc2, 0.8)]

        results = reranker.rerank("query", candidates, top_n=1)
        assert len(results) == 1
        assert results[0][0].metadata["chunk_id"] == "1"

    def test_reranker_fallback_on_uninitialized_model(self, monkeypatch):
        monkeypatch.setenv("ENABLE_RERANKER", "true")
        reranker = DocumentReranker(model_name="non-existent-model-xyz")
        # Ensure _load_model returns None safely without crashing
        reranker._initialization_attempted = True
        reranker._model = None

        doc1 = Document(page_content="Notice 1", metadata={"chunk_id": "1"})
        doc2 = Document(page_content="Notice 2", metadata={"chunk_id": "2"})
        candidates = [(doc1, 0.9), (doc2, 0.8)]

        results = reranker.rerank("query", candidates, top_n=2)
        assert len(results) == 2
        assert results[0][0].metadata["chunk_id"] == "1"

    def test_reranker_empty_candidates(self):
        reranker = DocumentReranker()
        assert reranker.rerank("query", []) == []
