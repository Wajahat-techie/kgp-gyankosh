"""
Unit Tests for Embeddings Module
Tests provider selection, fallback logic, and mock initialization.
"""

import sys
import pytest
from unittest.mock import MagicMock
from src.indexing.embeddings import get_embeddings_model


class TestEmbeddings:
    """Test suite for get_embeddings_model."""

    def test_default_sentence_transformers(self, monkeypatch):
        monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
        mock_st = MagicMock()
        monkeypatch.setitem(sys.modules, "sentence_transformers", mock_st)
        embeddings = get_embeddings_model()
        assert embeddings is not None

    def test_openai_missing_api_key_fallback(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        mock_st = MagicMock()
        monkeypatch.setitem(sys.modules, "sentence_transformers", mock_st)
        embeddings = get_embeddings_model()
        assert embeddings is not None

    def test_google_missing_api_key_fallback(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "google")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        mock_st = MagicMock()
        monkeypatch.setitem(sys.modules, "sentence_transformers", mock_st)
        embeddings = get_embeddings_model()
        assert embeddings is not None
