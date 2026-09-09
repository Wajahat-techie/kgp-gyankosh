"""
Embeddings Factory for KGP Gyankosh
Provides switchable embedding backends:
- Local 'sentence-transformers' (default: all-MiniLM-L6-v2) - free, offline, local-first
- Cloud 'openai' (e.g., text-embedding-3-small) - API key driven
Controlled 100% via environment variables with zero code changes required.
"""

import os
import logging
from typing import Optional
from langchain_core.embeddings import Embeddings

logger = logging.getLogger("kgp_gyankosh.embeddings")


def get_embeddings_model() -> Embeddings:
    """
    Instantiates and returns the configured LangChain Embeddings provider.
    Controlled by the EMBEDDING_PROVIDER environment variable ('sentence-transformers' or 'openai').
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers").lower().strip()
    
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            logger.warning("OPENAI_API_KEY is missing. Falling back to local sentence-transformers.")
            provider = "sentence-transformers"
        else:
            model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            logger.info(f"Initializing OpenAI Embeddings with model: {model_name}")
            try:
                from langchain_openai import OpenAIEmbeddings
                return OpenAIEmbeddings(model=model_name, api_key=api_key)
            except Exception as exc:
                logger.error(f"Failed to initialize OpenAIEmbeddings: {exc}. Falling back to local.")
                provider = "sentence-transformers"

    elif provider in ("google", "gemini"):
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            logger.warning("GOOGLE_API_KEY is missing. Falling back to local sentence-transformers.")
            provider = "sentence-transformers"
        else:
            model_name = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/text-embedding-004")
            logger.info(f"Initializing Google GenAI Embeddings with model: {model_name}")
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
            except Exception as exc:
                logger.error(f"Failed to initialize GoogleGenerativeAIEmbeddings: {exc}. Falling back to local.")
                provider = "sentence-transformers"

    elif provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model_name = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        logger.info(f"Initializing Ollama Embeddings: {model_name} at {base_url}")
        try:
            from langchain_ollama import OllamaEmbeddings
            return OllamaEmbeddings(model=model_name, base_url=base_url)
        except Exception as exc:
            logger.error(f"Failed to initialize OllamaEmbeddings: {exc}. Falling back to local.")
            provider = "sentence-transformers"

    # Default: Local Sentence Transformers (direct, fast, offline)
    model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    logger.info(f"Initializing local SentenceTransformers embeddings: '{model_name}'")
    
    try:
        from sentence_transformers import SentenceTransformer
        try:
            st_model = SentenceTransformer(model_name, device="cpu", local_files_only=True)
        except Exception:
            st_model = SentenceTransformer(model_name, device="cpu")
        
        class DirectSentenceTransformersEmbeddings(Embeddings):
            """Fast native SentenceTransformer embeddings wrapper."""
            def __init__(self, model):
                self.model = model
            def embed_documents(self, texts):
                embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                return embeddings.tolist()
            def embed_query(self, text):
                embedding = self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
                return embedding.tolist()

        return DirectSentenceTransformersEmbeddings(st_model)
    except Exception as exc:
        logger.error(f"Error loading SentenceTransformer '{model_name}': {exc}", exc_info=True)
        raise RuntimeError(f"Unable to initialize embedding provider: {exc}")
