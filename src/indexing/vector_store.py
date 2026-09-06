"""
Vector Store & Index Storage Manager for KGP Gyankosh
Manages FAISS dense vector storage and BM25 sparse keyword indices.
Supports persistent disk serialization, incremental change detection via MD5 manifests,
and zero-recomputation loading for Stage 2 queries.
"""

import os
import json
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

logger = logging.getLogger("kgp_gyankosh.vector_store")

MANIFEST_FILENAME = "index_manifest.json"
BM25_CORPUS_FILE = "bm25_corpus.pkl"
BM25_INDEX_FILE = "bm25_model.pkl"


def load_manifest(manifest_path: str) -> Dict[str, Any]:
    """Loads the indexing manifest tracking indexed files and hashes."""
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read index manifest at {manifest_path}: {e}")
    return {"indexed_files": {}, "total_chunks": 0, "embedding_model": None}


def save_manifest(manifest_path: str, data: Dict[str, Any]) -> None:
    """Persists the indexing manifest to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_vector_store(vector_store: FAISS, store_dir: str) -> None:
    """Saves the FAISS vector database to disk."""
    os.makedirs(store_dir, exist_ok=True)
    vector_store.save_local(store_dir)
    logger.info(f"FAISS vector store successfully saved to: {store_dir}")


def load_vector_store(store_dir: str, embeddings: Embeddings) -> Optional[FAISS]:
    """Loads an existing FAISS index from disk."""
    index_file = os.path.join(store_dir, "index.faiss")
    pkl_file = os.path.join(store_dir, "index.pkl")
    
    if not (os.path.exists(index_file) and os.path.exists(pkl_file)):
        logger.warning(f"No valid FAISS index found in {store_dir}")
        return None

    try:
        # allow_dangerous_deserialization=True is required by LangChain for local pickle loading
        vector_store = FAISS.load_local(
            store_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info(f"Loaded existing FAISS index from: {store_dir}")
        return vector_store
    except Exception as exc:
        logger.error(f"Failed to load FAISS index from {store_dir}: {exc}", exc_info=True)
        return None


def tokenize_text(text: str) -> List[str]:
    """Simple, fast lower-cased alphanumeric tokenizer for BM25 search."""
    import re
    tokens = re.findall(r"\b\w+\b", text.lower())
    return tokens


def save_bm25_store(chunks: List[Document], bm25_dir: str) -> None:
    """
    Tokenizes chunks, builds the BM25Okapi index, and serializes both
    the BM25 index and chunk objects to disk for fast Stage 2 loading.
    """
    os.makedirs(bm25_dir, exist_ok=True)
    
    tokenized_corpus = [tokenize_text(doc.page_content) for doc in chunks]
    bm25_model = BM25Okapi(tokenized_corpus)

    model_path = os.path.join(bm25_dir, BM25_INDEX_FILE)
    corpus_path = os.path.join(bm25_dir, BM25_CORPUS_FILE)

    with open(model_path, "wb") as f:
        pickle.dump(bm25_model, f)

    with open(corpus_path, "wb") as f:
        pickle.dump(chunks, f)

    logger.info(f"BM25 index and {len(chunks)} chunks saved to: {bm25_dir}")


def load_bm25_store(bm25_dir: str) -> Tuple[Optional[BM25Okapi], Optional[List[Document]]]:
    """
    Deserializes the cached BM25 model and document corpus in milliseconds.
    """
    model_path = os.path.join(bm25_dir, BM25_INDEX_FILE)
    corpus_path = os.path.join(bm25_dir, BM25_CORPUS_FILE)

    if not (os.path.exists(model_path) and os.path.exists(corpus_path)):
        logger.warning(f"BM25 index or corpus file not found in {bm25_dir}")
        return None, None

    try:
        with open(model_path, "rb") as f:
            bm25_model = pickle.load(f)
        with open(corpus_path, "rb") as f:
            chunks = pickle.load(f)
            
        logger.info(f"Loaded BM25 index with {len(chunks)} documents from: {bm25_dir}")
        return bm25_model, chunks
    except Exception as exc:
        logger.error(f"Error loading BM25 store from {bm25_dir}: {exc}", exc_info=True)
        return None, None
