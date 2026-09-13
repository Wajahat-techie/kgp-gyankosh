"""
Document ingestion and indexing pipeline for KGP Gyankosh.

Reads documents from the data directory (PDF, DOCX, TXT, Images),
extracts text with OCR support for scanned pages, splits into chunks,
generates dense vector embeddings (FAISS) and builds a BM25 sparse index.
Supports incremental indexing via file MD5 hash comparison.
"""

import os
import sys
import argparse
import datetime
import logging
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from src.ingestion.document_loader import (
    load_single_document,
    compute_file_hash,
    SUPPORTED_EXTENSIONS
)
from src.ingestion.chunker import chunk_documents
from src.indexing.embeddings import get_embeddings_model
from src.indexing.vector_store import (
    load_manifest,
    save_manifest,
    load_vector_store,
    save_vector_store,
    save_bm25_store,
    load_bm25_store,
    MANIFEST_FILENAME
)
from langchain_community.vectorstores import FAISS


def setup_logging(log_dir: str) -> logging.Logger:
    """Configures logging to both stdout and a timestamped file."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"index_run_{timestamp}.log")

    logger = logging.getLogger("kgp_gyankosh")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info(f"Indexing log started: {log_file}")
    return logger


def run_indexing(data_dir: str, rebuild: bool = False) -> None:
    """
    Main indexing pipeline: discovers documents, chunks text, and creates/updates
    FAISS vector store and BM25 index.
    """
    log_dir = os.getenv("INDEX_LOG_DIR", "output/index_logs")
    if not os.path.isabs(log_dir):
        log_dir = os.path.normpath(os.path.join(PROJECT_ROOT, log_dir))

    vector_store_dir = os.getenv("VECTOR_STORE_DIR", "output/vector_store")
    if not os.path.isabs(vector_store_dir):
        vector_store_dir = os.path.normpath(os.path.join(PROJECT_ROOT, vector_store_dir))

    bm25_store_dir = os.getenv("BM25_STORE_DIR", "output/bm25_store")
    if not os.path.isabs(bm25_store_dir):
        bm25_store_dir = os.path.normpath(os.path.join(PROJECT_ROOT, bm25_store_dir))

    manifest_path = os.path.join(PROJECT_ROOT, "output", MANIFEST_FILENAME)

    if not os.path.isabs(data_dir):
        data_dir = os.path.abspath(os.path.join(PROJECT_ROOT, data_dir))

    logger = setup_logging(log_dir)
    logger.info(f"Starting document indexing on: {data_dir} (rebuild={rebuild})")

    if not os.path.exists(data_dir):
        logger.error(f"Data directory does not exist: {data_dir}")
        sys.exit(1)

    # 1. Initialize Embeddings
    logger.info("Initializing embedding model...")
    embeddings = get_embeddings_model()

    # 2. Check Index Manifest for Incremental Indexing
    manifest = {"indexed_files": {}, "total_chunks": 0, "embedding_model": None} if rebuild else load_manifest(manifest_path)
    indexed_files = manifest.get("indexed_files", {})

    # Discover all candidate files
    candidate_files = []
    for root, _, files in os.walk(data_dir):
        for f in sorted(files):
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                candidate_files.append(os.path.join(root, f))

    if not candidate_files:
        logger.warning(f"No supported documents found in {data_dir}. Exiting.")
        return

    logger.info(f"Found {len(candidate_files)} supported document(s) in {data_dir}")

    # Identify new or changed files
    files_to_process = []
    unchanged_files = []

    for file_path in candidate_files:
        file_name = os.path.basename(file_path)
        current_hash = compute_file_hash(file_path)

        if not rebuild and file_name in indexed_files:
            if indexed_files[file_name].get("file_hash") == current_hash:
                unchanged_files.append(file_name)
                continue
        
        files_to_process.append((file_path, current_hash))

    logger.info(f"Skipping {len(unchanged_files)} unchanged file(s). Processing {len(files_to_process)} file(s).")

    if not files_to_process and not rebuild:
        logger.info("No changes detected. Indices are up to date.")
        return

    # Load and chunk new/modified documents
    new_chunks = []
    processed_files_metadata = {}

    for file_path, f_hash in files_to_process:
        file_name = os.path.basename(file_path)
        logger.info(f"Loading document: {file_name}")
        try:
            raw_docs = load_single_document(file_path)
            if not raw_docs:
                logger.warning(f"No readable content extracted from: {file_name}")
                continue

            chunks = chunk_documents(raw_docs)
            new_chunks.extend(chunks)

            processed_files_metadata[file_name] = {
                "file_hash": f_hash,
                "file_path": file_path,
                "chunks_count": len(chunks),
                "last_indexed_at": datetime.datetime.now().isoformat()
            }
            logger.info(f"Chunked '{file_name}': {len(chunks)} chunks.")
        except Exception as err:
            logger.error(f"Error processing document '{file_name}': {err}", exc_info=True)

    if not new_chunks and not rebuild:
        logger.warning("No new chunks extracted from pending files.")
        return

    # Update or Create FAISS Vector Store
    logger.info(f"Generating embeddings for {len(new_chunks)} chunks...")
    vector_store = None
    if not rebuild:
        vector_store = load_vector_store(vector_store_dir, embeddings)

    if vector_store is None or rebuild:
        if not new_chunks:
            logger.error("Cannot build index with 0 chunks.")
            return
        vector_store = FAISS.from_documents(new_chunks, embeddings)
        logger.info(f"Created fresh FAISS vector index with {len(new_chunks)} chunks.")
    else:
        vector_store.add_documents(new_chunks)
        logger.info(f"Added {len(new_chunks)} chunks to existing FAISS vector index.")

    save_vector_store(vector_store, vector_store_dir)

    # Rebuild BM25 index over all active chunks
    logger.info("Building BM25 keyword search index...")
    all_chunks = []
    
    if not rebuild:
        _, existing_corpus = load_bm25_store(bm25_store_dir)
        if existing_corpus:
            updated_names = set(processed_files_metadata.keys())
            all_chunks = [c for c in existing_corpus if c.metadata.get("source") not in updated_names]

    all_chunks.extend(new_chunks)
    save_bm25_store(all_chunks, bm25_store_dir)

    # Update manifest
    indexed_files.update(processed_files_metadata)
    manifest["indexed_files"] = indexed_files
    manifest["total_chunks"] = len(all_chunks)
    manifest["embedding_model"] = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    manifest["last_updated_at"] = datetime.datetime.now().isoformat()
    save_manifest(manifest_path, manifest)

    logger.info("Indexing completed successfully!")
    logger.info(f"Total documents: {len(indexed_files)} | Total chunks: {len(all_chunks)}")
    logger.info(f"Vector store: {vector_store_dir} | BM25 store: {bm25_store_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="KGP Gyankosh document ingestion and indexing utility."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.getenv("DATA_DIR", os.path.join(PROJECT_ROOT, "data")),
        help="Path to folder containing documents to index."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force full rebuild from scratch, ignoring manifest cache."
    )
    args = parser.parse_args()

    run_indexing(data_dir=args.data_dir, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
