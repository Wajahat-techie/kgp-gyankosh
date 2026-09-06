"""
Text Chunking Module for KGP Gyankosh
Splits administrative documents into semantically coherent chunks using
RecursiveCharacterTextSplitter with configurable size and overlap.
Enriches chunks with unique chunk identifiers and preserves source metadata.
"""

import os
import logging
from typing import List
from langchain_core.documents import Document

logger = logging.getLogger("kgp_gyankosh.chunker")
class RecursiveCharacterTextSplitter:
    """
    Fast, dependency-free implementation of RecursiveCharacterTextSplitter
    tuned for administrative circulars, policy clauses, and numbered orders.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100, separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n",
            "\r\n\r\n",
            "\n",
            "\r\n",
            ". ",
            "; ",
            " ",
            ""
        ]

    def split_text(self, text: str) -> List[str]:
        final_chunks = []
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        # Find first applicable separator
        chosen_sep = ""
        for sep in self.separators:
            if sep in text:
                chosen_sep = sep
                break

        splits = text.split(chosen_sep) if chosen_sep else list(text)

        current_doc = []
        total_len = 0
        sep_len = len(chosen_sep)

        for s in splits:
            s_len = len(s)
            if total_len + s_len + (sep_len if current_doc else 0) > self.chunk_size:
                if current_doc:
                    doc_str = chosen_sep.join(current_doc).strip()
                    if doc_str:
                        final_chunks.append(doc_str)
                    
                    # Handle overlap by keeping tail elements
                    overlap_len = 0
                    overlap_docs = []
                    for item in reversed(current_doc):
                        if overlap_len + len(item) <= self.chunk_overlap:
                            overlap_docs.insert(0, item)
                            overlap_len += len(item) + sep_len
                        else:
                            break
                    current_doc = overlap_docs
                    total_len = sum(len(x) for x in current_doc) + (sep_len * max(0, len(current_doc) - 1))

                # If single split item itself exceeds chunk_size, split recursively with finer separators
                if s_len > self.chunk_size:
                    sub_splitters = [sep for sep in self.separators if sep != chosen_sep]
                    sub_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                        separators=sub_splitters or [""]
                    )
                    sub_chunks = sub_splitter.split_text(s)
                    final_chunks.extend(sub_chunks)
                else:
                    current_doc.append(s)
                    total_len += s_len + (sep_len if len(current_doc) > 1 else 0)
            else:
                current_doc.append(s)
                total_len += s_len + (sep_len if len(current_doc) > 1 else 0)

        if current_doc:
            doc_str = chosen_sep.join(current_doc).strip()
            if doc_str:
                final_chunks.append(doc_str)

        return final_chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        result = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for chunk_str in chunks:
                meta = dict(doc.metadata)
                result.append(Document(page_content=chunk_str, metadata=meta))
        return result


def get_configured_text_splitter(
    chunk_size: int = None,
    chunk_overlap: int = None
) -> RecursiveCharacterTextSplitter:
    """Instantiates a RecursiveCharacterTextSplitter using environment variables or defaults."""
    size = chunk_size or int(os.getenv("CHUNK_SIZE", "500"))
    overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "100"))
    return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)


def chunk_documents(
    documents: List[Document],
    chunk_size: int = None,
    chunk_overlap: int = None
) -> List[Document]:
    """
    Splits a list of loaded documents into smaller overlapping chunks.
    
    Args:
        documents: Raw Document objects from document loader.
        chunk_size: Optional override for character limit per chunk.
        chunk_overlap: Optional override for overlap between chunks.
        
    Returns:
        List of chunked Document objects with unique chunk_id metadata.
    """
    if not documents:
        logger.warning("No documents provided for chunking.")
        return []

    splitter = get_configured_text_splitter(chunk_size, chunk_overlap)
    split_docs = splitter.split_documents(documents)

    # Assign deterministic, traceable chunk IDs
    source_chunk_counter = {}
    for chunk in split_docs:
        source = chunk.metadata.get("source", "unknown_doc")
        page = chunk.metadata.get("page", 1)
        
        counter_key = f"{source}_p{page}"
        current_idx = source_chunk_counter.get(counter_key, 0) + 1
        source_chunk_counter[counter_key] = current_idx

        # Structured chunk identifier e.g. "KGP_Notice_Leave_2026.pdf#p1_c1"
        chunk.metadata["chunk_id"] = f"{source}#p{page}_c{current_idx}"
        chunk.metadata["chunk_index"] = current_idx

    logger.info(
        f"Chunking complete: {len(documents)} parent documents split into {len(split_docs)} chunks."
    )
    return split_docs
