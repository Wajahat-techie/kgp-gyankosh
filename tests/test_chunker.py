"""
Unit Tests for Ingestion and Chunking Module
Tests RecursiveCharacterTextSplitter and chunk_documents metadata enrichment.
"""

import unittest
from langchain_core.documents import Document
from src.ingestion.chunker import (
    RecursiveCharacterTextSplitter,
    chunk_documents,
    get_configured_text_splitter
)


class TestTextSplitter(unittest.TestCase):
    """Test suite for RecursiveCharacterTextSplitter."""

    def test_short_text_no_split(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        text = "Order No. 42: Kashmiri Government Polytechnic Srinagar."
        chunks = splitter.split_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_text(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        assert splitter.split_text("") == []
        assert splitter.split_text("   ") == []

    def test_splitting_on_paragraphs(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=10)
        para1 = "Paragraph 1: Administrative council notification for college leaves."
        para2 = "Paragraph 2: Detailed guidelines for faculty and non-teaching staff."
        text = f"{para1}\n\n{para2}"
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2
        assert any("Paragraph 1" in c for c in chunks)
        assert any("Paragraph 2" in c for c in chunks)

    def test_split_documents(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
        doc = Document(
            page_content="Section 1: Academic Council.\n\nSection 2: Examination Schedule.\n\nSection 3: Fee Structure.",
            metadata={"source": "notices.pdf", "page": 1}
        )
        split_docs = splitter.split_documents([doc])
        assert len(split_docs) >= 3
        for chunk in split_docs:
            assert chunk.metadata["source"] == "notices.pdf"
            assert chunk.metadata["page"] == 1


class TestChunkDocuments:
    """Test suite for high-level chunk_documents function."""

    def test_empty_documents_input(self):
        assert chunk_documents([]) == []

    def test_chunk_id_and_metadata_enrichment(self):
        docs = [
            Document(
                page_content="Order No 101 of 2026: Sanction is hereby accorded to the grant of casual leave in favour of Er. Tariq Ahmad.",
                metadata={"source": "Leave_Orders.pdf", "page": 3}
            ),
            Document(
                page_content="Order No 102 of 2026: Reconstitution of Internal Anti-Ragging Committee for Academic Year 2026-27.",
                metadata={"source": "Committee_Notice.docx", "page": 1}
            )
        ]

        chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
        assert len(chunks) >= 2

        first_chunk = chunks[0]
        assert "chunk_id" in first_chunk.metadata
        assert first_chunk.metadata["chunk_id"] == "Leave_Orders.pdf#p3_c1"
        assert first_chunk.metadata["chunk_index"] == 1
        assert first_chunk.metadata["source"] == "Leave_Orders.pdf"

        second_chunk = chunks[1]
        assert second_chunk.metadata["chunk_id"] == "Committee_Notice.docx#p1_c1"
        assert second_chunk.metadata["chunk_index"] == 1

    def test_chunk_counter_increments_per_page(self):
        # A long document on the same page producing multiple chunks
        long_content = "Clause A: Detailed circular rules for procurement.\n\n" * 15
        doc = Document(
            page_content=long_content,
            metadata={"source": "Procurement.pdf", "page": 2}
        )
        chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1

        ids = [c.metadata["chunk_id"] for c in chunks]
        indices = [c.metadata["chunk_index"] for c in chunks]

        assert ids[0] == "Procurement.pdf#p2_c1"
        assert ids[1] == "Procurement.pdf#p2_c2"
        assert indices == list(range(1, len(chunks) + 1))
