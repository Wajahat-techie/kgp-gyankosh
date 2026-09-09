"""
Unit Tests for Conversational Memory Module
Tests sliding window turn management, dialogue formatting, and query reformulation fast-paths.
"""

import pytest
from unittest.mock import MagicMock
from src.memory.conversation_memory import ConversationMemory


class TestConversationMemory:
    """Test suite for ConversationMemory."""

    def test_add_turn_and_retrieval(self):
        memory = ConversationMemory(max_history_turns=3)
        memory.add_turn(
            user_query="Who is the Principal of KGP?",
            assistant_response="The Principal is Er. Tariq Ahmad.",
            sources=[{"title": "Staff_Directory.pdf", "page": 1}]
        )
        history = memory.get_history()
        assert len(history) == 1
        assert history[0]["user"] == "Who is the Principal of KGP?"
        assert history[0]["assistant"] == "The Principal is Er. Tariq Ahmad."
        assert len(history[0]["sources"]) == 1

    def test_sliding_window_limit(self):
        memory = ConversationMemory(max_history_turns=2)
        memory.add_turn("Q1", "A1")
        memory.add_turn("Q2", "A2")
        memory.add_turn("Q3", "A3")

        history = memory.get_history()
        assert len(history) == 2
        assert history[0]["user"] == "Q2"
        assert history[1]["user"] == "Q3"

    def test_clear_memory(self):
        memory = ConversationMemory()
        memory.add_turn("Q1", "A1")
        assert len(memory.get_history()) == 1
        memory.clear()
        assert len(memory.get_history()) == 0

    def test_format_history_for_prompt(self):
        memory = ConversationMemory()
        memory.add_turn("What is the fee?", "The fee is Rs 2500 per semester.")
        transcript = memory.format_history_for_prompt(max_turns=1)
        assert "User: What is the fee?" in transcript
        assert "Assistant: The fee is Rs 2500 per semester." in transcript

    def test_reformulate_query_empty_history(self):
        memory = ConversationMemory()
        # No history -> must return current query immediately without needing LLM
        query = "What is the examination schedule?"
        reformulated = memory.reformulate_query(query, llm_client=None)
        assert reformulated == query

    def test_reformulate_query_explicit_keywords_fast_path(self):
        memory = ConversationMemory()
        memory.add_turn("Tell me about leave rules.", "Leave rules follow J&K CSR.")
        
        # Contains explicit administrative keyword "order", so fast-path triggers
        explicit_query = "What is mentioned in Order No. 45?"
        reformulated = memory.reformulate_query(explicit_query, llm_client=None)
        assert reformulated == explicit_query

    def test_reformulate_query_with_llm_for_anaphora(self):
        memory = ConversationMemory()
        memory.add_turn("Who is the head of Civil Engineering?", "Er. Manzoor Ahmad.")

        # Follow up with pronoun "his"
        query_with_pronoun = "What is his contact number?"

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "What is the contact number of Er. Manzoor Ahmad?"
        mock_llm._llm.invoke.return_value = mock_response

        reformulated = memory.reformulate_query(query_with_pronoun, llm_client=mock_llm)
        assert reformulated == "What is the contact number of Er. Manzoor Ahmad?"
        assert mock_llm._llm.invoke.called
