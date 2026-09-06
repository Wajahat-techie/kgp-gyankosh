"""
Conversational Memory Module for KGP Gyankosh
Manages multi-turn conversation state and performs query reformulation/condensation
so follow-up questions (e.g., "what about the third one?") correctly resolve
against prior administrative context before triggering retrieval.
"""

import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("kgp_gyankosh.memory")


class ConversationMemory:
    """
    Tracks multi-turn conversation messages and handles contextual query reformulation.
    """

    def __init__(self, max_history_turns: int = 5):
        self.max_history_turns = max_history_turns
        self.history: List[Dict[str, Any]] = []

    def add_turn(
        self,
        user_query: str,
        assistant_response: str,
        sources: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Appends a completed question-answer turn to conversation memory."""
        self.history.append({
            "user": user_query,
            "assistant": assistant_response,
            "sources": sources or []
        })
        # Keep within sliding window
        if len(self.history) > self.max_history_turns:
            self.history.pop(0)

    def get_history(self) -> List[Dict[str, Any]]:
        """Returns the current list of conversation turns."""
        return self.history

    def clear(self) -> None:
        """Clears all conversation turns."""
        self.history = []
        logger.info("Conversation memory cleared.")

    def format_history_for_prompt(self, max_turns: int = 3) -> str:
        """Formats recent history as a readable dialogue transcript."""
        if not self.history:
            return ""

        recent = self.history[-max_turns:]
        lines = []
        for turn in recent:
            lines.append(f"User: {turn['user']}")
            # Truncate assistant response if overly long to conserve context window
            resp_snippet = turn["assistant"][:300].replace("\n", " ")
            lines.append(f"Assistant: {resp_snippet}")
        return "\n".join(lines)

    def reformulate_query(self, current_query: str, llm_client=None) -> str:
        """
        Resolves conversational references (anaphora, follow-ups) into a standalone search query.
        For example:
          Turn 1: "What does the attendance circular say?"
          Turn 2: "What is the condonation limit?"
          Reformulated: "What is the attendance condonation limit in KGP attendance circular?"
          
        If no history exists, returns the raw query unchanged.
        """
        if not self.history or llm_client is None:
            return current_query

        # Fast-path: if query has explicit administrative terms or lacks anaphoric references, skip LLM
        query_lower = current_query.lower()
        explicit_indicators = ["order", "circular", "notice", "notification", "dated", "no.", "no ", "meeting", "committee", "minutes", "rules", "policy"]
        if any(ind in query_lower for ind in explicit_indicators):
            return current_query

        words = current_query.strip().split()
        if len(words) > 10:
            return current_query

        # Only invoke LLM rewrite if query contains pronouns or contextual references
        anaphora_cues = {"it", "he", "she", "they", "this", "that", "these", "those", "his", "her", "their", "them", "same", "above", "earlier"}
        tokens = {w.strip("?,.!") for w in query_lower.split()}
        if not tokens.intersection(anaphora_cues) and len(words) >= 4:
            return current_query

        dialogue_context = self.format_history_for_prompt()
        reformulation_prompt = (
            "You are a query rewriting assistant for an administrative document search system.\n"
            "Given the following conversation history between an admin user and an assistant, "
            "rewrite the user's latest follow-up question into a single, standalone search query.\n"
            "Do NOT answer the question. Only output the standalone query without any preamble, quotes, or markdown.\n\n"
            f"Conversation History:\n{dialogue_context}\n\n"
            f"Latest Follow-up Question: {current_query}\n\n"
            "Standalone Search Query:"
        )

        try:
            standalone_query = llm_client.generate_raw(reformulation_prompt).strip()
            # Clean up any surrounding quotes or boilerplate
            standalone_query = standalone_query.strip('"\'- \n')
            if standalone_query:
                logger.info(f"Query reformulated: '{current_query}' -> '{standalone_query}'")
                return standalone_query
        except Exception as exc:
            logger.warning(f"Query reformulation failed ({exc}). Falling back to original query.")

        return current_query
