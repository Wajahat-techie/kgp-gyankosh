"""
LLM Client Factory & Hallucination Mitigation for KGP Gyankosh
Switchable LLM backend supporting:
- Fully local Ollama (default: llama3.1) - free, private, offline
- Cloud OpenAI (e.g. gpt-4o-mini) - API key driven
- Optional Google Gemini (gemini-1.5-flash)
Enforces strict hallucination mitigation and structured source citations.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()
logger = logging.getLogger("kgp_gyankosh.llm")

# System prompt engineered for strict administrative grounding & hallucination prevention
ADMIN_SYSTEM_PROMPT = """You are KGP Gyankosh, the official internal Enterprise Knowledge Assistant for Kashmir Government Polytechnic College, Srinagar.

CRITICAL OPERATIONAL RULES:
1. GROUNDING: Answer the question STRICTLY and SOLELY based on the provided Official Administrative Context passages below. 
2. NO HALLUCINATION: If the requested information is not explicitly stated in the provided context, or if the context is empty/irrelevant, you MUST reply with:
   "I could not find this information in the available official documents."
   Do NOT attempt to invent, speculate, or extrapolate policies, dates, fees, or rules from external knowledge.
3. FACTUAL CITATION: Whenever you state a rule, policy, fee, or date, reference the exact document title and page number where it appears.
4. INSTITUTION IDENTITY: Whenever mentioning the institution, ALWAYS format it prominently in bold uppercase as **KASHMIR GOVERNMENT POLYTECHNIC COLLEGE, SRINAGAR**.
5. STRUCTURE: Keep your response professional, well-structured (using bullet points where helpful), and concise.
6. SOURCES SECTION: Always conclude your response with a structured source list in the following format:
Sources:
- [Document_Name.pdf, Page X]
"""


def _extract_text(content: Any) -> str:
    """Safely extracts a clean string from various LLM response.content types (str, list of dicts, etc.)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
        return "".join(parts).strip()
    return str(content).strip()


class LLMClient:
    """
    Unified LLM Client providing switchable inference across Ollama, OpenAI, and Gemini.
    """

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower().strip()
        self.model_name = model
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))
        self._llm = self._initialize_llm()

    def _initialize_llm(self):
        """Instantiates the underlying LangChain chat model based on provider."""
        logger.info(f"Configuring LLM Client for provider: '{self.provider}'")

        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not api_key:
                logger.warning("OPENAI_API_KEY is missing! Falling back to local Ollama.")
                self.provider = "ollama"
            else:
                model_name = self.model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                logger.info(f"Connecting to OpenAI Chat with model: {model_name}")
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=model_name,
                    api_key=api_key,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )

        if self.provider == "google":
            api_key = os.getenv("GOOGLE_API_KEY", "").strip()
            if not api_key:
                logger.warning("GOOGLE_API_KEY is missing! Falling back to local Ollama.")
                self.provider = "ollama"
            else:
                model_name = self.model_name or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
                logger.info(f"Connecting to Google GenAI with model: {model_name}")
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=api_key,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )

        # Default: Local Ollama
        model_name = self.model_name or os.getenv("OLLAMA_MODEL", "llama3.1:latest")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        logger.info(f"Connecting to local Ollama instance at {base_url} with model: {model_name}")
        
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=model_name,
                base_url=base_url,
                temperature=self.temperature,
                num_predict=self.max_tokens
            )
        except Exception as exc:
            logger.error(f"Failed to initialize ChatOllama: {exc}. Using fallback wrapper.", exc_info=True)
            return None

    def generate_raw(self, prompt: str) -> str:
        """Executes a single raw completion without administrative system prompt (e.g. for query reformulation)."""
        if self._llm is None:
            return prompt

        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            return _extract_text(response.content)
        except Exception as exc:
            logger.warning(f"LLM raw generation error: {exc}")
            return prompt

    def generate_chat(
        self,
        query: str,
        conversation_history: str = "",
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Direct LLM chat mode: Allows using the underlying LLM (Llama 3.1, Gemini, GPT) 
        as a versatile general-purpose AI assistant for drafting documents, answering general queries,
        coding, explaining concepts, or general conversational interaction.
        """
        default_sys = (
            "You are KGP Gyankosh AI, the official intelligent AI Assistant for "
            "KASHMIR GOVERNMENT POLYTECHNIC COLLEGE, SRINAGAR. "
            "You can assist with general queries, drafting official notices/letters/circulars, "
            "explaining concepts, programming, education, and general conversation. "
            "Always be professional, polite, well-structured, and helpful."
        )
        sys_msg = system_prompt or default_sys
        
        messages = [SystemMessage(content=sys_msg)]
        if conversation_history:
            messages.append(HumanMessage(content=f"Previous Conversation:\n{conversation_history}"))
        messages.append(HumanMessage(content=query))
        
        try:
            if self._llm is None:
                raise RuntimeError("LLM backend is not initialized.")
            response = self._llm.invoke(messages)
            answer_text = _extract_text(response.content)
            return {
                "answer": answer_text,
                "sources": [],
                "provider": self.provider,
                "model": getattr(self._llm, "model", self.provider)
            }
        except Exception as exc:
            logger.error(f"Error invoking LLM in general chat mode: {exc}", exc_info=True)
            google_key = os.getenv("GOOGLE_API_KEY", "").strip()
            if self.provider == "ollama" and google_key:
                logger.info("Local Ollama failed. Auto-falling back to Google Gemini Cloud LLM in chat mode...")
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
                    fallback_llm = ChatGoogleGenerativeAI(
                        model=gemini_model,
                        google_api_key=google_key,
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens
                    )
                    resp = fallback_llm.invoke(messages)
                    return {
                        "answer": _extract_text(resp.content),
                        "sources": [],
                        "provider": "google",
                        "model": gemini_model
                    }
                except Exception as g_err:
                    logger.warning(f"Fallback to Gemini also failed: {g_err}")

            err_str = str(exc)
            if "10061" in err_str or "connection" in err_str.lower() or "refused" in err_str.lower():
                user_msg = (
                    "⚠️ **Ollama is not currently running on your machine.**\n\n"
                    "💡 **Fix**: In the sidebar dropdown, select **`✨ Cloud: Gemini Flash (Google)`** to chat immediately without needing local Ollama or high RAM."
                )
            else:
                user_msg = f"⚠️ Error generating response: {exc}. Please select '✨ Cloud: Gemini Flash (Google)' in the sidebar."
            return {
                "answer": user_msg,
                "sources": [],
                "provider": self.provider,
                "model": getattr(self._llm, "model", self.provider),
                "error": str(exc)
            }

    def generate_answer(
        self,
        query: str,
        context_documents: List[Document],
        conversation_history: str = ""
    ) -> Dict[str, Any]:
        """
        Synthesizes a strictly grounded answer from retrieved context passages.
        
        Args:
            query: Current user query.
            context_documents: Top passages from hybrid search and reranking.
            conversation_history: Formatted transcript of prior turns.
            
        Returns:
            Dict containing answer, source metadata list, provider, and model.
        """
        # Handle friendly greetings gracefully
        GREETINGS = {"hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "who are you", "what can you do", "help"}
        clean_q = query.strip().lower().rstrip("!?. ")
        if clean_q in GREETINGS:
            return {
                "answer": (
                    "👋 **Hello! Welcome to KASHMIR GOVERNMENT POLYTECHNIC COLLEGE, SRINAGAR.**\n\n"
                    "I am **KGP Gyankosh**, your AI Administrative Knowledge Assistant.\n\n"
                    "How can I help you today?\n"
                    "- 🏛️ **College Records Mode**: Ask about notices, admission circulars, leave rules, committee orders, or fee schedules.\n"
                    "- 🌐 **General AI Mode**: Switch to General AI Assistant in the sidebar to draft letters, summarize text, or ask any general question!"
                ),
                "sources": [],
                "provider": self.provider,
                "model": getattr(self._llm, "model", self.provider)
            }

        # If no documents were retrieved at all, avoid calling LLM and mitigate hallucination immediately
        if not context_documents:
            return {
                "answer": (
                    "I could not find this information in the available official college documents.\n\n"
                    "_Tip: You can switch to **General AI Assistant** mode in the sidebar to ask general questions or draft documents directly with the LLM!_"
                ),
                "sources": [],
                "provider": self.provider,
                "model": getattr(self._llm, "model", self.provider)
            }

        # Build context passage string and collect source citations
        context_blocks = []
        structured_sources = []
        seen_citations = set()

        for idx, doc in enumerate(context_documents, start=1):
            src_name = doc.metadata.get("source", "Unknown Document")
            page_num = doc.metadata.get("page", 1)
            chunk_id = doc.metadata.get("chunk_id", f"{src_name}#p{page_num}")
            rerank_score = doc.metadata.get("rerank_score")

            citation_key = f"{src_name} (Page {page_num})"
            if citation_key not in seen_citations:
                seen_citations.add(citation_key)
                structured_sources.append({
                    "source": src_name,
                    "page": page_num,
                    "file_path": doc.metadata.get("file_path", ""),
                    "chunk_id": chunk_id,
                    "rerank_score": round(rerank_score, 4) if rerank_score is not None else None,
                    "excerpt": doc.page_content[:200].replace("\n", " ") + "..."
                })

            context_blocks.append(
                f"--- Passage {idx} [Document: {src_name} | Page: {page_num}] ---\n{doc.page_content.strip()}"
            )

        context_text = "\n\n".join(context_blocks)

        user_content_parts = []
        if conversation_history:
            user_content_parts.append(f"Recent Conversation History:\n{conversation_history}\n")

        user_content_parts.append(f"Official Administrative Context:\n{context_text}\n")
        user_content_parts.append(f"User Question: {query}\n\nAnswer:")
        user_prompt = "\n".join(user_content_parts)

        messages = [
            SystemMessage(content=ADMIN_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]

        # Invoke LLM
        try:
            if self._llm is None:
                raise RuntimeError("LLM backend is not initialized.")
            
            response = self._llm.invoke(messages)
            answer_text = _extract_text(response.content)

            return {
                "answer": answer_text,
                "sources": structured_sources,
                "provider": self.provider,
                "model": getattr(self._llm, "model", self.provider)
            }

        except Exception as exc:
            logger.error(f"Error invoking LLM ({self.provider}): {exc}", exc_info=True)
            
            # If Ollama failed (e.g. out of memory / unreachable) and Google API key is available, automatically fall back to Gemini
            google_key = os.getenv("GOOGLE_API_KEY", "").strip()
            if self.provider == "ollama" and google_key:
                logger.info("Local Ollama failed (out of memory/unreachable). Auto-falling back to Google Gemini Cloud LLM...")
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
                    fallback_llm = ChatGoogleGenerativeAI(
                        model=gemini_model,
                        google_api_key=google_key,
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens
                    )
                    resp = fallback_llm.invoke(messages)
                    answer_text = _extract_text(resp.content)
                    return {
                        "answer": answer_text,
                        "sources": structured_sources,
                        "provider": "google",
                        "model": gemini_model
                    }
                except Exception as g_err:
                    logger.warning(f"Fallback to Gemini also failed: {g_err}")

            # Helpful administrative fallback when Ollama or API is unreachable
            fallback_msg = (
                f"⚠️ **LLM Service Error**: Could not connect to the '{self.provider}' language model service ({exc}).\n\n"
                f"💡 **Fix**: In the sidebar dropdown, select **`✨ Cloud: Gemini Flash (Google)`** for instant cloud inference with 0 local RAM required.\n\n"
                f"**Retrieved Official Excerpts (Direct Grounding):**\n"
            )
            for s in structured_sources[:2]:
                fallback_msg += f"- **{s['source']} (Page {s['page']})**: {s['excerpt']}\n"

            return {
                "answer": fallback_msg,
                "sources": structured_sources,
                "provider": self.provider,
                "model": self.provider,
                "error": str(exc)
            }
