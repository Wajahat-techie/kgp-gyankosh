"""
==============================================================================
KGP Gyankosh - Stage 2: Interactive Enterprise Knowledge Assistant (app.py)
Kashmir Government Polytechnic College, Srinagar - Administration Department
==============================================================================

Streamlit Query Application:
- Loads the pre-built FAISS vector database and BM25 index (zero query-time indexing).
- Enforces internal role-based access control via bcrypt authentication.
- Executes Advanced RAG: Query Reformulation -> Hybrid Search (RRF) -> Cross-Encoder Reranking -> Grounded LLM Response.
- Employs strict anti-hallucination mitigation and expandable source citations.
- Modern enterprise dark glassmorphic design system.
"""

import os
import sys
import re
import urllib.parse
import logging
import base64
from typing import Optional, List, Dict, Any, Tuple
import streamlit as st
from dotenv import load_dotenv

# Set page configuration first before any Streamlit widgets
st.set_page_config(
    page_title="KGP Gyankosh - Administrative Intelligence Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Seamlessly inject Streamlit Cloud secrets into os.environ for cloud deployment
try:
    for key, value in st.secrets.items():
        if isinstance(value, str):
            os.environ[key] = value
except Exception:
    pass

from src.indexing.embeddings import get_embeddings_model
from src.indexing.vector_store import (
    load_vector_store,
    load_bm25_store,
    load_manifest,
    MANIFEST_FILENAME
)
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.reranker import DocumentReranker
from src.memory.conversation_memory import ConversationMemory
from src.llm.client import LLMClient
from src.auth.authenticator import get_authenticator

logger = logging.getLogger("kgp_gyankosh.app")


def load_campus_bg_base64() -> str:
    bg_path = os.path.join(PROJECT_ROOT, "static", "campus_bg.jpg")
    if os.path.exists(bg_path):
        try:
            with open(bg_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.warning(f"Could not load campus background: {e}")
    return ""

CAMPUS_BG_BASE64 = load_campus_bg_base64()

if CAMPUS_BG_BASE64:
    APP_BG_STYLE = f"""
    background: 
        radial-gradient(ellipse at 15% 0%, rgba(6, 182, 212, 0.28) 0%, transparent 50%),
        radial-gradient(ellipse at 85% 10%, rgba(168, 85, 247, 0.3) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 90%, rgba(236, 72, 153, 0.2) 0%, transparent 55%),
        radial-gradient(ellipse at 80% 80%, rgba(59, 130, 246, 0.22) 0%, transparent 45%),
        linear-gradient(155deg, rgba(4, 7, 18, 0.9) 0%, rgba(8, 14, 32, 0.85) 45%, rgba(16, 12, 38, 0.9) 100%),
        url("data:image/jpeg;base64,{CAMPUS_BG_BASE64}") !important;
    background-size: cover !important;
    background-position: center center !important;
    background-repeat: no-repeat !important;
    background-attachment: fixed !important;
    """
else:
    APP_BG_STYLE = """
    background: 
        radial-gradient(ellipse at 15% 10%, rgba(6, 182, 212, 0.32) 0%, transparent 50%),
        radial-gradient(ellipse at 85% 15%, rgba(168, 85, 247, 0.3) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 85%, rgba(14, 165, 233, 0.25) 0%, transparent 55%),
        radial-gradient(ellipse at 90% 80%, rgba(236, 72, 153, 0.22) 0%, transparent 45%),
        linear-gradient(150deg, #030611 0%, #070e24 35%, #0e1232 70%, #050a1b 100%) !important;
    background-attachment: fixed !important;
    """

# ==============================================================================
# ENTERPRISE MODERN RADIANT GLASSMORPHISM THEME & CSS
# ==============================================================================
def load_custom_css(bg_style: str) -> str:
    """Loads external stylesheet from static/style.css with dynamic background styling."""
    css_path = os.path.join(PROJECT_ROOT, "static", "style.css")
    if os.path.exists(css_path):
        try:
            with open(css_path, "r", encoding="utf-8") as f:
                css_content = f.read()
            return f"<style>\n{css_content.replace('__BG_STYLE__', bg_style)}\n</style>"
        except Exception as e:
            logger.warning(f"Could not load custom CSS from {css_path}: {e}")
    return ""


CUSTOM_CSS = load_custom_css(APP_BG_STYLE)


# ==============================================================================
# DOCUMENT ACCESS & RESILIENT DOWNLOAD HELPERS
# ==============================================================================
def resolve_document_file_path(source_filename: str, manifest: dict = None) -> Optional[str]:
    """
    Resolves the physical document file path across local and cloud environments
    (Windows/Linux paths, case-insensitivity, and alias matching).
    """
    if not source_filename or source_filename == "Unknown Document":
        return None

    clean_name = os.path.basename(source_filename).strip().lower()

    # 1. Direct check in static/extracted_pages if it's already a single-page pdf name
    extracted_dir = os.path.join(PROJECT_ROOT, "static", "extracted_pages")
    extracted_candidate = os.path.join(extracted_dir, os.path.basename(source_filename))
    if os.path.exists(extracted_candidate):
        return extracted_candidate

    data_root = os.path.abspath(os.path.join(PROJECT_ROOT, "data"))
    if not os.path.exists(data_root):
        return None

    # 2. Search data/ recursively
    for root, _, files in os.walk(data_root):
        for f in files:
            f_lower = f.lower()
            if f_lower == clean_name:
                return os.path.join(root, f)
            # Handle Orders.pdf <-> Orders_11zon.pdf <-> Notices_and_orders.pdf alias
            if clean_name in ("orders.pdf", "orders_11zon.pdf", "notices_and_orders.pdf", "orders") and \
               f_lower in ("orders.pdf", "orders_11zon.pdf", "notices_and_orders.pdf"):
                return os.path.join(root, f)
            # Match basename without extension
            if os.path.splitext(f_lower)[0] == os.path.splitext(clean_name)[0]:
                return os.path.join(root, f)

    return None


def get_document_url_and_path(source_filename: str, manifest: dict = None):
    """
    Resolves the local file path and web-accessible static serving URL for an indexed document.
    Returns (url, file_path).
    """
    if not source_filename or source_filename == "Unknown Document":
        return None, None

    file_path = resolve_document_file_path(source_filename, manifest)
    if not file_path or not os.path.exists(file_path):
        return None, None

    try:
        data_root = os.path.abspath(os.path.join(PROJECT_ROOT, "data"))
        static_root = os.path.abspath(os.path.join(PROJECT_ROOT, "static"))
        
        if file_path.startswith(static_root):
            rel_path = os.path.relpath(file_path, static_root)
            clean_rel = rel_path.replace(os.sep, "/")
            url_path = "/app/static/" + "/".join(urllib.parse.quote(p) for p in clean_rel.split("/"))
            return url_path, file_path
        elif file_path.startswith(data_root):
            rel_path = os.path.relpath(file_path, data_root)
            clean_rel = rel_path.replace(os.sep, "/")
            url_path = "/app/static/docs/" + "/".join(urllib.parse.quote(p) for p in clean_rel.split("/"))
            return url_path, file_path
        return None, file_path
    except Exception as e:
        logger.warning(f"Error computing static URL for {file_path}: {e}")
        return None, file_path


def get_document_download_info(source_filename: str, page_num: int = None, manifest: dict = None):
    """
    Resolves the download URL, extracted path, and filename for an indexed document.
    For multi-page PDFs with a specified page_num, extracts ONLY that particular order page.
    Returns (download_url, download_filename, is_single_page, actual_file_path).
    """
    if not source_filename or source_filename == "Unknown Document":
        return None, None, False, None

    extracted_dir = os.path.join(PROJECT_ROOT, "static", "extracted_pages")
    base_name = os.path.splitext(os.path.basename(source_filename))[0]

    # 1. First check if single extracted page already exists in static/extracted_pages/
    cand_base_names = [base_name]
    if base_name.lower() in ("orders", "notices_and_orders", "orders_11zon"):
        cand_base_names = ["Orders", "orders", "Notices_and_orders", base_name]

    if page_num and page_num > 0:
        for c_base in cand_base_names:
            page_filename = f"{c_base}_page_{page_num}.pdf"
            page_filepath = os.path.join(extracted_dir, page_filename)
            if os.path.exists(page_filepath):
                page_url = f"/app/static/extracted_pages/{urllib.parse.quote(page_filename)}"
                return page_url, page_filename, True, page_filepath

    # 2. Check in static/docs/
    for sub in ["word", "pdf", ""]:
        cand_static = os.path.join(PROJECT_ROOT, "static", "docs", sub, os.path.basename(source_filename))
        if os.path.exists(cand_static):
            rel = os.path.relpath(cand_static, os.path.join(PROJECT_ROOT, "static")).replace(os.sep, "/")
            return f"/app/static/{rel}", os.path.basename(cand_static), False, cand_static

    # 3. Otherwise resolve parent file in data/
    file_path = resolve_document_file_path(source_filename, manifest)
    if not file_path or not os.path.exists(file_path):
        return None, os.path.basename(source_filename), False, None

    # Check if document is a PDF and a specific page is cited
    is_pdf = source_filename.lower().endswith(".pdf") or file_path.lower().endswith(".pdf")
    if is_pdf and page_num and page_num > 0:
        os.makedirs(extracted_dir, exist_ok=True)
        page_filename = f"{base_name}_page_{page_num}.pdf"
        page_filepath = os.path.join(extracted_dir, page_filename)

        # Generate single page on-demand if not already cached
        if not os.path.exists(page_filepath):
            try:
                from pypdf import PdfReader, PdfWriter
                reader = PdfReader(file_path)
                if 1 <= page_num <= len(reader.pages):
                    writer = PdfWriter()
                    writer.add_page(reader.pages[page_num - 1])
                    with open(page_filepath, "wb") as f_out:
                        writer.write(f_out)
            except Exception as err:
                logger.warning(f"Error extracting page {page_num} from {file_path}: {err}")

        if os.path.exists(page_filepath):
            page_url = f"/app/static/extracted_pages/{urllib.parse.quote(page_filename)}"
            return page_url, page_filename, True, page_filepath

    doc_url, _ = get_document_url_and_path(source_filename, manifest)
    return doc_url, os.path.basename(file_path), False, file_path


def linkify_answer_citations(answer_text: str, manifest: dict = None) -> str:
    """
    Cleans and transforms document citations in the synthesized LLM answer text into
    responsive, interactive download badges with direct links.
    """
    if not answer_text:
        return answer_text

    def _make_badge(doc_name: str, page_num_str: str = None):
        doc_name = doc_name.strip()
        page_num = int(page_num_str) if page_num_str and page_num_str.isdigit() else (1 if doc_name.lower().endswith(".pdf") else None)
        page_suffix = f" (Page {page_num})" if (page_num and doc_name.lower().endswith(".pdf")) else ""

        dl_url, dl_filename, is_page, actual_path = get_document_download_info(doc_name, page_num, manifest)
        icon = "📕" if doc_name.lower().endswith(".pdf") else "📘"

        if dl_url or (actual_path and os.path.exists(actual_path)):
            effective_url = dl_url or f"/app/static/docs/{urllib.parse.quote(dl_filename)}"
            badge_title = f"Click to download Page {page_num} of {doc_name}" if is_page else f"Click to download {doc_name}"
            return (
                f'<a href="{effective_url}" download="{dl_filename}" target="_blank" '
                f'style="display: inline-flex; align-items: center; gap: 4px; color: #38bdf8; font-weight: 700; background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.4); padding: 3px 10px; border-radius: 8px; margin: 2px 2px; text-decoration: none; cursor: pointer; transition: all 0.2s ease; vertical-align: middle;" '
                f'title="{badge_title}">'
                f'{icon} {doc_name}{page_suffix} <span style="font-size: 0.75rem; color: #38bdf8;">📥</span></a>'
            )
        return f'<span style="display: inline-flex; align-items: center; gap: 4px; color: #38bdf8; font-weight: 700; background: rgba(56, 189, 248, 0.12); border: 1px solid rgba(56, 189, 248, 0.25); padding: 3px 10px; border-radius: 8px; margin: 2px 2px; vertical-align: middle;">{icon} {doc_name}{page_suffix}</span>'

    # Pattern 1: [doc.pdf, Page 14], [doc.pdf (Page 14)], [doc.docx]
    def repl_p1(m):
        doc = m.group(1)
        p = m.group(2) or m.group(3) or m.group(4) or m.group(5)
        return _make_badge(doc, p)

    p1 = r'\[\s*([a-zA-Z0-9_\-\s\(\)]+?\.(?:docx|pdf|txt))\s*(?:,\s*(?:Page|p\.?)\s*(\d+)|\s*\((?:Page|p\.?)\s*(\d+)\)|\s*-\s*(?:Page|p\.?)\s*(\d+)|\s*(?:Page|p\.?)\s*(\d+))?\s*\]'
    answer_text = re.sub(p1, repl_p1, answer_text, flags=re.IGNORECASE)

    # Pattern 2: (doc.pdf, Page 14), (doc.pdf - Page 14)
    def repl_p2(m):
        doc = m.group(1)
        p = m.group(2) or m.group(3) or m.group(4)
        return _make_badge(doc, p)

    p2 = r'\(\s*([a-zA-Z0-9_\-\s]+?\.(?:docx|pdf|txt))\s*(?:,\s*(?:Page|p\.?)\s*(\d+)|\s*-\s*(?:Page|p\.?)\s*(\d+)|\s*(?:Page|p\.?)\s*(\d+))?\s*\)'
    answer_text = re.sub(p2, repl_p2, answer_text, flags=re.IGNORECASE)

    # Pattern 3: Raw filename with Page (e.g. Orders.pdf (Page 16), Orders.pdf, Page 16, Orders.pdf - Page 16)
    def repl_p3(m):
        doc = m.group(1)
        p = m.group(2) or m.group(3) or m.group(4) or m.group(5) or m.group(6)
        return _make_badge(doc, p)

    p3 = r'(?<!href=")(?<!/)\b([a-zA-Z0-9_\-]+\.(?:docx|pdf))\s*(?:\((?:Page|p\.?)\s*(\d+)\)|,\s*(?:Page|p\.?)\s*(\d+)|\s*-\s*(?:Page|p\.?)\s*(\d+)|\s*:\s*(?:Page|p\.?)\s*(\d+)|\s*(?:Page|p\.?)\s*(\d+))'
    answer_text = re.sub(p3, repl_p3, answer_text, flags=re.IGNORECASE)

    # Style college mentions in bold with beautiful typography
    college_regex = r'(?:\*{2})?Kashmir\s+(?:Govt\.?|Government)\s+Polytechnic\s+College(?:,?\s*Srinagar)?(?:\*{2})?'
    answer_text = re.sub(
        college_regex,
        '<strong class="kgp-college-badge">KASHMIR GOVERNMENT POLYTECHNIC COLLEGE, SRINAGAR</strong>',
        answer_text,
        flags=re.IGNORECASE
    )
    return answer_text


def render_citations_and_links(sources: list, manifest: dict, key_prefix: str = "src"):
    """
    Renders structured citation action cards with prominent, responsive download buttons
    and direct single-click extracted PDF page downloads.
    """
    if not sources:
        return

    with st.expander(f"📚 Official Source Citations & Download Verified Files ({len(sources)} documents)", expanded=True):
        st.markdown("<div style='margin-bottom: 8px; font-size: 0.82rem; color: #94a3b8;'>Verified official college documents retrieved for this answer:</div>", unsafe_allow_html=True)
        
        for idx, src in enumerate(sources):
            src_name = src.get("source", "Unknown Document")
            page_num = src.get("page", 1)
            rerank_score = src.get("rerank_score")
            score_text = f" • Match Score: `{rerank_score:.3f}`" if isinstance(rerank_score, (int, float)) else ""
            file_icon = "📕 PDF Order" if src_name.lower().endswith(".pdf") else "📘 Official Circular (DOCX)"

            dl_url, dl_filename, is_page, actual_path = get_document_download_info(src_name, page_num, manifest)

            # Container card for each source
            st.markdown(
                f"""
                <div class="download-card">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; flex-wrap: wrap; gap: 6px;">
                        <div style="font-size: 0.98rem; font-weight: 700; color: #f8fafc;">
                            {file_icon}: <span style="color: #38bdf8;">{src_name}</span>
                        </div>
                        <div style="font-size: 0.78rem; font-weight: 600; color: #34d399; background: rgba(16, 185, 129, 0.15); padding: 3px 10px; border-radius: 9999px; border: 1px solid rgba(16, 185, 129, 0.3);">
                            📄 Page {page_num}{score_text}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            col_excerpt, col_action = st.columns([2.8, 1.2])
            with col_excerpt:
                if src.get("excerpt"):
                    st.caption(f"🔎 **Excerpt:** _{src.get('excerpt')[:280]}..._")

            with col_action:
                btn_rendered = False
                if actual_path and os.path.exists(actual_path):
                    try:
                        with open(actual_path, "rb") as f_doc:
                            file_bytes = f_doc.read()
                        mime_type = "application/pdf" if dl_filename.endswith(".pdf") else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        btn_label = f"📥 Download Page {page_num}" if is_page else "📥 Download File"
                        st.download_button(
                            label=btn_label,
                            data=file_bytes,
                            file_name=dl_filename,
                            mime=mime_type,
                            key=f"dl_{key_prefix}_{idx}_{page_num}_{hash(src_name) % 10000}",
                            use_container_width=True
                        )
                        btn_rendered = True
                    except Exception as ex:
                        logger.warning(f"Could not prepare download for {actual_path}: {ex}")

                if not btn_rendered and dl_url:
                    st.markdown(
                        f'<a href="{dl_url}" download="{dl_filename}" target="_blank" '
                        f'style="display: block; text-align: center; background: linear-gradient(135deg, #0284c7 0%, #06b6d4 100%); color: #ffffff; padding: 8px 14px; border-radius: 12px; font-weight: 700; font-size: 0.84rem; text-decoration: none; box-shadow: 0 4px 15px rgba(6, 182, 212, 0.35);">'
                        f'📥 Download File</a>',
                        unsafe_allow_html=True
                    )
                elif not btn_rendered:
                    st.markdown(
                        '<div style="text-align: center; color: #94a3b8; font-size: 0.76rem; background: rgba(255,255,255,0.04); border: 1px dashed rgba(255,255,255,0.15); border-radius: 8px; padding: 6px 8px;">'
                        '🛡️ ISO Master Record</div>',
                        unsafe_allow_html=True
                    )

            if idx < len(sources) - 1:
                st.markdown("<hr style='margin: 10px 0; border: none; border-top: 1px solid rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

        st.caption("💡 _Click any download button to download that exact verified order page or circular directly to your device._")


# ==============================================================================
# CACHED RESOURCE INITIALIZATION (Stage 2: Never re-index or re-embed documents)
# ==============================================================================
@st.cache_resource(show_spinner="⚡ Initializing AI Knowledge Base & Retrieval Models...")
def initialize_system():
    """
    Loads pre-indexed vector store, BM25 index, reranker, and LLM backend once.
    This guarantees zero query-time indexing overhead.
    """
    vector_store_dir = os.getenv("VECTOR_STORE_DIR", "output/vector_store")
    if not os.path.isabs(vector_store_dir):
        vector_store_dir = os.path.normpath(os.path.join(PROJECT_ROOT, vector_store_dir))

    bm25_store_dir = os.getenv("BM25_STORE_DIR", "output/bm25_store")
    if not os.path.isabs(bm25_store_dir):
        bm25_store_dir = os.path.normpath(os.path.join(PROJECT_ROOT, bm25_store_dir))

    manifest_path = os.path.join(PROJECT_ROOT, "output", MANIFEST_FILENAME)

    embeddings = get_embeddings_model()
    vector_store = load_vector_store(vector_store_dir, embeddings)
    bm25_model, bm25_corpus = load_bm25_store(bm25_store_dir)
    manifest = load_manifest(manifest_path)

    reranker = DocumentReranker()
    try:
        reranker.preload()
    except Exception as re_err:
        logger.warning(f"Reranker preload deferred: {re_err}")
    default_provider = os.getenv("LLM_PROVIDER", "google").lower().strip()
    default_model = (
        os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite") if default_provider == "google"
        else (os.getenv("OLLAMA_MODEL", "llama3.1:latest") if default_provider == "ollama"
        else os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    )
    llm_client = get_cached_llm_client(default_provider, default_model)

    retriever = None
    if vector_store and bm25_model and bm25_corpus:
        retriever = HybridRetriever(
            vector_store=vector_store,
            bm25_model=bm25_model,
            bm25_corpus=bm25_corpus
        )

    return {
        "retriever": retriever,
        "reranker": reranker,
        "llm_client": llm_client,
        "manifest": manifest,
        "index_ready": (retriever is not None)
    }


_LLM_CLIENTS: Dict[str, LLMClient] = {}

def get_cached_llm_client(provider: str, model_name: str) -> LLMClient:
    """Returns a memoized active LLMClient instance for the requested provider and model."""
    cache_key = f"{provider}:{model_name}"
    if cache_key in _LLM_CLIENTS:
        return _LLM_CLIENTS[cache_key]

    os.environ["LLM_PROVIDER"] = provider
    if provider == "ollama":
        os.environ["OLLAMA_MODEL"] = model_name
    elif provider == "google":
        os.environ["GEMINI_MODEL"] = model_name
    elif provider == "openai":
        os.environ["OPENAI_MODEL"] = model_name

    client = LLMClient(provider=provider, model=model_name)
    _LLM_CLIENTS[cache_key] = client
    return client


# ==============================================================================
# SESSION STATE INITIALIZATION & MULTI-CHAT SESSION MANAGER
# ==============================================================================
import datetime

def get_active_session():
    """Ensures session state contains valid multi-turn chat sessions and returns active session data."""
    if "chat_sessions" not in st.session_state or not st.session_state.chat_sessions:
        s_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        st.session_state.chat_sessions = {
            s_id: {
                "id": s_id,
                "title": "New Chat",
                "created_at": datetime.datetime.now().strftime("%b %d, %H:%M"),
                "messages": [],
                "memory": ConversationMemory(max_history_turns=6)
            }
        }
        st.session_state.active_session_id = s_id

    if st.session_state.get("active_session_id") not in st.session_state.chat_sessions:
        st.session_state.active_session_id = list(st.session_state.chat_sessions.keys())[-1]

    return st.session_state.chat_sessions[st.session_state.active_session_id]


def render_landing_hero(total_files: int = 1162, total_chunks: int = 7801):
    """Renders the top institutional hero banner for the landing/login view."""
    st.markdown(
        f"""
        <div class="hero-card">
            <div style="font-size: 0.8rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #38bdf8; margin-bottom: 0.4rem;">
                INTERNAL ADMINISTRATIVE INTELLIGENCE PLATFORM
            </div>
            <h1 class="hero-main-title">KGP Gyankosh</h1>
            <div class="college-institution-title">
                KASHMIR GOVERNMENT POLYTECHNIC COLLEGE, SRINAGAR
            </div>
            <p class="hero-subtext">
                Internal Administrative Intelligence System for Official College Documents.
            </p>
            <div class="stats-strip">
                <div class="stat-box">
                    <span class="stat-num">{total_files:,}</span>
                    <span class="stat-label">Indexed Documents</span>
                </div>
                <div class="stat-box">
                    <span class="stat-num">{total_chunks:,}</span>
                    <span class="stat-label">Searchable Clauses</span>
                </div>
            </div>
            <div style="margin-top: 1.4rem;">
                <div class="landing-developer-badge">
                    <span>✨ Designed and Developed by <strong>Wajahat</strong></span>
                    <span style="opacity: 0.6;">•</span>
                    <span style="color: #38bdf8;">📞 9906457756</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )




def main():
    # Inject Custom CSS Theme
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Pre-warm AI models & indices in background while user views login screen
    if "prewarm_started" not in st.session_state:
        st.session_state.prewarm_started = True
        import threading
        threading.Thread(target=initialize_system, daemon=True).start()

    # --------------------------------------------------------------------------
    # 1. AUTHENTICATION LAYER
    # --------------------------------------------------------------------------
    try:
        authenticator = get_authenticator()
    except Exception as auth_err:
        st.error(f"Authentication system error: {auth_err}")
        return

    auth_status = st.session_state.get("authentication_status")

    if not auth_status:
        # Render Stunning Landing Page Hero Banner
        render_landing_hero()

        col_left, col_center, col_right = st.columns([1, 1.8, 1])
        with col_center:
            authenticator.login(location="main")
            auth_status = st.session_state.get("authentication_status")
            if auth_status is False:
                st.error("❌ Invalid credentials. Please check your username and password.")
            elif auth_status is None:
                st.caption("🔒 Please authenticate with your authorized college administrative credentials.")

        st.markdown(
            """
            <div class="app-bottom-credit-bar" style="margin-top: 2.5rem;">
                🏛️ <strong>KGP Gyankosh</strong> • Administrative Intelligence Assistant • Designed and Developed by <strong>Wajahat</strong> • 📞 <strong>9906457756</strong>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    # --------------------------------------------------------------------------
    # 2. AUTHENTICATED USER SESSION & FRESH WINDOW LOGIC
    # --------------------------------------------------------------------------
    username = st.session_state.get("username", "Admin")
    name = st.session_state.get("name", "Administrative Staff")

    # On brand new login, ensure a fresh conversation window is initialized
    if st.session_state.get("last_authenticated_user") != username:
        st.session_state.last_authenticated_user = username
        s_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        if "chat_sessions" not in st.session_state:
            st.session_state.chat_sessions = {}
        st.session_state.chat_sessions[s_id] = {
            "id": s_id,
            "title": "New Chat",
            "created_at": datetime.datetime.now().strftime("%b %d, %H:%M"),
            "messages": [],
            "memory": ConversationMemory(max_history_turns=6)
        }
        st.session_state.active_session_id = s_id

    active_session = get_active_session()

    system_resources = initialize_system()
    retriever = system_resources["retriever"]
    reranker = system_resources["reranker"]
    llm_client = system_resources["llm_client"]
    manifest = system_resources["manifest"]
    index_ready = system_resources["index_ready"]

    # --------------------------------------------------------------------------
    # 3. SIDEBAR WITH RECENT CHATS & CONTROLS
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                <img src="https://img.icons8.com/color/96/museum.png" width="46" style="filter: drop-shadow(0 0 10px rgba(99,102,241,0.4));" />
                <div>
                    <h2 style="font-size: 1.3rem; margin: 0; font-weight: 800; background: linear-gradient(135deg, #ffffff 0%, #38bdf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">KGP Gyankosh</h2>
                    <div style="font-family: 'Outfit', sans-serif; font-size: 0.72rem; color: #38bdf8; font-weight: 800; letter-spacing: 0.04em; line-height: 1.25; text-transform: uppercase; margin-top: 2px;">
                        KASHMIR GOVERNMENT POLYTECHNIC COLLEGE, SRINAGAR
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.divider()

        # User profile & logout
        st.markdown(f"**Logged in as:**\n👤 **{name}** (`{username}`)")
        authenticator.logout(location="sidebar")
        st.divider()

        # New Chat Button
        if st.button("➕ Start New Chat", use_container_width=True, type="primary"):
            new_s_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
            st.session_state.chat_sessions[new_s_id] = {
                "id": new_s_id,
                "title": "New Chat",
                "created_at": datetime.datetime.now().strftime("%b %d, %H:%M"),
                "messages": [],
                "memory": ConversationMemory(max_history_turns=6)
            }
            st.session_state.active_session_id = new_s_id
            st.rerun()

        # Recent Chats Session History Manager
        sessions_list = list(st.session_state.chat_sessions.items())
        with st.expander(f"🕒 Recent Chats ({len(sessions_list)})", expanded=True):
            for s_key, s_data in reversed(sessions_list):
                is_active = (s_key == st.session_state.active_session_id)
                prefix_icon = "🟢 " if is_active else "💬 "
                display_title = s_data.get("title", "New Chat")
                if len(display_title) > 22:
                    display_title = display_title[:20] + "..."
                msg_count = len(s_data.get("messages", []))

                col_s1, col_s2 = st.columns([4, 1])
                with col_s1:
                    if st.button(
                        f"{prefix_icon}{display_title}",
                        key=f"sess_btn_{s_key}",
                        use_container_width=True,
                        disabled=is_active,
                        help=f"{s_data.get('title')} ({msg_count} messages) - Created {s_data.get('created_at')}"
                    ):
                        st.session_state.active_session_id = s_key
                        st.rerun()
                with col_s2:
                    if len(st.session_state.chat_sessions) > 1:
                        if st.button("🗑️", key=f"del_btn_{s_key}", help="Delete this chat"):
                            del st.session_state.chat_sessions[s_key]
                            if st.session_state.active_session_id == s_key:
                                st.session_state.active_session_id = list(st.session_state.chat_sessions.keys())[-1]
                            st.rerun()

        st.divider()

        # 1. Operational Mode Selector
        st.markdown("<div style='font-size: 0.8rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;'>🎯 Select Operational Mode</div>", unsafe_allow_html=True)
        assistant_mode = st.radio(
            "Select Operational Mode",
            [
                "🏛️ College Records (RAG)",
                "🌐 General AI (Direct LLM)"
            ],
            index=0,
            label_visibility="collapsed",
            key="assistant_operational_mode"
        )
        st.divider()

        # Model options map (Gemini Flash default with Cloud speed)
        provider_models = {
            "✨ Cloud: Gemini Flash (Google)": ("google", os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")),
            "🦙 Local: Llama 3.1 (Ollama)": ("ollama", os.getenv("OLLAMA_MODEL", "llama3.1:latest")),
            "⚡ Cloud: GPT-4o-mini (OpenAI)": ("openai", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        }

        current_env = os.getenv("LLM_PROVIDER", "google").lower().strip()
        default_index = 0
        if current_env == "ollama":
            default_index = 1
        elif current_env == "openai":
            default_index = 2

        # 2. Unified Stable Model Selector
        is_rag_mode = ("College Records" in assistant_mode)
        title_color = "#38bdf8" if is_rag_mode else "#c084fc"
        title_text = "📚 Select LLM for College RAG:" if is_rag_mode else "🤖 Select LLM for General Chat:"
        caption_text = (
            "🔍 **RAG Mode**: Answers are grounded strictly in official college notices with citations and 1-click downloads."
            if is_rag_mode
            else "💬 **General AI Mode**: Direct conversation for drafting notices, coding, and general Q&A."
        )

        st.markdown(
            f"<div style='font-size: 0.82rem; font-weight: 700; color: {title_color}; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 4px;'>{title_text}</div>",
            unsafe_allow_html=True
        )
        selected_model_label = st.selectbox(
            "Active LLM Model",
            options=list(provider_models.keys()),
            index=default_index,
            label_visibility="collapsed",
            key="active_llm_selector",
            help="The model used to synthesize answers strictly from retrieved college notices and orders."
        )
        st.caption(caption_text)

        sel_provider, sel_model = provider_models[selected_model_label]
        llm_client = get_cached_llm_client(sel_provider, sel_model)
        st.divider()
        st.subheader("Indexed Knowledge Base")
        indexed_files = manifest.get("indexed_files", {})
        total_chunks = manifest.get("total_chunks", 0)

        if index_ready:
            st.success(f"🟢 Index Active: {len(indexed_files):,} files ({total_chunks:,} chunks)")
            with st.expander(f"📄 View Indexed Documents ({len(indexed_files)})"):
                for fname, fmeta in list(indexed_files.items())[:100]:
                    st.markdown(f"- **{fname}** ({fmeta.get('chunks_count', 0)} chunks)")
                if len(indexed_files) > 100:
                    st.caption(f"_...and {len(indexed_files) - 100} more files indexed on disk._")
        else:
            st.error("🔴 No index detected on disk!")
            st.caption("Run `python build_index.py` to ingest and index notices.")

        st.divider()

        # Clear Current Conversation Button
        if st.button("🗑️ Clear Current Chat", use_container_width=True):
            active_session["messages"] = []
            active_session["memory"].clear()
            st.rerun()

        # Technical Documentation & Reports Section
        reports_dir = os.path.join(PROJECT_ROOT, "docs", "reports")
        report_specs = [
            ("🏛️ System Architecture PDF", "KGP_Gyankosh_System_Architecture_and_Files_Reference.pdf"),
            ("📖 Line-by-Line Code Doc PDF", "KGP_Gyankosh_Line_By_Line_Code_Documentation.pdf"),
            ("📘 Complete Documentation PDF", "KGP_Gyankosh_Complete_Documentation.pdf"),
        ]
        available_reports = [
            (label, fname, os.path.join(reports_dir, fname))
            for label, fname in report_specs
            if os.path.exists(os.path.join(reports_dir, fname))
        ]
        if available_reports:
            with st.expander("📑 System Documentation Reports", expanded=False):
                st.markdown("<div style='font-size: 0.8rem; color: #94a3b8; margin-bottom: 8px;'>Download publication-ready technical reports:</div>", unsafe_allow_html=True)
                for r_idx, (r_label, r_fname, r_path) in enumerate(available_reports):
                    try:
                        with open(r_path, "rb") as rf:
                            pdf_bytes = rf.read()
                        st.download_button(
                            label=f"📥 {r_label}",
                            data=pdf_bytes,
                            file_name=r_fname,
                            mime="application/pdf",
                            key=f"sidebar_report_dl_{r_idx}",
                            use_container_width=True
                        )
                    except Exception as r_err:
                        logger.warning(f"Could not load report {r_fname}: {r_err}")

        # Developer Attribution Card in Sidebar
        st.divider()
        st.markdown(
            """
            <div class="developer-credit-card">
                <div class="developer-credit-title">SYSTEM ARCHITECT & DEVELOPER</div>
                <div class="developer-credit-name">Designed and Developed by Wajahat</div>
                <div class="developer-credit-phone">📞 9906457756</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------------------------
    # 4. MAIN CHAT INTERFACE
    # --------------------------------------------------------------------------
    # Top Bar Header
    st.markdown(
        f"""
        <div class="gov-badge-container" style="justify-content: flex-start; margin-bottom: 0.8rem;">
            <span class="gov-pill">🏛️ KGP Gyankosh</span>
            <span style="margin-left: auto; font-size: 0.78rem; color: #34d399; font-weight: 600; background: rgba(16, 185, 129, 0.12); padding: 4px 14px; border-radius: 9999px; border: 1px solid rgba(16, 185, 129, 0.3);">
                🟢 {len(indexed_files):,} Official Documents • {total_chunks:,} Clauses Live
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    is_general_ai = ("General AI" in assistant_mode)
    active_mode_badge = (
        '<span style="background: rgba(168, 85, 247, 0.16); color: #d8b4fe; font-size: 0.78rem; font-weight: 700; padding: 4px 14px; border-radius: 9999px; border: 1.5px solid rgba(168, 85, 247, 0.4);">🌐 Mode: General AI Chat</span>'
        if is_general_ai
        else '<span style="background: rgba(56, 189, 248, 0.16); color: #7dd3fc; font-size: 0.78rem; font-weight: 700; padding: 4px 14px; border-radius: 9999px; border: 1.5px solid rgba(56, 189, 248, 0.4);">🏛️ Mode: College Records (RAG)</span>'
    )
    active_model_badge = f'<span style="background: rgba(52, 211, 153, 0.16); color: #6ee7b7; font-size: 0.78rem; font-weight: 700; padding: 4px 14px; border-radius: 9999px; border: 1.5px solid rgba(52, 211, 153, 0.4);">🤖 Active Model: {selected_model_label}</span>'

    answer_stamp = (
        '<div class="official-answer-stamp"><span>🛡️ OFFICIAL VERIFIED RECORD</span><span>🏛️ KGP ARCHIVES GROUNDED</span></div>'
        if not is_general_ai
        else '<div class="official-answer-stamp" style="border-left-color: #c084fc; color: #d8b4fe;"><span>🌐 GENERAL AI RESPONSE</span><span>🤖 DIRECT LLM SYNTHESIS</span></div>'
    )

    # User input with mode-specific placeholder (docked at bottom by Streamlit)
    input_placeholder = (
        f"💬 Type your message here (e.g. 'Draft a circular for faculty meeting')... [Press Enter]"
        if is_general_ai
        else "🔍 Ask about college circulars, leave rules, notices, or admissions... [Press Enter]"
    )
    prompt = st.chat_input(input_placeholder)

    has_active_conversation = bool(active_session["messages"] or prompt)

    if has_active_conversation:
        # Compact top header during active conversation so messages remain visible
        st.markdown(
            f"""
            <div class="official-gov-banner" style="padding: 0.75rem 1.3rem; margin-bottom: 0.8rem;">
                <div class="official-header-row">
                    <div class="official-crest-group" style="gap: 10px;">
                        <span style="font-size: 1.5rem;">🏛️</span>
                        <div>
                            <div style="font-size: 0.68rem; font-weight: 800; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.08em;">GOVERNMENT OF JAMMU & KASHMIR</div>
                            <div style="font-family: 'Outfit', sans-serif; font-size: 1.05rem; font-weight: 800; color: #ffffff; text-transform: uppercase;">KASHMIR GOVERNMENT POLYTECHNIC COLLEGE, SRINAGAR</div>
                        </div>
                    </div>
                    <div class="official-status-pills">
                        {active_mode_badge}
                        {active_model_badge}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Prestigious official government banner when conversation starts
        st.markdown(
            f"""
            <div class="official-gov-banner">
                <div class="official-header-row">
                    <div class="official-crest-group">
                        <div class="official-crest-icon">🏛️</div>
                        <div>
                            <div class="official-dept-title">GOVERNMENT OF JAMMU & KASHMIR • DEPARTMENT OF SKILL DEVELOPMENT</div>
                            <h1 class="official-institution-name">KASHMIR GOVERNMENT POLYTECHNIC COLLEGE, SRINAGAR</h1>
                            <div class="official-sub-caption">Gogji-Bagh, Srinagar (J&K) - 190008 • Established 1958 • Official Institutional Intelligence Portal</div>
                        </div>
                    </div>
                    <div class="official-status-pills">
                        <span class="official-pill verified">🛡️ ISO Records Grounded</span>
                        <span class="official-pill records">📚 {len(indexed_files):,} Official Documents</span>
                        {active_model_badge}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    if not index_ready:
        st.warning(
            "⚠️ **Knowledge Base Index is Empty**: No active FAISS or BM25 index found. "
            "Please run `python build_index.py` from the terminal to ingest sample notices, then refresh this page."
        )
        return

    # Sanitize history: remove any trailing orphaned user message from interrupted/cancelled runs
    while active_session["messages"] and active_session["messages"][-1].get("role") == "user":
        active_session["messages"].pop()

    # Render current session conversation history
    for msg_idx, message in enumerate(active_session["messages"]):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(answer_stamp, unsafe_allow_html=True)
                content_with_links = linkify_answer_citations(message["content"], manifest)
                st.markdown(content_with_links, unsafe_allow_html=True)
                if message.get("sources"):
                    render_citations_and_links(message["sources"], manifest, key_prefix=f"hist_{active_session['id']}_{msg_idx}")
            else:
                st.markdown(message["content"])

    def scroll_to_bottom():
        """Smoothly auto-scrolls down to reveal newly rendered queries, spinner, and answers without hiding behind the bottom bar."""
        js = """
        <script>
            function doScroll() {
                try {
                    const doc = window.parent.document;
                    const scrollContainers = [
                        doc.querySelector('[data-testid="stMain"]'),
                        doc.querySelector('section.main'),
                        doc.querySelector('[data-testid="stAppViewContainer"]'),
                        doc.documentElement,
                        doc.body
                    ];
                    for (const el of scrollContainers) {
                        if (el && el.scrollHeight > el.clientHeight) {
                            el.scrollTo({ top: el.scrollHeight + 2000, behavior: 'smooth' });
                        }
                    }
                    const lastMsg = doc.querySelector('[data-testid="stChatMessage"]:last-of-type');
                    if (lastMsg) {
                        lastMsg.scrollIntoView({ behavior: 'smooth', block: 'end' });
                    }
                } catch (e) {}
            }
            setTimeout(doScroll, 60);
            setTimeout(doScroll, 250);
        </script>
        """
        st.components.v1.html(js, height=0)

    if active_session["messages"]:
        scroll_to_bottom()

    # Empty State Guidance: only shown when no active conversation or input exists
    if not has_active_conversation:
        st.markdown(
            f"""
            <div class="official-welcome-deck">
                <div class="official-welcome-title">🏛️ Institutional Administrative Intelligence Assistant</div>
                <div class="official-welcome-desc">
                    Welcome to the official executive digital assistant of <strong>Kashmir Government Polytechnic College, Srinagar</strong>. 
                    Ask questions grounded strictly across <strong>{len(indexed_files):,}</strong> authenticated circulars, administrative duty orders, leave policies, and exam rosters with verified single-click document downloads.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    if prompt:
        # Prevent any duplicate/orphaned user bubbles
        while active_session["messages"] and active_session["messages"][-1].get("role") == "user":
            active_session["messages"].pop()

        # Update session title if first query
        if active_session.get("title") == "New Chat":
            clean_title = prompt.strip().replace("\n", " ")
            active_session["title"] = (clean_title[:28] + "...") if len(clean_title) > 28 else clean_title

        # Display user query
        active_session["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        scroll_to_bottom()

        # Process query through selected mode (RAG vs Direct LLM)
        with st.chat_message("assistant"):
            try:
                if "General AI" in assistant_mode:
                    with st.spinner("⚡ Formulating AI response..."):
                        history_prompt = active_session["memory"].format_history_for_prompt()
                        result = llm_client.generate_chat(
                            query=prompt,
                            conversation_history=history_prompt
                        )
                        answer_text = result["answer"]
                        sources = []
                else:
                    with st.spinner("⚡ Searching verified college records & synthesizing answer..."):
                        # 1. Multi-turn Query Reformulation
                        reformulated_query = active_session["memory"].reformulate_query(prompt, llm_client)

                        # 2. Hybrid Retrieval (Vector + BM25 Reciprocal Rank Fusion)
                        candidates = retriever.retrieve(reformulated_query)

                        # 3. Cross-Encoder Reranking
                        reranked_docs = reranker.rerank(reformulated_query, candidates)
                        top_documents = [doc for doc, score in reranked_docs]

                        # 4. Synthesize Answer with Strict Anti-Hallucination Prompt
                        history_prompt = active_session["memory"].format_history_for_prompt()
                        result = llm_client.generate_answer(
                            query=prompt,
                            context_documents=top_documents,
                            conversation_history=history_prompt
                        )

                        answer_text = result["answer"]
                        sources = result["sources"]

                # Display final response with official verification stamp & formatted citations
                st.markdown(answer_stamp, unsafe_allow_html=True)
                content_with_links = linkify_answer_citations(answer_text, manifest)
                st.markdown(content_with_links, unsafe_allow_html=True)

                if sources:
                    render_citations_and_links(sources, manifest, key_prefix=f"curr_{active_session['id']}_{len(active_session['messages'])}")

                # Update memory & session state
                active_session["memory"].add_turn(prompt, answer_text, sources)
                active_session["messages"].append({
                    "role": "assistant",
                    "content": answer_text,
                    "sources": sources
                })
                scroll_to_bottom()
            except Exception as err:
                st.error(f"⚠️ Query error: {err}")
                logger.error(f"Error handling query: {err}", exc_info=True)

    # Developer Credit Bar at Interface Bottom
    st.markdown(
        """
        <div class="app-bottom-credit-bar">
            🏛️ <strong>KGP Gyankosh</strong> • Administrative Intelligence Assistant • Designed and Developed by <strong>Wajahat</strong> • 📞 <strong>9906457756</strong>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

