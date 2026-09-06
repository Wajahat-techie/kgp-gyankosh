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
        radial-gradient(ellipse at 20% 0%, rgba(56, 189, 248, 0.22) 0%, transparent 55%),
        radial-gradient(ellipse at 80% 0%, rgba(129, 140, 248, 0.24) 0%, transparent 55%),
        radial-gradient(ellipse at 50% 100%, rgba(6, 182, 212, 0.18) 0%, transparent 55%),
        linear-gradient(155deg, rgba(6, 10, 22, 0.88) 0%, rgba(9, 16, 36, 0.82) 45%, rgba(15, 14, 38, 0.88) 100%),
        url("data:image/jpeg;base64,{CAMPUS_BG_BASE64}") !important;
    background-size: cover !important;
    background-position: center center !important;
    background-repeat: no-repeat !important;
    background-attachment: fixed !important;
    """
else:
    APP_BG_STYLE = """
    background: 
        radial-gradient(ellipse at 20% 10%, rgba(56, 189, 248, 0.28) 0%, transparent 50%),
        radial-gradient(ellipse at 85% 15%, rgba(168, 85, 247, 0.25) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 85%, rgba(14, 165, 233, 0.22) 0%, transparent 55%),
        radial-gradient(ellipse at 90% 80%, rgba(16, 185, 129, 0.18) 0%, transparent 45%),
        linear-gradient(150deg, #050812 0%, #091024 35%, #0f1533 70%, #070d1c 100%) !important;
    background-attachment: fixed !important;
    """

# ==============================================================================
# ENTERPRISE MODERN DARK GLASSMORPHISM THEME & CSS
# ==============================================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* Global Font & Reset - preserve Streamlit Material icon font ligatures */
html, body {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

[data-testid*="Icon"], [data-testid*="icon"], .material-symbols-rounded, .material-icons {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons', sans-serif !important;
}

h1, h2, h3, h4, .hero-main-title {
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: -0.02em;
}

/* Background Atmosphere - KGP College Campus with Scrim */
.stApp {
    __BG_STYLE__
    color: #f1f5f9;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Custom Sleek Scrollbar */
::-webkit-scrollbar {
    width: 7px;
    height: 7px;
}
::-webkit-scrollbar-track {
    background: #090d16;
}
::-webkit-scrollbar-thumb {
    background: #1e293b;
    border-radius: 9999px;
}
::-webkit-scrollbar-thumb:hover {
    background: #6366f1;
}

/* ==============================================================================
   LANDING PAGE & HERO HEADER
   ============================================================================== */
.gov-badge-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    margin-bottom: 1.2rem;
    flex-wrap: wrap;
}

.gov-pill {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.18) 0%, rgba(139, 92, 246, 0.12) 100%);
    border: 1px solid rgba(99, 102, 241, 0.4);
    color: #c7d2fe;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 6px 16px;
    border-radius: 9999px;
    box-shadow: 0 0 15px rgba(99, 102, 241, 0.2);
}

.dept-pill {
    background: rgba(6, 182, 212, 0.12);
    border: 1px solid rgba(6, 182, 212, 0.35);
    color: #67e8f9;
    font-size: 0.76rem;
    font-weight: 600;
    padding: 6px 16px;
    border-radius: 9999px;
}

.hero-card {
    background: rgba(19, 27, 46, 0.75);
    border: 1px solid rgba(99, 102, 241, 0.3);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 24px;
    padding: 2.4rem 2rem 1.8rem 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6), 0 0 45px rgba(79, 70, 229, 0.15);
    text-align: center;
    position: relative;
    overflow: hidden;
}

.hero-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 10%;
    right: 10%;
    height: 2px;
    background: linear-gradient(90deg, transparent, #6366f1, #06b6d4, #a855f7, transparent);
}

.hero-main-title {
    font-size: 3rem;
    font-weight: 800;
    line-height: 1.15;
    margin: 0.4rem 0 0.5rem 0;
    background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 35%, #818cf8 70%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-description {
    font-size: 1.2rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 0.4rem;
}

.college-institution-title {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
    font-size: 1.42rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    background: linear-gradient(135deg, #ffffff 10%, #38bdf8 55%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0.4rem 0 0.6rem 0;
    filter: drop-shadow(0 2px 12px rgba(56, 189, 248, 0.3));
}

.kgp-college-badge {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
    font-weight: 800 !important;
    letter-spacing: 0.03em;
    background: linear-gradient(135deg, #ffffff 10%, #38bdf8 60%, #818cf8 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    display: inline;
}

.hero-subtext {
    font-size: 0.94rem;
    color: #94a3b8;
    max-width: 760px;
    margin: 0 auto 1.6rem auto;
    line-height: 1.6;
}

/* Statistics Strip */
.stats-strip {
    display: flex;
    justify-content: center;
    gap: 18px;
    max-width: 520px;
    margin: 1.5rem auto 0 auto;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
}

.stat-box {
    flex: 1;
    min-width: 150px;
    background: rgba(11, 15, 25, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 12px 14px;
    display: flex;
    flex-direction: column;
    align-items: center;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}

.stat-box:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.45);
    box-shadow: 0 8px 20px rgba(99, 102, 241, 0.15);
}

.stat-num {
    font-family: 'Outfit', sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.stat-label {
    font-size: 0.72rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 3px;
    font-weight: 600;
}

/* ==============================================================================
   AUTHENTICATION FORM STYLING
   ============================================================================== */
[data-testid="stForm"] {
    background: rgba(19, 27, 46, 0.85) !important;
    border: 1px solid rgba(99, 102, 241, 0.35) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border-radius: 22px !important;
    padding: 2.2rem !important;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5), 0 0 35px rgba(99, 102, 241, 0.15) !important;
}

[data-testid="stForm"] h2 {
    color: #ffffff !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    margin-bottom: 1.2rem !important;
    text-align: center;
}

/* Form Inputs (Scoped to avoid overriding BaseWeb selectbox internal inputs) */
[data-testid="stForm"] input[type="text"],
[data-testid="stForm"] input[type="password"],
[data-testid="stTextInput"] input {
    background: #090e1c !important;
    border: 1px solid rgba(56, 189, 248, 0.25) !important;
    border-radius: 12px !important;
    color: #f8fafc !important;
    font-size: 0.95rem !important;
    padding: 0.72rem 1rem !important;
    transition: all 0.2s ease !important;
}

[data-testid="stForm"] input[type="text"]:focus,
[data-testid="stForm"] input[type="password"]:focus,
[data-testid="stTextInput"] input:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.25) !important;
    background: #0e172e !important;
}

/* Buttons */
.stButton > button, [data-testid="stForm"] button[kind="secondaryFormSubmit"], button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    padding: 0.68rem 1.6rem !important;
    font-size: 0.95rem !important;
    box-shadow: 0 4px 20px rgba(6, 182, 212, 0.35) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    letter-spacing: 0.02em;
}

.stButton > button:hover, [data-testid="stForm"] button[kind="secondaryFormSubmit"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(6, 182, 212, 0.55), 0 0 20px rgba(59, 130, 246, 0.35) !important;
    filter: brightness(1.15);
}

/* ==============================================================================
   SIDEBAR CUSTOMIZATION & CONTROL SUITE
   ============================================================================== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #070c1a 0%, #0b1328 45%, #121034 100%) !important;
    border-right: 1px solid rgba(56, 189, 248, 0.22) !important;
    box-shadow: 6px 0 30px rgba(0, 0, 0, 0.5) !important;
}

.sidebar-badge {
    display: inline-block;
    background: rgba(56, 189, 248, 0.15);
    border: 1px solid rgba(56, 189, 248, 0.35);
    color: #7dd3fc;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 8px;
    margin-bottom: 4px;
}

.sidebar-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(56, 189, 248, 0.18);
    border-radius: 14px;
    padding: 12px 14px;
    margin-bottom: 10px;
    backdrop-filter: blur(12px);
}

/* ==============================================================================
   SELECTBOX & DROPDOWN - STUCK-FREE, SMOOTH, & HIGH-CONTRAST
   ============================================================================== */
div[data-baseweb="select"] {
    border-radius: 12px !important;
    cursor: pointer !important;
}

div[data-baseweb="select"] > div {
    background: linear-gradient(145deg, rgba(14, 22, 44, 0.95) 0%, rgba(18, 28, 56, 0.9) 100%) !important;
    border: 1px solid rgba(56, 189, 248, 0.4) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    min-height: 46px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35) !important;
    transition: all 0.22s ease !important;
    cursor: pointer !important;
}

div[data-baseweb="select"] > div:hover {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 20px rgba(56, 189, 248, 0.32) !important;
    transform: translateY(-1px);
}

div[data-baseweb="select"] svg {
    color: #38bdf8 !important;
}

/* Dropdown popover menu overlay */
div[data-baseweb="popover"] {
    z-index: 99999999 !important;
}

div[data-baseweb="popover"] > div,
ul[role="listbox"] {
    background: rgba(10, 17, 34, 0.97) !important;
    border: 1px solid rgba(56, 189, 248, 0.45) !important;
    border-radius: 14px !important;
    box-shadow: 0 24px 50px rgba(0, 0, 0, 0.8), 0 0 25px rgba(56, 189, 248, 0.22) !important;
    backdrop-filter: blur(28px) !important;
    -webkit-backdrop-filter: blur(28px) !important;
    padding: 6px !important;
}

li[role="option"] {
    color: #cbd5e1 !important;
    background: transparent !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    padding: 11px 15px !important;
    border-radius: 9px !important;
    margin: 3px 2px !important;
    transition: all 0.16s ease !important;
    cursor: pointer !important;
}

li[role="option"]:hover,
li[role="option"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.28) 0%, rgba(99, 102, 241, 0.24) 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: 1px solid rgba(56, 189, 248, 0.45) !important;
    box-shadow: 0 2px 10px rgba(56, 189, 248, 0.2) !important;
}

/* Operational Mode Radio Buttons */
div[data-testid="stRadio"] > div {
    background: rgba(11, 18, 36, 0.75) !important;
    border: 1px solid rgba(56, 189, 248, 0.2) !important;
    border-radius: 14px !important;
    padding: 6px 8px !important;
    gap: 6px !important;
}

div[data-testid="stRadio"] label {
    background: rgba(15, 23, 42, 0.55) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 10px !important;
    padding: 8px 12px !important;
    margin: 2px 0 !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}

div[data-testid="stRadio"] label:hover {
    background: rgba(56, 189, 248, 0.12) !important;
    border-color: rgba(56, 189, 248, 0.35) !important;
}

div[data-testid="stRadio"] label[data-checked="true"],
div[data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.22) 0%, rgba(99, 102, 241, 0.2) 100%) !important;
    border: 1px solid rgba(56, 189, 248, 0.6) !important;
    box-shadow: 0 0 16px rgba(56, 189, 248, 0.2) !important;
}

div[data-testid="stRadio"] label span {
    color: #f1f5f9 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}

/* ==============================================================================
   CHAT INTERFACE & MESSAGES - RADIANT FROSTED GLASS
   ============================================================================== */
[data-testid="stChatMessage"] {
    background: rgba(12, 19, 36, 0.78) !important;
    border: 1px solid rgba(56, 189, 248, 0.2) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border-radius: 20px !important;
    padding: 1.3rem 1.6rem !important;
    margin-bottom: 1.1rem !important;
    box-shadow: 0 12px 36px rgba(0, 0, 0, 0.45) !important;
    transition: all 0.25s ease;
}

[data-testid="stChatMessage"]:hover {
    border-color: rgba(56, 189, 248, 0.45) !important;
    box-shadow: 0 16px 42px rgba(0, 0, 0, 0.55), 0 0 25px rgba(56, 189, 248, 0.2) !important;
}

/* Assistant message distinctive border */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: linear-gradient(135deg, rgba(11, 18, 36, 0.9) 0%, rgba(16, 26, 54, 0.84) 100%) !important;
    border-left: 4px solid #38bdf8 !important;
}

/* User message distinctive border */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, rgba(26, 20, 56, 0.82) 0%, rgba(13, 20, 38, 0.86) 100%) !important;
    border-left: 4px solid #a855f7 !important;
}

/* Bottom Sticky Bar Container (Mobile & Desktop fix) */
[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"],
.stChatFloatingInputContainer {
    background: rgba(7, 11, 20, 0.94) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
}

/* Chat Input Bar & Placeholder */
[data-testid="stChatInput"] {
    border-radius: 20px !important;
    background: #0d1527 !important;
    border: 1.5px solid rgba(56, 189, 248, 0.5) !important;
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.65), 0 0 25px rgba(56, 189, 248, 0.2) !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #38bdf8 !important;
    box-shadow: 0 14px 45px rgba(0, 0, 0, 0.75), 0 0 32px rgba(56, 189, 248, 0.45) !important;
}

[data-testid="stChatInput"] > div {
    background: transparent !important;
}

[data-testid="stChatInput"] textarea,
[data-testid="stChatInputTextArea"],
textarea[data-testid="stChatInputTextArea"] {
    background-color: transparent !important;
    color: #f8fafc !important;
    -webkit-text-fill-color: #f8fafc !important;
    caret-color: #38bdf8 !important;
    font-size: 1rem !important; /* 16px prevents iOS mobile zoom glitch */
    line-height: 1.5 !important;
}

[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stChatInputTextArea"]::placeholder,
textarea[data-testid="stChatInputTextArea"]::placeholder {
    color: #94a3b8 !important;
    -webkit-text-fill-color: #94a3b8 !important;
    opacity: 1 !important;
    font-size: 0.94rem !important;
}

[data-testid="stChatInput"] textarea::-webkit-input-placeholder {
    color: #94a3b8 !important;
    -webkit-text-fill-color: #94a3b8 !important;
}

[data-testid="stChatInput"] textarea::-moz-placeholder {
    color: #94a3b8 !important;
    opacity: 1 !important;
}

[data-testid="stChatInput"] button {
    color: #38bdf8 !important;
    border: none !important;
}

[data-testid="stChatInput"] button:hover {
    color: #7dd3fc !important;
    transform: scale(1.08) !important;
}

/* Chat Message Text Colors (Ensures mobile high-contrast readability) */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div,
[data-testid="stChatMessage"] li {
    color: #f1f5f9 !important;
    line-height: 1.65;
}

[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] b {
    color: #ffffff !important;
    font-weight: 700;
}

/* Mobile Responsiveness Enhancements */
@media (max-width: 768px) {
    .hero-main-title {
        font-size: 2rem !important;
    }
    .hero-card {
        padding: 1.5rem 1rem !important;
        border-radius: 18px !important;
    }
    [data-testid="stChatMessage"] {
        padding: 1rem 1.1rem !important;
        border-radius: 16px !important;
        margin-bottom: 0.8rem !important;
    }
    [data-testid="stChatInput"] {
        border-radius: 16px !important;
    }
    [data-testid="stChatInput"] textarea {
        font-size: 16px !important;
    }
}

iframe[height="0"],
iframe[title*="html"] {
    display: none !important;
}

/* Expander Cards (Citations & Info) */
[data-testid="stExpander"] {
    background: rgba(15, 23, 42, 0.65) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    margin-top: 0.6rem !important;
}

[data-testid="stExpander"] details summary {
    font-weight: 600 !important;
    color: #93c5fd !important;
}

/* Query Chips (Starters) */
.chip-container {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin: 1.2rem 0 1.6rem 0;
}

.query-chip {
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #c7d2fe;
    font-size: 0.84rem;
    font-weight: 500;
    padding: 7px 16px;
    border-radius: 9999px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: all 0.2s ease;
}

.query-chip:hover {
    background: rgba(99, 102, 241, 0.25);
    border-color: #818cf8;
    color: #ffffff;
    transform: translateY(-1px);
}

.credential-badge {
    background: rgba(11, 15, 25, 0.7);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 14px;
    padding: 14px 18px;
    margin-bottom: 10px;
}

/* Document Direct Download Links */
.doc-download-link {
    font-weight: 700 !important;
    color: #38bdf8 !important;
    text-decoration: underline !important;
    text-decoration-color: rgba(56, 189, 248, 0.45) !important;
    text-underline-offset: 4px;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: all 0.2s ease;
}

.doc-download-link:hover {
    color: #7dd3fc !important;
    text-decoration-color: #38bdf8 !important;
}

.doc-download-badge {
    font-size: 0.72rem;
    color: #38bdf8;
    background: rgba(56, 189, 248, 0.14);
    border: 1px solid rgba(56, 189, 248, 0.32);
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 600;
    text-decoration: none !important;
    display: inline-flex;
    align-items: center;
    gap: 3px;
    vertical-align: middle;
    transition: all 0.15s ease;
}

.doc-download-link:hover .doc-download-badge {
    background: rgba(56, 189, 248, 0.28);
    border-color: #38bdf8;
    color: #ffffff;
    transform: translateY(-1px);
}
</style>
"""
CUSTOM_CSS = CUSTOM_CSS.replace("__BG_STYLE__", APP_BG_STYLE)


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

    data_root = os.path.abspath(os.path.join(PROJECT_ROOT, "data"))
    if not os.path.exists(data_root):
        return None

    clean_name = os.path.basename(source_filename).strip().lower()
    
    # 1. Search data/ recursively
    for root, _, files in os.walk(data_root):
        for f in files:
            f_lower = f.lower()
            if f_lower == clean_name:
                return os.path.join(root, f)
            # Handle Orders.pdf <-> Orders_11zon.pdf alias
            if clean_name in ("orders.pdf", "orders_11zon.pdf") and f_lower in ("orders.pdf", "orders_11zon.pdf"):
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
        rel_path = os.path.relpath(file_path, data_root)
        clean_rel = rel_path.replace(os.sep, "/")
        encoded_parts = [urllib.parse.quote(part) for part in clean_rel.split("/")]
        url_path = "app/static/docs/" + "/".join(encoded_parts)
        return url_path, file_path
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

    file_path = resolve_document_file_path(source_filename, manifest)
    if not file_path or not os.path.exists(file_path):
        return None, None, False, None

    # Check if document is a PDF and a specific page is cited
    is_pdf = source_filename.lower().endswith(".pdf") or file_path.lower().endswith(".pdf")
    if is_pdf and page_num and page_num > 0:
        extracted_dir = os.path.join(PROJECT_ROOT, "static", "extracted_pages")
        os.makedirs(extracted_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
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
            page_url = f"app/static/extracted_pages/{urllib.parse.quote(page_filename)}"
            return page_url, page_filename, True, page_filepath

    doc_url, _ = get_document_url_and_path(source_filename, manifest)
    return doc_url, os.path.basename(file_path), False, file_path


def linkify_answer_citations(answer_text: str, manifest: dict = None) -> str:
    """
    Cleans and styles document citations in the answer text using clickable download badges.
    Clicking a multi-page PDF citation downloads specifically that cited order page.
    """
    if not answer_text:
        return answer_text

    pattern = r'\[([^\]\n]+?\.(?:docx|pdf|txt|png|jpg|jpeg))\s*(?:,\s*Page\s*(\d+))?\]'

    def _replace(match):
        doc_name = match.group(1).strip()
        page_num_str = match.group(2)
        page_num = int(page_num_str) if page_num_str and page_num_str.isdigit() else 1
        page_suffix = f", Page {page_num}" if page_num_str else ""

        dl_url, dl_filename, is_page, _ = get_document_download_info(doc_name, page_num, manifest)
        if dl_url:
            badge_title = f"Click to download Page {page_num} of {doc_name}" if is_page else f"Click to download {doc_name}"
            return (
                f'<a href="{dl_url}" download="{dl_filename}" target="_blank" '
                f'style="display: inline-block; color: #38bdf8; font-weight: 600; background: rgba(56, 189, 248, 0.12); padding: 2px 10px; border-radius: 6px; border: 1px solid rgba(56, 189, 248, 0.25); margin: 2px 0; text-decoration: none; cursor: pointer; transition: all 0.15s ease;" '
                f'title="{badge_title}">'
                f'📄 {doc_name}{page_suffix} <span style="font-size: 0.72rem; color: #a5b4fc;">📥</span></a>'
            )
        return f'<span style="display: inline-block; color: #38bdf8; font-weight: 600; background: rgba(56, 189, 248, 0.12); padding: 2px 10px; border-radius: 6px; border: 1px solid rgba(56, 189, 248, 0.25); margin: 2px 0;">📄 {doc_name}{page_suffix}</span>'

    text = re.sub(pattern, _replace, answer_text, flags=re.IGNORECASE)

    # Style college mentions in bold with beautiful typography
    college_regex = r'(?:\*{2})?Kashmir\s+(?:Govt\.?|Government)\s+Polytechnic\s+College(?:,?\s*Srinagar)?(?:\*{2})?'
    text = re.sub(
        college_regex,
        '<strong class="kgp-college-badge">KASHMIR GOVERNMENT POLYTECHNIC COLLEGE, SRINAGAR</strong>',
        text,
        flags=re.IGNORECASE
    )
    return text


def render_citations_and_links(sources: list, manifest: dict, key_prefix: str = "src"):
    """
    Renders structured citations with guaranteed Streamlit native download buttons
    and direct single-click extracted PDF page download links.
    """
    if not sources:
        return

    with st.expander(f"📚 Source Citations & Download Official Files ({len(sources)} documents)", expanded=True):
        for idx, src in enumerate(sources):
            src_name = src.get("source", "Unknown Document")
            page_num = src.get("page", 1)
            rerank_score = src.get("rerank_score")
            score_text = f" | Rerank Score: `{rerank_score}`" if rerank_score is not None else ""

            dl_url, dl_filename, is_page, actual_path = get_document_download_info(src_name, page_num, manifest)

            col_info, col_btn = st.columns([3, 1])
            with col_info:
                st.markdown(
                    f"""
                    <div style="font-size: 0.95rem; font-weight: 700; color: #38bdf8;">
                        📄 {src_name}
                        <span style="color: #94a3b8; font-size: 0.82rem; font-weight: 500;"> (Page {page_num})</span>
                        <span style="color: #34d399; font-size: 0.76rem; font-weight: 600;">{score_text}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if src.get("excerpt"):
                    st.caption(f"_{src.get('excerpt')[:250]}..._")

            with col_btn:
                if actual_path and os.path.exists(actual_path):
                    try:
                        with open(actual_path, "rb") as f_doc:
                            file_bytes = f_doc.read()
                        mime_type = "application/pdf" if dl_filename.endswith(".pdf") else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        btn_label = f"📥 Page {page_num}" if is_page else "📥 Download"
                        st.download_button(
                            label=btn_label,
                            data=file_bytes,
                            file_name=dl_filename,
                            mime=mime_type,
                            key=f"dl_{key_prefix}_{idx}_{page_num}",
                            use_container_width=True
                        )
                    except Exception as ex:
                        logger.warning(f"Could not prepare download for {actual_path}: {ex}")
                elif dl_url:
                    st.markdown(f'<a href="{dl_url}" download="{dl_filename}" target="_blank" class="doc-download-badge">📥 Download</a>', unsafe_allow_html=True)

            if idx < len(sources) - 1:
                st.markdown("<hr style='margin: 8px 0; border: none; border-top: 1px solid rgba(255,255,255,0.07);'>", unsafe_allow_html=True)

        st.caption("💡 _Click any download button to retrieve that exact cited order page or document directly to your device._")


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
# SESSION STATE INITIALIZATION
# ==============================================================================
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory(max_history_turns=6)

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


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
        return

    # --------------------------------------------------------------------------
    # 2. AUTHENTICATED USER SESSION & SIDEBAR
    # --------------------------------------------------------------------------
    system_resources = initialize_system()
    retriever = system_resources["retriever"]
    reranker = system_resources["reranker"]
    llm_client = system_resources["llm_client"]
    manifest = system_resources["manifest"]
    index_ready = system_resources["index_ready"]

    # Sidebar
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
        username = st.session_state.get("username", "Admin")
        name = st.session_state.get("name", "Administrative Staff")
        st.markdown(f"**Logged in as:**\n👤 **{name}** (`{username}`)")
        authenticator.logout(location="sidebar")
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
            "🔍 **RAG Mode**: Answers are grounded strictly in 1,162 official college notices with citations."
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

        # Clear Conversation Button
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.chat_messages = []
            st.session_state.memory.clear()
            st.rerun()

    # --------------------------------------------------------------------------
    # 3. MAIN CHAT INTERFACE
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
        '<span style="background: rgba(168, 85, 247, 0.16); color: #d8b4fe; font-size: 0.78rem; font-weight: 700; padding: 3px 12px; border-radius: 9999px; border: 1px solid rgba(168, 85, 247, 0.35);">🌐 Mode: General AI Chat</span>'
        if is_general_ai
        else '<span style="background: rgba(56, 189, 248, 0.16); color: #7dd3fc; font-size: 0.78rem; font-weight: 700; padding: 3px 12px; border-radius: 9999px; border: 1px solid rgba(56, 189, 248, 0.35);">🏛️ Mode: College Records (RAG)</span>'
    )
    active_model_badge = f'<span style="background: rgba(52, 211, 153, 0.16); color: #6ee7b7; font-size: 0.78rem; font-weight: 700; padding: 3px 12px; border-radius: 9999px; border: 1px solid rgba(52, 211, 153, 0.35);">🤖 Active Model: {selected_model_label}</span>'
    mode_description = (
        f"Direct AI intelligence for drafting official notices, writing letters, explaining concepts, or general Q&A with <strong>{sel_model}</strong>."
        if is_general_ai
        else f"Institutional knowledge search grounded strictly in 1,162 official college notices & circulars with citations via <strong>{sel_model}</strong>."
    )

    if st.session_state.chat_messages:
        # Compact top header during active conversation so messages remain visible
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem; padding: 0.65rem 1.2rem; background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 14px; backdrop-filter: blur(12px);">
                <div style="font-family: 'Outfit', sans-serif; font-size: 1.05rem; font-weight: 800; text-transform: uppercase;">
                    🏛️ <span style="background: linear-gradient(135deg, #ffffff 15%, #38bdf8 70%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">KASHMIR GOVERNMENT POLYTECHNIC COLLEGE</span>
                </div>
                <div style="display: flex; gap: 8px; align-items: center;">
                    {active_mode_badge}
                    {active_model_badge}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Full banner when no conversation has started
        st.markdown(
            f"""
            <div style="margin-bottom: 1rem; padding: 1.1rem 1.4rem; background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 16px; backdrop-filter: blur(12px);">
                <div style="font-family: 'Outfit', sans-serif; font-size: 1.45rem; font-weight: 800; letter-spacing: 0.04em; text-transform: uppercase; line-height: 1.3;">
                    🏛️ <span style="background: linear-gradient(135deg, #ffffff 15%, #38bdf8 60%, #818cf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800;">KASHMIR GOVERNMENT POLYTECHNIC COLLEGE, SRINAGAR</span>
                </div>
                <div style="font-size: 0.86rem; color: #cbd5e1; margin-top: 4px;">
                    {mode_description}
                </div>
                <div style="display: flex; gap: 8px; margin-top: 10px; align-items: center; flex-wrap: wrap;">
                    {active_mode_badge}
                    {active_model_badge}
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

    # Render previous conversation history
    for msg_idx, message in enumerate(st.session_state.chat_messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                content_with_links = linkify_answer_citations(message["content"], manifest)
                st.markdown(content_with_links, unsafe_allow_html=True)
                if message.get("sources"):
                    render_citations_and_links(message["sources"], manifest, key_prefix=f"hist_{msg_idx}")
            else:
                st.markdown(message["content"])

    def scroll_to_bottom():
        """Smoothly auto-scrolls down to reveal newly rendered queries, spinner, and answers."""
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
                            el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
                        }
                    }
                    const lastMsg = doc.querySelector('[data-testid="stChatMessage"]:last-of-type') || doc.querySelector('[data-testid="stChatInput"]');
                    if (lastMsg) {
                        lastMsg.scrollIntoView({ behavior: 'smooth', block: 'end' });
                    }
                } catch (e) {}
            }
            setTimeout(doScroll, 40);
            setTimeout(doScroll, 250);
        </script>
        """
        st.components.v1.html(js, height=0)

    if st.session_state.chat_messages:
        scroll_to_bottom()

    # Empty State Guidance
    if not st.session_state.chat_messages:
        st.markdown(
            """
            <div style="text-align: center; margin: 3.5rem auto 2rem auto; max-width: 600px;">
                <div style="font-size: 2.4rem; margin-bottom: 0.5rem;">💡</div>
                <h2 style="font-family: 'Outfit', sans-serif; font-size: 1.85rem; font-weight: 700; background: linear-gradient(135deg, #ffffff 30%, #38bdf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;">
                    How can I assist you today?
                </h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    # User input with mode-specific placeholder
    input_placeholder = (
        f"💬 Type your message here (e.g. 'Draft a circular for faculty meeting')... [Press Enter]"
        if is_general_ai
        else "🔍 Ask about college circulars, leave rules, notices, or admissions... [Press Enter]"
    )
    if prompt := st.chat_input(input_placeholder):
        # Display user query
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        scroll_to_bottom()

        # Process query through selected mode (RAG vs Direct LLM)
        with st.chat_message("assistant"):
            try:
                if "General AI" in assistant_mode:
                    with st.spinner("Thinking..."):
                        history_prompt = st.session_state.memory.format_history_for_prompt()
                        result = llm_client.generate_chat(
                            query=prompt,
                            conversation_history=history_prompt
                        )
                        answer_text = result["answer"]
                        sources = []
                else:
                    with st.spinner("Thinking..."):
                        # 1. Multi-turn Query Reformulation
                        reformulated_query = st.session_state.memory.reformulate_query(prompt, llm_client)

                        # 2. Hybrid Retrieval (Vector + BM25 Reciprocal Rank Fusion)
                        candidates = retriever.retrieve(reformulated_query)

                        # 3. Cross-Encoder Reranking
                        reranked_docs = reranker.rerank(reformulated_query, candidates)
                        top_documents = [doc for doc, score in reranked_docs]

                        # 4. Synthesize Answer with Strict Anti-Hallucination Prompt
                        history_prompt = st.session_state.memory.format_history_for_prompt()
                        result = llm_client.generate_answer(
                            query=prompt,
                            context_documents=top_documents,
                            conversation_history=history_prompt
                        )

                        answer_text = result["answer"]
                        sources = result["sources"]

                # Display final response with formatted citations
                content_with_links = linkify_answer_citations(answer_text, manifest)
                st.markdown(content_with_links, unsafe_allow_html=True)

                if sources:
                    render_citations_and_links(sources, manifest, key_prefix=f"curr_{len(st.session_state.chat_messages)}")

                # Update memory & session state
                st.session_state.memory.add_turn(prompt, answer_text, sources)
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "sources": sources
                })
                scroll_to_bottom()
            except Exception as err:
                st.error(f"⚠️ Query error: {err}")
                logger.error(f"Error handling query: {err}", exc_info=True)


if __name__ == "__main__":
    main()
