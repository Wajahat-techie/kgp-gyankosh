"""
create_diagrams.py
Generates crystal-clear, high-resolution workflow diagrams for KGP Gyankosh
using Pillow with TrueType fonts (Segoe UI / Arial).
"""

import os
import sys
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")

def get_font(name="segoeui.ttf", size=18, bold=False):
    font_file = "segoeuib.ttf" if bold else name
    font_path = os.path.join("C:\\Windows\\Fonts", font_file)
    if not os.path.exists(font_path):
        font_path = os.path.join("C:\\Windows\\Fonts", "arialbd.ttf" if bold else "arial.ttf")
    try:
        return ImageFont.truetype(font_path, size)
    except Exception:
        return ImageFont.load_default()

def draw_card(draw, xy, title, subtitle="", items=None, bg_color="#ffffff", border_color="#3b82f6", title_color="#1e3a8a", text_color="#334155", badge=None, badge_bg="#e0f2fe", badge_color="#0369a1"):
    x0, y0, x1, y1 = xy
    # Draw card background
    draw.rounded_rectangle([x0, y0, x1, y1], radius=10, fill=bg_color, outline=border_color, width=2)
    
    # Title & Badge
    f_title = get_font("segoeuib.ttf", 16, bold=True)
    f_sub = get_font("segoeui.ttf", 12)
    f_item = get_font("segoeui.ttf", 13)
    f_badge = get_font("segoeuib.ttf", 11, bold=True)

    draw.text((x0 + 14, y0 + 12), title, fill=title_color, font=f_title)
    
    if badge:
        bbox = draw.textbbox((0, 0), badge, font=f_badge)
        bw = bbox[2] - bbox[0] + 16
        bh = bbox[3] - bbox[1] + 8
        bx1 = x1 - 12
        bx0 = bx1 - bw
        by0 = y0 + 10
        by1 = by0 + bh
        draw.rounded_rectangle([bx0, by0, bx1, by1], radius=6, fill=badge_bg)
        draw.text((bx0 + 8, by0 + 3), badge, fill=badge_color, font=f_badge)

    curr_y = y0 + 36
    if subtitle:
        draw.text((x0 + 14, curr_y), subtitle, fill="#64748b", font=f_sub)
        curr_y += 18

    if items:
        for it in items:
            draw.text((x0 + 14, curr_y), it, fill=text_color, font=f_item)
            curr_y += 18

def draw_arrow(draw, start, end, color="#2563eb", width=3):
    x0, y0 = start
    x1, y1 = end
    draw.line([x0, y0, x1, y1], fill=color, width=width)
    # Draw arrow head
    if x0 == x1:  # Vertical
        if y1 > y0:  # Down
            draw.polygon([(x1 - 6, y1 - 8), (x1 + 6, y1 - 8), (x1, y1 + 2)], fill=color)
        else:  # Up
            draw.polygon([(x1 - 6, y1 + 8), (x1 + 6, y1 + 8), (x1, y1 - 2)], fill=color)
    elif y0 == y1:  # Horizontal
        if x1 > x0:  # Right
            draw.polygon([(x1 - 8, y1 - 6), (x1 - 8, y1 + 6), (x1 + 2, y1)], fill=color)
        else:  # Left
            draw.polygon([(x1 + 8, y1 - 6), (x1 + 8, y1 + 6), (x1 - 2, y1)], fill=color)


def generate_stage1_diagram(output_path=None):
    """Diagram 1: Offline Ingestion & Indexing Pipeline"""
    if output_path is None:
        output_path = os.path.join(STATIC_DIR, "diagram_stage1_indexing.png")
    w, h = 1500, 850
    img = Image.new("RGB", (w, h), color="#f8fafc")
    draw = ImageDraw.Draw(img)

    # Header Banner
    draw.rounded_rectangle([20, 20, w - 20, 90], radius=12, fill="#0f172a")
    f_banner = get_font("segoeuib.ttf", 24, bold=True)
    f_bannersub = get_font("segoeui.ttf", 13)
    draw.text((45, 30), "STAGE 1: OFFLINE DOCUMENT INGESTION & DUAL INDEXING PIPELINE", fill="#ffffff", font=f_banner)
    draw.text((45, 62), "Batch script (build_index.py) executed offline to parse, OCR, chunk, and index 1,158 administrative college records into FAISS and BM25", fill="#94a3b8", font=f_bannersub)

    # Step 1: Data Source
    draw_card(
        draw, [40, 130, 440, 260],
        title="1. Administrative Records",
        subtitle="c:/.../kgp-gyankosh/data/",
        items=[
            "• 1,157 DOCX Circulars (data/word/)",
            "• 85-Page Orders Book (data/pdf/Orders.pdf)",
            "• Policies, leave sanctions, exams & fee orders"
        ],
        border_color="#0284c7", title_color="#0369a1",
        badge="INPUT DATA", badge_bg="#e0f2fe", badge_color="#0369a1"
    )

    # Arrow 1 -> 2
    draw_arrow(draw, (440, 195), (520, 195))

    # Step 2: Document Loader & Inspection
    draw_card(
        draw, [520, 130, 960, 260],
        title="2. Polymorphic Document Loader",
        subtitle="src/ingestion/document_loader.py",
        items=[
            "• Checks file hashes against index_manifest.json",
            "• Auto-detects native text vs scanned pages",
            "• Incremental skip: 1,158 unchanged files bypassed"
        ],
        border_color="#6366f1", title_color="#4338ca",
        badge="INSPECTION", badge_bg="#ede9fe", badge_color="#6d28d9"
    )

    # Branches to OCR and Native Text
    draw_arrow(draw, (960, 175), (1040, 175))
    draw_arrow(draw, (960, 215), (1040, 310))

    # Step 3A: Native Text Parser
    draw_card(
        draw, [1040, 130, 1460, 230],
        title="3A. Native Text Parser (DOCX)",
        subtitle="python-docx & pypdfium2",
        items=[
            "• Direct text extraction from 1,157 DOCX files",
            "• Paragraph formatting & tables preserved"
        ],
        border_color="#10b981", title_color="#047857",
        badge="NATIVE TEXT", badge_bg="#d1fae5", badge_color="#065f46"
    )

    # Step 3B: OCR Engine
    draw_card(
        draw, [1040, 255, 1460, 375],
        title="3B. Scanned OCR Engine",
        subtitle="src/ingestion/ocr_loader.py (RapidOCR)",
        items=[
            "• Scanned pages from Orders.pdf",
            "• RapidOCR (ONNX Runtime) + Tesseract fallback",
            "• Persistent OCR Cache: output/ocr_cache/"
        ],
        border_color="#f59e0b", title_color="#b45309",
        badge="OCR PIPELINE", badge_bg="#fef3c7", badge_color="#92400e"
    )

    # Converge 3A & 3B down to Step 4
    draw_arrow(draw, (1250, 230), (1250, 245))
    draw_arrow(draw, (1250, 375), (1250, 420))
    draw_arrow(draw, (1250, 420), (780, 420))
    draw_arrow(draw, (780, 420), (780, 450))

    # Step 4: Recursive Text Chunker
    draw_card(
        draw, [500, 450, 1060, 570],
        title="4. Recursive Text Chunker",
        subtitle="src/ingestion/chunker.py",
        items=[
            "• Target Chunk Size: 500 characters",
            "• Chunk Overlap: 100 characters",
            "• Preserves paragraph boundaries, dates & order numbers",
            "• Total Chunks Generated: 8,166 clause segments"
        ],
        border_color="#8b5cf6", title_color="#6d28d9",
        badge="CHUNKING", badge_bg="#ede9fe", badge_color="#5b21b6"
    )

    # Diverge Step 4 to Dense Embeddings & Sparse BM25
    draw_arrow(draw, (620, 570), (320, 620))
    draw_arrow(draw, (940, 570), (1180, 620))

    # Step 5A: Dense Vector Embedding & FAISS Store
    draw_card(
        draw, [80, 620, 580, 770],
        title="5A. Dense Embeddings & FAISS",
        subtitle="src/indexing/embeddings.py & vector_store.py",
        items=[
            "• Model: sentence-transformers (all-MiniLM-L6-v2)",
            "• 384-dimensional dense semantic vector space",
            "• Serialized Target: output/vector_store/index.faiss"
        ],
        border_color="#0284c7", title_color="#0369a1",
        badge="DENSE VECTOR", badge_bg="#e0f2fe", badge_color="#0369a1"
    )

    # Step 5B: Sparse BM25 Lexical Index
    draw_card(
        draw, [920, 620, 1420, 770],
        title="5B. Sparse BM25 Lexical Index",
        subtitle="src/retrieval/hybrid_search.py (rank_bm25)",
        items=[
            "• BM25Okapi inverted token frequency index",
            "• Exact match for order IDs, dates & figures",
            "• Serialized Target: output/bm25_store/bm25_model.pkl"
        ],
        border_color="#10b981", title_color="#047857",
        badge="SPARSE BM25", badge_bg="#d1fae5", badge_color="#065f46"
    )

    # Output Manifest
    draw_arrow(draw, (580, 695), (660, 695))
    draw_arrow(draw, (920, 695), (840, 695))

    draw_card(
        draw, [660, 640, 840, 750],
        title="Index Manifest",
        subtitle="output/index_manifest.json",
        items=["• 1,158 Files Cataloged", "• 8,166 Total Chunks"],
        border_color="#475569", title_color="#1e293b",
        badge="SYNC", badge_bg="#f1f5f9", badge_color="#475569"
    )

    # Footer note
    f_ft = get_font("segoeui.ttf", 11)
    draw.text((45, 805), "Strict Two-Stage Design: Heavy ingestion runs once offline; query app loads pre-computed stores in sub-second time with 0% query indexing lag.", fill="#64748b", font=f_ft)

    img.save(output_path, "PNG")
    print(f"Generated Stage 1 diagram: {output_path}")


def generate_stage2_diagram(output_path=None):
    """Diagram 2: Online Query & Retrieval Workflow"""
    if output_path is None:
        output_path = os.path.join(STATIC_DIR, "diagram_stage2_retrieval.png")
    w, h = 1500, 850
    img = Image.new("RGB", (w, h), color="#f8fafc")
    draw = ImageDraw.Draw(img)

    # Header Banner
    draw.rounded_rectangle([20, 20, w - 20, 90], radius=12, fill="#0f172a")
    f_banner = get_font("segoeuib.ttf", 24, bold=True)
    f_bannersub = get_font("segoeui.ttf", 13)
    draw.text((45, 30), "STAGE 2: ONLINE QUERY RETRIEVAL & CONVERSATIONAL WORKFLOW", fill="#ffffff", font=f_banner)
    draw.text((45, 62), "Real-time user query flow: Authentication -> Memory Rewrite -> Hybrid Search -> Neural Reranker -> Grounded Answer & Single-Page PDF Download", fill="#94a3b8", font=f_bannersub)

    # Step 1: User & Auth
    draw_card(
        draw, [40, 130, 440, 260],
        title="1. Staff User & Auth Gate",
        subtitle="src/auth/authenticator.py (app.py)",
        items=[
            "• College administrative login gate",
            "• Salted BCrypt password verification",
            "• Role-based access: Admin, Staff, Faculty",
            "• Intranet / LAN deployment (http://localhost:8501)"
        ],
        border_color="#0284c7", title_color="#0369a1",
        badge="SECURITY", badge_bg="#e0f2fe", badge_color="#0369a1"
    )

    draw_arrow(draw, (440, 195), (530, 195))

    # Step 2: Query & Conversational Memory
    draw_card(
        draw, [530, 130, 970, 260],
        title="2. Natural Language Query & Memory",
        subtitle="src/memory/conversation_memory.py",
        items=[
            "• User enters question (e.g. 'Vacation order for July')",
            "• Tracks multi-turn dialogue history (up to 6 turns)",
            "• Reformulates ambiguous follow-ups into standalone search terms"
        ],
        border_color="#8b5cf6", title_color="#6d28d9",
        badge="MEMORY", badge_bg="#ede9fe", badge_color="#5b21b6"
    )

    draw_arrow(draw, (970, 195), (1050, 195))

    # Step 3: Hybrid Search Retrieval
    draw_card(
        draw, [1050, 130, 1460, 310],
        title="3. Hybrid Search Retriever",
        subtitle="src/retrieval/hybrid_search.py",
        items=[
            "• FAISS Dense Search: Top-30 semantic vectors",
            "• BM25 Sparse Search: Top-30 lexical token matches",
            "• Reciprocal Rank Fusion (RRF, k=60, alpha=0.5)",
            "• Fuses semantic intent + exact order codes",
            "• Output: Top-10 Ranked Hybrid Passages"
        ],
        border_color="#10b981", title_color="#047857",
        badge="HYBRID RRF", badge_bg="#d1fae5", badge_color="#065f46"
    )

    draw_arrow(draw, (1255, 310), (1255, 360))
    draw_arrow(draw, (1255, 360), (750, 360))
    draw_arrow(draw, (750, 360), (750, 400))

    # Step 4: Cross-Encoder Neural Reranker
    draw_card(
        draw, [480, 400, 1020, 540],
        title="4. Cross-Encoder Neural Reranker",
        subtitle="src/retrieval/reranker.py (ms-marco-MiniLM-L-6-v2)",
        items=[
            "• Joint bidirectional self-attention over (Query, Passage) pairs",
            "• Accurately eliminates false positives & passage drift",
            "• Re-scores top-10 hybrid candidates with deep cross-attention",
            "• Selects Top-6 most authoritative, relevant administrative clauses"
        ],
        border_color="#f59e0b", title_color="#b45309",
        badge="NEURAL RERANK", badge_bg="#fef3c7", badge_color="#92400e"
    )

    draw_arrow(draw, (750, 540), (750, 590))

    # Step 5: Switchable LLM Inference
    draw_card(
        draw, [60, 590, 680, 740],
        title="5. Switchable LLM Inference Engine",
        subtitle="src/llm/client.py",
        items=[
            "• Google Gemini (gemini-3.5-flash-lite) [Active]",
            "• Local Ollama (llama3.1:latest) [100% Offline, Free]",
            "• OpenAI (gpt-4o-mini) [Cloud API]",
            "• Strict System Prompt: Grounded solely in retrieved clauses",
            "• Anti-hallucination guard: Declines unsupported claims"
        ],
        border_color="#6366f1", title_color="#4338ca",
        badge="LLM ENGINE", badge_bg="#ede9fe", badge_color="#6d28d9"
    )

    draw_arrow(draw, (680, 665), (780, 665))

    # Step 6: Grounded Output & Single-Page PDF Downloader
    draw_card(
        draw, [780, 590, 1440, 740],
        title="6. Grounded Answer & Direct Download",
        subtitle="Streamlit app.py (Single-Page PDF Extractor)",
        items=[
            "• Source citations with document title and page numbers",
            "• Direct click-to-download on document name itself",
            "• Multi-Page PDF Handling: Extracts ONLY cited single page",
            "  (e.g. Orders_page_67.pdf @ 400 KB instead of full 36 MB PDF)",
            "• Complete Word documents (.docx) downloaded on single click"
        ],
        border_color="#0284c7", title_color="#0369a1",
        badge="OUTPUT & SERVING", badge_bg="#e0f2fe", badge_color="#0369a1"
    )

    # Footer note
    f_ft = get_font("segoeui.ttf", 11)
    draw.text((45, 805), "Zero Query Indexing Overhead: Pre-indexed FAISS & BM25 indices load in ~0.2s. Total query response time 1.2s - 2.5s.", fill="#64748b", font=f_ft)

    img.save(output_path, "PNG")
    print(f"Generated Stage 2 diagram: {output_path}")


def generate_file_execution_flow_diagram(output_path=None):
    """Diagram 3: File-by-File Execution Sequence Flowchart"""
    if output_path is None:
        output_path = os.path.join(STATIC_DIR, "diagram_file_execution_flow.png")
    w, h = 1500, 1220
    img = Image.new("RGB", (w, h), color="#f8fafc")
    draw = ImageDraw.Draw(img)

    # Header Banner
    draw.rounded_rectangle([25, 20, w - 25, 95], radius=12, fill="#0f172a")
    f_title = get_font("segoeuib.ttf", 22, bold=True)
    f_sub = get_font("segoeui.ttf", 12.5)
    draw.text((45, 30), "FILE-BY-FILE CODE EXECUTION FLOW ON USER QUERY", fill="#ffffff", font=f_title)
    draw.text((45, 64), "Sequential code execution trace showing exactly which source file, function, and data store executes at each stage", fill="#38bdf8", font=f_sub)

    def draw_exec_box(xy, step_num, file_name, method_name, bullets, color="#0284c7", badge="STEP"):
        x0, y0, x1, y1 = xy
        # Outer Card
        draw.rounded_rectangle([x0, y0, x1, y1], radius=10, fill="#ffffff", outline=color, width=2)
        # Header banner inside card
        draw.rounded_rectangle([x0, y0, x1, y0 + 36], radius=10, fill=color)
        f_st = get_font("segoeuib.ttf", 15, bold=True)
        f_fn = get_font("segoeuib.ttf", 13, bold=True)
        f_lbl = get_font("segoeuib.ttf", 12)
        f_txt = get_font("segoeui.ttf", 12)

        draw.text((x0 + 16, y0 + 7), f"STEP {step_num}:  {file_name}", fill="#ffffff", font=f_st)

        # Badge
        f_bdg = get_font("segoeuib.ttf", 11, bold=True)
        bbox = draw.textbbox((0, 0), badge, font=f_bdg)
        bw = bbox[2] - bbox[0] + 16
        draw.rounded_rectangle([x1 - bw - 14, y0 + 6, x1 - 12, y0 + 30], radius=6, fill="#ffffff")
        draw.text((x1 - bw - 6, y0 + 9), badge, fill=color, font=f_bdg)

        # Function Line
        draw.text((x0 + 16, y0 + 46), "Function / Method:", fill="#64748b", font=f_lbl)
        draw.text((x0 + 155, y0 + 46), method_name, fill="#0f172a", font=f_fn)

        # Bullet items
        for i, line in enumerate(bullets):
            draw.text((x0 + 16, y0 + 72 + i * 21), line, fill="#334155", font=f_txt)

    def draw_down_arrow(start_x, start_y, end_y, text=""):
        draw.line([start_x, start_y, start_x, end_y], fill="#2563eb", width=3)
        draw.polygon([(start_x - 6, end_y - 8), (start_x + 6, end_y - 8), (start_x, end_y + 2)], fill="#2563eb")
        if text:
            f_lbl = get_font("segoeuib.ttf", 11, bold=True)
            draw.text((start_x + 12, (start_y + end_y) / 2 - 8), text, fill="#2563eb", font=f_lbl)

    # Step 1: app.py
    draw_exec_box(
        [50, 115, 1450, 245],
        step_num="1",
        file_name="app.py",
        method_name="st.chat_input()  ->  main()",
        bullets=[
            "• User types natural language prompt into Streamlit chat bar (e.g. 'Order for summer vacations 2026').",
            "• app.py verifies authenticated session from config/auth_config.yaml and dispatches to RAG pipeline.",
            "• Adds user message to st.session_state.chat_messages for display in chat UI."
        ],
        color="#0284c7", badge="USER INPUT"
    )

    draw_down_arrow(750, 245, 275, "Calls Memory")

    # Step 2: src/memory/conversation_memory.py
    draw_exec_box(
        [50, 275, 1450, 410],
        step_num="2",
        file_name="src/memory/conversation_memory.py",
        method_name="st.session_state.memory.reformulate_query(prompt, llm_client)",
        bullets=[
            "• Inspects previous 6 dialogue turns for contextual references (pronouns like 'it', 'that order', 'the principal').",
            "• If context is required, invokes src/llm/client.py to rewrite into a standalone search query.",
            "• Output: 'What is the official college order regarding Declaration of Summer Vacations 2026?'"
        ],
        color="#7c3aed", badge="QUERY REFORMULATION"
    )

    draw_down_arrow(750, 410, 440, "Passes Standalone Query")

    # Step 3: src/retrieval/hybrid_search.py
    draw_exec_box(
        [50, 440, 1450, 605],
        step_num="3",
        file_name="src/retrieval/hybrid_search.py",
        method_name="retriever.retrieve(reformulated_query, top_k=30)",
        bullets=[
            "• Dense Branch: Calls src/indexing/vector_store.py -> searches output/vector_store/index.faiss (all-MiniLM-L6-v2).",
            "• Sparse Branch: Queries output/bm25_store/bm25_model.pkl for exact administrative tokens, order numbers & dates.",
            "• Reciprocal Rank Fusion (RRF): Combines dense and sparse scores with constant k=60 and alpha=0.5 weighting.",
            "• Output: Top-30 hybrid candidate passages, pre-filtered to Top-10 candidates."
        ],
        color="#059669", badge="HYBRID SEARCH (RRF)"
    )

    draw_down_arrow(750, 605, 635, "Passes Top-10 Candidates")

    # Step 4: src/retrieval/reranker.py
    draw_exec_box(
        [50, 635, 1450, 770],
        step_num="4",
        file_name="src/retrieval/reranker.py",
        method_name="reranker.rerank(query, raw_candidates, top_n=6)",
        bullets=[
            "• Feeds (Query, Passage) pairs simultaneously into cross-encoder/ms-marco-MiniLM-L-6-v2.",
            "• Computes bidirectional cross-attention across full token sequences to eliminate semantic drift.",
            "• Output: Top-6 most authoritative, relevant administrative clauses with document and page metadata."
        ],
        color="#d97706", badge="NEURAL RERANKING"
    )

    draw_down_arrow(750, 770, 800, "Passes Top-6 Clauses")

    # Step 5: src/llm/client.py
    draw_exec_box(
        [50, 800, 1450, 935],
        step_num="5",
        file_name="src/llm/client.py",
        method_name="llm_client.generate_rag_response(query, context_documents, ...)",
        bullets=[
            "• Connects to active backend from .env (Google Gemini gemini-3.5-flash-lite / Ollama llama3.1 / OpenAI gpt-4o-mini).",
            "• Injects grounded system prompt: answers strictly derived from provided administrative clauses, no hallucinations.",
            "• Mandatory source citations: attaches document title and page number (e.g. [Orders.pdf, Page 67])."
        ],
        color="#4f46e5", badge="GROUNDED GENERATION"
    )

    draw_down_arrow(750, 935, 965, "Delivers Answer & Citations")

    # Step 6: app.py (Output & Downloader)
    draw_exec_box(
        [50, 965, 1450, 1150],
        step_num="6",
        file_name="app.py",
        method_name="linkify_answer_citations() & render_citations_and_links()",
        bullets=[
            "• Formats assistant answer in Streamlit UI and converts document mentions into clickable citation badges.",
            "• Calls get_document_download_info(): for multi-page PDFs (Orders.pdf), invokes pypdf to extract ONLY the cited page",
            "  into static/extracted_pages/Orders_page_67.pdf (400 KB single page instead of the full 36 MB PDF).",
            "• Renders direct single-click browser download links on document titles and badges in the Streamlit web interface.",
            "• Updates conversation memory state: st.session_state.memory.add_turn(prompt, answer_text, sources)."
        ],
        color="#0284c7", badge="UI & DOWNLOAD DELIVERY"
    )

    # Footer note
    f_ft = get_font("segoeui.ttf", 11)
    draw.text((45, 1175), "Execution completed in 1.2s - 2.5s. All operations operate on pre-computed local stores with zero query-time re-indexing.", fill="#64748b", font=f_ft)

    img.save(output_path, "PNG")
    print(f"Generated File Execution Flow diagram: {output_path}")


if __name__ == "__main__":
    generate_stage1_diagram()
    generate_stage2_diagram()
    generate_file_execution_flow_diagram()
