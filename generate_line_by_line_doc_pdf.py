"""
generate_line_by_line_doc_pdf.py
Compiles comprehensive, file-by-file, line-by-line documentation
of the entire KGP Gyankosh codebase into an authoritative, publication-ready PDF.
"""

import os
import sys
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable, Image as RLImage
)
from reportlab.pdfgen import canvas
import create_diagrams


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and draw 'Page X of Y' headers and footers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Running Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "KGP Gyankosh: Complete Codebase & Line-by-Line Technical Documentation")
            self.drawRightString(612 - 54, 750, "Kashmir Government Polytechnic College, Srinagar")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, 742, 612 - 54, 742)

        # Running Footer (all pages)
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 45, 612 - 54, 45)
        self.drawString(54, 32, "Confidential — Kashmir Govt. Polytechnic College | IIT Patna Capstone")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 32, page_text)
        self.restoreState()


def build_pdf(filename="KGP_Gyankosh_Line_By_Line_Code_Documentation.pdf"):
    # Ensure diagrams are freshly generated
    diag1_path = os.path.join("static", "diagram_stage1_indexing.png")
    diag2_path = os.path.join("static", "diagram_stage2_retrieval.png")
    diag3_path = os.path.join("static", "diagram_file_execution_flow.png")

    if not os.path.exists(diag1_path) or not os.path.exists(diag2_path):
        create_diagrams.generate_stage1_diagram(diag1_path)
        create_diagrams.generate_stage2_diagram(diag2_path)
    if not os.path.exists(diag3_path):
        create_diagrams.generate_execution_flow_diagram(diag3_path)

    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=50,
        rightMargin=50,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Color Tokens
    c_primary = colors.HexColor("#091024")     # Midnight Navy
    c_accent = colors.HexColor("#0284c7")      # Institutional Cyan / Blue
    c_sub = colors.HexColor("#1e293b")         # Text dark slate
    c_border = colors.HexColor("#cbd5e1")      # Border light slate
    c_bg_light = colors.HexColor("#f8fafc")    # Off-white row background
    c_code_bg = colors.HexColor("#f1f5f9")     # Code block background
    c_code_text = colors.HexColor("#0f172a")

    # Typography Styles
    style_title = ParagraphStyle(
        'DocTitle', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=22, leading=26,
        textColor=c_primary, spaceAfter=4
    )
    style_sub = ParagraphStyle(
        'DocSub', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=11, leading=15,
        textColor=c_accent, spaceAfter=8
    )
    style_meta = ParagraphStyle(
        'DocMeta', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8.5, leading=12.5,
        textColor=colors.HexColor("#475569"), spaceAfter=10
    )
    style_h1 = ParagraphStyle(
        'Heading1_Custom', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=13, leading=17,
        textColor=c_primary, spaceBefore=12, spaceAfter=6, keepWithNext=True
    )
    style_h2 = ParagraphStyle(
        'Heading2_Custom', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=10.5, leading=14,
        textColor=c_accent, spaceBefore=9, spaceAfter=4, keepWithNext=True
    )
    style_h3 = ParagraphStyle(
        'Heading3_Custom', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=9, leading=12,
        textColor=c_primary, spaceBefore=6, spaceAfter=2, keepWithNext=True
    )
    style_body = ParagraphStyle(
        'Body_Custom', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8.5, leading=12.5,
        textColor=c_sub, spaceAfter=5
    )
    style_caption = ParagraphStyle(
        'Caption_Custom', parent=styles['Normal'],
        fontName='Helvetica-Oblique', fontSize=8, leading=11,
        textColor=colors.HexColor("#64748b"), alignment=1, spaceBefore=4, spaceAfter=8
    )
    style_code_box = ParagraphStyle(
        'CodeBox', parent=styles['Normal'],
        fontName='Courier', fontSize=7.2, leading=9.8,
        textColor=c_code_text
    )
    style_th = ParagraphStyle(
        'TH_Custom', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=8, leading=10.5,
        textColor=colors.white
    )
    style_td = ParagraphStyle(
        'TD_Custom', parent=styles['Normal'],
        fontName='Helvetica', fontSize=7.5, leading=10,
        textColor=c_sub
    )
    style_td_code = ParagraphStyle(
        'TD_Code', parent=styles['Normal'],
        fontName='Courier-Bold', fontSize=7, leading=9.5,
        textColor=colors.HexColor("#0369a1")
    )

    story = []

    def add_code_block(title, explanation, code_snippet):
        """Helper to create a structured code analysis block."""
        block = [
            Paragraph(f"<b>{title}</b>", style_h3),
            Paragraph(explanation, style_body),
            Spacer(1, 2)
        ]
        if code_snippet:
            formatted_code = code_snippet.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>&nbsp;&nbsp;")
            t_code = Table([[Paragraph(formatted_code, style_code_box)]], colWidths=[512])
            t_code.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), c_code_bg),
                ('BOX', (0,0), (-1,-1), 0.5, c_border),
                ('TOPPADDING', (0,0), (-1,-1), 5),
                ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                ('LEFTPADDING', (0,0), (-1,-1), 7),
                ('RIGHTPADDING', (0,0), (-1,-1), 7),
            ]))
            block.append(t_code)
            block.append(Spacer(1, 6))
        return block

    # =========================================================================
    # SECTION 1: COVER & EXECUTIVE SUMMARY
    # =========================================================================
    story.append(Paragraph("KGP Gyankosh: Complete Technical Documentation", style_title))
    story.append(Paragraph("Comprehensive File Directory, System Architecture, & Line-by-Line Code Breakdown", style_sub))
    story.append(Paragraph(
        "<b>Institution:</b> Kashmir Government Polytechnic College, Srinagar<br/>"
        "<b>Capstone Project:</b> IIT Patna — Advanced Enterprise RAG & Generative AI Platform<br/>"
        "<b>Scope of Document:</b> Exhaustive documentation of all codebase files, functional operations, internal logic, algorithms, and line-by-line code walk-through.",
        style_meta
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_accent, spaceBefore=2, spaceAfter=8))

    story.append(Paragraph("1. Executive Overview & Two-Stage System Architecture", style_h1))
    story.append(Paragraph(
        "<b>KGP Gyankosh</b> is an internal administrative intelligence assistant developed for Kashmir Government Polytechnic College, Srinagar. "
        "The system completely decouples the heavy offline indexing pipeline from the online interactive search engine: "
        "<br/><br/>"
        "• <b>Stage 1: Offline Batch Ingestion & Dual Indexing Pipeline</b>: Executed via <code>build_index.py</code>. "
        "Recursively crawls <code>data/</code>, calculates MD5 hashes for incremental change tracking, routes scanned PDFs through RapidOCR, "
        "splits documents into 500-character semantic chunks with 100-character overlap, and synchronizes a dense FAISS vector index (MiniLM embeddings) "
        "and a sparse BM25 inverted lexical index. "
        "<br/><br/>"
        "• <b>Stage 2: Online Multi-Turn Interactive RAG Application</b>: Executed via <code>app.py</code>. "
        "Delivers sub-second response times with zero query-time document re-indexing. Incorporates BCrypt role-based authentication, "
        "asynchronous background pre-warming, multi-turn sliding dialogue memory with query reformulation, Hybrid Reciprocal Rank Fusion (RRF, k=60), "
        "deep neural Cross-Encoder reranking (MS-MARCO), strict anti-hallucination prompting, and automated single-page order PDF extraction.",
        style_body
    ))
    story.append(Spacer(1, 6))

    # =========================================================================
    # SECTION 2: ARCHITECTURE DIAGRAMS
    # =========================================================================
    story.append(Paragraph("2. Architectural Workflows & Data Flow Diagrams", style_h1))
    story.append(Paragraph(
        "The following diagrams illustrate the offline document processing pipeline, online query flow, and file-by-file execution trace.",
        style_body
    ))

    if os.path.exists(diag1_path):
        story.append(RLImage(diag1_path, width=512, height=512 * (850 / 1500)))
        story.append(Paragraph("Figure 1: Stage 1 Ingestion, OCR, Chunking, and Dual-Indexing Pipeline (build_index.py)", style_caption))

    story.append(PageBreak())

    if os.path.exists(diag2_path):
        story.append(RLImage(diag2_path, width=512, height=512 * (850 / 1500)))
        story.append(Paragraph("Figure 2: Stage 2 Online Multi-Turn Query Retrieval, Neural Reranking, Grounding, and Order Download (app.py)", style_caption))

    if os.path.exists(diag3_path):
        story.append(Spacer(1, 8))
        story.append(RLImage(diag3_path, width=512, height=512 * (1220 / 1500)))
        story.append(Paragraph("Figure 3: Sequential File-by-File Code Execution Flow (app.py -> memory -> hybrid search -> reranker -> llm)", style_caption))

    story.append(PageBreak())

    # =========================================================================
    # SECTION 3: MASTER FILE INVENTORY TABLE
    # =========================================================================
    story.append(Paragraph("3. Master File Inventory & Architectural Role", style_h1))
    story.append(Paragraph(
        "Complete directory of all codebase modules, configuration files, and persistent artifact directories:",
        style_body
    ))

    file_table_data = [
        [Paragraph("File Path", style_th), Paragraph("Subsystem", style_th), Paragraph("Operational Function & Responsibility", style_th)],
        [Paragraph("app.py", style_td_code), Paragraph("Frontend / Application", style_td), Paragraph("Streamlit web interface, BCrypt login gate, aurora dark glassmorphism theme, background pre-warming, mode selector (RAG vs General AI), auto-scroll, and single-page PDF delivery.", style_td)],
        [Paragraph("build_index.py", style_td_code), Paragraph("Offline Pipeline", style_td), Paragraph("Command-line indexing script. MD5 incremental change detection, polymorphic document extraction, semantic chunking, and FAISS + BM25 disk persistence.", style_td)],
        [Paragraph("src/auth/authenticator.py", style_td_code), Paragraph("Authentication", style_td), Paragraph("Role-based access control (RBAC), salted BCrypt password hashing, and cached YAML credential store loader.", style_td)],
        [Paragraph("src/ingestion/document_loader.py", style_td_code), Paragraph("Ingestion Engine", style_td), Paragraph("Polymorphic document loader supporting DOCX, PDF, TXT, and scanned image formats. Integrates OCR routing.", style_td)],
        [Paragraph("src/ingestion/ocr_loader.py", style_td_code), Paragraph("OCR Engine", style_td), Paragraph("Pure-Python RapidOCR (ONNX Runtime) with optional Tesseract fallback. Disk-based page-level JSON cache.", style_td)],
        [Paragraph("src/ingestion/chunker.py", style_td_code), Paragraph("Chunking Engine", style_td), Paragraph("Recursive character text chunker splitting documents into 500-char blocks with 100-char overlap, preserving administrative clause boundaries.", style_td)],
        [Paragraph("src/indexing/embeddings.py", style_td_code), Paragraph("Embedding Layer", style_td), Paragraph("LangChain embeddings abstraction supporting local SentenceTransformers (all-MiniLM-L6-v2) and OpenAI (text-embedding-3-small).", style_td)],
        [Paragraph("src/indexing/vector_store.py", style_td_code), Paragraph("Storage Layer", style_td), Paragraph("Manages FAISS dense vector store, BM25Okapi lexical index, and MD5 catalog manifest serialization and deserialization.", style_td)],
        [Paragraph("src/retrieval/hybrid_search.py", style_td_code), Paragraph("Hybrid Retrieval", style_td), Paragraph("Combines FAISS dense vector search and BM25 sparse keyword matching using Reciprocal Rank Fusion (RRF, k=60, alpha=0.5).", style_td)],
        [Paragraph("src/retrieval/reranker.py", style_td_code), Paragraph("Neural Reranking", style_td), Paragraph("Deep cross-encoder model (cross-encoder/ms-marco-MiniLM-L-6-v2) performing full self-attention reranking on query-passage pairs.", style_td)],
        [Paragraph("src/memory/conversation_memory.py", style_td_code), Paragraph("Conversational Memory", style_td), Paragraph("Maintains sliding window of last 6 turns and uses LLM for multi-turn contextual query reformulation.", style_td)],
        [Paragraph("src/llm/client.py", style_td_code), Paragraph("LLM Interface", style_td), Paragraph("Multi-backend client (Google Gemini Cloud default, local Ollama, OpenAI) with automatic cloud failover and strict anti-hallucination prompting.", style_td)],
        [Paragraph(".env & .env.example", style_td_code), Paragraph("Configuration", style_td), Paragraph("Environment parameters: active LLM provider, API keys, models, chunk sizes, reranking thresholds, and paths.", style_td)],
        [Paragraph(".streamlit/config.toml", style_td_code), Paragraph("UI Config", style_td), Paragraph("Enables static document serving, headless mode, custom dark theme colors, and browser settings.", style_td)],
        [Paragraph("config/auth_config.yaml", style_td_code), Paragraph("Credential Store", style_td), Paragraph("User credential store containing BCrypt-hashed passwords, emails, full names, and roles (admin, staff, faculty).", style_td)],
        [Paragraph("requirements.txt", style_td_code), Paragraph("Dependencies", style_td), Paragraph("Pinned library dependencies: streamlit, langchain, faiss-cpu, rank-bm25, sentence-transformers, google-genai, rapidocr, reportlab, etc.", style_td)],
        [Paragraph("data/ (word, pdf)", style_td_code), Paragraph("Data Stores", style_td), Paragraph("Raw document repository: 1,157 official polytechnic DOCX circulars + 85-page scanned Orders.pdf archive.", style_td)],
        [Paragraph("output/ (vector, bm25, ocr)", style_td_code), Paragraph("Index Stores", style_td), Paragraph("Serialized indexing artifacts: index.faiss, index.pkl, bm25_model.pkl, bm25_corpus.pkl, index_manifest.json, and OCR cache.", style_td)],
    ]

    t_inv = Table(file_table_data, colWidths=[120, 95, 297])
    t_inv.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_light]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_inv)

    story.append(PageBreak())

    # =========================================================================
    # SECTION 4: FILE-BY-FILE & LINE-BY-LINE DETAILED CODE ANALYSIS
    # =========================================================================
    story.append(Paragraph("4. Exhaustive File-by-File & Line-by-Line Code Breakdown", style_h1))
    story.append(Paragraph(
        "The following sections provide an exhaustive, line-by-line and block-by-block technical analysis "
        "of every Python module and configuration file comprising the system.",
        style_body
    ))

    # --- 4.1 app.py ---
    story.append(Paragraph("4.1. app.py — Streamlit Frontend & Enterprise Application Engine", style_h2))
    story.append(Paragraph(
        "<b>app.py</b> is the primary web application entry point (1,244 lines). It orchestrates user authentication, "
        "custom glassmorphic styling, background model warming, conversational state management, and real-time query execution.",
        style_body
    ))
    story.extend(add_code_block(
        "Lines 1-40: Initialization & Page Configuration",
        "Configures page title, institutional icon (🏛️), wide layout, and expanded sidebar. "
        "Ensures project root is prepended to <code>sys.path</code> and loads <code>.env</code>.",
        "st.set_page_config(\n    page_title=\"KGP Gyankosh - Administrative Intelligence Assistant\",\n    page_icon=\"🏛️\",\n    layout=\"wide\",\n    initial_sidebar_state=\"expanded\"\n)\nload_dotenv(os.path.join(PROJECT_ROOT, \".env\"))"
    ))
    story.extend(add_code_block(
        "Lines 50-95: Campus Background & Radiant Glassmorphism Atmosphere",
        "Loads <code>static/campus_bg.jpg</code> as base64 and layers radiant cyber-cyan (#38bdf8) and royal violet (#818cf8) "
        "radial gradients over a deep obsidian base (#070b14) to create a high-contrast enterprise theme.",
        "def load_campus_bg_base64() -> str:\n    bg_path = os.path.join(PROJECT_ROOT, \"static\", \"campus_bg.jpg\")\n    with open(bg_path, \"rb\") as f:\n        return base64.b64encode(f.read()).decode(\"utf-8\")"
    ))
    story.extend(add_code_block(
        "Lines 96-550: Scoped CSS & Non-Blocking Selectbox Dropdown Engine",
        "Scoped text inputs to avoid breaking Streamlit's internal BaseWeb selectbox dropdown. "
        "Applies frosted glass styling to dropdown menus with high z-index (99999999), custom radio toggle pills, and vibrant citation badges.",
        "div[data-baseweb=\"select\"] > div {\n    background: linear-gradient(145deg, rgba(14,22,44,0.95), rgba(18,28,56,0.9)) !important;\n    border: 1px solid rgba(56, 189, 248, 0.4) !important;\n}\ndiv[data-baseweb=\"popover\"] { z-index: 99999999 !important; }"
    ))
    story.extend(add_code_block(
        "Lines 565-610: Dynamic Single-Page PDF Order Extraction (pypdf)",
        "When an order is cited (e.g. <code>Orders.pdf, Page 67</code>), intercepts the request and extracts ONLY that single page "
        "into <code>static/extracted_pages/Orders_page_67.pdf</code> using <code>pypdf</code>. "
        "The user downloads a lightweight ~400 KB PDF instead of the full 36 MB orders catalog.",
        "from pypdf import PdfReader, PdfWriter\nreader = PdfReader(file_path)\nwriter = PdfWriter()\nwriter.add_page(reader.pages[page_num - 1])\nwith open(page_filepath, \"wb\") as f_out:\n    writer.write(f_out)"
    ))
    story.extend(add_code_block(
        "Lines 710-755: Cached System Resource Initialization",
        "Decorated with <code>@st.cache_resource</code>. Loads FAISS index, BM25 store, manifest catalog, reranker, "
        "and default Gemini client once into memory, guaranteeing 0 query-time indexing overhead.",
        "@st.cache_resource(show_spinner=\"⚡ Initializing AI Knowledge Base...\")\ndef initialize_system():\n    embeddings = get_embeddings_model()\n    vector_store = load_vector_store(vector_store_dir, embeddings)\n    bm25_model, bm25_corpus = load_bm25_store(bm25_store_dir)\n    return {\"retriever\": retriever, \"reranker\": reranker, ...}"
    ))
    story.extend(add_code_block(
        "Lines 820-840: Background Thread Model Pre-Warming",
        "During initial landing page view (while entering credentials), spawns a daemon thread running <code>initialize_system()</code>. "
        "By the time credentials are submitted, heavy models are already warm in memory, making login instant.",
        "if \"prewarm_started\" not in st.session_state:\n    st.session_state.prewarm_started = True\n    import threading\n    threading.Thread(target=initialize_system, daemon=True).start()"
    ))
    story.extend(add_code_block(
        "Lines 860-880: Memoized LLM Client Factory (_LLM_CLIENTS)",
        "Caches initialized LLM instances by <code>(provider, model_name)</code> key. "
        "Eliminates module reloads and allows instant, 0 ms model switching in the dropdown.",
        "_LLM_CLIENTS: Dict[str, LLMClient] = {}\ndef get_cached_llm_client(provider: str, model_name: str) -> LLMClient:\n    key = f\"{provider}:{model_name}\"\n    if key in _LLM_CLIENTS:\n        return _LLM_CLIENTS[key]\n    client = LLMClient(provider=provider, model=model_name)\n    _LLM_CLIENTS[key] = client\n    return client"
    ))
    story.extend(add_code_block(
        "Lines 1150-1240: Dual-Mode Execution, Auto-Scroll, & Citation Rendering",
        "Toggles between <b>College Records (RAG)</b> and <b>General AI Chat</b>. "
        "Injects smooth auto-scroll JS so queries and answers appear in view without manual scrolling. Citations are linkified to direct download badges.",
        "if prompt := st.chat_input(input_placeholder):\n    st.session_state.chat_messages.append({\"role\": \"user\", \"content\": prompt})\n    scroll_to_bottom()\n    candidates = retriever.retrieve(reformulated_query)\n    reranked = reranker.rerank(reformulated_query, candidates)\n    result = llm_client.generate_answer(prompt, top_documents)"
    ))

    story.append(PageBreak())

    # --- 4.2 build_index.py ---
    story.append(Paragraph("4.2. build_index.py — Stage 1 Offline Batch Ingestion & Indexing Engine", style_h2))
    story.append(Paragraph(
        "<b>build_index.py</b> is the offline batch processing engine (265 lines). "
        "It reads documents from <code>data/</code>, computes MD5 file hashes for incremental detection, "
        "extracts text, chunks semantic segments, and creates the FAISS vector database and BM25 lexical index.",
        style_body
    ))
    story.extend(add_code_block(
        "Lines 40-75: Incremental MD5 Change Detection",
        "Calculates hashlib.md5 hashes for every file in data/. Compares against index_manifest.json. "
        "Files with matching hashes are skipped, enabling incremental updates in under 2 seconds.",
        "def compute_file_md5(file_path: str) -> str:\n    hasher = hashlib.md5()\n    with open(file_path, \"rb\") as f:\n        for chunk in iter(lambda: f.read(65536), b\"\"):\n            hasher.update(chunk)\n    return hasher.hexdigest()"
    ))
    story.extend(add_code_block(
        "Lines 80-160: Batch Processing & Document Chunking Coordinator",
        "Discovers .docx, .pdf, .txt files, invokes DocumentLoader for parsing, "
        "passes extracted documents to Chunker (500 chars, 100 overlap), and aggregates chunks.",
        "loader = DocumentLoader()\nchunker = DocumentChunker(chunk_size=500, chunk_overlap=100)\nall_chunks = []\nfor file_path in discovered_files:\n    docs = loader.load(file_path)\n    chunks = chunker.split_documents(docs)\n    all_chunks.extend(chunks)"
    ))
    story.extend(add_code_block(
        "Lines 165-240: Dual-Index Serialization (FAISS + BM25)",
        "Computes dense embeddings via SentenceTransformers and builds FAISS index. "
        "Tokenizes corpus and serializes BM25Okapi model to disk along with manifest catalog.",
        "embeddings = get_embeddings_model()\nvector_store = FAISS.from_documents(all_chunks, embeddings)\nsave_vector_store(vector_store, vector_store_dir)\nsave_bm25_store(all_chunks, bm25_store_dir)\nsave_manifest(manifest_path, manifest_data)"
    ))

    # --- 4.3 src/auth/authenticator.py ---
    story.append(Paragraph("4.3. src/auth/authenticator.py — Access Control & BCrypt Security Gate", style_h2))
    story.append(Paragraph(
        "<b>src/auth/authenticator.py</b> enforces internal role-based access control (RBAC). "
        "Uses bcrypt-hashed passwords and manages encrypted cookie sessions via streamlit-authenticator.",
        style_body
    ))
    story.extend(add_code_block(
        "Lines 18-35: Password Hashing & Cached Config Loader",
        "Provides bcrypt hashing helper. Uses @st.cache_data to prevent repetitive disk I/O on every Streamlit rerun.",
        "def hash_password(password: str) -> str:\n    return bcrypt.hashpw(password.encode(\"utf-8\"), bcrypt.gensalt()).decode(\"utf-8\")\n\n@st.cache_data(show_spinner=False)\ndef load_auth_config(config_path=None) -> dict:\n    with open(path, \"r\", encoding=\"utf-8\") as f:\n        return yaml.safe_load(f)"
    ))

    story.append(PageBreak())

    # --- 4.4 Ingestion Subsystem ---
    story.append(Paragraph("4.4. Ingestion Subsystem: document_loader.py, ocr_loader.py, chunker.py", style_h2))
    story.extend(add_code_block(
        "src/ingestion/document_loader.py: Polymorphic Ingestion",
        "Detects file format. Extracts text and tables from Word (.docx) using python-docx. "
        "Extracts native PDF text using pypdf. Routes scanned / image pages to OCRLoader.",
        "if ext == \".docx\": return self._load_docx(file_path)\nelif ext == \".pdf\": return self._load_pdf(file_path)\nelif ext in [\".png\", \".jpg\", \".jpeg\"]: return self.ocr_loader.load(file_path)"
    ))
    story.extend(add_code_block(
        "src/ingestion/ocr_loader.py: RapidOCR Engine & Persistent Page Caching",
        "Uses Pure-Python RapidOCR (ONNX Runtime). Maintains persistent JSON cache in output/ocr_cache/ "
        "so the 85-page Orders.pdf is OCR-processed only once across rebuilds.",
        "cache_key = f\"{os.path.basename(file_path)}_pages.json\"\nif os.path.exists(cache_file):\n    return self._load_cache(cache_file)\n# Extract and run RapidOCR per page\nocr = RapidOCR()\nresult, _ = ocr(page_image)"
    ))
    story.extend(add_code_block(
        "src/ingestion/chunker.py: Recursive Character Splitting",
        "Uses LangChain RecursiveCharacterTextSplitter with separators [\"\\n\\n\", \"\\n\", \". \", \" \"]. "
        "Guarantees that administrative clauses, fee tables, and order sections remain intact.",
        "splitter = RecursiveCharacterTextSplitter(\n    chunk_size=500, chunk_overlap=100,\n    separators=[\"\\n\\n\", \"\\n\", \". \", \" \"]\n)"
    ))

    # --- 4.5 Indexing Subsystem ---
    story.append(Paragraph("4.5. Indexing Subsystem: embeddings.py, vector_store.py", style_h2))
    story.extend(add_code_block(
        "src/indexing/embeddings.py: Unified Embeddings Provider",
        "Switchable via EMBEDDING_PROVIDER (.env). Default: local sentence-transformers (all-MiniLM-L6-v2) "
        "running on CPU with normalized vectors for cosine distance.",
        "class DirectSentenceTransformersEmbeddings(Embeddings):\n    def embed_documents(self, texts):\n        return self.model.encode(texts, normalize_embeddings=True).tolist()\n    def embed_query(self, text):\n        return self.model.encode([text], normalize_embeddings=True)[0].tolist()"
    ))
    story.extend(add_code_block(
        "src/indexing/vector_store.py: FAISS Vector Store & BM25 Storage",
        "Manages index.faiss persistence and BM25Okapi tokenized index serialization via pickle.",
        "vector_store = FAISS.load_local(store_dir, embeddings, allow_dangerous_deserialization=True)\nwith open(model_path, \"wb\") as f: pickle.dump(bm25_model, f)"
    ))

    story.append(PageBreak())

    # --- 4.6 Retrieval Subsystem ---
    story.append(Paragraph("4.6. Retrieval Subsystem: hybrid_search.py, reranker.py", style_h2))
    story.extend(add_code_block(
        "src/retrieval/hybrid_search.py: Reciprocal Rank Fusion (RRF)",
        "Retrieves top-k from FAISS and top-k from BM25. Combines ranks using RRF formula: "
        "Score(d) = sum(1 / (k + rank_i)). Ensures exact order numbers and semantic concepts both score highly.",
        "def rrf(dense_results, sparse_results, k=60, alpha=0.5):\n    rrf_scores = defaultdict(float)\n    for rank, (doc, _) in enumerate(dense_results):\n        rrf_scores[doc.metadata['chunk_id']] += alpha * (1.0 / (k + rank + 1))\n    for rank, (doc, _) in enumerate(sparse_results):\n        rrf_scores[doc.metadata['chunk_id']] += (1 - alpha) * (1.0 / (k + rank + 1))\n    return sorted(candidates, key=lambda x: rrf_scores[x.metadata['chunk_id']], reverse=True)"
    ))
    story.extend(add_code_block(
        "src/retrieval/reranker.py: Deep Neural Cross-Encoder Reranking",
        "Uses cross-encoder/ms-marco-MiniLM-L-6-v2. Jointly evaluates [query, passage] through bidirectional "
        "transformer attention layers, re-scoring top-10 candidates down to top-6 authoritative passages.",
        "pairs = [[query, doc.page_content] for doc, _ in candidates]\nscores = model.predict(pairs)\nfor (doc, _), score in zip(candidates, scores):\n    doc.metadata[\"rerank_score\"] = float(score)\n    reranked.append((doc, float(score)))\nreturn reranked[:limit]"
    ))

    # --- 4.7 Memory & LLM Subsystem ---
    story.append(Paragraph("4.7. Intelligence Subsystem: conversation_memory.py, client.py", style_h2))
    story.extend(add_code_block(
        "src/memory/conversation_memory.py: Sliding Window Buffer & Query Reformulation",
        "Stores last 6 conversation turns. When follow-up questions are submitted (e.g. 'When does it expire?'), "
        "uses the LLM to resolve pronoun references into a standalone query for retrieval.",
        "prompt = REFORMULATION_PROMPT.format(history=recent_history, question=query)\nstandalone_query = llm_client.generate_raw(prompt).strip()"
    ))
    story.extend(add_code_block(
        "src/llm/client.py: Switchable Backend, Anti-Hallucination & Auto-Fallback",
        "Enforces strict administrative grounding prompt. If context does not answer the query, replies: "
        "'I could not find this information in the available official documents.' "
        "Includes automatic fallback to Google Gemini Cloud if local Ollama runs out of RAM.",
        "ADMIN_SYSTEM_PROMPT = \"\"\"You are KGP Gyankosh... Ground answers STRICTLY in context. If unmentioned, state 'I could not find this information in official documents.' Conclude with Sources: - [Doc, Page X]\"\"\"\n\n# Auto-fallback to Gemini if local Ollama fails (out of RAM)\nif self.provider == 'ollama' and google_key:\n    fallback_llm = ChatGoogleGenerativeAI(model=gemini_model, ...)"
    ))

    # =========================================================================
    # SECTION 5: CONCLUSION & RUNNING COMMANDS
    # =========================================================================
    story.append(Spacer(1, 10))
    story.append(Paragraph("5. Deployment & Execution Commands", style_h1))
    story.append(Paragraph(
        "<b>1. Run the interactive Streamlit assistant:</b><br/>"
        "<code>streamlit run app.py</code><br/><br/>"
        "<b>2. Rebuild offline document index from data/:</b><br/>"
        "<code>python build_index.py</code><br/><br/>"
        "<b>3. Run automated retrieval test harness:</b><br/>"
        "<code>python scratch_test_queries.py</code><br/><br/>"
        "<b>4. Re-compile this technical documentation PDF:</b><br/>"
        "<code>python generate_line_by_line_doc_pdf.py</code>",
        style_body
    ))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated documentation PDF: {filename} ({os.path.getsize(filename):,} bytes)")


if __name__ == "__main__":
    out_file = "KGP_Gyankosh_Line_By_Line_Code_Documentation.pdf"
    if len(sys.argv) > 1:
        out_file = sys.argv[1]
    build_pdf(out_file)
