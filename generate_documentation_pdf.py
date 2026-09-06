"""
generate_documentation_pdf.py
Compiles the complete KGP Gyankosh codebase files inventory, usage guide, 
and diagrammatic workflow into a high-grade publication-ready PDF document.
Embeds high-resolution visual flowcharts for Stage 1 and Stage 2 architectures.
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
    """Canvas that performs two passes to compute total page count for footer numbering."""
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
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "KGP Gyankosh: Complete Files Reference & Diagrammatic Architecture")
            self.drawRightString(612 - 54, 750, "Kashmir Government Polytechnic College")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, 742, 612 - 54, 742)

        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 45, 612 - 54, 45)
        self.drawString(54, 32, "Confidential — Internal Administrative Intelligence System | IIT Patna Capstone")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 32, page_text)
        self.restoreState()


def build_pdf(filename="KGP_Gyankosh_System_Architecture_and_Files_Reference.pdf"):
    # Ensure diagrams are freshly generated
    diag1_path = os.path.join("static", "diagram_stage1_indexing.png")
    diag2_path = os.path.join("static", "diagram_stage2_retrieval.png")
    create_diagrams.generate_stage1_diagram(diag1_path)
    create_diagrams.generate_stage2_diagram(diag2_path)

    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor("#0f172a")     # Deep Slate
    c_accent = colors.HexColor("#0284c7")      # Institutional Blue
    c_sub = colors.HexColor("#334155")         # Body dark
    c_border = colors.HexColor("#cbd5e1")      # Border gray
    c_bg_light = colors.HexColor("#f8fafc")    # Light background

    # Typography styles
    style_title = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=c_primary,
        alignment=0,
        spaceAfter=3
    )
    
    style_sub = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=c_accent,
        spaceAfter=10
    )

    style_meta = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=10
    )

    style_h1 = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=c_primary,
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )

    style_h2 = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=c_accent,
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True
    )

    style_body = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12.5,
        textColor=c_sub,
        spaceAfter=5
    )

    style_caption = ParagraphStyle(
        'Caption_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748b"),
        alignment=1,
        spaceBefore=4,
        spaceAfter=10
    )

    style_th = ParagraphStyle(
        'TH_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10.5,
        textColor=colors.white
    )

    style_td = ParagraphStyle(
        'TD_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        textColor=c_sub
    )

    style_td_code = ParagraphStyle(
        'TD_Code',
        parent=styles['Normal'],
        fontName='Courier-Bold',
        fontSize=7,
        leading=9.5,
        textColor=colors.HexColor("#0369a1")
    )

    story = []

    # --------------------------------------------------------------------------
    # COVER / HEADER SECTION
    # --------------------------------------------------------------------------
    story.append(Paragraph("KGP Gyankosh: Enterprise Knowledge Assistant", style_title))
    story.append(Paragraph("Complete Codebase File Directory, Usage Specifications & Diagrammatic Architecture Workflows", style_sub))
    story.append(Paragraph(
        "<b>Institution:</b> Kashmir Government Polytechnic College, Srinagar<br/>"
        "<b>Project:</b> IIT Patna — Generative AI & Agentic AI Capstone (Advanced RAG)<br/>"
        "<b>Active Dataset:</b> 1,157 College DOCX Circulars + 85-Page Orders Book (PDF) | 8,166 Indexed Clauses",
        style_meta
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_accent, spaceBefore=2, spaceAfter=10))

    # --------------------------------------------------------------------------
    # 1. EXECUTIVE OVERVIEW
    # --------------------------------------------------------------------------
    story.append(Paragraph("1. Executive Overview & System Architecture", style_h1))
    story.append(Paragraph(
        "<b>KGP Gyankosh</b> is an internal administrative intelligence system designed specifically for authorized staff, "
        "department heads, and academic officers of Kashmir Government Polytechnic College, Srinagar. "
        "The system operates on a strict <b>Two-Stage Architecture</b>, completely decoupling the offline document ingestion and "
        "indexing pipeline from the online conversational query interface. "
        "Heavy operations (OCR scanning, text chunking, and dense embedding generation) occur once offline via <code>build_index.py</code>, "
        "ensuring sub-second search speeds and zero query-time indexing overhead in the online application.",
        style_body
    ))
    story.append(Spacer(1, 4))

    # --------------------------------------------------------------------------
    # 2. DIAGRAMMATIC WORKFLOW 1: STAGE 1 OFFLINE INGESTION & INDEXING
    # --------------------------------------------------------------------------
    story.append(Paragraph("2. Diagrammatic Workflow: Stage 1 Ingestion & Indexing Pipeline", style_h1))
    story.append(Paragraph(
        "The diagram below illustrates the complete offline ingestion and dual-indexing pipeline. "
        "Documents in <code>data/</code> are discovered, checked incrementally via MD5 hashes, parsed through native text or RapidOCR engines, "
        "split into semantic chunks, and concurrently indexed into a dense FAISS vector store and sparse BM25 inverted lexical index.",
        style_body
    ))

    if os.path.exists(diag1_path):
        img_w = 504
        img_h = 504 * (850 / 1500)
        story.append(RLImage(diag1_path, width=img_w, height=img_h))
        story.append(Paragraph("Figure 1: Stage 1 Offline Document Ingestion, OCR, Chunking, and Dual-Indexing Pipeline (build_index.py)", style_caption))

    story.append(PageBreak())

    # --------------------------------------------------------------------------
    # 3. DIAGRAMMATIC WORKFLOW 2: STAGE 2 ONLINE RETRIEVAL & SERVING
    # --------------------------------------------------------------------------
    story.append(Paragraph("3. Diagrammatic Workflow: Stage 2 Online Query & Retrieval Engine", style_h1))
    story.append(Paragraph(
        "The diagram below outlines the end-to-end real-time query execution workflow. "
        "When an authenticated staff member submits a natural language question, the system tracks conversational history, "
        "executes Hybrid Search (FAISS + BM25 via Reciprocal Rank Fusion), re-ranks the top candidates using a deep neural Cross-Encoder, "
        "and injects the most authoritative clauses into the LLM with strict grounding constraints. "
        "Multi-page order citations are dynamically extracted and served as single-page order PDFs upon click.",
        style_body
    ))

    if os.path.exists(diag2_path):
        img_w = 504
        img_h = 504 * (850 / 1500)
        story.append(RLImage(diag2_path, width=img_w, height=img_h))
        story.append(Paragraph("Figure 2: Stage 2 Online Multi-Turn Query Retrieval, Neural Reranking, Grounding, and Single-Page Order Download (app.py)", style_caption))

    story.append(PageBreak())

    # --------------------------------------------------------------------------
    # 4. DIAGRAMMATIC WORKFLOW 3: FILE-BY-FILE CODE EXECUTION SEQUENCE
    # --------------------------------------------------------------------------
    diag3_path = os.path.join("static", "diagram_file_execution_flow.png")
    story.append(Paragraph("4. Diagrammatic Workflow: File-by-File Code Execution Trace", style_h1))
    story.append(Paragraph(
        "The diagram below traces the exact file-by-file execution sequence that runs when a user enters a query in the chat interface. "
        "It illustrates which Python module, function, and underlying index file executes at each step in the pipeline.",
        style_body
    ))

    if os.path.exists(diag3_path):
        img_w = 504
        img_h = 504 * (1220 / 1500)
        story.append(RLImage(diag3_path, width=img_w, height=img_h))
        story.append(Paragraph("Figure 3: Sequential File-by-File Code Execution Trace (app.py -> memory -> hybrid search -> reranker -> llm -> app.py)", style_caption))

    story.append(PageBreak())

    # --------------------------------------------------------------------------
    # 5. MASTER FILE INVENTORY TABLE
    # --------------------------------------------------------------------------
    story.append(Paragraph("5. Master File Inventory & Usage Specifications", style_h1))
    story.append(Paragraph(
        "The following master reference table documents every file, module, configuration, and data store comprising KGP Gyankosh.",
        style_body
    ))

    file_rows = [
        [Paragraph("File Path", style_th), Paragraph("Subsystem / Category", style_th), Paragraph("Operational Function & Purpose", style_th)],
        
        # Root Applications & Config
        [Paragraph("app.py", style_td_code), 
         Paragraph("Web Application (Streamlit)", style_td), 
         Paragraph("Main web application. Renders glassmorphic UI, BCrypt login gate, operational mode toggle (RAG vs General AI), multi-turn chat memory, and direct single-page order PDF downloads.", style_td)],
        
        [Paragraph("build_index.py", style_td_code), 
         Paragraph("Offline Indexing Pipeline", style_td), 
         Paragraph("Command-line batch script. Discovers files in data/, computes MD5 hashes for incremental indexing, runs text extraction and OCR, splits chunks, and serializes FAISS & BM25 indices.", style_td)],

        [Paragraph(".env", style_td_code), 
         Paragraph("System Configuration", style_td), 
         Paragraph("Active environment configuration. Sets LLM provider (Ollama, Gemini, OpenAI), API keys, DATA_DIR=data, embedding models, reranker settings, and retrieval thresholds.", style_td)],

        [Paragraph(".env.example", style_td_code), 
         Paragraph("Config Template", style_td), 
         Paragraph("Reference template showing all environment variables with explanations for evaluators and developers.", style_td)],

        [Paragraph(".streamlit/config.toml", style_td_code), 
         Paragraph("Streamlit Server Config", style_td), 
         Paragraph("Enables static file serving (enableStaticServing=true), headless mode, dark institutional color palette, and disables unnecessary file watchers.", style_td)],

        [Paragraph("config/auth_config.yaml", style_td_code), 
         Paragraph("Authentication Store", style_td), 
         Paragraph("Active role-based access control (RBAC) credential store. Contains user profiles, salted BCrypt password hashes, and access roles (admin, staff, faculty). Git-ignored for security.", style_td)],

        [Paragraph("config/auth_config.yaml.example", style_td_code), 
         Paragraph("Auth Template", style_td), 
         Paragraph("Reference credential template with pre-computed BCrypt hashes for default evaluation accounts.", style_td)],

        [Paragraph("requirements.txt", style_td_code), 
         Paragraph("Dependencies", style_td), 
         Paragraph("Pinned Python dependencies: streamlit, langchain, faiss-cpu, rank-bm25, sentence-transformers, google-genai, rapidocr-onnxruntime, pypdf, python-docx, reportlab, etc.", style_td)],

        [Paragraph("scratch_test_queries.py", style_td_code), 
         Paragraph("Test Harness", style_td), 
         Paragraph("Standalone test harness validating FAISS vector retrieval, BM25 keyword search, and cross-encoder reranking accuracy across sample administrative questions.", style_td)],

        [Paragraph("create_diagrams.py", style_td_code), 
         Paragraph("Visual Graphics Engine", style_td), 
         Paragraph("Python script using Pillow and TrueType fonts to generate crystal-clear, high-resolution vector workflow diagrams for Stage 1 and Stage 2 architectures.", style_td)],

        [Paragraph("generate_documentation_pdf.py", style_td_code), 
         Paragraph("PDF Documentation Engine", style_td), 
         Paragraph("ReportLab compilation script that generates this publication-grade PDF documentation report with embedded diagrams and comprehensive file tables.", style_td)],

        # Ingestion Subsystem
        [Paragraph("src/ingestion/document_loader.py", style_td_code), 
         Paragraph("Document Ingestion", style_td), 
         Paragraph("Polymorphic document loader. Automatically detects file types (.docx, .pdf, .txt, .png/.jpg) and routes scanned document pages to the OCR engine.", style_td)],

        [Paragraph("src/ingestion/ocr_loader.py", style_td_code), 
         Paragraph("Scanned OCR Engine", style_td), 
         Paragraph("Optical Character Recognition engine using Pure-Python RapidOCR (ONNX Runtime) with optional Tesseract fallback. Manages persistent OCR caching in output/ocr_cache/.", style_td)],

        [Paragraph("src/ingestion/chunker.py", style_td_code), 
         Paragraph("Text Chunking", style_td), 
         Paragraph("Recursive character text splitter. Divides text into 500-character chunks with 100-character overlap while preserving paragraph and administrative clause boundaries.", style_td)],

        # Indexing Subsystem
        [Paragraph("src/indexing/embeddings.py", style_td_code), 
         Paragraph("Embedding Abstraction", style_td), 
         Paragraph("Unified embedding interface supporting local HuggingFace SentenceTransformers (all-MiniLM-L6-v2) and cloud OpenAI embeddings (text-embedding-3-small).", style_td)],

        [Paragraph("src/indexing/vector_store.py", style_td_code), 
         Paragraph("FAISS Vector Store", style_td), 
         Paragraph("FAISS vector store manager. Handles dense vector indexing, cosine similarity persistence, on-disk serialization (index.faiss, index.pkl), and fast index loading.", style_td)],

        # Retrieval Subsystem
        [Paragraph("src/retrieval/hybrid_search.py", style_td_code), 
         Paragraph("Hybrid Retrieval & RRF", style_td), 
         Paragraph("Orchestrates dual retrieval: FAISS dense semantic search and BM25 sparse keyword search. Combines ranked lists using Reciprocal Rank Fusion (RRF, k=60, alpha=0.5).", style_td)],

        [Paragraph("src/retrieval/reranker.py", style_td_code), 
         Paragraph("Neural Cross-Encoder", style_td), 
         Paragraph("Deep self-attention reranker (cross-encoder/ms-marco-MiniLM-L-6-v2). Evaluates query-passage token interactions to filter top-10 candidates down to top-6 authoritative clauses.", style_td)],

        # Memory & LLM
        [Paragraph("src/memory/conversation_memory.py", style_td_code), 
         Paragraph("Conversational Memory", style_td), 
         Paragraph("Maintains multi-turn dialogue history up to 6 turns and uses the LLM to rewrite ambiguous follow-up questions ('What is the deadline for it?') into standalone queries.", style_td)],

        [Paragraph("src/llm/client.py", style_td_code), 
         Paragraph("Switchable LLM Engine", style_td), 
         Paragraph("Unified inference client supporting Google Gemini (default), local Ollama (Llama 3.1), and OpenAI. Injects anti-hallucination guardrails and enforces mandatory citations.", style_td)],

        [Paragraph("src/auth/authenticator.py", style_td_code), 
         Paragraph("Authentication Gate", style_td), 
         Paragraph("Wraps streamlit-authenticator to deliver BCrypt cookie session management, secure login forms, and role verification without storing plaintext passwords.", style_td)],

        # Data & Output Stores
        [Paragraph("data/word/ (*.docx)", style_td_code), 
         Paragraph("Active Administrative Data", style_td), 
         Paragraph("Contains 1,157 official college Word circulars, notices, committee orders, fee schedules, and leave sanctions issued by Kashmir Government Polytechnic College.", style_td)],

        [Paragraph("data/pdf/Orders.pdf", style_td_code), 
         Paragraph("Historical Orders Book", style_td), 
         Paragraph("85-page comprehensive official scanned orders compilation covering administrative decisions, transfer sanctions, disciplinary actions, and committee notifications.", style_td)],

        [Paragraph("output/index_manifest.json", style_td_code), 
         Paragraph("Indexing Catalog", style_td), 
         Paragraph("Persistent JSON catalog storing file paths, MD5 file hashes, chunk counts, and timestamps for all 1,158 indexed files. Enables instantaneous incremental updates.", style_td)],

        [Paragraph("output/ocr_cache/Orders_pdf_pages.json", style_td_code), 
         Paragraph("OCR Cache", style_td), 
         Paragraph("Pre-computed OCR transcriptions for all 85 pages of Orders.pdf. Eliminates redundant OCR processing on index rebuilds.", style_td)],

        [Paragraph("output/vector_store/ (faiss, pkl)", style_td_code), 
         Paragraph("Dense Vector Index", style_td), 
         Paragraph("Pre-computed FAISS vector index (8,166 embedding vectors) and pickled document metadata chunk store for rapid similarity retrieval.", style_td)],

        [Paragraph("output/bm25_store/ (model, corpus)", style_td_code), 
         Paragraph("Lexical BM25 Index", style_td), 
         Paragraph("Serialized BM25 token frequencies, inverse document frequency statistics, and corpus pickle for exact administrative code and number matching.", style_td)],

        [Paragraph("static/extracted_pages/ (*.pdf)", style_td_code), 
         Paragraph("Single-Page PDF Cache", style_td), 
         Paragraph("Pre-extracted lightweight single-page PDFs for each order in Orders.pdf (e.g. Orders_page_67.pdf). Served directly when users click a cited order in the app.", style_td)],

        [Paragraph("static/docs", style_td_code), 
         Paragraph("Static Serving Junction", style_td), 
         Paragraph("Filesystem junction linking static/docs directly to data/ to allow Streamlit's web server to serve document downloads directly.", style_td)]
    ]

    t_files = Table(file_rows, colWidths=[125, 110, 269])
    t_files.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_light]),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4.5),
        ('RIGHTPADDING', (0,0), (-1,-1), 4.5),
    ]))

    story.append(t_files)
    story.append(Spacer(1, 10))

    # --------------------------------------------------------------------------
    # 5. SINGLE-PAGE EXTRACTION & SYSTEM HIGHLIGHTS
    # --------------------------------------------------------------------------
    story.append(Paragraph("5. Technical Implementation Highlights", style_h1))

    story.append(Paragraph("5.1. Single-Page PDF Order Extraction & Instant Download", style_h2))
    story.append(Paragraph(
        "A critical innovation in KGP Gyankosh is the automatic on-demand single-page PDF extraction mechanism. "
        "Official polytechnic records often contain large multi-page consolidated books (e.g. <code>Orders.pdf</code>, 36 MB, 85 pages). "
        "When an administrative user asks for a specific order (such as Order No. 63 of 2026), the system cites <code>Orders.pdf (Page 67)</code>. "
        "Instead of forcing the user to download the entire 36 MB PDF, the function <code>get_document_download_info()</code> "
        "intercepts the citation, extracts that exact single page using <code>pypdf</code> into <code>static/extracted_pages/Orders_page_67.pdf</code>, "
        "and serves it as a lightweight (~400 KB) standalone PDF file directly to the user's browser. "
        "Word documents (.docx) are already individual order files and download directly upon click.",
        style_body
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("5.2. Hybrid Reciprocal Rank Fusion (RRF) & Cross-Encoder Reranking", style_h2))
    story.append(Paragraph(
        "Polytechnic circulars demand both semantic topic understanding (e.g., 'rules for attendance shortage condonation') "
        "and exact administrative code matching (e.g. order numbers like <code>KGP/EST/2026/301</code>, committee member names, or fee amounts). "
        "KGP Gyankosh combines dense vector embeddings (FAISS) with sparse BM25 lexical token matching via Reciprocal Rank Fusion (RRF, k=60, alpha=0.5). "
        "The top-10 hybrid candidates are then processed by a neural Cross-Encoder (<code>cross-encoder/ms-marco-MiniLM-L-6-v2</code>) "
        "using bidirectional self-attention to filter out irrelevant text and promote the top-6 authoritative clauses to the LLM.",
        style_body
    ))
    story.append(Spacer(1, 10))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated documentation PDF: {filename} ({os.path.getsize(filename):,} bytes)")

if __name__ == "__main__":
    out_pdf = "KGP_Gyankosh_System_Architecture_and_Files_Reference.pdf"
    if len(sys.argv) > 1:
        out_pdf = sys.argv[1]
    build_pdf(out_pdf)
