# KGP Gyankosh: Administrative Knowledge Assistant with Advanced RAG
### Internal Document Search & Intelligence Platform for Kashmir Government Polytechnic College, Srinagar
**Capstone Project: IIT Patna — Generative AI & Agentic AI for Developers**  
*Enterprise Administrative Knowledge Assistant with Advanced Two-Stage RAG*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://kgp-gyankosh.streamlit.app/)
[![Tests](https://img.shields.io/badge/pytest-26%20passed-success)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue)](requirements.txt)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

- **Live Cloud Deployment**: [https://kgp-gyankosh.streamlit.app/](https://kgp-gyankosh.streamlit.app/)
- **GitHub Repository**: [https://github.com/Wajahat-techie/kgp-gyankosh](https://github.com/Wajahat-techie/kgp-gyankosh)

---

## 1. Project Overview & Institutional Context

### Institutional Background
**Kashmir Government Polytechnic (KGP) College, Srinagar** (established in 1958 under the Department of Skill Development, Government of Jammu & Kashmir) is one of the oldest technical diploma institutions in the region. The administrative offices—including the Principal's Secretariat, Academic Section, Examination Wing, and Departmental Heads—handle thousands of official records every academic year:
- Government orders, institutional circulars, and departmental sanction notifications.
- J&K State Board of Technical Education (SBOTE) notifications, examination schedules, and syllabus revisions.
- Student attendance policies, academic guidelines, and admission rosters.
- Faculty service records, committee notifications, and civil service leave sanctions governed by J&K Civil Service Regulations (CSR).

### The Core Problem Solved
Administrative staff and academic counselors routinely spend hours manually searching for information scattered across physical filing cabinets and multi-page digital records:
1. **Scanned Documents Without Text Layers**: Many official government orders and notices are uploaded as scanned image PDFs with stamps and signatures. Standard keyword searches (`Ctrl+F` in Adobe or Windows Search) fail completely because there is no selectable text layer.
2. **High Precision Requirements in Governance**: Answering queries like *"What is the condonation limit for shortage of attendance?"* or *"Who is eligible for tuition fee exemption under the TFW scheme?"* requires citing exact clause numbers, dates, and order codes. Approximate answers are unacceptable in an administrative setting.
3. **Risk of Hallucination with Generic LLMs**: Standard cloud chatbots have no access to internal college records and frequently fabricate plausible-sounding rules or deadlines.
4. **Data Privacy & Operational Flexibility**: Sensitive internal administrative circulars require a solution that can run either fully offline on local campus hardware without leaking records to external APIs, or on secure cloud infrastructure for remote access.

**KGP Gyankosh** is an end-to-end, two-stage Advanced Retrieval-Augmented Generation (RAG) platform developed to solve these operational challenges.

---

## 2. Project Objectives & System Scope

1. **Comprehensive Multi-Format Document Ingestion**:
   - Automatically parse, OCR, and index official records across PDF, DOCX, and scanned image formats.
2. **Advanced Two-Stage RAG Pipeline**:
   - Combine sparse lexical search (BM25) with dense semantic search (FAISS) via Reciprocal Rank Fusion (RRF) and Cross-Encoder reranking.
3. **Dual Deployment Flexibility (Zero Vendor Lock-In)**:
   - **Online Cloud**: Hosted live on Streamlit Cloud using Google Gemini Flash or OpenAI GPT-4o-mini for fast cloud reasoning.
   - **Offline Local**: Self-hosted via Ollama with Llama 3.1 (8B) for zero cloud cost and 100% on-campus data privacy.
4. **Strict Grounding & Zero-Hallucination Policy**:
   - Zero tolerance for fabricated rules; every response must cite the verified document name and page number, with 1-click downloads of the original file.
5. **Role-Based Administrative Security**:
   - Salted bcrypt password authentication across 4 administrative tiers (Admin, Academic, Exam, Faculty).

---

## 3. System Architecture: Two-Stage Hybrid Pipeline

The system is cleanly separated into two distinct stages: an **Offline Ingestion & Indexing Engine** and an **Online Retrieval & Generation Web App**. This guarantees zero query-time indexing overhead and sub-second response times.

```
==================================================================================================
                     STAGE 1: OFFLINE DOCUMENT INGESTION & INDEXING PIPELINE
                                   (Executed via build_index.py)
==================================================================================================

   [ College Administrative Documents ]
   (Digital PDFs, Scanned Image PDFs, DOCX, TXT)
                  │
                  ▼
   ┌───────────────────────────────┐
   │       Document Loader         │  Auto-detects format & checks for digital text layer
   └───────────────────────────────┘
                  │
                  ├──► [ Scanned Page / Image ] ──► [ RapidOCR (ONNX) + pypdfium2 ] ──┐
                  │                                  (Fallback to Tesseract)           │
                  │                                                                    ▼
                  └──► [ Digital Text Stream ] ───────────────────────────────► [ Document Stream ]
                                                                                      │
                                                                                      ▼
                                                                       ┌──────────────────────────────┐
                                                                       │   Recursive Text Chunker     │
                                                                       │  (500 chars, 100 overlap,    │
                                                                       │   chunk IDs & MD5 tracking)  │
                                                                       └──────────────────────────────┘
                                                                                      │
                                       ┌──────────────────────────────────────────────┴──────────────────────────────┐
                                       ▼                                                                             ▼
                          ┌─────────────────────────┐                                                   ┌─────────────────────────┐
                          │    Dense Embeddings     │                                                   │       BM25 Corpus       │
                          │   (all-MiniLM-L6-v2)    │                                                   │    Tokenizer & Model    │
                          └─────────────────────────┘                                                   └─────────────────────────┘
                                       │                                                                             │
                                       ▼                                                                             ▼
                          ┌─────────────────────────┐                                                   ┌─────────────────────────┐
                          │   FAISS Vector Store    │                                                   │       BM25 Store        │
                          │ (output/vector_store/)  │                                                   │  (output/bm25_store/)   │
                          └─────────────────────────┘                                                   └─────────────────────────┘
                                       │                                                                             │
                                       └──────────────────────────────┬──────────────────────────────────────────────┘
                                                                      ▼
                                                        [ output/index_manifest.json ]
                                                   (Tracks MD5 hashes for incremental updates)

==================================================================================================
                     STAGE 2: ONLINE RETRIEVAL & INTERACTIVE QUERY APPLICATION
                                (Executed via Streamlit app.py)
==================================================================================================

     [ Authorized College Staff ]
                  │
                  ▼
     ┌─────────────────────────┐
     │   Role-Based Login      │  Verifies bcrypt salted hash against config/auth_config.yaml
     └─────────────────────────┘
                  │ (Authenticated: admin, academic, exam, faculty)
                  ▼
     [ User Query / Follow-up ]
                  │
                  ▼
     ┌─────────────────────────┐
     │  Conversational Memory  │  Resolves pronouns & anaphora into a standalone search query
     └─────────────────────────┘
                  │
                  ▼
     ┌─────────────────────────────────────────────────────────────────────────────────────────┐
     │                                HYBRID SEARCH RETRIEVER                                  │
     │   ┌─────────────────────────────────────────┐ ┌───────────────────────────────────────┐ │
     │   │     Dense Vector Retrieval (FAISS)      │ │      Sparse Keyword Match (BM25)      │ │
     │   │  Captures semantic and conceptual intent│ │  Captures order nos, acronyms, names  │ │
     │   └─────────────────────────────────────────┘ └───────────────────────────────────────┘ │
     │                        │                                          │                     │
     │                        └────────────────────┬─────────────────────┘                     │
     │                                             ▼                                           │
     │                            Reciprocal Rank Fusion (RRF, k=60)                           │
     └─────────────────────────────────────────────────────────────────────────────────────────┘
                                                   │
                                                   ▼ (Top-10 Hybrid Candidates)
     ┌─────────────────────────────────────────────────────────────────────────────────────────┐
     │                                CROSS-ENCODER RERANKER                                   │
     │                       (cross-encoder/ms-marco-MiniLM-L-6-v2)                            │
     │             Scores full (Query, Passage) attention to eliminate false positives         │
     └─────────────────────────────────────────────────────────────────────────────────────────┘
                                                   │
                                                   ▼ (Top-4 Highly Relevant Passages)
     ┌─────────────────────────────────────────────────────────────────────────────────────────┐
     │                                SWITCHABLE LLM BACKEND                                   │
     │       [Google Gemini Flash] (Default) | [Ollama Llama 3.1] | [OpenAI GPT-4o-mini]       │
     │                                                                                         │
     │   System Prompt Constraints:                                                            │
     │   1. Ground answers strictly on retrieved official text.                                │
     │   2. Refuse out-of-domain queries ("I could not find this in official documents").      │
     │   3. Include exact source filename and page numbers for every stated fact.              │
     └─────────────────────────────────────────────────────────────────────────────────────────┘
                                                   │
                                                   ▼
     ┌─────────────────────────────────────────────────────────────────────────────────────────┐
     │                                     STREAMLIT UI                                        │
     │   - Formatted grounded answer with official institutional verification badge            │
     │   - Clickable citation cards with direct single-page PDF or full document downloads     │
     │   - Multi-turn conversation history with session reset control                          │
     │   - Dynamic index status badge showing active document & clause count                   │
     └─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Ingestion, Chunking & Indexing Pipeline

### Multi-Format Parsing Architecture
- **Digital PDFs**: Extracted using `pypdf` page by page, preserving original page numbers.
- **Word Documents (`.docx`)**: Parsed using `python-docx`, capturing paragraphs and structured tables.
- **Scanned Orders & Notices**: Automatic text-density checks identify pages without a native text layer. Scanned pages are passed to a pure-Python ONNX-based OCR engine (`RapidOCR` + `pypdfium2`) with fallback to Tesseract.
- **Persistent OCR Cache**: OCR outputs are serialized to `output/ocr_cache/` so subsequent indexing runs execute in milliseconds.
- **Incremental Delta Indexing**: `build_index.py` computes MD5 hashes for all files and compares them against `output/index_manifest.json`. Only new or modified circulars are processed during routine updates.

### Semantic Chunking Strategy
- **Splitter**: `RecursiveCharacterTextSplitter`
- **Chunk Size**: `500 characters` | **Chunk Overlap**: `100 characters`
- **Why 500 Characters?**: Government orders and circulars are structured in compact, clause-specific paragraphs. A 500-character boundary preserves individual rules (e.g., minimum attendance criteria, late fee penalties) without diluting the semantic meaning across unrelated sections.
- **Metadata Schema per Chunk**:
  ```json
  {
    "source": "Orders.pdf",
    "page": 14,
    "chunk_id": "Orders.pdf#p14_c2",
    "chunk_index": 2,
    "file_type": "pdf",
    "is_ocr": true
  }
  ```

---

## 4. Key Features & Functionality

### 1. Dual Operational Modes
The application provides two distinct operational modes accessible from the sidebar radio toggle:
- **College Records (RAG Mode)**: Queries run through the full hybrid retrieval and reranking pipeline across **1,158 official documents** (8,166 clauses). Answers are grounded strictly in retrieved text with mandatory document and page citations. Out-of-scope questions trigger an anti-hallucination refusal.
- **General AI (Direct LLM Mode)**: Bypasses the document index and acts as an administrative drafting assistant. Staff can draft official government circulars, write formal memos, format emails according to J&K civil service tone, or ask general polytechnic academic and technical questions.

### 2. Multi-Backend LLM Selector
Users can dynamically switch between three LLM backends in the sidebar without restarting the server:
- **Google Gemini** (`gemini-3.5-flash-lite` / `gemini-1.5-flash`): High-speed cloud reasoning with generous free tier allowances.
- **Local Ollama** (`llama3.1:latest`): Completely offline, private, zero-token-fee inference running locally on the workstation or campus server.
- **OpenAI** (`gpt-4o-mini`): High-accuracy cloud fallback.

### 3. Dense-Sparse Hybrid Search with RRF
Neither vector search nor keyword search alone is sufficient for administrative documents:
- Dense vector search (`all-MiniLM-L6-v2`) understands semantic concepts (e.g. *"financial help for school fees"* matches *"Child Education Allowance"*).
- Sparse search (BM25) ensures that exact order reference numbers (e.g. `Order No: 12 of 2026`) and institutional abbreviations (`SBOTE`, `NSS`, `JKBOSE`) are never missed.
- The two result lists are merged using **Reciprocal Rank Fusion (RRF)**:
  $$\text{RRF Score}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{w_m}{60 + \text{rank}_m(d)}$$

### 4. Cross-Encoder Reranker
Bi-encoder retrieval generates a broad candidate pool (top 10). The Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) evaluates the full `(query, passage)` pair with joint self-attention, re-ranking documents so that only the top-4 truly relevant clauses are presented to the LLM.

### 5. Multi-Turn Conversational Memory & Query Reformulation
When staff ask follow-up questions (e.g., Turn 1: *"Who is the Incharge of Computer Engineering?"* → Turn 2: *"When was he appointed?"*), the memory module detects anaphora and reformulates Turn 2 into a self-contained query (*"When was Er. Shabir Ahmad Ahanger appointed as Incharge of Computer Engineering?"*) before running retrieval.

### 6. Interactive Citation & Direct Document Download Cards
For every retrieved source, the interface renders:
- An expandable citation card showing the document title, page number, match score, and verified text excerpt.
- A **direct download button** (`📥 Download Page X` or `📥 Download Order`) allowing staff to download the exact single-page PDF slice or original document directly from `static/docs/` or `static/extracted_pages/`.

### 7. Incremental Indexing with MD5 Manifest
Running `build_index.py` does not re-process unchanged documents. It checks MD5 file hashes stored in `output/index_manifest.json`, processing only newly added or modified files to keep indexing fast.

---

## 5. Repository Structure

```
kgp-gyankosh/
├── README.md                           # Comprehensive documentation and capstone report
├── PROJECT_ARCHITECTURE_AND_FEATURES.md# Deep architectural and implementation reference
├── requirements.txt                    # Python dependencies (with PyTorch CPU wheel)
├── pytest.ini                          # Pytest configuration
├── .env.example                        # Template for environment configuration
├── .env                                # Active local configuration (git-ignored)
├── .gitignore                          # Excludes secrets, caches, and large indices
├── .github/
│   └── workflows/
│       └── keepalive.yaml              # Scheduled GitHub Actions ping for Streamlit Cloud
├── config/
│   ├── auth_config.yaml.example        # Template containing demo accounts & bcrypt hashes
│   └── auth_config.yaml                # Active credentials file (git-ignored)
├── data/
│   ├── sample_notices/                 # Core test administrative notices
│   ├── pdf/                            # Official scanned and digital PDF circulars
│   └── word/                           # Official Microsoft Word (.docx) circulars
├── src/
│   ├── __init__.py
│   ├── auth/
│   │   ├── __init__.py
│   │   └── authenticator.py            # Streamlit-authenticator bcrypt login wrapper
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── ocr_loader.py               # RapidOCR + pypdfium2 with Tesseract fallback
│   │   ├── document_loader.py          # Multi-format parser (PDF, DOCX, TXT)
│   │   └── chunker.py                  # Recursive text splitter with metadata tracking
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── embeddings.py               # Embedding provider factory (Local, Google, OpenAI)
│   │   └── vector_store.py             # FAISS and BM25 persistence & manifest manager
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── hybrid_search.py            # FAISS + BM25 Reciprocal Rank Fusion (RRF)
│   │   └── reranker.py                 # Cross-encoder reranker (ms-marco-MiniLM)
│   ├── memory/
│   │   ├── __init__.py
│   │   └── conversation_memory.py      # Multi-turn history & query reformulation
│   └── llm/
│       ├── __init__.py
│       └── client.py                   # LLM factory (Gemini, Ollama, OpenAI) with anti-hallucination prompt
├── static/
│   ├── style.css                       # Enterprise dark theme and responsive layout styles
│   ├── campus_bg.jpg                   # KGP campus photograph background
│   └── docs/                           # Served static files for citation downloads
├── tests/
│   ├── __init__.py
│   ├── test_chunker.py                 # Tests for text chunking, overlap, and metadata
│   ├── test_embeddings.py              # Tests for embedding provider fallbacks
│   ├── test_memory.py                  # Tests for conversation sliding window & reformulation
│   └── test_retrieval.py               # Tests for BM25, RRF fusion, and CrossEncoder
├── scripts/
│   ├── test_queries.py                 # CLI test script for query verification
│   ├── create_diagrams.py              # Architecture diagram generator
│   ├── generate_documentation_pdf.py   # Publication-ready project report PDF generator
│   └── generate_line_by_line_doc_pdf.py# Line-by-line codebase PDF report generator
├── build_index.py                      # Stage 1: Batch offline indexing CLI
├── app.py                              # Stage 2: Interactive Streamlit web application
└── output/
    ├── index_manifest.json             # Manifest registering 1,158 documents & 8,166 clauses
    ├── index_logs/                     # Timestamped execution logs for indexing runs
    ├── ocr_cache/                      # Persistent disk cache for extracted OCR text
    ├── vector_store/                   # Serialized FAISS vector index (index.faiss, index.pkl)
    └── bm25_store/                     # Serialized BM25 model and tokenized corpus
```

---

## 6. Setup & Installation Guide

### Prerequisites
- **Python**: Version 3.10, 3.11, 3.12, or 3.14.
- **Git**: Installed and available in terminal PATH.
- **Hardware**: Any modern multi-core CPU and 4 GB RAM. (A GPU is not required; PyTorch CPU and FAISS run efficiently).
- **OCR**: Pure-Python RapidOCR runs out-of-the-box via ONNX. Installing system Tesseract is completely optional.
- **Optional Local LLM**: If using Ollama locally, install [Ollama](https://ollama.com/) and run `ollama pull llama3.1`.

### Step 1: Clone the Repository
```bash
git clone https://github.com/Wajahat-techie/kgp-gyankosh.git
cd kgp-gyankosh
```

### Step 2: Create and Activate Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Linux or macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
The `requirements.txt` file is pre-configured with the PyTorch CPU extra-index URL to ensure clean, fast installation:
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Copy the `.env.example` template to `.env`:
```bash
cp .env.example .env
```
Open `.env` and set your desired configuration. For example, to use Google Gemini:
```env
LLM_PROVIDER=google
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
```

### Step 5: Configure Authentication Accounts
Copy `config/auth_config.yaml.example` to `config/auth_config.yaml`:
```bash
cp config/auth_config.yaml.example config/auth_config.yaml
```

The system comes pre-configured with four test administrative accounts (all passwords hashed with bcrypt):

| Role / Designation | Username | Demo Password | Scope & Department |
| :--- | :--- | :--- | :--- |
| **Admin** | `admin_kgp` | Principal / Head of Administration | `KgpAdmin@2026` |
| **Staff** | `clerk_academic` | Academic Section In-Charge | `Academic@2026` |
| **Staff** | `exam_wing` | Examination Wing Controller | `Exam@2026` |
| **Faculty** | `hod_polytechnic` | Head of Department / Senior Faculty | `Faculty@2026` |

---

## 9. User Interface & Administrative Design System

- **Institutional Styling**: Modern dark glassmorphic design system (`static/style.css`) incorporating official Jammu & Kashmir Government crest and college branding.
- **Typography**: Google Fonts (`Outfit` for headings and `Plus Jakarta Sans` for body text).
- **Live Metric Badge**: Real-time status counter displaying the number of active indexed documents and searchable clauses.
- **Interactive Suggestion Chips**: Quick-click query prompts for common administrative inquiries.
- **Responsive Citation Cards**: Structured source cards showing verified document titles, page numbers, match scores, excerpts, and download buttons.
- **Non-Overlapping Bottom Dock**: Floating query input bar with auto-scroll script.

---

## 10. Automated Testing & Verification

The repository includes **26 automated unit and integration tests** built with `pytest`:

```bash
pytest -v
```

### Test Coverage
- `tests/test_chunker.py` (7 tests): Text chunk limits, overlap consistency, and deterministic chunk ID assignment (`source#pX_cY`).
- `tests/test_embeddings.py` (3 tests): SentenceTransformers embedding initialization, dimension validation (384), and provider fallback logic.
- `tests/test_memory.py` (7 tests): Multi-turn sliding window memory, conversation clearing, and LLM-assisted query reformulation.
- `tests/test_retrieval.py` (9 tests): Alphanumeric BM25 tokenization, Reciprocal Rank Fusion weights, and Cross-Encoder reranker fallbacks.

**Status**: 26 passed in ~0.86s across Python 3.10 through 3.14.

### Developer CLI Benchmark Harness
```bash
python scripts/test_queries.py
```

---

## 11. Production Cloud Deployment & Optimizations

Live deployment: **[https://kgp-gyankosh.streamlit.app/](https://kgp-gyankosh.streamlit.app/)**

### Production Optimizations:
1. **PyTorch CPU Build Optimization**: `requirements.txt` points to `--extra-index-url https://download.pytorch.org/whl/cpu`. This reduces the PyTorch package size from ~3GB (CUDA) to ~180MB, completely eliminating out-of-memory cloud container crashes.
2. **Pre-Built Ingestion Stores**: Pre-serialized FAISS and BM25 index stores in `output/` allow the cloud web app to cold-start in under 3 seconds with zero query-time indexing lag.
3. **Resource Caching (`@st.cache_resource`)**: Singletons for vector stores, reranker models, and LLM clients cached in memory.
4. **Automated Keepalive Workflow**: A GitHub Actions workflow (`.github/workflows/keepalive.yaml`) pings the Streamlit deployment every 3 days to prevent automatic app hibernation.

---

## 12. Demonstrated Administrative Use Cases

| # | Administrative Query | Verified Finding | Cited Source |
| :--- | :--- | :--- | :--- |
| **1** | *"Who has been assigned as Incharge Computer Engineering?"* | Er. Shabir Ahmad Ahanger assigned as Incharge Computer Engineering. | `Orders.pdf` (Page 22, Order No: 23/2026) |
| **2** | *"Who was appointed to the Sports Committee for the academic session 2026?"* | Constitution of institutional sports committee with designated head and faculty departmental representatives. | `Orders.pdf` (Page 3, Order No: 03 of 2026) |
| **3** | *"What are the internal physical verification guidelines for college stores and workshops?"* | Directs HoDs and Section Heads to conduct internal physical verification of equipment, stock registers, and assets by 30th April 2026. | `Orders.pdf` (Page 4, Order No: 05 of 2026) |
| **4** | *"What are the criteria and points for Guest Faculty engagement in 2025?"* | 80 points for academic merit, 15 points for higher qualifications, 5 points for interaction. | `Advertisement Guest Faculty KGP 2025.docx` (Page 1) |
| **5** | *"What is the condonation limit for shortage of attendance?"* | Up to 10% condonation permissible by the Principal for verified medical illness or official institutional deputation. | `Notices_and_orders.pdf` (Page 1) |

---

## 13. Engineering Challenges & Technical Solutions

- **Challenge 1: OCR Degradation on Old Scanned Circulars**
  - *Problem*: Physical stamps, signatures, and low scan resolution corrupted text extraction.
  - *Solution*: Implemented text-density filters, fallback to pure-Python RapidOCR via ONNX, and persistent disk-based MD5 caching.
- **Challenge 2: Query-Document Terminology Mismatch**
  - *Problem*: Clerks ask questions in colloquial language while circulars use formal bureaucratic terminology.
  - *Solution*: Dense vector search (`all-MiniLM-L6-v2`) maps semantic synonyms, while BM25 preserves exact circular numbers, merged via Reciprocal Rank Fusion.
- **Challenge 3: Cloud Memory & Cold-Start Limits**
  - *Problem*: Running heavy deep transformer models on free cloud tiers caused memory overruns.
  - *Solution*: Decoupled heavy Cross-Encoder reranker calls, used CPU-only PyTorch wheels, pre-computed search stores, and cached singleton models.

---

## 11. Engineering Decisions & Design Trade-offs

### 1. Why a Strict Two-Stage Architecture?
*Decision*: Separate document ingestion and indexing (`build_index.py`) from online query retrieval (`app.py`).  
*Rationale*: Parsing hundreds of PDFs, performing OCR on scanned circulars, and generating vector embeddings takes several minutes. Performing these steps on application startup or during user queries would create unacceptable wait times and exhaust web server memory. Running indexing offline produces persistent FAISS and BM25 store files that the Streamlit app loads into memory in less than a second.

### 2. Why Dense + Sparse Hybrid Search with Reciprocal Rank Fusion?
*Decision*: Combine FAISS dense semantic search with BM25Okapi lexical matching using RRF.  
*Rationale*: Pure vector embeddings excel at semantic paraphrasing (*"tuition fee relief guidelines"* matching *"Tuition Fee Waiver (TFW) Regulations"*), but often miss exact administrative identifiers such as circular numbers (`KGP/ADM/2026/89`), acronyms (`SBOTE`, `JKBOSE`), or monetary figures (`Rs. 2,200`). BM25 handles exact tokens reliably. Combining both candidate lists with RRF ($k=60$) balances conceptual understanding with exact lexical precision.

### 3. Why Cross-Encoder Reranking?
*Decision*: Add `cross-encoder/ms-marco-MiniLM-L-6-v2` as a second-stage ranking filter.  
*Rationale*: Bi-encoders encode queries and documents separately, which is fast for initial candidate filtering but cannot capture fine-grained token-level interactions. The Cross-Encoder processes the query and passage together through joint transformer attention layers. While too computationally expensive to run over thousands of documents, running it over the top 10 hybrid candidates takes under 100ms and significantly improves the precision of context passed to the LLM.

### 4. Why RapidOCR (ONNX) Instead of External Tesseract Binaries?
*Decision*: Default to `rapidocr-onnxruntime` + `pypdfium2`, keeping Tesseract as an optional fallback.  
*Rationale*: Standard Tesseract OCR and `pdf2image` require installing external operating system binaries (`tesseract-ocr` and `poppler-utils`). This complicates local Windows/Mac setups and frequently causes deployment failures on containerized cloud platforms. RapidOCR runs entirely within the Python/ONNX runtime, requires zero OS-level binary installations, and handles both English text and numerical tabular data with high fidelity.

### 5. Why Local Bcrypt YAML Authentication for this Capstone?
*Decision*: Implement authentication with `streamlit-authenticator` using salted bcrypt hashes in a Git-ignored YAML file.  
*Rationale*: Enterprise SAML or Active Directory integrations introduce external infrastructure dependencies that cannot be easily reviewed or run by an evaluator. Local bcrypt authentication demonstrates secure credential handling (salted hashes, session cookies, role verification) while remaining completely self-contained and reproducible.

---

## 12. Operational Scope & Known Limitations

1. **OCR Quality Dependency**: Text extraction from physical circulars is dependent on scan resolution and document condition. Faint carbon copies, skewed scans, or low-DPI photos may contain minor OCR character errors.
2. **Complex Multi-Page Tables**: While the chunker preserves paragraph structure and basic tables, complex nested financial tables spanning across page boundaries may require manual cross-verification against the original PDF (facilitated by the direct download buttons in citation cards).
3. **Authentication Boundary**: Authentication uses salted bcrypt configuration files rather than institutional Single Sign-On (SSO). Passwords and session cookies are managed locally.
4. **Local LLM Hardware Requirements**: Running the local Ollama backend with `llama3.1:latest` requires a machine with at least 8 GB of available system RAM or a dedicated 6 GB+ VRAM GPU. For resource-constrained workstations, the default Google Gemini backend provides immediate cloud inference at zero hardware cost.

---

## 15. Project Evaluation Rubric Cross-Reference

This table maps each requirement of the **IIT Patna Capstone Project 2 (Enterprise Knowledge Assistant with Advanced RAG)** specification to its implementation in this repository:

| Project 2 Requirement | Implementation in KGP Gyankosh | Source File Reference | Verification Status |
| :--- | :--- | :--- | :--- |
| **Document Ingestion (≥2 formats)** | Ingests PDF, DOCX, TXT, and scanned image notices | `src/ingestion/document_loader.py` | Verified |
| **OCR for Scanned Documents** | Pure-Python RapidOCR (ONNX) + pypdfium2 with persistent cache | `src/ingestion/ocr_loader.py` | Verified |
| **Text Chunking & Metadata** | Recursive character text splitting (500 chars, 100 overlap) tracking file, page, and chunk ID | `src/ingestion/chunker.py` | Verified (`test_chunker.py`) |
| **Dense Embeddings** | Sentence-Transformers (`all-MiniLM-L6-v2`) with Google and OpenAI fallbacks | `src/indexing/embeddings.py` | Verified (`test_embeddings.py`) |
| **Local Vector Store** | Serialized FAISS vector database (`index.faiss`, `index.pkl`) | `src/indexing/vector_store.py` | Verified |
| **Incremental Re-indexing** | MD5 hash comparison via manifest to process only new/modified files | `build_index.py`, `output/index_manifest.json` | Verified |
| **Hybrid Search (Vector + BM25)** | Combines FAISS dense retrieval with BM25Okapi using Reciprocal Rank Fusion | `src/retrieval/hybrid_search.py` | Verified (`test_retrieval.py`) |
| **Reranking Step** | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) scoring query-passage pairs | `src/retrieval/reranker.py` | Verified (`test_retrieval.py`) |
| **Conversational Memory** | Sliding window context tracking with LLM-assisted query reformulation | `src/memory/conversation_memory.py` | Verified (`test_memory.py`) |
| **Source Citations** | Expandable citation cards with document name, page number, excerpt, and match score | `app.py`, `src/llm/client.py` | Verified |
| **Document Download Links** | Direct download buttons for verified single-page PDF extracts or full source files | `app.py`, `static/docs/` | Verified |
| **Hallucination Mitigation** | Strict grounding prompt refusing unverified facts with fallback response | `src/llm/client.py` | Verified |
| **Switchable LLM Providers** | Seamless runtime switching between Google Gemini, local Ollama, and OpenAI | `src/llm/client.py`, `app.py` | Verified |
| **Role-Based Access Control** | Salted bcrypt password authentication across 4 administrative roles | `src/auth/authenticator.py`, `config/auth_config.yaml` | Verified |
| **Interactive User Interface** | Responsive Streamlit interface with dark glassmorphism styling and live metrics | `app.py`, `static/style.css` | Verified |
| **Automated Test Suite** | 26 unit and integration tests executed with pytest | `tests/`, `pytest.ini` | Verified (26/26 passed) |
| **Production Cloud Deployment** | Hosted on Streamlit Community Cloud with GitHub Actions keepalive workflow | `.github/workflows/keepalive.yaml` | Verified Live |
| **Real Institutional Dataset** | Ingested repository of official administrative documents and indexed clauses | `output/index_manifest.json`, `data/` | Verified |
| **Documentation & Diagrams** | Complete system guide, setup manual, architecture diagrams, and test queries | `README.md`, `PROJECT_ARCHITECTURE_AND_FEATURES.md` | Verified |

---

## 16. Local Setup & Quickstart Guide

### Step 1: Clone the Repository
```bash
git clone https://github.com/Wajahat-techie/kgp-gyankosh.git
cd kgp-gyankosh
```

### Step 2: Create a Virtual Environment
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux / macOS:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables & LLM Provider

> [!IMPORTANT]
> **API Key Setup & Model Choice**:
> 1. Copy the template file:
>    ```bash
>    cp .env.example .env
>    ```
> 2. Open `.env` in any text editor and choose your configuration:
>    - **Option A — Cloud LLM (Google Gemini / OpenAI)**:
>      Paste your free Gemini API key from [Google AI Studio](https://aistudio.google.com/) (`GOOGLE_API_KEY=your_key_here`) or OpenAI key (`OPENAI_API_KEY=your_key_here`).
>    - **Option B — Fully Offline Local LLM (Ollama / Llama 3.1)**:
>      If you prefer 100% privacy with zero token costs and no internet API calls, simply start Ollama locally (`ollama run llama3.1`) and set `LLM_PROVIDER=ollama` in `.env`. No API keys required!

```env
# -----------------------------------------------------------
# OPTION 1: Google Gemini Cloud (Default - Free Google Studio Key)
# -----------------------------------------------------------
LLM_PROVIDER=google
GOOGLE_API_KEY=your_google_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# -----------------------------------------------------------
# OPTION 2: 100% Offline Local Inference (No API Key Required!)
# -----------------------------------------------------------
# LLM_PROVIDER=ollama
# OLLAMA_MODEL=llama3.1:latest
# OLLAMA_BASE_URL=http://localhost:11434

# -----------------------------------------------------------
# OPTION 3: OpenAI Cloud API
# -----------------------------------------------------------
# LLM_PROVIDER=openai
# OPENAI_API_KEY=your_openai_api_key_here
# OPENAI_MODEL=gpt-4o-mini
```

### Step 5: Build or Rebuild the Search Indices
```bash
python build_index.py --rebuild
```

### Step 6: Launch the Web App
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser and log in with any demo account (e.g., `admin_kgp` / `KgpAdmin@2026`).

---

## 17. Project Links & Resources

- 🌐 **Live Web Application**: [https://kgp-gyankosh.streamlit.app/](https://kgp-gyankosh.streamlit.app/)
- 💻 **GitHub Repository**: [https://github.com/Wajahat-techie/kgp-gyankosh](https://github.com/Wajahat-techie/kgp-gyankosh)
- 📊 **Presentation Slide Deck**: [Download PPTX (17.8 MB)](docs/KGP_Gyankosh_Presentation.pptx)
- 📝 **Presentation Speaker Notes**: [Download Notes (DOCX)](docs/KGP_Gyankosh_Presentation_Notes.docx)
- 📑 **System Architecture & Files Reference PDF**: [Download PDF](docs/reports/KGP_Gyankosh_System_Architecture_and_Files_Reference.pdf)
- 📖 **Line-by-Line Code Documentation PDF**: [Download PDF](docs/reports/KGP_Gyankosh_Line_By_Line_Code_Documentation.pdf)

---

## 18. License

This capstone project is developed as part of the **IIT Patna — Certification in Generative AI & Agentic AI for Developers**.

Released under the [MIT License](LICENSE).
