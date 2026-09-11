# KGP Gyankosh: Enterprise Knowledge Assistant with Advanced RAG
### Internal Administrative Intelligence System for Kashmir Government Polytechnic College, Srinagar
*Capstone Project: IIT Patna — Generative AI & Agentic AI for Developers (Project 2: Enterprise Knowledge Assistant with Advanced RAG)*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://kgp-gyankosh.streamlit.app/)
[![Tests](https://img.shields.io/badge/pytest-26%20passed-success)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue)](requirements.txt)
[![Official Records](https://img.shields.io/badge/records-1%2C158%20Docs%20%7C%208%2C166%20Clauses-green)](output/index_manifest.json)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

- **Live Cloud Deployment**: [https://kgp-gyankosh.streamlit.app/](https://kgp-gyankosh.streamlit.app/)
- **GitHub Repository**: [https://github.com/Wajahat-techie/kgp-gyankosh](https://github.com/Wajahat-techie/kgp-gyankosh)

---

## 1. Project Overview & Institutional Context

### Institutional Background
**Kashmir Government Polytechnic (KGP) College, Srinagar** (established in 1958 under the Department of Skill Development, Government of Jammu & Kashmir) is one of the oldest technical diploma institutions in the region. The administrative offices—including the Principal's Secretariat, Academic Section, Examination Wing, and Departmental Heads—handle thousands of official records every academic year:
- Government orders, sanction orders, and financial release circulars (such as Child Education Allowances).
- J&K State Board of Technical Education (SBOTE) notifications, examination schedules, and syllabus revisions.
- Student attendance policies, fee concession circulars, and admission rosters.
- Faculty service records, committee proceedings, and civil service leave sanctions governed by J&K Civil Service Regulations.

### The Problem
Administrative staff and academic counselors routinely spend hours manually locating and cross-referencing information scattered across physical files and multi-page digital records:
1. **Scanned Documents Without Text Layers**: Many official government orders and notices are uploaded as scanned image PDFs or scanned photos. Standard keyword searches (Windows Explorer, Adobe Acrobat) fail completely because there is no selectable text layer.
2. **High Precision Requirements**: Answering queries like *"What is the condonation limit for shortage of attendance?"* or *"Who is eligible for fee exemption under the TFW scheme?"* requires citing exact clause numbers, dates, and order codes. Vague answers are unacceptable in an administrative setting.
3. **Risk of Hallucination with Generic LLMs**: Standard cloud chatbots have no access to internal college records and frequently fabricate plausible-sounding rules or deadlines.
4. **Data Privacy & Operational Flexibility**: Sensitive internal administrative circulars require a solution that can run either fully offline on the campus LAN without leaking records to external APIs, or on secure cloud infrastructure using encrypted API backends.

### The Solution: KGP Gyankosh
**KGP Gyankosh** is an internal enterprise-grade Retrieval-Augmented Generation (RAG) platform developed to solve these operational challenges. The system:
- Ingests and processes digital PDFs, scanned document images, Word files (`.docx`), and text notices using a built-in OCR pipeline.
- Employs a **Two-Stage Architecture** separating offline batch indexing from online real-time querying, eliminating startup latency.
- Uses **Dense-Sparse Hybrid Retrieval** (FAISS + BM25 merged via Reciprocal Rank Fusion) combined with a **Cross-Encoder Reranker** (`ms-marco-MiniLM-L-6-v2`) to accurately retrieve relevant clauses.
- Enforces **strict anti-hallucination prompting** so the LLM answers only from verified passages and provides clear source citations with page numbers.
- Offers direct single-page or full document downloads for every cited source.
- Features **role-based bcrypt authentication** across four administrative roles.
- Supports switchable LLM backends: Google Gemini (default cloud), local Ollama with Llama 3.1 (100% offline and private), and OpenAI.

---

## 2. System Architecture & Query Pipeline

KGP Gyankosh separates the resource-heavy ingestion and embedding operations from the fast, interactive query application. Pre-computed indices are stored on disk and loaded in sub-second time by the web interface.

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
   │       Document Loader         │  Auto-detects format & checks for embedded digital text
   └───────────────────────────────┘
                  │
                  ├──► [ Scanned Page / Image ] ──► [ Pure-Python RapidOCR (ONNX) + pypdfium2 ] ──┐
                  │                                  (Fallback to Tesseract if installed)         │
                  │                                                                               ▼
                  └──► [ Digital Text Stream ] ────────────────────────────────────────► [ Document Stream ]
                                                                                                  │
                                                                                                  ▼
                                                                                   ┌──────────────────────────────┐
                                                                                   │   Recursive Text Chunker     │
                                                                                   │  (500 chars, 100 overlap,    │
                                                                                   │   metadata & hash tracking)  │
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
     │        [Google Gemini 3.5 Flash] (Default) | [Ollama Llama 3.1] | [OpenAI GPT-4o-mini]  │
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
     │   - Live index statistics (1,158 documents, 8,166 clauses active)                       │
     └─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Technology Stack & Component Justifications

| Component | Technology | Version | Engineering Justification |
| :--- | :--- | :--- | :--- |
| **Runtime Environment** | Python | 3.10 – 3.14 | Modern standard for ML/RAG pipelines; tested on Windows and Linux. |
| **Web Framework** | Streamlit | `>=1.30.0` | Fast deployment of data-intensive web apps with built-in state handling. |
| **Dense Vector Index** | FAISS (`faiss-cpu`) | `>=1.7.4` | In-memory similarity search with fast serialization to disk; no external database server needed. |
| **Sparse Keyword Search** | BM25 (`rank_bm25`) | `>=0.2.2` | Implements BM25Okapi for exact lexical matches on institutional codes (e.g. `KGP/ADM/2026/104`, `SBOTE`). |
| **Dense Embeddings** | `sentence-transformers` | `>=2.2.2` | Local `all-MiniLM-L6-v2` produces compact 384-dimensional embeddings at zero inference cost. Fallbacks for Google GenAI and OpenAI. |
| **Passage Reranking** | Cross-Encoder | `>=2.2.2` | `ms-marco-MiniLM-L-6-v2` applies cross-attention between query and passage, pruning false positives before LLM context creation. |
| **LLM Inference** | Google Gemini / Ollama / OpenAI | Variable | Configurable via `.env` or UI selector. Gemini Flash provides fast cloud responses; Ollama (Llama 3.1) ensures 100% offline data privacy on-campus. |
| **OCR Processing** | RapidOCR (`rapidocr-onnxruntime`) + `pypdfium2` | `>=1.3.0` | Pure-Python ONNX runtime OCR requiring zero external C++ binaries (Tesseract/Poppler), with disk caching. |
| **Document Loaders** | `pypdf`, `python-docx` | `>=3.0.0` | Native parsers for digital PDF notices and Word circulars preserving structural paragraphs. |
| **Authentication** | `streamlit-authenticator` + `bcrypt` | `>=0.3.0` | Salted bcrypt password hashing prevents plaintext password leakage, keeping auth in Git-ignored YAML. |
| **PyTorch Distribution** | PyTorch CPU (`--extra-index-url`) | `>=2.0.0` | Uses the lightweight CPU wheel (~180MB) instead of the heavy CUDA wheel (~3GB) to prevent container crashes on Streamlit Cloud. |
| **Automated Testing** | `pytest` | `>=7.4.0` | 26 passing unit and integration tests covering chunking, embeddings, memory, tokenization, and RRF fusion. |
| **Cloud Keepalive** | GitHub Actions | `keepalive.yaml` | Scheduled workflow pinging the Streamlit Cloud deployment every 3 days to prevent automatic sleeping. |

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
| **Principal / Admin** | `admin_kgp` | `KgpAdmin@2026` | Full administrative oversight |
| **Academic Section** | `clerk_academic` | `Academic@2026` | Admissions, syllabus, attendance rules |
| **Examination Wing** | `exam_wing` | `Exam@2026` | SBOTE exams, fee dates, evaluation |
| **HOD / Senior Faculty** | `hod_polytechnic` | `Faculty@2026` | Departmental circulars & leave sanction |

### Step 6: Verify System with Automated Tests
Run the complete test suite to confirm that chunking, embeddings, memory, tokenization, and retrieval functions pass:
```bash
pytest -v
```
Expected output:
```text
tests/test_chunker.py::TestTextSplitter::test_empty_text PASSED
tests/test_chunker.py::TestTextSplitter::test_short_text_no_split PASSED
...
tests/test_retrieval.py::TestDocumentReranker::test_reranker_empty_candidates PASSED
============================== 26 passed in 1.37s ==============================
```

---

## 7. Environment Variables Reference

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `LLM_PROVIDER` | `google` | Active LLM backend: `google`, `ollama`, or `openai`. |
| `GOOGLE_API_KEY` | `""` | API key from Google AI Studio (required if `LLM_PROVIDER=google`). |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Gemini model name (e.g. `gemini-3.5-flash-lite` or `gemini-1.5-flash`). |
| `OLLAMA_MODEL` | `llama3.1:latest` | Ollama model tag when using local offline LLM. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | HTTP endpoint of local Ollama service. |
| `OPENAI_API_KEY` | `""` | OpenAI API key (required if `LLM_PROVIDER=openai`). |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI chat completion model. |
| `EMBEDDING_PROVIDER` | `sentence-transformers` | `sentence-transformers` (local CPU), `google`, or `openai`. |
| `EMBEDDING_MODEL_NAME`| `all-MiniLM-L6-v2` | Hugging Face model name for local dense embeddings. |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Hugging Face Cross-Encoder model name. |
| `ENABLE_RERANKER` | `true` | Toggle cross-encoder reranking step on or off. |
| `HYBRID_ALPHA` | `0.5` | Weight balance: 0.0 (pure BM25) to 1.0 (pure FAISS vector). 0.5 is balanced. |
| `CHUNK_SIZE` | `500` | Target character size for text chunks. |
| `CHUNK_OVERLAP` | `100` | Character overlap between adjacent chunks to preserve boundary context. |
| `DATA_DIR` | `data/sample_notices` | Source folder containing administrative circulars to index. |
| `VECTOR_STORE_DIR` | `output/vector_store` | Destination folder for serialized FAISS index. |
| `BM25_STORE_DIR` | `output/bm25_store` | Destination folder for serialized BM25 index. |
| `AUTH_CONFIG_PATH` | `config/auth_config.yaml` | Location of administrative credentials file. |

---

## 8. Execution Guide: Two-Stage Workflow

### Stage 1: Document Ingestion & Indexing (`build_index.py`)
Run the indexing script whenever new college circulars or notices are added:

```bash
# Standard incremental run (processes only new or modified files)
python build_index.py

# Force a full rebuild from scratch
python build_index.py --rebuild

# Index from a custom directory
python build_index.py --data-dir path/to/notices
```

**Indexing Actions:**
1. Loads files from the data folder (PDF, DOCX, TXT, scanned images).
2. Uses cached OCR text for scanned documents or runs RapidOCR if not yet cached.
3. Chunks text using `RecursiveCharacterTextSplitter` and embeds chunks using `all-MiniLM-L6-v2`.
4. Updates FAISS vector store in `output/vector_store/` and BM25 index in `output/bm25_store/`.
5. Records file MD5 hashes in `output/index_manifest.json` and writes execution logs to `output/index_logs/`.

*(Note: Pre-built index stores for 1,158 official college documents and 8,166 clauses are already included in `output/`, so you can immediately launch Stage 2 without re-indexing).*

### Stage 2: Launch Interactive Web Application (`app.py`)
Launch the Streamlit web application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`. Log in using any of the credentials from the [Authentication Table](#step-5-configure-authentication-accounts).

### Stage 3: Headless CLI Query Testing
To test retrieval accuracy and generation directly from the command line without opening a browser:
```bash
python scripts/test_queries.py
```

---

## 9. Switching LLM Backends

Switching between cloud and local offline models requires **no code modifications**; simply adjust your `.env` or select from the sidebar dropdown:

### 1. Google Gemini (Cloud — Default)
```env
LLM_PROVIDER=google
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
EMBEDDING_PROVIDER=sentence-transformers
```

### 2. Local Ollama (100% Offline & Free)
Make sure the Ollama daemon is running:
```bash
ollama run llama3.1
```
Configure `.env`:
```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:latest
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_PROVIDER=sentence-transformers
```

### 3. OpenAI (Cloud)
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-proj-your_openai_key_here
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_PROVIDER=openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```
*(Note: If you switch embedding providers from sentence-transformers to OpenAI, run `python build_index.py --rebuild` to regenerate the vector index with the new embedding dimensionality).*

---

## 10. Sample Test Cases & Grounded Outputs

The following representative test cases demonstrate the system's accuracy, conversational memory, anti-hallucination guardrails, and administrative drafting capabilities.

### Test Case 1: Exact Fact Retrieval with Source Citation
- **User Query**: *"What is the minimum attendance required for diploma engineering students, and can the Principal condone shortage?"*
- **System Retrieval**: Top candidate retrieved from `KGP_Order_Attendance_and_Leave_Rules_2026.pdf` (Page 1) with rerank score `0.892`.
- **Grounded Response**:
  > Under the official attendance policy of Kashmir Government Polytechnic College, Srinagar:
  > - All registered diploma engineering students must maintain a mandatory minimum attendance of **75%** across theory lectures and practical laboratory sessions to qualify for the SBOTE semester examinations.
  > - The Principal is authorized to condone an attendance shortage of up to **10%** (i.e. between 65.0% and 74.9%) exclusively in verified cases of prolonged medical illness or official institutional deputation (NCC, NSS, or recognized sports).
  > - Any attendance aggregate below 65.0% cannot be condoned under any circumstances, and the candidate will be detained.
  >
  > **Sources**:
  > - `KGP_Order_Attendance_and_Leave_Rules_2026.pdf` (Page 1)

### Test Case 2: Multi-Turn Dialogue & Pronoun Resolution (Conversational Memory)
- **Turn 1 User**: *"What is the policy for medical leave condonation?"*
- **Turn 1 Response**: Explains condonation criteria as cited above.
- **Turn 2 User**: *"What is the deadline to submit certificates for it?"*
- **Internal Query Reformulation**: *"What is the deadline to submit medical certificates for attendance condonation in KGP attendance rules?"*
- **Grounded Response**:
  > Medical certificates issued by a Government Hospital or Registered Medical Practitioner must be submitted to the concerned Head of Department within **three (3) working days** of the student resuming classes. Retrospective submissions at the end of the academic semester will not be accepted.
  >
  > **Sources**:
  > - `KGP_Order_Attendance_and_Leave_Rules_2026.pdf` (Page 1)

### Test Case 3: Anti-Hallucination on Out-of-Scope Query
- **User Query**: *"What is the mess fee refund policy for PhD research scholars staying in hostel block C?"*
- **System Retrieval**: Reranker scores all retrieved chunks below relevance threshold.
- **Grounded Response**:
  > **I could not find this information in the available official documents.**
  > *(The indexed college records cover three-year polytechnic diploma courses and administrative staff orders. There are no guidelines or policies regarding PhD research scholar hostels in the available documents).*
  >
  > **Sources**:
  > - None (Query ungrounded in available documents)

### Test Case 4: Administrative Circular Drafting (General AI Mode)
- **Mode Toggle**: Switched to `🌐 General AI (Direct LLM)`.
- **User Query**: *"Draft an official circular notifying students of Computer and Civil Engineering about the upcoming semester fee submission deadline of 25th October 2026."*
- **Generated Response**:
  > Generates a formal Jammu & Kashmir Government administrative circular complete with institutional header, reference code format (`KGP/ACAD/2026/...`), subject line, numbered operative paragraphs, late fee penalty clauses, and standard distribution copy endorsements (HODs, Notice Board, Accounts Section, Principal's Secretariat).

---

## 11. Engineering Decisions & Design Trade-offs

### 1. Why a Strict Two-Stage Architecture?
*Decision*: Separate document ingestion and indexing (`build_index.py`) from online query retrieval (`app.py`).  
*Rationale*: Parsing hundreds of PDFs, performing OCR on scanned circulars, and generating vector embeddings takes several minutes. Performing these steps on application startup or during user queries would create unacceptable wait times and exhaust web server memory. Running indexing offline produces persistent FAISS and BM25 store files that the Streamlit app loads into memory in less than a second.

### 2. Why Dense + Sparse Hybrid Search with Reciprocal Rank Fusion?
*Decision*: Combine FAISS dense semantic search with BM25Okapi lexical matching using RRF.  
*Rationale*: Pure vector embeddings excel at semantic paraphrasing (*"allowance for school expenses"* matching *"Child Education Allowance"*), but often miss exact administrative identifiers such as circular numbers (`KGP/ADM/2026/89`), acronyms (`SBOTE`, `JKBOSE`), or monetary figures (`Rs. 2,200`). BM25 handles exact tokens reliably. Combining both candidate lists with RRF ($k=60$) balances conceptual understanding with exact lexical precision.

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

## 13. Project Evaluation Rubric Cross-Reference

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
| **Real Institutional Dataset** | Ingested repository of 1,158 official administrative documents and 8,166 clauses | `output/index_manifest.json`, `data/` | Verified |
| **Documentation & Diagrams** | Complete system guide, setup manual, architecture diagrams, and test queries | `README.md`, `PROJECT_ARCHITECTURE_AND_FEATURES.md` | Verified |

---

## 14. License & Institutional Attribution

This capstone project is developed as part of the **IIT Patna — Executive M.Tech / Certification in Generative AI & Agentic AI for Developers**.  
Institutional document records and administrative context: **Kashmir Government Polytechnic College, Srinagar**, Department of Skill Development, Government of Jammu & Kashmir.

Released under the [MIT License](LICENSE).
