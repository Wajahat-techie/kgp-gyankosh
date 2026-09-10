# KGP Gyankosh: Enterprise Knowledge Assistant with Advanced RAG
### Internal Administrative Intelligence System for Kashmir Government Polytechnic College, Srinagar
*Capstone Project: IIT Patna — Generative AI & Agentic AI for Developers (Project 2: Enterprise Knowledge Assistant with Advanced RAG)*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://kgp-gyankosh.streamlit.app/)
[![Tests](https://img.shields.io/badge/pytest-26%20passed-success)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue)](requirements.txt)
[![Official Records](https://img.shields.io/badge/records-1%2C158%20Docs%20%7C%208%2C166%20Clauses-green)](output/index_manifest.json)

**🌐 Live Cloud Web Application**: [https://kgp-gyankosh.streamlit.app/](https://kgp-gyankosh.streamlit.app/)

---

## 1. Executive Summary & Problem Statement

### Administrative Context
**Kashmir Government Polytechnic (KGP) College, Srinagar** is one of the premier technical diploma institutions in the region. The administration department manages hundreds of critical public notices, government orders, examination notifications, fee revision circulars, student welfare policies, and attendance regulations issued throughout the academic session.

### Problem Statement
Administrative staff, departmental clerks, and academic advisors currently spend significant manual effort locating, verifying, and cross-referencing information across physical notice archives and multi-page scanned PDF orders. Common queries such as:
- *"What is the exact condonation percentage allowed for attendance shortage?"*
- *"What is the deadline for odd semester exam form submission with and without late fee?"*
- *"Who is eligible for a full tuition fee waiver?"*

require manual searching through voluminous files. Furthermore, many official circulars are **scanned document images with no digital text layer**, rendering standard keyword search useless.

### Solution Overview: KGP Gyankosh
**KGP Gyankosh** (ज्ञानकोश / Repository of Knowledge) is an enterprise-grade, internal **Advanced Retrieval-Augmented Generation (RAG)** system designed specifically for the KGP administration department. It enables authorized college staff to ask natural-language questions and receive grounded, accurate, and source-cited answers derived solely from official college documents.

---

## 2. High-Level Architecture

KGP Gyankosh is built on a strict **Two-Stage Architecture**, completely separating offline document indexing from online query retrieval, with pre-built search stores for instant cloud and local execution:

```
==================================================================================================
                     STAGE 1: OFFLINE INGESTION & INDEXING PIPELINE
                          (Executed via build_index.py once or on doc updates)
==================================================================================================

  [ Administrative Documents ]
  (PDFs, DOCX, TXT, Scanned Images)
              │
              ▼
  ┌───────────────────────┐
  │   Document Loader     │ ──► Auto-detects scanned PDFs & images
  └───────────────────────┘
              │
              ├──► [ Scanned Page? ] ──► [ Pure-Python RapidOCR + pypdfium2 / Tesseract ] ──┐
              │                                                                             │
              └──► [ Native Text Layer ] ───────────────────────────────────────────────────┴──► [ Plain Text + Metadata ]
                                                                                       │
                                                                                       ▼
                                                                          ┌──────────────────────────┐
                                                                          │  Recursive Text Chunker  │
                                                                          │ (500 chars / 100 overlap)│
                                                                          └──────────────────────────┘
                                                                                       │
                                          ┌────────────────────────────────────────────┴─────────────────┐
                                          ▼                                                              ▼
                             ┌─────────────────────────┐                                   ┌──────────────────────────┐
                             │    Dense Embeddings     │                                   │       BM25 Corpus        │
                             │   (all-MiniLM-L6-v2)    │                                   │    Tokenizer & Model     │
                             └─────────────────────────┘                                   └──────────────────────────┘
                                          │                                                              │
                                          ▼                                                              ▼
                             ┌─────────────────────────┐                                   ┌──────────────────────────┐
                             │    FAISS Vector Store   │                                   │        BM25 Index        │
                             │ (output/vector_store/)  │                                   │   (output/bm25_store/)   │
                             └─────────────────────────┘                                   └──────────────────────────┘
                                          │                                                              │
                                          └───────────────────────┬──────────────────────────────────────┘
                                                                  ▼
                                                    [ output/index_manifest.json ]
                                              (Tracks file hashes for incremental indexing)

==================================================================================================
                     STAGE 2: ONLINE RETRIEVAL & CONVERSATIONAL QUERY APP
       (Executed via Streamlit app.py on College LAN or Cloud Web App via Gemini / OpenAI)
==================================================================================================

    [ Authorized Admin Staff ]
                │
                ▼
    ┌─────────────────────────┐
    │  Streamlit Auth Layer   │ ──► Verifies bcrypt password against config/auth_config.yaml
    └─────────────────────────┘
                │ (Authenticated Session)
                ▼
    [ Admin Question / Follow-up ]
                │
                ▼
    ┌─────────────────────────┐
    │  Conversational Memory  │ ──► Reformulates follow-ups into standalone search queries
    └─────────────────────────┘
                │
                ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                         HYBRID SEARCH RETRIEVER                             │
    │  ┌───────────────────────────────┐     ┌─────────────────────────────────┐  │
    │  │ Dense Semantic Search (FAISS) │     │ Sparse Keyword Search (BM25)    │  │
    │  └───────────────────────────────┘     └─────────────────────────────────┘  │
    │                  │                                      │                   │
    │                  └───────────────────┬──────────────────┘                   │
    │                                      ▼                                      │
    │                   Reciprocal Rank Fusion (RRF Scoring)                      │
    └─────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼ (Top-K Hybrid Candidates)
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                       CROSS-ENCODER RERANKER                                │
    │            (cross-encoder/ms-marco-MiniLM-L-6-v2)                           │
    │           Scores (Query, Passage) pairs via deep self-attention             │
    └─────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼ (Top-N Most Relevant Clauses)
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                         SWITCHABLE LLM BACKEND                              │
    │   [LLM_PROVIDER=google] (Gemini) | [ollama] (Local) | [openai] (Cloud API)  │
    │                                                                             │
    │  System Prompt Enforces:                                                    │
    │  1. Strict grounding solely in retrieved passages.                          │
    │  2. Absolute hallucination mitigation ("I could not find this...").         │
    │  3. Mandatory document title and page number citations.                     │
    └─────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                             STREAMLIT UI                                    │
    │  - Grounded Answer                                                          │
    │  - Expandable Source Citations (Document Name, Page Number, Rerank Score)  │
    │  - Multi-Turn Dialogue History                                              │
    │  - Reset / Clear Conversation Option                                        │
    └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Technology Stack & Component Rationale

| Layer | Technology | Justification / Rationale |
| :--- | :--- | :--- |
| **Language** | Python 3.10 – 3.14 | Standard ecosystem for enterprise AI pipelines. |
| **RAG Framework** | LangChain (`langchain-core`, `langchain-community`) | Modular document representations, embeddings abstraction, and chat message flow. |
| **Dense Vector Store** | **FAISS** (`faiss-cpu`) | High-performance in-memory vector index with instant local disk serialization; zero external server dependencies. |
| **Sparse Keyword Search** | **BM25** (`rank_bm25`) | BM25Okapi algorithm provides exact keyword/code/acronym recall (e.g., `SBOTE`, `KGP/ADM/2026/104-A`). |
| **Dense Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`), Google GenAI (`text-embedding-004`), OpenAI (`text-embedding-3-small`) | Local zero-cost 384-dim dense semantic embeddings or cloud API embeddings switchable via `EMBEDDING_PROVIDER`. |
| **Reranking** | `Cross-Encoder` (`cross-encoder/ms-marco-MiniLM-L-6-v2`) | Full query-passage cross-attention to eliminate false positives from hybrid retrieval before LLM context generation. |
| **LLM Backends** | **Google Gemini** (`gemini-3.5-flash-lite`), **Ollama** (`llama3.1`), **OpenAI** (`gpt-4o-mini`) | Switchable via `.env` or UI selector. Gemini provides fast cloud inference; Ollama provides 100% offline air-gapped data privacy; OpenAI gives enterprise flexibility. |
| **PyTorch Optimization** | PyTorch CPU (`--extra-index-url .../whl/cpu`) | CPU-only wheel (~180 MB download vs. ~3 GB CUDA wheel) preventing container memory/disk exhaustion on Streamlit Community Cloud. |
| **OCR Ingestion** | `rapidocr-onnxruntime` + `pypdfium2` (fallback to `pytesseract` + `pdf2image`) | High-speed, pure-Python / ONNX OCR engine requiring zero external system binaries; extracts text from scanned PDF circulars and image notices with persistent disk caching. |
| **User Interface** | **Streamlit** + Custom CSS (`static/style.css`) | Institutional J&K Government branding, glowing document download cards (`static/docs/`), interactive suggestion chips, live statistics pill, and sticky query bar. |
| **Access Control** | `streamlit-authenticator` + `bcrypt` | Secure administrative authentication with salted bcrypt password hashing stored in YAML config. |
| **Automated Testing** | **pytest** (`tests/`) | 26 passing unit and integration tests verifying chunking, embeddings, conversational memory, and hybrid retrieval. |
| **CI / Cloud Keepalive** | **GitHub Actions** (`.github/workflows/keepalive.yaml`) | Automated cron ping workflow ensuring 24/7 uptime for the live Streamlit Community Cloud deployment. |

---

### 3.1 Multipurpose Enterprise Applications of the LLM Layer

While standard search engines only return raw links, the integrated LLM backend (**Google Gemini / Ollama / OpenAI**) performs a wide spectrum of intelligent administrative tasks:

1. **Context-Aware Grounded Q&A (Anti-Hallucination Retrieval)**:
   - Synthesizes clear, accurate responses grounded *strictly* in retrieved college notices.
   - Enforces explicit document names, order reference codes, and page citations.
   - Gracefully declines out-of-scope questions to prevent misinformation.

2. **Conversational Query Reformulation & Pronoun Resolution**:
   - Resolves multi-turn follow-up questions (e.g., *"What about for diploma lateral entry students?"*) into self-contained search queries using conversational history.

3. **Administrative Notice & Circular Drafting**:
   - Assists departmental clerks and HODs in drafting official office orders, exam notifications, and leave approvals following standard Jammu & Kashmir government formatting and institutional tone.

4. **Multi-Page Executive Summarization & Key Takeaways**:
   - Automatically extracts critical action items, eligibility conditions, deadline schedules, and fee breakdowns from lengthy government gazettes and complex multi-page circulars.

5. **Cross-Policy Synthesis & Comparative Analysis**:
   - Compares past circulars against updated regulations to identify policy revisions (e.g., shifts in attendance condonation rules or revised scholarship income ceilings).

6. **Student Helpdesk & Simplified Language Adaptation**:
   - Converts dense, legalistic administrative jargon into simple, student-friendly explanations and FAQ guides.

---

## 4. Repository Structure

```
kgp-gyankosh/
├── README.md                         # Complete project documentation and capstone report
├── PROJECT_ARCHITECTURE_AND_FEATURES.md # Deep architectural inventory and codebase reference
├── requirements.txt                  # Python dependencies (CPU-optimized PyTorch)
├── pytest.ini                        # Pytest configuration
├── .env.example                      # Documented configuration template
├── .env                              # Active environment configuration
├── .gitignore                        # Exclusion of temporary caches and private keys
├── .github/
│   └── workflows/
│       └── keepalive.yaml            # Streamlit Cloud keepalive ping automation
├── config/
│   ├── auth_config.yaml.example      # Template for authentication accounts
│   └── auth_config.yaml              # Active bcrypt administrative credentials
├── data/
│   ├── sample_notices/               # 7 representative administrative documents
│   ├── pdf/                          # Official scanned and digital PDF circulars
│   └── word/                         # Official Microsoft Word (.docx) circulars
├── src/
│   ├── ingestion/
│   │   ├── ocr_loader.py             # OCR engine (RapidOCR / Tesseract fallback)
│   │   ├── document_loader.py        # Unified multi-format parser with metadata tracking
│   │   └── chunker.py                # High-speed recursive text splitting with chunk IDs
│   ├── indexing/
│   │   ├── embeddings.py             # Switchable SentenceTransformers / Gemini / OpenAI embeddings
│   │   └── vector_store.py           # FAISS & BM25 persistence with MD5 manifest tracking
│   ├── retrieval/
│   │   ├── hybrid_search.py          # Vector + BM25 Reciprocal Rank Fusion (RRF)
│   │   └── reranker.py               # Cross-encoder reranking (ms-marco-MiniLM)
│   ├── memory/
│   │   └── conversation_memory.py    # Multi-turn history & query reformulation
│   ├── llm/
│   │   └── client.py                 # Switchable Gemini / Ollama / OpenAI client with anti-hallucination prompt
│   └── auth/
│       └── authenticator.py          # Streamlit-authenticator bcrypt login manager
├── static/
│   ├── style.css                     # Enterprise dark glassmorphism design system
│   ├── campus_bg.jpg                 # Polytechnic campus visual background
│   └── docs/                         # Static served documents for direct download citations
├── tests/
│   ├── test_chunker.py               # Unit tests for text chunking & metadata
│   ├── test_embeddings.py            # Unit tests for embedding provider factory
│   ├── test_memory.py                # Unit tests for conversation memory & reformulation
│   └── test_retrieval.py             # Unit tests for BM25, RRF fusion, and CrossEncoder
├── scripts/
│   ├── test_queries.py               # Headless CLI test harness for query evaluation
│   ├── create_diagrams.py            # Architectural diagram generation script
│   ├── generate_documentation_pdf.py # Project report PDF generator
│   └── generate_line_by_line_doc_pdf.py # Line-by-line codebase PDF generator
├── build_index.py                    # Stage 1: Batch offline indexing CLI
├── app.py                            # Stage 2: Streamlit web application
└── output/
    ├── index_manifest.json           # Registry of 1,158 documents and 8,166 clauses
    ├── index_logs/                   # Execution logs per indexing run
    ├── ocr_cache/                    # Precomputed OCR text cache for scanned documents
    ├── vector_store/                 # Serialized FAISS index (index.faiss, index.pkl)
    └── bm25_store/                   # Serialized BM25 model & tokenized corpus
```

---

## 5. Setup & Installation Instructions

### Prerequisites
- **Python 3.10+** (Tested on Python 3.10, 3.11, 3.12, 3.14)
- **Git**
- *(Built-in Pure-Python OCR)*: Scanned document OCR runs out-of-the-box using `rapidocr-onnxruntime` and `pypdfium2` (no external software installation needed). Standard `Tesseract` and `Poppler` remain optionally supported.
- *(Optional for Local Offline LLM)*: [Ollama](https://ollama.com/) with `ollama pull llama3.1`.

### Step 1: Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/your-username/kgp-gyankosh.git
cd kgp-gyankosh

python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Copy the `.env.example` template to `.env`:
```bash
cp .env.example .env
```
Edit `.env` to suit your desired setup (see [Environment Variables Reference](#6-environment-variables-reference) below).

### Step 4: Configure Authentication
Copy `config/auth_config.yaml.example` to `config/auth_config.yaml`:
```bash
cp config/auth_config.yaml.example config/auth_config.yaml
```
Pre-configured default credentials for evaluation:
- **Administrator**: Username `admin_kgp` / Password `KgpAdmin@2026` (Principal / Head of Admin)
- **Academic Staff**: Username `clerk_academic` / Password `Academic@2026` (Academic Section In-Charge)
- **Examination Wing**: Username `exam_wing` / Password `Exam@2026` (Controller of Examinations)
- **HOD / Faculty**: Username `hod_polytechnic` / Password `Faculty@2026` (Senior Faculty)

### Step 5: Run Automated Verification Tests
Verify system integrity by executing the complete test suite:
```bash
pytest -v
```
*(All 26 unit and integration tests validate text chunking, embedding fallbacks, conversation memory reformulation, BM25 tokenization, and reciprocal rank fusion).*

---

## 6. Environment Variables Reference

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `LLM_PROVIDER` | `google` | Selected LLM backend: `google` (default Gemini), `ollama` (local, free), or `openai` (cloud). |
| `GOOGLE_API_KEY` | `""` | Google Gemini API Key (required if `LLM_PROVIDER=google`). |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Google Gemini model name. |
| `OLLAMA_MODEL` | `llama3.1:latest` | Ollama model name when `LLM_PROVIDER=ollama`. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Endpoint URL of local Ollama daemon. |
| `OPENAI_API_KEY` | `""` | OpenAI API Key (required if `LLM_PROVIDER=openai`). |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI chat model. |
| `EMBEDDING_PROVIDER` | `sentence-transformers` | `sentence-transformers` (local), `google`, or `openai`. |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Hugging Face embedding model name for local embeddings. |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model for passage reranking. |
| `ENABLE_RERANKER` | `true` | Toggle cross-encoder reranking stage. |
| `HYBRID_ALPHA` | `0.5` | RRF balance: 0.0 (pure BM25) to 1.0 (pure FAISS vector). 0.5 is balanced. |
| `CHUNK_SIZE` | `500` | Target chunk character limit for text splitting. |
| `CHUNK_OVERLAP` | `100` | Overlap in characters between adjacent chunks. |
| `DATA_DIR` | `data/sample_notices` | Source directory containing administrative documents. |
| `VECTOR_STORE_DIR` | `output/vector_store` | Target storage folder for FAISS index files. |
| `BM25_STORE_DIR` | `output/bm25_store` | Target storage folder for serialized BM25 index. |
| `AUTH_CONFIG_PATH` | `config/auth_config.yaml` | Path to local authentication configuration. |
| `TESSERACT_CMD` | `""` | Path to `tesseract.exe` (leave blank if in system PATH). |
| `POPPLER_PATH` | `""` | Path to Poppler `Library/bin` folder. |

---

## 7. Execution Guide: Two-Stage Workflow

### Stage 1: Build & Update Index (`build_index.py`)
Run the indexing pipeline whenever new circulars or orders are added to `data/sample_notices/`:

```bash
# Standard incremental indexing (processes only new or modified files)
python build_index.py

# Force full rebuild from scratch
python build_index.py --rebuild

# Index from a custom document folder
python build_index.py --data-dir path/to/notices
```

**Output:**
- Inspect console output or timestamped execution log in `output/index_logs/index_run_YYYYMMDD_HHMMSS.log`.
- Serializes FAISS index to `output/vector_store/` and BM25 index to `output/bm25_store/`.
- Updates `output/index_manifest.json` with file hashes for future incremental runs.

### Stage 2: Run Local Web Application (`app.py`)
Launch the Streamlit interface locally or on your campus LAN:
```bash
streamlit run app.py
```
Access the application in your web browser at:
`http://localhost:8501` (or local network IP `http://192.168.x.x:8501`).

### Stage 3: Live Cloud Web Deployment (Streamlit Community Cloud)
The repository is production-ready for instantaneous cloud deployment:
1. **Repository Linkage**: Connect your GitHub repository (`kgp-gyankosh`) to [Streamlit Community Cloud](https://share.streamlit.io/).
2. **Pre-Built Indices**: The repository contains pre-built FAISS and BM25 index artifacts in `output/`, eliminating query-time indexing overhead and build timeouts.
3. **CPU-Optimized PyTorch**: `requirements.txt` is configured with the PyTorch CPU extra-index, ensuring clean builds well within Streamlit Cloud's container RAM and disk quotas.
4. **Cloud Secrets Configuration**: In the Streamlit Cloud Dashboard under **App Settings → Secrets**, configure:
   ```toml
   LLM_PROVIDER = "google"
   GOOGLE_API_KEY = "AIzaSy..."
   GEMINI_MODEL = "gemini-3.5-flash-lite"
   EMBEDDING_PROVIDER = "sentence-transformers"
   ```
5. **Continuous Keepalive**: The repository includes `.github/workflows/keepalive.yaml`, which runs automated scheduled pings every 3 days to prevent Streamlit Cloud from putting the application to sleep.
6. **Live URL**: [https://kgp-gyankosh.streamlit.app/](https://kgp-gyankosh.streamlit.app/)

---

## 8. Switching Between Local (Ollama) and Cloud (OpenAI)

Switching requires **zero code changes** — simply edit `.env`:

### 1. To run with Google Gemini (Active & Recommended):
```env
LLM_PROVIDER=google
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
EMBEDDING_PROVIDER=sentence-transformers
```

### 2. To run 100% locally and free (Ollama):
```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:latest
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_PROVIDER=sentence-transformers
```
Ensure Ollama is running in the background:
```bash
ollama run llama3.1
```

### 3. To switch to OpenAI API:
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_PROVIDER=openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```
*(If switching embedding providers, re-run `python build_index.py --rebuild` to regenerate the vector index with the new embedding dimension).*

---

## 9. Sample Inputs & Grounded Outputs (Evaluation Test Cases)

### Test Case 1: Fact Retrieval with Exact Citation
- **User Question:** *"What is the minimum attendance required for diploma engineering students, and can the Principal condone shortage?"*
- **KGP Gyankosh Grounded Answer:**
  > According to the official attendance policy of Kashmir Government Polytechnic College, Srinagar:
  > - All registered diploma engineering students must maintain a mandatory minimum overall attendance of **75%** in both theory lectures and laboratory practicals to be eligible for the SBOTE Semester Final Examinations.
  > - The Principal holds the discretionary authority to condone an attendance deficiency of up to **10%** (i.e. between 65.0% and 74.9%) exclusively under verified circumstances such as prolonged illness/hospitalization or official deputation in sports/NCC/NSS.
  > - Any attendance aggregate strictly below 65.0% is non-condonable under any circumstances, and the candidate will be detained.
  > 
  > **Sources:**
  > - `KGP_Order_Attendance_and_Leave_Rules_2026.pdf` (Page 1)

### Test Case 2: Multi-Turn Contextual Follow-Up (Conversational Memory)
- **User Question (Turn 2):** *"What is the deadline to submit medical certificates for it?"*
- **Reformulated Internal Query:** *"What is the deadline to submit medical certificates for attendance condonation in KGP attendance rules?"*
- **KGP Gyankosh Grounded Answer:**
  > Medical leave applications and certificates from a Government Hospital or Registered Medical Practitioner must be submitted to the respective Head of Department within **three (3) working days** of resumption of classes. Delayed applications submitted at the end of the semester will not be entertained.
  > 
  > **Sources:**
  > - `KGP_Order_Attendance_and_Leave_Rules_2026.pdf` (Page 1)

### Test Case 3: Anti-Hallucination & Unsupported Claim Handling
- **User Question:** *"What is the mess fee refund policy for students residing in the PhD research scholar hostel?"*
- **KGP Gyankosh Grounded Answer:**
  > **I could not find this information in the available official documents.**
  > *(The indexed college records contain guidelines for 3-Year Diploma programmes, but do not contain policies or provisions regarding PhD research scholar hostels).*
  > 
  > **Sources:**
  > - None (Query ungrounded in available documents)

---

## 10. Key Architectural & Design Decisions

### 1. Why a Strict Two-Stage Architecture?
Combining document loading/OCR/chunking directly into the query application introduces severe latency (10–30+ seconds per query) and causes memory thrashing in multi-user environments. Structurally isolating indexing into an offline batch script (`build_index.py`) ensures:
- Pre-computed embeddings and indices are loaded into memory in sub-second time.
- Heavy OCR and chunking operations run once and never block online staff queries.
- Zero query-time indexing overhead.

### 2. Why Hybrid Search (FAISS + BM25) with Reciprocal Rank Fusion?
Pure vector search frequently struggles with exact official identifiers (e.g. order numbers like `KGP/EXAM/2026/89`, acronyms like `SBOTE`, `JKBOSE`, or specific fee figures like `Rs. 2,200`). BM25 provides precise lexical matching. Merging them via **Reciprocal Rank Fusion (RRF)** ensures that documents matching either deep semantic concepts OR exact administrative codes rank at the top.

### 3. Why Cross-Encoder Reranking?
Dual-encoders (bi-encoders) embed queries and documents independently to maximize search speed. While efficient for initial candidate filtering, they lack inter-token interaction. The **Cross-Encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) performs cross-attention over `(query, passage)` pairs simultaneously, re-scoring the top-10 hybrid candidates to eliminate irrelevant passages before context injection into the LLM.

### 4. Why Role-Based Authentication with Local BCrypt Hashing?
Government polytechnic circulars often contain internal administrative directives not intended for public internet browsing. `streamlit-authenticator` with salted bcrypt password hashing provides robust internal protection without exposing passwords in plain text. Storing credentials in `config/auth_config.yaml` (strictly gitignored) prevents credential leakage to version control.

### 5. Why Dual Support for Local (Ollama) and Cloud (Google Gemini / OpenAI)?
The system is built with a plug-and-play LLM abstraction layer:
- **Offline Campus LAN (Ollama `llama3.1`)**: Ensures complete data residency, zero API subscription overhead, and air-gapped privacy for confidential administrative records.
- **Online Cloud Deployment (Google Gemini / OpenAI)**: Enables effortless one-click hosting on cloud platforms (such as Streamlit Community Cloud or Hugging Face Spaces) without requiring local GPU servers or dedicated hardware.

---

## 11. Limitations & Operational Scope

1. **Flexible Deployment Scope (Campus LAN & Cloud Web App)**: The system natively supports both internal intranet deployment (e.g. `http://192.168.1.50:8501` using local Ollama) and secure public cloud deployment (e.g., Streamlit Community Cloud using Google Gemini / OpenAI with encrypted secrets).
2. **Authentication Scope**: Authentication relies on an encrypted local YAML configuration file (`config/auth_config.yaml`) with salted bcrypt password hashing rather than enterprise Active Directory / SAML Single Sign-On (SSO). This provides robust security tailored for institutional and departmental capstone scope.
3. **Scanned Document Quality**: OCR performance is dependent on image resolution and clarity of physical scans. Faint photocopies or distorted carbon-copies may require manual inspection.

---

## 12. Submission Checklist Cross-Reference

| Official Project 2 Requirement | Implementation Location in KGP Gyankosh | Verification Status |
| :--- | :--- | :--- |
| **Document Ingestion (≥2 formats)** | `src/ingestion/document_loader.py` (PDF, DOCX, TXT, PNG) | Completed |
| **OCR Ingestion for Scanned Docs** | `src/ingestion/ocr_loader.py` (`rapidocr-onnxruntime` + `pypdfium2` / `pytesseract`) | Completed |
| **Document Chunking** | `src/ingestion/chunker.py` (`RecursiveCharacterTextSplitter`) | Completed |
| **Dense Embeddings** | `src/indexing/embeddings.py` (`sentence-transformers` / `openai`) | Completed |
| **Local Vector Store** | `src/indexing/vector_store.py` (FAISS serialization) | Completed |
| **Incremental Re-indexing** | `build_index.py` (`output/index_manifest.json` MD5 tracking) | Completed |
| **Hybrid Search (Vector + BM25)** | `src/retrieval/hybrid_search.py` (Reciprocal Rank Fusion) | Completed |
| **Reranking Step** | `src/retrieval/reranker.py` (`cross-encoder/ms-marco-MiniLM`) | Completed |
| **Conversational Memory** | `src/memory/conversation_memory.py` (Query reformulation) | Completed |
| **Source Citations** | `src/llm/client.py` & `app.py` (Expandable citation cards) | Completed |
| **Hallucination Mitigation** | `src/llm/client.py` (Strict system prompt & empty fallback) | Completed |
| **Switchable LLM Backend** | `src/llm/client.py` (Google Gemini default vs Ollama vs OpenAI via `.env`) | Completed |
| **Access Control Layer** | `src/auth/authenticator.py` (`streamlit-authenticator` + `bcrypt`) | Completed |
| **Streamlit User Interface** | `app.py` & `static/style.css` (Dark glassmorphism UI, suggestion chips, live badges) | Completed |
| **Document Download Cards** | `app.py` & `static/docs/` (Responsive document cards for direct citation downloads) | Completed |
| **Automated Testing Suite** | `tests/` & `pytest.ini` (26 unit and integration tests passing) | Completed |
| **Cloud Deployment & Keepalive** | `.github/workflows/keepalive.yaml` (Streamlit Cloud zero-downtime hosting) | Completed |
| **Logging & Error Handling** | `output/index_logs/` & standard logging across all modules | Completed |
| **Official Administrative Data** | `data/` (1,158 official documents, 8,166 clauses indexed in FAISS and BM25) | Completed |
| **No Committed Secrets** | `.gitignore` protecting `.env` and `config/auth_config.yaml` | Completed |
| **Documentation & Diagram** | `README.md` & `PROJECT_ARCHITECTURE_AND_FEATURES.md` (Comprehensive documentation) | Completed |

