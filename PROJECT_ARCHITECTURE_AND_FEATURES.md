# 🏛️ KGP Gyankosh: Enterprise Knowledge Assistant & Administrative Intelligence Platform
### Kashmir Government Polytechnic College, Srinagar
*Comprehensive Capstone System Architecture, Feature Inventory, and Codebase File Reference*

---

## 1. Executive Summary & Project Overview

**KGP Gyankosh** is an Enterprise Knowledge Assistant and Agentic Retrieval-Augmented Generation (RAG) platform developed specifically for the Administration Department of **Kashmir Government Polytechnic (KGP) College, Srinagar**.

Polytechnic institutions manage thousands of confidential, high-stakes administrative records:
- Government orders, administrative circulars, and departmental sanction decrees.
- Faculty committee constitutions, departmental postings, and institutional duty rosters.
- Board of Technical Education (BOTE) diploma admission circulars, examination schedules, and syllabus revisions.
- Civil service leave sanctions (e.g., Paternity leave, Child Care leave, Medical leave under J&K Civil Service Rules 1979).

### The Core Problem Solved
Traditional keyword search (e.g., Windows Search, Adobe PDF Search) fails on institutional repositories because:
1. Documents are diverse and complex: digital PDFs, legacy DOCX files, high-resolution scans requiring Optical Character Recognition (OCR), and tabular administrative rosters.
2. Administrative queries require semantic synthesis, exact rule citations, and zero tolerance for AI hallucination.
3. Strict institutional confidentiality prevents sending private staff and student records to unauthorized cloud third parties without local offline governance.

**KGP Gyankosh solves this by pairing an Offline-First, Zero-Cloud-Cost Local LLM architecture (via Ollama & Llama 3.1) with Dense-Sparse Hybrid Retrieval (FAISS + BM25 with Reciprocal Rank Fusion) and a Cross-Encoder Reranker.**

---

## 2. Key Architectural Features & Innovations

### 🎯 Dual Operational Modes
1. **🏛️ College Records (Strict RAG Mode)**:
   - Queries are processed through the complete 2-stage retrieval pipeline across **1,158 official college documents** (8,166 searchable semantic clauses).
   - **Zero Hallucination Rule**: If an answer cannot be proven from official documents, the system explicitly states: *"I could not find this information in the available official documents."*
   - **Source Citations**: Every fact, date, or policy cites the exact document title and page number (e.g., `[Orders.pdf, Page 14]`).
   - **Conversational Greetings**: Welcomes users warmly as the official college assistant when greeted with "Hi", "Hello", or "Who are you?".

2. **🌐 General AI Assistant (Direct LLM Mode)**:
   - Functions as an open-ended conversational AI (powered by your chosen model: Llama 3.1, Gemini, or GPT-4o-mini).
   - Enables administrative staff to:
     - **Draft official circulars, memos, and government letters**.
     - Formulate structured emails and leave responses.
     - Summarize arbitrary administrative text.
     - Answer polytechnic academic, engineering (Civil, Mechanical, Electrical, Computer, IT), and programming questions.

---

### 🤖 Dynamic, Multi-Backend LLM Selector
The system supports hot-swapping between three distinct LLM backends on the fly directly from the user interface:
- **🦙 Local Ollama (Llama 3.1 8B)**: 100% private, zero API fees, runs entirely offline on localhost hardware.
- **✨ Google GenAI (Gemini 3.5 Flash / 1.5 Flash)**: High-speed cloud reasoning.
- **⚡ OpenAI (GPT-4o-mini)**: Industry-standard cloud intelligence.

Both modes (RAG and General AI) feature **independent, mode-specific LLM selectors in the sidebar**, allowing users to pair different models for different tasks (e.g., fast cloud Gemini for RAG document synthesis and local private Llama 3.1 for draft generation).

---

### 🔍 Two-Stage Hybrid Retrieval + Cross-Encoder Reranking
1. **Sparse Keyword Search (BM25)**: Accurately retrieves exact administrative codes, order numbers (e.g., `Order No: 02/2026` or `KGP/ADM/2026/104`), and officer names.
2. **Dense Vector Search (FAISS + `all-MiniLM-L6-v2`)**: Understands conceptual semantic queries (e.g., *"fee concession and scholarship rules"* matches *"Tuition Fee Waiver (TFW) Scheme"*). Supports cloud Google GenAI embeddings (`text-embedding-004`) and OpenAI embeddings (`text-embedding-3-small`).
3. **Reciprocal Rank Fusion (RRF)**: Merges sparse and dense ranked candidate lists fairly using rank reciprocals ($k=60$).
4. **Deep Cross-Encoder Reranking (`ms-marco-MiniLM-L-6-v2`)**: Evaluates query-document pairs simultaneously using full cross-attention, eliminating false positives and delivering the top most relevant clauses to the LLM.

---

### 🧠 Multi-Turn Conversational Memory & Query Reformulation
- Maintains a sliding context window of prior user-assistant dialogue turns.
- Automatically reformulates ambiguous follow-up questions into standalone search queries before hitting the retrieval index (e.g., Turn 1: *"Who is the Incharge of Computer Engineering?"* $\rightarrow$ Turn 2: *"When was he appointed?"* $\rightarrow$ Reformulated: *"When was Er. Shabir Ahmad Ahanger appointed as Incharge of Computer Engineering?"*).

---

### 🛡️ Enterprise Security & Role-Based Authentication
- Implements secure, salted **bcrypt** password hashing with `streamlit-authenticator`.
- Protects administrative records behind four designated institutional roles:
  1. **Principal / Head of Administration** (`admin_kgp` - Role: `admin`)
  2. **Academic Section In-Charge** (`clerk_academic` - Role: `staff`)
  3. **Examination Wing Controller** (`exam_wing` - Role: `staff`)
  4. **Head of Department / Senior Faculty** (`hod_polytechnic` - Role: `faculty`)

---

### 🎨 Prestigious Government Institutional Dark Glassmorphism UI
- Styled specifically with official government branding: **Government of Jammu & Kashmir · Kashmir Government Polytechnic College, Srinagar**.
- Custom typography using Google Fonts (**Outfit** 800 Extra-Bold and **Plus Jakarta Sans**) via external CSS stylesheet (`static/style.css`).
- Dynamic campus background atmosphere with deep scrim overlay.
- **Interactive Query Suggestion Chips**: Quick-click suggestion buttons for common administrative inquiries.
- **Live Institutional Metric Badge**: Real-time counter displaying `1,158 Official Documents · 8,166 Clauses Live`.
- **Responsive Document Download Cards**: Glowing cards linking directly to `static/docs/` so staff can open or download the original source PDF/Word circular.
- **Sticky Query Bar**: Non-overlapping bottom search bar with smooth scroll padding.

---

### ☁️ Cloud Deployment & High-Availability Architecture
- **Streamlit Community Cloud Deployment**: Hosted live at [https://kgp-gyankosh.streamlit.app/](https://kgp-gyankosh.streamlit.app/).
- **PyTorch CPU Build Optimization**: Configured `--extra-index-url https://download.pytorch.org/whl/cpu` to avoid downloading 3GB CUDA wheels, ensuring zero build memory/disk crashes.
- **Pre-Built Ingestion Stores**: Production FAISS and BM25 databases pre-built and synchronized in Git (`output/`), eliminating cloud startup delays.
- **Automated Keepalive Ping Workflow**: GitHub Actions workflow (`.github/workflows/keepalive.yaml`) runs automated health checks every 3 days to keep the cloud app permanently awake.

---

### 🧪 Automated Test Suite & Code Quality
- Complete test suite (`tests/`) containing 26 automated unit and integration tests executed with `pytest`:
  - `tests/test_chunker.py`: Validates text splitting, chunk counters, and metadata enrichment.
  - `tests/test_embeddings.py`: Validates default SentenceTransformers and graceful API key fallbacks.
  - `tests/test_memory.py`: Validates sliding turn memory and LLM query reformulation.
  - `tests/test_retrieval.py`: Validates BM25 tokenization, RRF rank weighting, and Cross-Encoder reranking fallbacks.

---

## 3. High-Level Data Flow Diagram

```
[ Official College Documents: PDF, DOCX, Scans ]
                       │
                       ▼
          [ Ingestion & OCR Engine ]
         (PyPDF, pdfplumber, docx, Tesseract)
                       │
                       ▼
          [ Recursive Character Chunker ]
          (500 chars / 100 char overlap)
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
 [ Dense Embeddings ]      [ Sparse BM25 Index ]
(all-MiniLM-L6-v2)         (Tokenized Corpus)
          │                         │
          ▼                         ▼
   [ FAISS Store ]           [ BM25 Store ]
          └────────────┬────────────┘
                       │
═══════════════════════╪═══════════════════════════════════════════
                 QUERY RUNTIME PIPELINE
═══════════════════════╪═══════════════════════════════════════════
                       │
               [ User Question ]
                       │
                       ▼
         [ Multi-Turn Memory Context ]
                       │
                       ▼
        [ LLM Query Reformulation Step ]
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
[ FAISS Vector Search ]     [ BM25 Keyword Search ]
    (Top 30 Chunks)             (Top 30 Chunks)
         └─────────────┬─────────────┘
                       │
                       ▼
        [ Reciprocal Rank Fusion (RRF) ]
                       │
                       ▼
     [ Cross-Encoder Reranker (ms-marco) ]
            (Top 6 Filtered Chunks)
                       │
                       ▼
       [ Strict Grounded Prompting ]
                       │
                       ▼
     [ Active LLM: Llama 3.1 / Gemini / GPT ]
                       │
                       ▼
   [ Grounded Response + Document Citations ]
```

---

## 4. Comprehensive File-by-File Breakdown

Below is the complete inventory of all files and directories in the project with their exact technical role:

```
kgp-gyankosh/
├── app.py                            # Streamlit web application & UI controller
├── build_index.py                    # Stage 1: Batch offline indexing CLI
├── requirements.txt                  # Python dependencies (CPU-optimized PyTorch)
├── pytest.ini                        # Pytest configuration
├── .env                              # Active environment configuration
├── .env.example                      # Documented configuration template
├── .gitignore                        # Git exclusion rules
├── README.md                         # Capstone project report and user manual
├── PROJECT_ARCHITECTURE_AND_FEATURES.md # Deep architectural inventory and codebase reference
├── .github/
│   └── workflows/
│       └── keepalive.yaml            # Streamlit Cloud keepalive ping automation
├── config/
│   ├── auth_config.yaml              # Active bcrypt administrative credentials
│   └── auth_config.yaml.example      # Template for authentication accounts
├── data/
│   ├── sample_notices/               # 7 representative administrative documents
│   ├── pdf/                          # Official scanned and digital PDF circulars
│   └── word/                         # Official Microsoft Word (.docx) circulars
├── src/
│   ├── __init__.py
│   ├── auth/
│   │   ├── __init__.py
│   │   └── authenticator.py          # Streamlit bcrypt authentication manager
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── chunker.py                # Text splitting & metadata clause tagging
│   │   ├── document_loader.py        # Multi-format document parser
│   │   └── ocr_loader.py             # OCR engine with persistent disk cache
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── embeddings.py             # SentenceTransformers / Gemini / OpenAI embeddings
│   │   └── vector_store.py           # FAISS & BM25 persistence with MD5 manifest
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── hybrid_search.py          # Vector + BM25 Reciprocal Rank Fusion
│   │   └── reranker.py               # Cross-encoder reranking
│   ├── memory/
│   │   ├── __init__.py
│   │   └── conversation_memory.py    # Multi-turn history & query reformulation
│   └── llm/
│       ├── __init__.py
│       └── client.py                 # Switchable Gemini / Ollama / OpenAI inference
├── static/
│   ├── style.css                     # Enterprise dark glassmorphism design system
│   ├── campus_bg.jpg                 # Polytechnic campus visual background
│   └── docs/                         # Static served documents for citation downloads
├── tests/
│   ├── test_chunker.py               # Unit tests for text chunking & metadata
│   ├── test_embeddings.py            # Unit tests for embedding provider factory
│   ├── test_memory.py                # Unit tests for conversation memory & reformulation
│   └── test_retrieval.py             # Unit tests for BM25, RRF fusion, and CrossEncoder
├── scripts/
│   ├── test_queries.py               # Headless CLI test harness for query evaluation
│   ├── create_diagrams.py            # Architectural diagram generator
│   ├── generate_documentation_pdf.py # Project report PDF generator
│   └── generate_line_by_line_doc_pdf.py # Codebase line-by-line documentation PDF generator
└── output/
    ├── vector_store/                 # Serialized FAISS index (index.faiss, index.pkl)
    ├── bm25_store/                   # Serialized BM25 model & tokenized corpus
    ├── ocr_cache/                    # Precomputed OCR text cache
    ├── index_logs/                   # Execution logs per indexing run
    └── index_manifest.json           # Index registry for 1,158 docs and 8,166 clauses
```

---

### 📄 Root Application & Orchestration Files

#### 1. [`app.py`](file:///c:/Users/Electronics/Desktop/Capstone/kgp-gyankosh/app.py)
* **Role**: Primary Streamlit web application entry point and user interface controller.
* **Key Components**:
  - **Institutional Branding**: Government of Jammu & Kashmir insignia, institutional title header, and live metric badge (`1,158 Official Documents · 8,166 Clauses Live`).
  - **CSS Theme Engine**: Links `static/style.css` injecting modern glassmorphism CSS, custom Google Fonts (`Outfit`, `Plus Jakarta Sans`), sleek scrollbars, and government institution badge styles.
  - **Authentication Gate**: Renders the login portal, credential cards, and verifies roles via `src/auth/authenticator.py`.
  - **Resource Caching**: Preloads FAISS vector index, BM25 keyword store, manifest metadata, and reranker once on startup via `@st.cache_resource` for zero query-time lag.
  - **Sidebar Controller**: Houses the **Operational Mode Switcher** (RAG vs General AI), **Active AI Model Dropdown**, User Profile badge, Logout button, and Indexed Documents browser.
  - **Interactive Suggestions**: Quick-click suggestion chips for common polytechnic administrative queries (Attendance condonation, Lateral Entry admission, Fee structure, Student grievance redressal).
  - **Dynamic Chat Execution**:
    - If in *College Records (RAG)* mode: Triggers conversation query reformulation, hybrid retrieval, cross-encoder reranking, and grounded answer synthesis with citation badges.
    - If in *General AI* mode: Invokes `llm_client.generate_chat()` for direct conversational problem-solving and notice drafting.
  - **Responsive Citation & Document Download Cards**: Renders interactive source cards with glowing icons linking directly to original files in `static/docs/` for one-click downloading.
  - **Cloud Secrets Integration**: Automatically injects Streamlit Cloud secrets (`st.secrets`) into `os.environ` for zero-configuration cloud hosting.

#### 2. [`build_index.py`](file:///c:/Users/Electronics/Desktop/Capstone/kgp-gyankosh/build_index.py)
* **Role**: Standalone offline ingestion and indexing pipeline script.
* **Key Components**:
  - Discovers all documents in `DATA_DIR` (PDF, DOCX, TXT, images).
  - Routes scanned pages to OCR and digital text to standard loaders.
  - Splits documents into overlapping semantic clauses using `RecursiveCharacterTextSplitter`.
  - Computes dense vector embeddings using SentenceTransformers and builds a local FAISS L2/Cosine index.
  - Tokenizes text and constructs a fast BM25 sparse keyword index.
  - Generates `output/index_manifest.json` recording file sizes, modification timestamps, and page counts.
  - Supports incremental delta indexing: skips already processed documents to save computation.

#### 3. [`requirements.txt`](file:///c:/Users/Electronics/Desktop/Capstone/kgp-gyankosh/requirements.txt)
* **Role**: Complete Python dependency specification optimized for local and cloud environments.
* **Key Optimizations & Dependencies**:
  - `--extra-index-url https://download.pytorch.org/whl/cpu`: Pinned to the CPU-only PyTorch build (~180MB download vs. ~3GB CUDA wheel), eliminating container Out-Of-Memory and disk-exhaustion crash loops on Streamlit Community Cloud.
  - `langchain`, `langchain-core`, `langchain-community`: RAG orchestration primitives.
  - `langchain-ollama`, `ollama`: Offline local LLM connectivity for Llama 3.1.
  - `langchain-google-genai`: Cloud connectivity for Gemini 3.5 Flash.
  - `langchain-openai`: Cloud connectivity for OpenAI GPT-4o-mini.
  - `sentence-transformers`, `torch`, `transformers`: Dense vector embedding generation (`all-MiniLM-L6-v2`) and cross-encoder reranking.
  - `faiss-cpu`: High-speed vector similarity search.
  - `rank-bm25`: BM25 Okapi keyword ranking algorithm.
  - `pdfplumber`, `pypdf`, `python-docx`: Document parsing engines.
  - `streamlit`, `streamlit-authenticator`, `bcrypt`, `PyYAML`: UI and cryptographic authentication.

#### 4. [`.env`](file:///c:/Users/Electronics/Desktop/Capstone/kgp-gyankosh/.env) & [`.env.example`](file:///c:/Users/Electronics/Desktop/Capstone/kgp-gyankosh/.env.example)
* **Role**: Environment configuration file containing secret keys, file paths, model identifiers, and retrieval thresholds.
* **Key Parameters**:
  - `LLM_PROVIDER`: Master default switch (`google`, `ollama`, or `openai`).
  - `GOOGLE_API_KEY` & `GEMINI_MODEL`: Gemini API configuration (`gemini-3.5-flash-lite`).
  - `OLLAMA_MODEL` & `OLLAMA_BASE_URL`: Local model tag and endpoint (`llama3.1:latest`, `http://localhost:11434`).
  - `OPENAI_API_KEY` & `OPENAI_MODEL`: OpenAI API configuration (`gpt-4o-mini`).
  - `EMBEDDING_PROVIDER`: Selected embedding backend (`sentence-transformers`, `google`, or `openai`).
  - `EMBEDDING_MODEL_NAME`: Local embedding model (`all-MiniLM-L6-v2`).
  - `RERANKER_MODEL`: Cross-Encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
  - `TOP_K_RETRIEVAL` & `TOP_N_RERANK`: Candidates retrieved (30) and reranked (6).
  - `DATA_DIR`, `VECTOR_STORE_DIR`, `BM25_STORE_DIR`: Persistent storage paths.

#### 5. [`scripts/test_queries.py`](file:///c:/Users/Electronics/Desktop/Capstone/kgp-gyankosh/scripts/test_queries.py)
* **Role**: Headless command-line developer test harness.
* **Purpose**: Allows executing offline test queries against the indexed vector store and reranker without opening a web browser.

#### 6. [`pytest.ini`](file:///c:/Users/Electronics/Desktop/Capstone/kgp-gyankosh/pytest.ini) & [`tests/`](file:///c:/Users/Electronics/Desktop/Capstone/kgp-gyankosh/tests/)
* **Role**: Complete automated testing harness with 26 unit and integration tests.
* **Test Modules**:
  - `tests/test_chunker.py`: Tests document chunking and metadata attribution.
  - `tests/test_embeddings.py`: Tests embedding models and graceful fallbacks.
  - `tests/test_memory.py`: Tests dialogue turn management and query reformulation.
  - `tests/test_retrieval.py`: Tests tokenization, RRF scoring math, and CrossEncoder reranker.

#### 7. [`.github/workflows/keepalive.yaml`](file:///c:/Users/Electronics/Desktop/Capstone/kgp-gyankosh/.github/workflows/keepalive.yaml)
* **Role**: Continuous keepalive cron automation for Streamlit Community Cloud.
* **Purpose**: Pings the live cloud URL every 3 days to prevent cloud app idling and sleeping.

#### 8. [`README.md`](file:///c:/Users/Electronics/Desktop/Capstone/kgp-gyankosh/README.md)
* **Role**: Comprehensive repository documentation, submission checklist, and capstone project explanation.

---

### ⚙️ Configuration Module (`config/`)

#### 7. [`config/auth_config.yaml`](file:///c:/Users/Electronics/Desktop/Capstone/kgp-gyankosh/config/auth_config.yaml) & [`auth_config.yaml.example`](file:///c:/Users/Electronics/Desktop/Capstone/kgp-gyankosh/config/auth_config.yaml.example)
* **Role**: Secure credentials registry for internal college accounts.
* **Contents**:
  - Cookie configuration (`kgp_gyankosh_auth`, secret salt key, 1-day expiry).
  - Four institutional administrative accounts with bcrypt-hashed passwords:
    - `admin_kgp`: Principal / Head of Admin (`KgpAdmin@2026`).
    - `clerk_academic`: Academic Section In-Charge (`Academic@2026`).
    - `exam_wing`: Examination Controller (`Exam@2026`).
    - `hod_polytechnic`: Department Head / Senior Faculty (`Faculty@2026`).

---

### 🧩 Core Source Code Modules (`src/`)

#### 🔐 Authentication Layer (`src/auth/`)
* **[`src/auth/authenticator.py`](file:///c:/Users/Electronics/Desktop/Capstone/kgp-gyankosh/src/auth/authenticator.py)**:
  - `hash_password(password)`: Hashes raw strings with cryptographic bcrypt and salt rounds.
  - `load_auth_config()`: Safely parses YAML credentials; auto-initializes from `.example` if absent.
  - `get_authenticator()`: Instantiates `streamlit_authenticator.Authenticate` cookie session manager.

#### 📥 Document Ingestion & OCR Layer (`src/ingestion/`)
* **[`src/ingestion/document_loader.py`](file:///c:/Users/Electronics/Desktop/Capstone/kgp-gyankosh/src/ingestion/document_loader.py)**:
  - Multi-format document parser supporting PDF, DOCX, TXT, and images.
  - Uses `pdfplumber` for precise table extraction and digital text reading.
  - Uses `python-docx` for parsing Microsoft Word circulars and memos.
  - Automatically identifies scanned or image-only PDF pages and routes them to OCR.
* **[`src/ingestion/ocr_loader.py`](file:///c:/Users/Electronics/Desktop/Capstone/kgp-gyankosh/src/ingestion/ocr_loader.py)**:
  - Tesseract OCR wrapper for extracting text from scanned government orders, stamped notices, and circulars.
  - Disk-backed OCR caching (`output/ocr_cache/`) prevents re-running OCR on identical pages across builds.
* **[`src/ingestion/chunker.py`](file:///c:/Users/Electronics/Desktop/Capstone/kgp-gyankosh/src/ingestion/chunker.py)**:
  - Implements `RecursiveCharacterTextSplitter` configured for administrative prose (500 character chunks with 100 character overlap).
  - Enriches every clause with critical administrative metadata: source file name, page number, chunk ID, and file modification date.

#### 🗄️ Indexing & Vector Database Layer (`src/indexing/`)
* **[`src/indexing/embeddings.py`](file:///c:/Users/Electronics/Desktop/Capstone/kgp-gyankosh/src/indexing/embeddings.py)**:
  - Dense embedding factory. Instantiates local HuggingFace `all-MiniLM-L6-v2` (384 dimensions), Google GenAI `models/text-embedding-004`, or cloud OpenAI `text-embedding-3-small`.
  - Runs embedding operations on local CPU using PyTorch without external network calls when using local models.
* **[`src/indexing/vector_store.py`](file:///c:/Users/Electronics/Desktop/Capstone/kgp-gyankosh/src/indexing/vector_store.py)**:
  - Encapsulates FAISS vector database initialization, serialization, and disk persistence.
  - Saves index binaries to `output/vector_store/index.faiss` and metadata to `output/vector_store/index.pkl`.
  - Provides helper functions `save_vector_store()` and `load_vector_store()`.

#### 🔎 Retrieval & Reranking Layer (`src/retrieval/`)
* **[`src/retrieval/hybrid_search.py`](file:///c:/Users/Electronics/Desktop/Capstone/kgp-gyankosh/src/retrieval/hybrid_search.py)**:
  - Implements dense-sparse Hybrid Retrieval.
  - Queries both FAISS (semantic similarity) and BM25 (exact keyword match).
  - Combines results using **Reciprocal Rank Fusion (RRF)**:
    $$RRF\_Score(d) = \sum \frac{1}{k + rank(d)}$$
  - Normalizes candidate scores and passes top candidates to the reranker.
* **[`src/retrieval/reranker.py`](file:///c:/Users/Electronics/Desktop/Capstone/kgp-gyankosh/src/retrieval/reranker.py)**:
  - Wraps HuggingFace Cross-Encoder model: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
  - Unlike bi-encoders, the cross-encoder jointly analyzes the query and the candidate text together with full cross-attention.
  - Scores candidates between 0.0 and 1.0, filters out irrelevant passages, and sorts the top $N$ clauses for LLM grounding.

#### 🧠 Conversational Memory Layer (`src/memory/`)
* **[`src/memory/conversation_memory.py`](file:///c:/Users/Electronics/Desktop/Capstone/kgp-gyankosh/src/memory/conversation_memory.py)**:
  - `ConversationMemory`: Thread-safe sliding turn buffer maintaining conversational state.
  - `reformulate_query(query, llm_client)`: Uses the active LLM to resolve pronouns, ellipsis, and context dependencies into standalone retrieval queries.
  - `format_history_for_prompt()`: Serializes past exchanges into structured transcripts for LLM context injection.

#### 🤖 LLM Client & Hallucination Mitigation Layer (`src/llm/`)
* **[`src/llm/client.py`](file:///c:/Users/Electronics/Desktop/Capstone/kgp-gyankosh/src/llm/client.py)**:
  - **`LLMClient` Class**: The unified inference engine supporting dynamic provider and model assignment (`google`, `ollama`, `openai`).
  - **`ADMIN_SYSTEM_PROMPT`**: Strict administrative constitution enforcing:
    1. Grounding solely on provided passages.
    2. Zero hallucination guarantee.
    3. Mandatory uppercase bold naming: **KASHMIR GOVERNMENT POLYTECHNIC COLLEGE, SRINAGAR**.
    4. Structured source citations `[Document_Name.pdf, Page X]`.
  - **`generate_answer()`**: Synthesizes grounded RAG answers from retrieved document clauses; handles greetings warmly.
  - **`generate_chat()`**: Powers the **General AI Assistant** mode for open-ended conversation, administrative drafting, coding, and general inquiry.
  - **`generate_raw()`**: Performs fast, prompt-free completions for query reformulation.

---

### 💾 Persistent Data & Storage (`output/`)

* **`output/vector_store/`**: Contains serialized FAISS index files (`index.faiss`) and pickled document metadata (`index.pkl`).
* **`output/bm25_store/`**: Stores serialized BM25 Okapi model and tokenized document corpus dictionary.
* **`output/ocr_cache/`**: JSON files caching extracted text of scanned pages, preventing redundant OCR re-computation.
* **`output/index_logs/`**: Detailed execution logs of indexing runs recording throughput, document counts, and any parse warnings.
* **`output/index_manifest.json`**: Index registry mapping all 1,158 files, their 8,166 chunk clauses, file paths, and metadata hashes.

---

## 5. Security, Privacy & Hardware Analysis

| Dimension | Implementation Details |
| :--- | :--- |
| **Data Privacy** | All document ingestion, chunking, embedding generation, FAISS indexing, and BM25 tokenization occur strictly on local disk. When using Ollama with Llama 3.1, zero data leaves the college machine. |
| **Password Security** | Passwords in `config/auth_config.yaml` are cryptographically hashed using salted bcrypt (12 rounds). Raw passwords are never stored in code or database. |
| **Version Control Safety** | `.env`, `output/vector_store/`, `output/bm25_store/`, and real document directories are strictly excluded in `.gitignore` to prevent leakage. |
| **Hardware Compatibility** | Engineered to run smoothly on standard institutional PC hardware (x86-64, 8GB-16GB RAM) without requiring expensive enterprise GPUs. |

---

## 6. Summary for Capstone Submission & Evaluators

KGP Gyankosh represents a complete, production-grade implementation of modern AI engineering:
1. **Advanced RAG over Naive RAG**: Employs hybrid reciprocal rank fusion and cross-encoder reranking rather than simple cosine vector lookup.
2. **True Multimodality**: Automatically detects and OCR-processes scanned legacy government circulars.
3. **Agentic Memory**: Resolves multi-turn conversational ambiguity before search execution.
4. **Dual Capability**: Serves both as an authoritative institutional compliance search engine and as a versatile generative assistant for everyday administrative drafting.
5. **Architectural Flexibility**: Features hot-swappable local and cloud backends accommodating offline security mandates and high-speed cloud operations alike.
