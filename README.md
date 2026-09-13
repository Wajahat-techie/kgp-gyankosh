# KGP Gyankosh: Administrative Knowledge Assistant with Advanced RAG
### Internal Document Search & Intelligence Platform for Kashmir Government Polytechnic College, Srinagar
**Capstone Project: IIT Patna — Generative AI & Agentic AI for Developers**  
*Designed and Developed by **Wajahat** (Department of Computer Engineering)*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://kgp-gyankosh.streamlit.app/)
[![Tests](https://img.shields.io/badge/pytest-26%20passed-success)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue)](requirements.txt)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

- 🌐 **Live Web Application**: [https://kgp-gyankosh.streamlit.app/](https://kgp-gyankosh.streamlit.app/)
- 💻 **GitHub Repository**: [https://github.com/Wajahat-techie/kgp-gyankosh](https://github.com/Wajahat-techie/kgp-gyankosh)
- 📊 **Presentation Slide Deck**: [Download PPTX](docs/KGP_Gyankosh_Presentation.pptx) | [Download Notes DOCX](docs/KGP_Gyankosh_Presentation_Notes.docx)
- 📑 **Technical Reports**: [System Architecture PDF](docs/reports/KGP_Gyankosh_System_Architecture_and_Files_Reference.pdf) | [Complete Documentation PDF](docs/reports/KGP_Gyankosh_Complete_Documentation.pdf)

---

## 1. About the Project & Motivation

Hi! I am **Wajahat**, and this repository contains **KGP Gyankosh**, an administrative AI assistant and Retrieval-Augmented Generation (RAG) system that I designed and built for **Kashmir Government Polytechnic (KGP) College, Srinagar** (established in 1958 under the Department of Skill Development, Government of Jammu & Kashmir).

### Why I Built This
Having observed the day-to-day administrative workflow at our college, I realized that a huge amount of staff time is spent hunting down specific government orders, circulars, and notifications across paper binders and disorganized file folders:
- **Scanned and image-only orders**: Many official orders are physically stamped, signed, and scanned. Standard text search (like `Ctrl+F` in Adobe or Windows Search) cannot find words inside them.
- **Strict precision required**: In government administration, approximations are not acceptable. If a clerk asks about Child Education Allowance rates, attendance shortage condonation limits, or guest faculty remuneration points, the answer must cite the exact order number, date, and clause.
- **Risk of AI hallucinations**: Off-the-shelf LLMs have no access to internal college records and will confidently guess incorrect dates, fees, or policies.
- **Privacy and offline needs**: Public educational institutions often need the ability to run completely offline on local machines (via Ollama) without uploading internal documents to third-party cloud services, while still supporting fast cloud inference (via Google Gemini) for remote access.

To solve this, I built **KGP Gyankosh** — an end-to-end, two-stage Advanced RAG pipeline combining dense vector embeddings, BM25 keyword matching, Reciprocal Rank Fusion (RRF), Cross-Encoder reranking, OCR fallback, and strict anti-hallucination guardrails.

---

## 2. Project Presentation Slide Deck (20 Slides)

I prepared a 20-slide presentation deck covering the problem, technical architecture, engineering choices, test results, and live deployment.

- 📥 **PowerPoint Presentation File**: [`docs/KGP_Gyankosh_Presentation.pptx`](docs/KGP_Gyankosh_Presentation.pptx) *(Widescreen 16:9 with full speaker notes)*
- 📝 **Presentation Notes Document**: [`docs/KGP_Gyankosh_Presentation_Notes.docx`](docs/KGP_Gyankosh_Presentation_Notes.docx)

### Slide-by-Slide Overview

| Slide # | Title | Key Talking Points & Content Summary |
| :--- | :--- | :--- |
| **01** | **Title & Project Overview** | Project name, institutional context (KGP Srinagar, Estd 1958), Capstone Project 2 for IIT Patna GenAI program. |
| **02** | **The Real-World Problem** | Information fragmentation across physical files and scanned PDFs; why basic keyword search fails; high precision needs in governance. |
| **03** | **The Solution: KGP Gyankosh** | Core system capabilities: multi-format ingestion, OCR layer, hybrid search (FAISS + BM25), Cross-Encoder reranking, and 1-click citation downloads. |
| **04** | **System Architecture** | Two-stage design: Stage 1 Offline Ingestion (`build_index.py`) and Stage 2 Interactive Web App (`app.py`). |
| **05** | **Document Ingestion & OCR** | Multi-format loader supporting digital/scanned PDFs, DOCX, and TXT; pure-Python RapidOCR + pypdfium2 with persistent caching. |
| **06** | **Recursive Text Chunking** | Chunk size (500 chars), overlap (100 chars), semantic separator hierarchy, and deterministic chunk ID assignment (`source#pX_cY`). |
| **07** | **Dense Semantic Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`, 384 dimensions); local CPU inference; fallbacks for Google GenAI and OpenAI. |
| **08** | **Sparse Lexical Search (BM25)** | BM25Okapi implementation for exact administrative tokens, order numbers, employee names, and official acronyms (`BOTE`, `SBOTE`). |
| **09** | **Dense-Sparse Hybrid Retrieval (RRF)** | Reciprocal Rank Fusion formula with constant $k=60$ merging top vector and BM25 candidates into a balanced top-10 pool. |
| **10** | **Cross-Encoder Reranking** | Second-stage scoring with `ms-marco-MiniLM-L-6-v2` applying joint cross-attention to eliminate false positives and output top-4 passages. |
| **11** | **Conversational Memory** | Multi-turn sliding window history; LLM-assisted query reformulation resolving pronouns and context into standalone search strings. |
| **12** | **Strict Grounding & Anti-Hallucination** | System prompt guardrails; mandatory refusal message if facts are not found in official records; verified source page references. |
| **13** | **Switchable LLM Backends** | Dynamic sidebar switcher: Google Gemini Flash (default cloud), local Ollama with Llama 3.1 (100% private), and OpenAI GPT-4o-mini. |
| **14** | **Role-Based Authentication** | Salted bcrypt password hashing with `streamlit-authenticator` across 4 administrative roles (Admin, Academic, Exam, Faculty). |
| **15** | **User Interface & Design** | Custom dark glassmorphic UI (`static/style.css`), live metric badge, suggestion chips, responsive citation cards, and non-overlapping input dock. |
| **16** | **Automated Testing & QA** | 26 unit and integration tests passing in `pytest` (chunking, embeddings, memory, retrieval); CLI evaluation script `test_queries.py`. |
| **17** | **Live Online Cloud Deployment** | Streamlit Cloud live link; PyTorch CPU build optimization (reduced size from 3GB to 180MB); GitHub Actions keepalive workflow. |
| **18** | **Demonstrated Administrative Cases** | Verified real-world test cases: attendance condonation, Child Education Allowance, student expulsion/suspension orders, guest faculty criteria. |
| **19** | **Engineering Challenges & Solutions** | Resolving degraded scanned OCR, bridging formal administrative terminology with colloquial queries, and staying within cloud memory limits. |
| **20** | **Summary & Future Roadmap** | Key achievements delivered; future BOTE examination portal integration; multilingual support for Urdu and Kashmiri. |

---

## 3. System Architecture & Pipeline Flow

The system uses a strict **Two-Stage Architecture** to guarantee that user queries return in under a second without any indexing latency during web sessions:

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

## 4. Key Technical Features

### 1. Dense + Sparse Hybrid Search with Reciprocal Rank Fusion
Neither vector embeddings nor keyword matching alone is sufficient for administrative records:
- **Vector search (`all-MiniLM-L6-v2`)** understands conceptual semantics (e.g., searching for *"financial allowance for school fees"* matches *"Child Education Allowance"* even without word overlap).
- **BM25 search (`rank_bm25`)** guarantees exact matches on specific circular codes (e.g. `Order No: 02/2026`), roll numbers, acronyms (`BOTE`, `PEM`, `NSS`), and names.
- Both streams are combined using **Reciprocal Rank Fusion (RRF)**:
  $$\text{RRF Score}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{w_m}{60 + \text{rank}_m(d)}$$

### 2. Cross-Encoder Reranking
While bi-encoders encode queries and passages independently, the Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) performs joint cross-attention across the full `(query, passage)` pair. Reranking the top 10 hybrid candidates prunes irrelevant chunks and sends only the top 4 highest-confidence passages to the LLM.

### 3. Multi-Turn Conversational Memory with Query Reformulation
When users ask follow-up questions (e.g., Turn 1: *"Who is the Incharge of Computer Engineering?"* followed by Turn 2: *"When was he appointed?"*), the conversation memory module automatically reformulates Turn 2 into a self-contained search query (*"When was Er. Shabir Ahmad Ahanger appointed as Incharge of Computer Engineering?"*) before retrieval.

### 4. Interactive Citation Badges & 1-Click Document Downloads
In addition to text citations, every cited order generates:
- An expandable citation card showing the source file, page number, match score, and excerpt.
- A **direct download button** that serves either the exact extracted single-page PDF slice or the complete original document.

### 5. Dual Operational Modes
- **🏛️ College Records (Strict RAG Mode)**: Strictly grounded in college circulars with mandatory citations and anti-hallucination refusal.
- **🌐 General AI (Direct LLM Mode)**: Functions as an administrative assistant to draft formal circulars, memos, leave approvals, and answer technical questions.

### 6. Role-Based Bcrypt Authentication
Configured with salted bcrypt hashes in `config/auth_config.yaml` supporting 4 distinct user roles:
- `admin_kgp` (Principal / Head of Administration)
- `clerk_academic` (Academic Section In-Charge)
- `exam_wing` (Examination Controller)
- `hod_polytechnic` (Head of Department / Senior Faculty)

---

## 5. Technology Stack

| Layer | Component / Library | Version | Role in System |
| :--- | :--- | :--- | :--- |
| **Language** | Python | 3.10 – 3.14 | Core backend language |
| **Frontend** | Streamlit | `>=1.30.0` | Reactive web interface with custom CSS |
| **Vector DB** | FAISS (`faiss-cpu`) | `>=1.7.4` | In-memory dense similarity search & disk serialization |
| **Keyword Index** | Rank-BM25 | `>=0.2.2` | BM25Okapi sparse lexical scoring |
| **Embeddings** | Sentence-Transformers | `>=2.2.2` | Local `all-MiniLM-L6-v2` embeddings (384 dims) |
| **Reranker** | Cross-Encoder | `>=2.2.2` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **LLM Inference** | Google Gemini / Ollama / OpenAI | Switchable | Cloud speed (Gemini Flash) or 100% offline privacy (Llama 3.1) |
| **OCR Engine** | RapidOCR + pypdfium2 | `>=1.3.0` | Pure Python ONNX OCR with disk caching |
| **Document Loaders** | pypdf, python-docx | `>=3.0.0` | Parsing digital PDFs and Word files |
| **Authentication** | streamlit-authenticator + bcrypt | `>=0.3.0` | Salted password hashing and session tokens |
| **PyTorch Wheel** | PyTorch CPU (`--extra-index-url`) | `>=2.0.0` | Lightweight ~180MB build for cloud hosting |
| **Test Suite** | pytest | `>=7.4.0` | 26 automated unit and integration tests |

---

## 6. Project Structure

```
kgp-gyankosh/
├── README.md                           # Main documentation and personal capstone report
├── PROJECT_ARCHITECTURE_AND_FEATURES.md# Deep technical architecture documentation
├── app.py                              # Streamlit web application & UI
├── build_index.py                      # Offline document ingestion and indexing pipeline
├── requirements.txt                    # Production dependencies (PyTorch CPU optimized)
├── pytest.ini                          # Pytest configuration
├── .env.example                        # Configuration template for public cloning
├── .env                                # Local environment settings (no secrets committed)
├── .gitignore                          # Excludes secrets, caches, and local files
├── .github/
│   └── workflows/
│       └── keepalive.yaml              # GitHub Actions ping to prevent app hibernation
├── config/
│   ├── auth_config.yaml.example        # Template auth config with demo bcrypt hashes
│   └── auth_config.yaml                # Role-based login credentials
├── data/                               # College notices and orders directory
│   ├── pdf/                            # PDF administrative orders and circulars
│   └── word/                           # Microsoft Word (.docx) circulars
├── docs/                               # Presentation slides and technical reports
│   ├── KGP_Gyankosh_Presentation.pptx  # 20-slide presentation deck with speaker notes
│   ├── KGP_Gyankosh_Presentation_Notes.docx # Complete slide notes and talking points
│   └── reports/                        # PDF reports on system architecture and code
├── output/                             # Generated indices and metadata
│   ├── vector_store/                   # Serialized FAISS vector index (index.faiss, index.pkl)
│   ├── bm25_store/                     # Serialized BM25 model and corpus (pkl)
│   ├── index_manifest.json             # Document metadata and MD5 hash tracker
│   └── ocr_cache/                      # Persistent OCR text caches
├── src/                                # Core application source code
│   ├── auth/                           # Authentication handler
│   ├── ingestion/                      # Document loading, chunking, and OCR
│   ├── indexing/                       # Embeddings and vector store management
│   ├── retrieval/                      # Hybrid search (RRF) and Cross-Encoder reranker
│   ├── memory/                         # Conversation memory & query reformulation
│   └── llm/                            # LLM client factory & grounding prompts
├── static/                             # Web assets and custom styling
│   ├── style.css                       # Modern dark glassmorphic CSS design system
│   ├── campus_bg.jpg                   # Institutional campus background image
│   └── docs/                           # Web-accessible mirrors for document downloads
└── tests/                              # Automated test suite (26 passing tests)
```

---

## 7. Local Setup & Quickstart Guide

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
*(The `requirements.txt` is configured with `--extra-index-url https://download.pytorch.org/whl/cpu` to install the lightweight CPU-only PyTorch build, saving ~2.8 GB of disk space).*

### Step 4: Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` to set your preferred LLM provider:
```env
# Default: Google Gemini (Free API Key from Google AI Studio)
LLM_PROVIDER=google
GOOGLE_API_KEY=your_google_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# Optional: Local Ollama (100% Free & Offline)
# LLM_PROVIDER=ollama
# OLLAMA_MODEL=llama3.1:latest
# OLLAMA_BASE_URL=http://localhost:11434

# Optional: OpenAI
# LLM_PROVIDER=openai
# OPENAI_API_KEY=your_openai_api_key_here
# OPENAI_MODEL=gpt-4o-mini
```

### Step 5: Build or Rebuild the Search Indices
Run the offline ingestion pipeline to chunk all documents in `data/` and generate the FAISS and BM25 search stores:
```bash
python build_index.py --rebuild
```

### Step 6: Launch the Streamlit Web App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 8. Demo User Accounts

You can log in with any of the following pre-configured administrative accounts (hashes stored in `config/auth_config.yaml`):

| Username | Role | Display Name | Password |
| :--- | :--- | :--- | :--- |
| `admin_kgp` | `admin` | Principal / Head of Administration | `KgpAdmin@2026` |
| `clerk_academic` | `staff` | Academic Section In-Charge | `Academic@2026` |
| `exam_wing` | `staff` | Examination Controller | `Exam@2026` |
| `hod_polytechnic` | `faculty` | Head of Department / Senior Faculty | `Faculty@2026` |

---

## 9. Automated Testing & Verification

The codebase includes **26 automated unit and integration tests** verifying every component of the RAG pipeline.

Run the test suite with:
```bash
pytest -v
```

### Test Coverage Summary
- `tests/test_chunker.py`: Verifies recursive chunk boundary splitting, overlap preservation, and deterministic chunk ID assignment.
- `tests/test_embeddings.py`: Tests SentenceTransformers embedding generation, output dimension validation (384), and provider fallback logic.
- `tests/test_memory.py`: Validates multi-turn conversation buffer windowing and LLM-assisted query reformulation.
- `tests/test_retrieval.py`: Tests BM25 tokenization, Reciprocal Rank Fusion scoring, and Cross-Encoder reranking.

**Result**: 26 passed in ~3.8s across Python 3.10 – 3.14.

---

## 10. Sample Questions to Try

Here are a few verified sample queries grounded in the indexed administrative documents:

1. **Faculty In-Charge Orders**:
   - *"Who has been assigned as the Incharge of Computer Engineering Department, and by which order?"*
   - *(Grounds to Order No: 23/2026, dated 05-05-2026, appointing Er. Shabir Ahmad Ahanger).*
2. **Disciplinary & Hostel Actions**:
   - *"What disciplinary actions were taken regarding the misconduct incident in the hostel on 11-05-2026?"*
   - *(Grounds to Order No: 30 of 2026, citing debarred and suspended students).*
3. **Child Education Allowance (CEA)**:
   - *"How much total Child Education Allowance was sanctioned under Order No 15 of 2026?"*
   - *(Grounds to Order No 15 of 2026, dated 29-04-2026, citing ₹13,16,250 released for 17 employees).*
4. **Student Picnic Approvals**:
   - *"Which faculty members were assigned to accompany the 6th Semester Computer Engineering picnic?"*
   - *(Grounds to Order No: 22/2026, dated 02-05-2026, listing Er. Irfan Ahmad Sofi and accompanying faculty).*
5. **Anti-Hallucination Refusal Test**:
   - *"What is the policy for PhD hostel room allocation and mess refunds?"*
   - *(Refuses gracefully because KGP is a 3-year diploma polytechnic college and no PhD records exist).*

---

## 11. Author & Contact

**Wajahat**  
Department of Computer Engineering  
Kashmir Government Polytechnic College, Srinagar  
📧 Email: `kgpolysgr58@gmail.com`  
📞 Contact: `+91 9906457756`  
🔗 Project Live URL: [https://kgp-gyankosh.streamlit.app/](https://kgp-gyankosh.streamlit.app/)

---

## 12. License

This project is licensed under the [MIT License](LICENSE).

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
