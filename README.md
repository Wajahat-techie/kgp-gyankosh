# KGP Gyankosh: Administrative Knowledge Assistant with Advanced RAG
### Internal Document Search & Intelligence Platform for Kashmir Government Polytechnic College, Srinagar
**Capstone Project: IIT Patna — Generative AI & Agentic AI for Developers**  
*Enterprise Administrative Knowledge Assistant with Advanced Two-Stage RAG*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://kgp-gyankosh.streamlit.app/)
[![Tests](https://img.shields.io/badge/pytest-26%20passed-success)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue)](requirements.txt)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

- 🌐 **Live Web Application**: [https://kgp-gyankosh.streamlit.app/](https://kgp-gyankosh.streamlit.app/)
- 💻 **GitHub Repository**: [https://github.com/Wajahat-techie/kgp-gyankosh](https://github.com/Wajahat-techie/kgp-gyankosh)
- 📊 **Presentation Slide Deck**: [Download PPTX (17.8 MB)](docs/KGP_Gyankosh_Presentation.pptx) | [Download Presentation Notes (DOCX)](docs/KGP_Gyankosh_Presentation_Notes.docx)
- 📑 **Technical Reports**: [System Architecture PDF](docs/reports/KGP_Gyankosh_System_Architecture_and_Files_Reference.pdf) | [Complete Documentation PDF](docs/reports/KGP_Gyankosh_Complete_Documentation.pdf)

> [!IMPORTANT]
> **Quick LLM Setup Notice**:
> - **To run with Cloud AI (Fastest)**: Copy `.env.example` to `.env` and add your **Google Gemini API Key** (`GOOGLE_API_KEY=...`) or **OpenAI API Key** (`OPENAI_API_KEY=...`).
> - **To run 100% Free & Offline (Zero Cloud Cost / Full Data Privacy)**: Install [Ollama](https://ollama.com/), run `ollama run llama3.1`, and set `LLM_PROVIDER=ollama` in `.env`. **No API keys or internet connection required!**

---

## 1. Project Background & Institutional Motivation

**Kashmir Government Polytechnic (KGP) College, Srinagar** (established in 1958 under the Department of Skill Development, Government of Jammu & Kashmir) is one of the premier technical diploma institutions in the region. The administrative offices—including the Principal's Secretariat, Academic Section, Examination Wing, and Departmental Heads—handle thousands of official records every academic year:
- Government orders, sanction decrees, and financial release circulars (such as Child Education Allowances).
- J&K State Board of Technical Education (SBOTE) notifications, examination schedules, and syllabus revisions.
- Student attendance policies, disciplinary proceedings, and admission rosters.
- Faculty service records, committee notifications, and civil service leave sanctions governed by J&K Civil Service Regulations (CSR).

### The Core Problem Solved
Administrative staff and academic counselors routinely spend hours manually searching for information scattered across physical filing cabinets and multi-page digital records:
1. **Scanned Documents Without Text Layers**: Many official government orders and notices are uploaded as scanned image PDFs with stamps and signatures. Standard keyword searches (`Ctrl+F` in Adobe or Windows Search) fail completely because there is no selectable text layer.
2. **High Precision Requirements in Governance**: Answering queries like *"What is the condonation limit for shortage of attendance?"* or *"Who is eligible for Child Education Allowance?"* requires citing exact clause numbers, dates, and order codes. Approximate answers are unacceptable in an administrative setting.
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

## 5. Retrieval & Reranking Strategy

### Why Single-Method Search Fails
- **Vector Search alone**: Struggles with exact administrative order numbers (e.g., `Order No: 23/2026`), acronyms (`BOTE`, `PEM`, `NSS`), and monetary figures.
- **BM25 Keyword Search alone**: Completely misses questions phrased with colloquial synonyms (e.g., *"financial help for school fees"* failing to match *"Child Education Allowance"*).

### Dense-Sparse Hybrid Search with Reciprocal Rank Fusion (RRF)
1. **Dense Vector Search (Semantic Understanding)**:
   - Model: `all-MiniLM-L6-v2` (SentenceTransformers, 384 dimensions)
   - Store: FAISS dense vector database
   - Function: Captures semantic concepts, natural language synonyms, and intent.
2. **Sparse Keyword Search (Lexical Precision)**:
   - Algorithm: `BM25Okapi` with alphanumeric tokenization
   - Store: Serialized BM25 model & tokenized corpus
   - Function: Exact matching for circular reference numbers, employee names, and official abbreviations.
3. **Reciprocal Rank Fusion**:
   $$\text{RRF Score}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{w_m}{60 + \text{rank}_m(d)}$$
   Combines candidate lists into a balanced top-10 pool.

### Cross-Encoder Reranker
- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Evaluates the full `(query, passage)` pair with joint self-attention across transformer layers.
- Filters out false positives from the candidate pool and delivers the top-4 highest-confidence passages to the LLM.

---

## 6. Conversational Memory & Zero-Hallucination Guardrails

### Multi-Turn Context & Query Reformulation
Administrative inquiries frequently involve follow-up questions with pronouns:
- **Turn 1 User**: *"Who is the Incharge of Computer Engineering?"*
- **Turn 1 Assistant**: Explains that Er. Shabir Ahmad Ahanger was assigned via Order No: 23/2026.
- **Turn 2 User**: *"When was he appointed?"*
- **Query Reformulation**: The memory module detects anaphora and reformulates Turn 2 into:
  `"When was Er. Shabir Ahmad Ahanger appointed as Incharge of Computer Engineering?"`
- **Result**: Direct hit on the appointment order in the retrieval database.

### Strict Grounding & Anti-Hallucination Policy
- **System Prompt Rules**:
  1. Answer strictly and solely based on the retrieved official administrative context.
  2. If the requested fact is missing from the indexed records, reply with:
     > *"I could not find this information in the available official documents."*
  3. Reference exact document filenames and page numbers for every stated fact.
  4. Never invent rules, dates, or policies.
- **Interactive Citation & Download Badges**:
  Every citation rendered in the answer includes a direct download link allowing staff to download that specific extracted single-page PDF slice or the full original document for audit verification.

---

## 7. Dual Operational Modes & Model Flexibility

### Dual Operational Modes (Sidebar Toggle)
1. **🏛️ College Records (Strict RAG Mode)**:
   - Full hybrid retrieval and reranking active.
   - Grounded strictly in college circulars with mandatory citations and 1-click download cards.
2. **🌐 General AI Assistant (Direct LLM Mode)**:
   - Bypasses retrieval to act as an open administrative assistant.
   - Helps staff draft formal government circulars, memos, meeting notices, and emails according to J&K administrative conventions.

### Multi-Backend LLM Selector (No Vendor Lock-In)
- **✨ Google Gemini Flash** (`gemini-2.5-flash` / `gemini-1.5-flash`): High-speed cloud reasoning; default for online web deployment.
- **🦙 Local Ollama** (`llama3.1:latest`): 100% offline, private, zero-token-fee inference on college workstation or LAN server.
- **⚡ OpenAI** (`gpt-4o-mini`): High-accuracy cloud fallback.

---

## 8. Role-Based Access Control (RBAC) & Security

Implemented with `streamlit-authenticator` using salted **bcrypt password hashes** in `config/auth_config.yaml`:

| Role | Username | Full Name / Scope | Default Password |
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
| **2** | *"What disciplinary action was taken regarding the hostel incident on 11-05-2026?"* | 3 students debarred from hostel for academic year; 1 student suspended for 20 days and ordered to repair victim's phone; 6 students suspended for 20 days. | `Orders.pdf` (Page 27, Order No: 30 of 2026) |
| **3** | *"How much Child Education Allowance was sanctioned under Order No 15 of 2026?"* | Total of ₹13,16,250 released in favour of 17 institutional employees under GO (I) No. 473-F. | `Orders.pdf` (Page 13, Order No: 15 of 2026) |
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

## 14. Presentation Slide Deck Overview

The complete 20-slide presentation deck is included in this repository:
- 📊 [`docs/KGP_Gyankosh_Presentation.pptx`](docs/KGP_Gyankosh_Presentation.pptx)
- 📝 [`docs/KGP_Gyankosh_Presentation_Notes.docx`](docs/KGP_Gyankosh_Presentation_Notes.docx)

| Slide # | Slide Title | Visual / Content Focus |
| :--- | :--- | :--- |
| **01** | Title & Project Overview | KGP Gyankosh branding, Capstone Project 2, IIT Patna GenAI program. |
| **02** | The Real-World Problem | Physical file silos, lack of OCR text layers, precision requirements in governance. |
| **03** | The Solution: KGP Gyankosh | Hybrid search, two-stage architecture, 1-click downloads, zero hallucination. |
| **04** | Project Objectives & System Scope | 5 core pillars: Ingestion, Hybrid RAG, Dual Deployment, Strict Grounding, RBAC. |
| **05** | System Architecture: Two-Stage Pipeline | Flow diagram separating Stage 1 (Ingestion) from Stage 2 (Interactive App). |
| **06** | Dataset Scale & Document Ingestion | Multi-format loader (PDF, DOCX, TXT), RapidOCR, MD5 manifest delta indexing. |
| **07** | Semantic Chunking Strategy | 500-char chunks, 100-char overlap, rich metadata tracking (`chunk_id`, `page`). |
| **08** | Hybrid Search (Vector + BM25) | Parallel tracks: Dense FAISS (`all-MiniLM-L6-v2`) + Sparse BM25Okapi. |
| **09** | Two-Stage Ranking (RRF & Cross-Encoder) | Reciprocal Rank Fusion ($k=60$) + `ms-marco-MiniLM-L-6-v2` cross-attention reranker. |
| **10** | Conversational Memory & Query Reformulation | Sliding window dialogue buffer and LLM anaphora query rewriting. |
| **11** | Zero-Hallucination Guardrails | Strict grounding policy, explicit refusal fallback, verified citation badges. |
| **12** | Dual Operational Modes | Side-by-side: 🏛️ College Records (RAG) vs. 🌐 General AI (Direct LLM Drafting). |
| **13** | Model Flexibility (Cloud Speed & Local Privacy)| Google Gemini Flash, OpenAI GPT-4o-mini, and offline local Ollama Llama 3.1. |
| **14** | Security & Role-Based Access Control | Salted bcrypt password hashing across 4 administrative tiers. |
| **15** | User Interface & Administrative Design System | Dark glassmorphism, Google Fonts, live metrics badge, non-overlapping input dock. |
| **16** | Automated Testing & QA | 26 automated unit tests passing in pytest; CLI benchmark harness. |
| **17** | Live Online Cloud Deployment | Streamlit Cloud deployment, PyTorch CPU optimization, keepalive workflow. |
| **18** | Demonstrated Administrative Cases | 5 verified real-world governance test cases with exact citations. |
| **19** | Engineering Challenges & Solutions | Resolving noisy OCR, vocabulary mismatch, and cloud memory optimization. |
| **20** | Summary, Roadmap & Conclusion | Delivered achievements, future BOTE portal integration, Urdu/Kashmiri support. |

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
- 📑 **System Architecture PDF**: [Download PDF](docs/reports/KGP_Gyankosh_System_Architecture_and_Files_Reference.pdf)
- 📖 **Complete Documentation PDF**: [Download PDF](docs/reports/KGP_Gyankosh_Complete_Documentation.pdf)

---

## 18. License & Institutional Attribution

This capstone project is developed as part of the **IIT Patna — Executive M.Tech / Certification in Generative AI & Agentic AI for Developers**.  
Institutional document records and administrative context: **Kashmir Government Polytechnic College, Srinagar**, Department of Skill Development, Government of Jammu & Kashmir.

Released under the [MIT License](LICENSE).
