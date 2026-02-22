# 🎓 VAIT — Virtual Academic Intelligence Terminal

> **"We Never Let You Wait for Anything."**

VAIT is a production-grade **Retrieval-Augmented Generation (RAG)** AI assistant built for **Vasireddy Venkatadri International Technology (VVIT)**. It delivers accurate, source-cited answers about academic calendars, regulations, syllabi, notices, examinations, placements, departments, and more — all powered by local LLM inference with zero cloud dependency.

VAIT is **not** a general-purpose chatbot. It is a **strict institutional knowledge retrieval system** that refuses to answer when context is insufficient, never guesses, and never fabricates information.

---

## 📌 Table of Contents

- [Tech Stack](#-tech-stack)
- [Architecture](#-architecture)
- [Key Features](#-key-features)
- [RAG Pipeline](#-rag-pipeline)
- [Project Structure](#-project-structure)
- [API Endpoints](#-api-endpoints)
- [Getting Started](#-getting-started)
- [Management CLI](#-management-cli)
- [Knowledge Base](#-knowledge-base)
- [Configuration](#-configuration)
- [Changelog](#-changelog)

---

## 🛠 Tech Stack

### Backend (Python)

| Technology | Version | Purpose |
|---|---|---|
| **FastAPI** | 0.109.2 | Async web framework |
| **Uvicorn** | 0.27.1 | ASGI server |
| **Pydantic** | 2.6.1 | Data validation & settings |
| **FAISS (faiss-cpu)** | 1.7.4 | Vector similarity search |
| **Ollama** | External | Local LLM inference (phi3:mini) + embeddings (nomic-embed-text) |
| **NumPy** | 1.26.4 | Numerical operations |
| **PyPDF2** | 3.0.1 | PDF text extraction |
| **python-docx** | 1.1.0 | DOCX text extraction |
| **httpx** | 0.26.0 | Async HTTP client |
| **BeautifulSoup4** | 4.12.3 | HTML parsing for web crawling |
| **watchdog** | 4.0.0 | File system monitoring |
| **python-dotenv** | 1.0.1 | Environment variable management |
| **colorama** | 0.4.6 | Colored logging output |

### Frontend (JavaScript)

| Technology | Version | Purpose |
|---|---|---|
| **React** | ^19.2.4 | UI framework |
| **React DOM** | ^19.2.4 | DOM rendering |
| **React Router DOM** | ^7.13.0 | Client-side routing |
| **Vite** | ^7.3.1 | Build tool & dev server |
| **@vitejs/plugin-react** | ^5.1.4 | React support for Vite |

---

## 🏗 Architecture

```
┌───────────────────────┐        ┌──────────────────────────────────────────┐
│    React Frontend     │───────▶│           FastAPI Backend                │
│    (Vite, port 3002)  │ /api/  │           (Uvicorn, port 8000)          │
└───────────────────────┘ vait/  │                                          │
                                 │   Routes → Controller → RAG Service      │
                                 │                                          │
                                 │   ┌────────────┐   ┌────────────────┐   │
                                 │   │ Embedding   │   │  LLM Service   │   │
                                 │   │ Service     │   │  (Ollama)      │   │
                                 │   └──────┬──────┘   └───────┬────────┘   │
                                 │          │                  │             │
                                 │          ▼                  ▼             │
                                 │   ┌───────────┐    ┌──────────────┐      │
                                 │   │  FAISS    │    │   Ollama     │      │
                                 │   │  Index    │    │  (localhost  │      │
                                 │   │  (disk)   │    │   :11434)    │      │
                                 │   └───────────┘    └──────────────┘      │
                                 └──────────────────────────────────────────┘
```

**Design Principles:**
- **Clean separation** — Routes → Controllers → Services
- **Fully offline** — FAISS + Ollama = no internet required
- **Data sovereignty** — All documents and queries stay on institutional hardware
- **Zero hallucination** — Strict retrieval threshold + authority hierarchy + refusal mechanism

---

## ✨ Key Features

### 🤖 RAG Engine
- **19-step processing pipeline** with query normalization, intent classification, authority-weighted scoring, and hallucination guards
- **FAISS vector search** with cosine similarity (IndexFlatIP, L2-normalized)
- **Strict similarity threshold** (0.65) — refuses to answer when confidence is low
- **Authority-weighted scoring** — Regulations (×1.15) > Syllabi/Notices (×1.05) > Website/Social (×1.00)
- **LRU response cache** with 600s TTL (100-entry capacity)
- **Content deduplication** — hash-based + Jaccard word-overlap filtering (0.90 threshold)
- **Adjacent chunk merging** from the same source document
- **Confidence scoring** — High (≥0.85), Medium (≥0.75), Low (<0.75)
- **Structured answers** — Title, explanation, bullet points, source citations
- **Performance metrics** — embed_ms, retrieval_ms, generation_ms, total_ms

### 📄 Document Ingestion
- Batch CLI ingestion for **PDF**, **DOCX**, and **TXT** files
- Smart character-based chunking (~400 tokens / 1600 chars, 200-char overlap)
- Metadata inference from folder structure (document_type, authority_level, academic_year, department)
- Content-hash deduplication during ingestion

### 🌐 Website Crawling
- Domain-restricted crawling (only `vvitguntur.com` and `vvitu.ac.in`)
- Configurable depth limit (default 2) and max pages (default 200)
- Intelligent HTML cleaning — removes scripts, nav, headers, footers
- Polite crawling with 0.5s delay between requests
- URL filtering to skip login pages, admin panels, and media files

### 📱 Social Media Ingestion
- Structured JSON ingestion from **Instagram**, **LinkedIn**, and **YouTube**
- Text cleaning (zero-width chars, whitespace normalization)
- Human-readable text block composition with platform metadata

### 👁 File Watcher (Auto-Ingestion)
- Real-time file monitoring via **watchdog**
- Watches `uploads/`, `knowledge/notices/`, `announcements/`, `social/`
- Supports .pdf, .docx, .txt, .json
- 3-second debounce to prevent duplicate events
- Incremental FAISS ingestion without full rebuild

### 💬 Frontend UI
- Animated intro splash screen with phased VAIT branding
- Chat-based conversational interface
- **Department selector** — CSE, AI, ECE, IT, Mechanical, Civil
- **Academic year selector** — 2023-24, 2024-25, 2025-26
- **Conversation history** with time-grouped labels (Today, Yesterday, This Week, This Month, Older)
- **Category filtering** — Academic, Exams, Administration, Placements
- Conversation rename, delete, clear, and JSON export
- Suggestion chips for new conversations
- Confidence badge display (color-coded: High/Medium/Low)
- Source citation tags per message
- Typing animation & character counter (1000-char limit)
- Auto-scroll to latest message
- Mock AI fallback for development

### 🔐 Admin Features
- Single & bulk document ingestion via API
- File upload ingestion endpoint (.txt)
- Social media entries ingestion API
- Website crawl & reindex trigger
- Knowledge base statistics & clearing
- Comprehensive health dashboard

---

## 🔄 RAG Pipeline

The complete 19-step retrieval and generation flow:

```
1.  Query arrives → Normalize (expand shorthand)
2.  Cache check → LRU cache with 600s TTL
3.  Intent classification → Rule-based keyword matching (6 categories)
4.  Embedding → Ollama nomic-embed-text (768 dimensions)
5.  FAISS search → Top-K=6 nearest neighbors (cosine similarity)
6.  Threshold filter → Discard below 0.65 → refusal if none survive
7.  Deduplication → Remove by content hash
8.  Authority-weighted scoring → authority level + source boost + intent boost
9.  Adjusted threshold guard → top score < 0.55 → refusal
10. Near-duplicate removal → Jaccard overlap > 0.90 → discard lower
11. Adjacent chunk merging → Merge contiguous same-document chunks
12. Context cap → Top 4 chunks, max 8000 characters
13. LLM generation → Ollama phi3:mini with system prompt
14. Hallucination guard → Post-generation validation
15. Answer structure enforcement → Title, explanation, bullets, sources
16. Response polish → Clean spacing, dedup sentences, 600-word cap
17. Citation formatting → Append structured source block
18. Cache store → Save for future identical queries
19. Log → Structured entry to query log
```

---

## 📁 Project Structure

```
vait/
├── vait-backend/
│   ├── manage.py                      # Management CLI (reindex, ingest)
│   ├── requirements.txt               # Python dependencies
│   ├── app/
│   │   ├── main.py                    # FastAPI app setup & startup
│   │   ├── controllers/
│   │   │   └── chat_controller.py     # Chat request handling
│   │   ├── routes/
│   │   │   ├── chat.py                # Chat & debug API routes
│   │   │   └── admin.py               # Admin API routes
│   │   ├── services/
│   │   │   ├── embedding_service.py   # FAISS index & embeddings
│   │   │   ├── llm_service.py         # Ollama LLM interaction
│   │   │   ├── ollama_service.py      # Ollama health & helpers
│   │   │   ├── rag_service.py         # Full RAG pipeline
│   │   │   ├── website_crawler.py     # Domain-restricted web crawler
│   │   │   ├── social_ingestor.py     # Social media JSON ingestion
│   │   │   └── file_watcher.py        # Auto-ingestion via watchdog
│   │   └── utils/
│   │       ├── config.py              # Pydantic settings & config
│   │       └── text_chunker.py        # Character-based text chunking
│   ├── scripts/
│   │   └── ingest_documents.py        # Batch document ingestion CLI
│   ├── knowledge/                     # Institutional documents
│   │   ├── academic-calendar/
│   │   ├── departments/
│   │   ├── examinations/
│   │   ├── notices/
│   │   ├── regulations/
│   │   └── syllabus/
│   ├── social/                        # Social media JSON files
│   │   ├── instagram_posts.json
│   │   ├── linkedin_posts.json
│   │   └── youtube_updates.json
│   ├── data/
│   │   ├── raw_docs/                  # Input documents
│   │   ├── vector_store/              # FAISS index & metadata
│   │   └── logs/                      # Application & ingestion logs
│   ├── prompts/
│   │   └── vait_system_prompt.txt     # LLM system prompt
│   └── uploads/                       # File upload directory
│
├── vait-frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js                 # Vite config with proxy
│   └── src/
│       ├── App.jsx                    # Root component with routing
│       ├── main.jsx                   # Entry point
│       ├── components/
│       │   ├── ChatWindow.jsx         # Main chat area
│       │   ├── ChatInput.jsx          # Message input with char counter
│       │   ├── MessageBubble.jsx      # Individual message display
│       │   ├── Sidebar.jsx            # History & navigation sidebar
│       │   ├── TopBar.jsx             # Top navigation bar
│       │   ├── HistoryList.jsx        # Grouped conversation history
│       │   ├── DepartmentSelector.jsx # Department picker
│       │   └── YearSelector.jsx       # Academic year picker
│       ├── context/
│       │   └── ChatContext.jsx        # State management
│       ├── pages/
│       │   ├── Chat.jsx               # Chat page
│       │   └── Intro.jsx              # Animated intro splash
│       ├── styles/
│       │   └── index.css              # Global styles
│       └── utils/
│           └── mockAI.js              # Mock AI for development
│
└── README.md                          # ← You are here
```

---

## 🔌 API Endpoints

### Chat & Query

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Root health check |
| `GET` | `/health` | Comprehensive health dashboard with metrics |
| `POST` | `/api/vait/chat` | Primary chat — returns reply, sources, confidence, intent, performance |
| `POST` | `/api/vait/chat/detailed` | Chat with additional `is_refusal` flag |
| `GET` | `/api/vait/debug?query=...` | Raw retrieval diagnostics (no LLM call) |
| `GET` | `/api/vait/stats` | Knowledge base statistics |
| `GET` | `/api/vait/system_summary` | Full system self-summary |

### Admin

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/vait/admin/ingest` | Single document ingestion |
| `POST` | `/api/vait/admin/ingest/bulk` | Bulk document ingestion |
| `POST` | `/api/vait/admin/ingest/file` | File upload ingestion (.txt) |
| `POST` | `/api/vait/admin/ingest/social` | Social media entries ingestion |
| `POST` | `/api/vait/admin/reindex/websites` | Trigger website crawl & reindex |
| `GET` | `/api/vait/admin/knowledge-base/stats` | Detailed knowledge base stats |
| `DELETE` | `/api/vait/admin/knowledge-base/clear` | Clear entire knowledge base |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **Ollama** installed and running at `localhost:11434`
  - Models required: `phi3:mini` (LLM), `nomic-embed-text` (embeddings)

### Backend Setup

```bash
cd vait-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Pull Ollama models
ollama pull phi3:mini
ollama pull nomic-embed-text

# Ingest documents into the knowledge base
python scripts/ingest_documents.py

# Start the backend server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend Setup

```bash
cd vait-frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will be available at `http://localhost:3002/vait/` and proxies API requests to the backend.

---

## 🧰 Management CLI

```bash
cd vait-backend

# Crawl and reindex website content
python manage.py reindex_websites [--depth 2] [--max-pages 200]

# Ingest social media JSON files
python manage.py ingest_social

# Full reindex: documents + websites + social media
python manage.py reindex_all [--force]
```

---

## 📚 Knowledge Base

### Document Types & Authority Levels

| Source | Authority | Score Multiplier |
|---|---|---|
| Regulations (R20, R23) | 🔴 High | ×1.15 |
| Syllabi, Notices, Calendars | 🟡 Medium | ×1.05 |
| Website Content, Social Media | 🟢 Low | ×1.00 |

### Supported Formats
- **PDF** — via PyPDF2
- **DOCX** — via python-docx
- **TXT** — built-in
- **JSON** — social media structured data

### Intent Categories
Academic · Admissions · Examinations · Placements · Events · Infrastructure · General

---

## ⚙ Configuration

Key settings managed via `.env` + Pydantic Settings:

| Setting | Default | Description |
|---|---|---|
| Ollama URL | `localhost:11434` | Ollama server address |
| LLM Model | `phi3:mini` | Language model for generation |
| Embedding Model | `nomic-embed-text` | Embedding model (768-dim) |
| Embedding Dimensions | `768` | Vector dimensions |
| Similarity Threshold | `0.65` | Minimum retrieval score |
| Adjusted Threshold | `0.55` | Minimum authority-weighted score |
| Context Cap | `8000` chars | Maximum context for LLM |
| Max Chunks | `4` | Maximum retrieved chunks |
| Cache Size | `100` entries | LRU cache capacity |
| Cache TTL | `600` seconds | Cache time-to-live |
| Max Response Words | `600` | Word cap on answers |
| CORS Origins | `*` | Allowed CORS origins |
| Backend Port | `8000` | FastAPI server port |
| Frontend Port | `3002` | Vite dev server port |
| File Watcher | `disabled` | Auto-ingestion monitoring |
| Auto Reindex on Startup | `disabled` | Startup reindex trigger |

---

## 📝 Changelog

### v1.1.0 — Model Migration (February 2026)
- **Switched LLM from Mistral to phi3:mini** for better RAM efficiency
- Centralized model configuration in `config.py` + `.env`
- Added memory error handling (graceful response when model exceeds available RAM)
- Removed all OpenAI references (fully offline, no API keys needed)
- Cleaned legacy fallback logic

### v1.0.0 — Initial Release (February 2026)
- Full RAG pipeline with 19-step processing
- FastAPI backend with FAISS vector store & Ollama integration
- React 19 frontend with Vite 7
- Document ingestion (PDF, DOCX, TXT)
- Website crawler (domain-restricted to VVIT)
- Social media ingestion (Instagram, LinkedIn, YouTube)
- File watcher for automatic incremental ingestion
- Management CLI for reindexing operations
- Admin API for knowledge base management
- Chat UI with conversation history, department/year selectors, confidence badges, source citations
- Animated intro splash screen
- Mock AI fallback for development mode

---

## 👥 Team

Built for **VVIT (Vasireddy Venkatadri International Technology)** — Guntur, Andhra Pradesh, India.

---

## 📄 License

This project is proprietary to VVIT. All rights reserved.
