# VAIT — Project History & Architecture Documentation

> **VAIT** — VVIT's Artificial Intelligence Technology  
> Official AI Assistant for Vasireddy Venkatadri International Technology (VVIT)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Phase 1 — Foundation Setup](#phase-1--foundation-setup)
3. [Phase 2 — Ingestion Pipeline](#phase-2--ingestion-pipeline)
4. [Phase 3 — API Integration & RAG Runtime](#phase-3--api-integration--rag-runtime)
5. [Architectural Decisions Record](#architectural-decisions-record)
6. [File Reference](#file-reference)

---

## Project Overview

VAIT is a production-grade Retrieval-Augmented Generation (RAG) system designed to serve thousands of students and faculty at VVIT. It answers institutional queries — academic calendars, regulations, syllabi, notices, examination rules — with **strict grounding** in verified documents.

### Core Design Principles

| Principle | Implementation |
|-----------|---------------|
| **No hallucinations** | Every response is grounded in retrieved context; if context is weak, VAIT refuses. |
| **Authority hierarchy** | Chunks are ranked by `authority_level` (high > medium > low) before LLM generation. |
| **Transparency** | Every answer includes the source document names and a confidence score. |
| **Consistent refusal** | A single, institutional refusal message is used verbatim when context is insufficient. |
| **Offline-first knowledge** | Knowledge is ingested offline via a CLI script; the runtime only reads FAISS + metadata. |

### Technology Stack

| Layer | Technology |
|-------|-----------|
| Web framework | FastAPI (Python 3.10+) |
| Vector database | FAISS (faiss-cpu) |
| Embeddings | OpenAI `text-embedding-3-large` (3072 dimensions) |
| LLM | OpenAI `gpt-4-turbo-preview` |
| Tokenisation | tiktoken (cl100k_base) |
| Configuration | pydantic-settings + python-dotenv |

**Explicitly excluded:** JavaScript, Node.js, ChromaDB, Pinecone, LangChain.

---

## Phase 1 — Foundation Setup

### What Was Built

- Modular backend directory structure following clean-architecture patterns.
- Core application scaffold with FastAPI, lifespan management, and CORS middleware.
- Centralised configuration module (`app/utils/config.py`) with validated settings.
- Directory layout for data storage: `data/raw_docs/`, `data/vector_store/`, `data/logs/`.
- Knowledge directory tree organised by document category: `knowledge/regulations/`, `knowledge/syllabus/`, etc.

### Why It Was Built

The structure was designed for **separation of concerns** — ingestion logic is isolated in `scripts/`, runtime services in `app/services/`, and HTTP interface in `app/routes/`. This ensures each layer can be tested and maintained independently.

### Files Involved

| File | Purpose |
|------|---------|
| `app/main.py` | FastAPI application factory, lifespan manager, logging setup |
| `app/utils/config.py` | All configuration constants, Settings class, directory paths |
| `app/__init__.py` | Package marker |
| `requirements.txt` | Locked Python dependencies |
| `.env` | Environment secrets (OPENAI_API_KEY) |

### Architectural Decisions

1. **FAISS over ChromaDB** — FAISS was selected because it provides direct control over the similarity computation (IndexFlatIP with L2-normalised vectors for cosine similarity), has zero external service dependencies, and fits the offline-ingestion / online-retrieval pattern perfectly. ChromaDB adds an abstraction layer and a separate process that are unnecessary for VAIT's deployment model.

2. **pydantic-settings for configuration** — Provides type-safe, validated configuration with automatic `.env` loading and environment variable override, reducing the risk of misconfiguration in production.

3. **Lifespan context manager** — The FAISS index and metadata are loaded once at startup and shared across requests via a global `rag_service` instance, eliminating redundant I/O on every query.

### Lessons Learned

- Keep the project root path computation (`Path(__file__).parent.parent.parent`) in one place to avoid inconsistencies.
- Validate critical configuration (API keys) at startup, not at first use, to surface issues immediately.

---

## Phase 2 — Ingestion Pipeline

### What Was Built

- Offline CLI script (`scripts/ingest_documents.py`) that scans document directories, extracts text from PDF / DOCX / TXT files, chunks it, derives metadata, generates embeddings, builds a FAISS index, and persists both the index and metadata to disk.
- Intelligent text chunker (`app/utils/text_chunker.py`) with tiktoken-based token counting, paragraph-aware splitting, and configurable overlap.
- Metadata inference engine that derives `document_type`, `authority_level`, `academic_year`, and `department` from the file's directory path.
- Rate-limit-resilient embedding generation with exponential backoff.

### Why It Was Built

The ingestion pipeline is **offline by design**. Running it separately from the API server ensures:
- No impact on query latency during re-ingestion.
- Deterministic, reproducible index builds.
- The ability to inspect and validate the index before deploying it.

### Files Involved

| File | Purpose |
|------|---------|
| `scripts/ingest_documents.py` | Full ingestion pipeline: scan → extract → chunk → embed → index → save |
| `app/utils/text_chunker.py` | Reusable chunking utility (also used by the runtime RAG service) |
| `data/vector_store/vait.index` | FAISS binary index (output) |
| `data/vector_store/metadata.json` | Chunk text + metadata (output) |
| `data/logs/ingestion.log` | Detailed ingestion log |

### Metadata Schema

Each chunk in `metadata.json` carries:

```json
{
  "text": "…chunk content…",
  "document_type": "regulation | syllabus | notice | website",
  "academic_year": "2025-26",
  "department": "CSE | ECE | General | …",
  "authority_level": "high | medium | low",
  "source_file": "filename.pdf"
}
```

### Chunking Strategy

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Chunk size | ~400 tokens | Balances context richness with embedding quality; exceeding ~512 tokens degrades embedding precision for `text-embedding-3-large`. |
| Overlap | ~50 tokens | Ensures continuity across chunk boundaries for multi-sentence answers. |
| Min chunk | 30 tokens | Discards extremely small trailing chunks that add noise. |
| Tokeniser | cl100k_base (tiktoken) | Exact token counting compatible with OpenAI models. |

### Why FAISS over ChromaDB

| Criterion | FAISS | ChromaDB |
|-----------|-------|----------|
| Runtime dependency | Single library, no server | Requires a background process or embedded server |
| Index control | Full control over index type (Flat, IVF, HNSW) | Abstracted away |
| Cosine similarity | Achieved via L2-normalised vectors + IndexFlatIP | Built-in but opaque |
| Persistence | Simple file-based (`.index` binary) | SQLite + Parquet |
| Suitability | Best for offline-ingestion + online-retrieval | Better for frequent dynamic updates |
| Complexity | Minimal API surface | Higher-level API with more moving parts |

For VAIT's use-case — infrequent batch ingestion with high-frequency reads — FAISS is the optimal choice.

### Lessons Learned

- Always sort OpenAI embedding response items by `index` — the API does not guarantee order.
- Atomic file writes (write to `.tmp`, then rename) prevent corruption if the process is interrupted mid-save.
- Companion `.json` files alongside documents allow manual metadata overrides without modifying the script.

---

## Phase 3 — API Integration & RAG Runtime

### What Was Built

1. **Configuration validation** — `.env` is loaded via python-dotenv at import time; `OPENAI_API_KEY` is validated on startup with a masked log message.

2. **Embedding service** (`app/services/embedding_service.py`) — Async OpenAI embedding calls with exponential-backoff retry for rate-limit and transient errors.

3. **LLM service** (`app/services/llm_service.py`) — Async chat completion calls with retry logic, low temperature (0.3) for factual responses, and a structured contextual prompt.

4. **RAG service** (`app/services/rag_service.py`) — The core pipeline:
   - Embed user query
   - Search FAISS index (Top-K = 4)
   - Apply strict similarity threshold (0.75 cosine similarity)
   - Refuse if no chunks meet threshold
   - Prioritise qualified chunks by `authority_level` (high > medium > low)
   - Build grounded context
   - Inject system prompt + context into LLM call
   - Return `{ reply, sources, confidence }`

5. **Confidence scoring**:
   - **High**: top similarity score ≥ 0.85
   - **Medium**: top similarity score ≥ 0.75
   - **Low**: below 0.75

6. **Refusal logic** — A single, consistent refusal message: *"This information is not available in the official VAIT records at this time."*

7. **Chat controller** (`app/controllers/chat_controller.py`) — Orchestrates RAG service calls and returns structured results.

8. **Chat API** (`app/routes/chat.py`):
   - `POST /api/vait/chat` → `{ reply, sources, confidence }`
   - `POST /api/vait/chat/detailed` → Adds `is_refusal` flag
   - `GET /api/vait/stats` → Knowledge base statistics

9. **Structured logging** — Every query is logged to `data/logs/query.log` as JSON-lines: `timestamp`, `question`, `confidence`, `sources`, `is_refusal`, `top_score`, `threshold`.

10. **System prompt** (`prompts/vait_system_prompt.txt`) — Hardened institutional prompt with absolute grounding rules, source citation requirements, authority-level awareness, and the exact refusal message.

### Safety & Edge Cases

| Scenario | Handling |
|----------|----------|
| Empty query | Returns structured error immediately (no API call) |
| Extremely long query (>2000 chars) | Rejected with clear message |
| FAISS index missing | RAG service initialises with empty index; queries return informative message |
| `metadata.json` missing | Same as above |
| API rate limit (OpenAI) | Exponential backoff with up to 3–5 retries |
| API failure (timeout, connection error) | Caught and returned as structured error, never a crash |
| No chunks above threshold | Exact refusal message returned |

### Files Involved

| File | Purpose |
|------|---------|
| `app/services/rag_service.py` | Core RAG pipeline with retrieval, scoring, authority sorting, confidence |
| `app/services/embedding_service.py` | Async embedding generation with retry |
| `app/services/llm_service.py` | Async LLM generation with retry |
| `app/controllers/chat_controller.py` | Business logic orchestration |
| `app/routes/chat.py` | HTTP API endpoints |
| `app/routes/admin.py` | Administrative endpoints (ingest, stats, clear) |
| `prompts/vait_system_prompt.txt` | Hardened system prompt |
| `data/logs/query.log` | Structured query audit log |
| `data/logs/vait.log` | Application-wide debug log |

### Architectural Decisions

1. **No question classification API call** — Removed the separate `classify_question` LLM call that categorised queries as factual/policy/procedure/definition. This saves one API round-trip per query and reduces latency. The confidence metric (derived from similarity scores) provides more actionable signal than a classification label.

2. **Authority-weighted context sorting** — Chunks are sorted by `(authority_weight, similarity_score)` descending before being injected into the LLM context. This ensures that official regulations take precedence over informal website content, even if both match the query.

3. **Confidence scoring from similarity** — Rather than asking the LLM to self-assess confidence, we derive it directly from FAISS similarity scores. This is deterministic, cost-free, and not susceptible to LLM overconfidence.

4. **Structured logging to query.log** — JSON-lines format enables easy ingestion into monitoring tools. Each entry is self-contained with timestamp, question, confidence, sources, and threshold.

### Lessons Learned

- Never let logging failures propagate — wrap all log writes in try/except.
- Validate API keys at startup, not at first request, to provide clear error messages.
- Returning structured errors (`{ reply, sources, confidence }`) even for failures keeps the frontend contract consistent.
- System prompts should be loaded from files, not hardcoded, to allow institutional updates without code changes.

---

## File Reference

```
vait-backend/
├── .env                            # Environment secrets
├── requirements.txt                # Python dependencies
├── PROJECT_HISTORY.md              # This document
├── app/
│   ├── main.py                     # FastAPI app factory + lifespan
│   ├── controllers/
│   │   └── chat_controller.py      # Business logic orchestration
│   ├── routes/
│   │   ├── chat.py                 # Chat API endpoints
│   │   └── admin.py                # Admin API endpoints
│   ├── services/
│   │   ├── embedding_service.py    # OpenAI embedding generation
│   │   ├── llm_service.py          # OpenAI LLM generation
│   │   └── rag_service.py          # Core RAG pipeline
│   └── utils/
│       ├── config.py               # Configuration + validation
│       └── text_chunker.py         # Token-aware text chunking
├── data/
│   ├── raw_docs/                   # Input documents for ingestion
│   ├── vector_store/
│   │   ├── vait.index              # FAISS binary index
│   │   └── metadata.json           # Chunk text + metadata
│   └── logs/
│       ├── query.log               # Structured query audit log
│       ├── vait.log                # Application debug log
│       └── ingestion.log           # Ingestion pipeline log
├── knowledge/                      # Organised knowledge documents
│   ├── academic-calendar/
│   ├── departments/
│   ├── examinations/
│   ├── notices/
│   ├── regulations/
│   └── syllabus/
├── prompts/
│   └── vait_system_prompt.txt      # Hardened system prompt
└── scripts/
    └── ingest_documents.py         # Offline ingestion CLI
```

---

*Document maintained as part of the VAIT project portfolio.*  
*Last updated: February 2026*
