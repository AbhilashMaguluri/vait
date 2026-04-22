# VAIT Backend — Technical Report

> **Version:** 2.0 (Production-Grade RAG)
> **Author:** VAIT Engineering Team
> **Date:** February 2026
> **Stack:** Python 3.10+ · FastAPI · FAISS · OpenRouter Embeddings · Groq (primary LLM) + OpenRouter (fallback LLM)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Ingestion Pipeline](#3-ingestion-pipeline)
4. [Retrieval Logic](#4-retrieval-logic)
5. [API Endpoints](#5-api-endpoints)
6. [Logging & Diagnostics](#6-logging--diagnostics)
7. [Design Decisions](#7-design-decisions)
8. [Future Improvements](#8-future-improvements)

---

## 1. System Overview

### What is VAIT?

**VAIT** (Virtual Academic Intelligence Terminal) is a Retrieval-Augmented Generation (RAG) system built for **Vasireddy Venkatadri International Technology (VVIT)**. It provides students and staff with authoritative, source-cited answers to institutional queries — academic calendars, regulations, syllabi, notices, and departmental information.

VAIT is **not** a general-purpose chatbot. It is a **strict institutional knowledge retrieval system** that refuses to answer when context is insufficient, never guesses, and never fabricates information.

### Why RAG Architecture?

| Concern | RAG Solution |
|---------|-------------|
| **Hallucination** | LLM answers *only* from retrieved context; refusal if below similarity threshold |
| **Updatability** | Re-run ingestion to refresh knowledge — no model retraining required |
| **Auditability** | Every answer cites source documents; retrieval scores are logged |
| **Cost** | Embeddings are computed once at ingestion; only the LLM call is per-query |
| **Offline capability** | FAISS index + local embeddings work entirely without internet |

### Why Local Embeddings?

- **Zero per-query API cost** for embedding generation
- **No internet dependency** at runtime — the system works fully offline
- **Privacy** — institutional documents never leave the server
- `all-MiniLM-L6-v2` provides strong semantic similarity at 384 dimensions with minimal compute

### Why Groq + OpenRouter Fallback?

- **Reliability** — primary Groq generation with automatic OpenRouter fallback
- **Cloud scalability** — handles production traffic without local model hosting
- **Provider resilience** — graceful degradation across provider outages/timeouts
- **Operational simplicity** — no local model runtime management in production

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         VAIT RAG PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐    ┌────────────────┐    ┌───────────────┐               │
│  │   User   │───▶│   Embedding    │───▶│  FAISS Index  │               │
│  │  Query   │    │  (MiniLM-L6)   │    │  (IndexFlatIP)│               │
│  └──────────┘    └────────────────┘    └───────┬───────┘               │
│                                                │                        │
│                                      Top-K = 6 │ Cosine Similarity     │
│                                                ▼                        │
│                                     ┌────────────────────┐             │
│                                     │   Retrieval Filter  │             │
│                                     │  • threshold ≥ 0.65 │             │
│                                     │  • deduplication     │             │
│                                     │  • authority sort    │             │
│                                     │  • adjusted scoring  │             │
│                                     └─────────┬──────────┘             │
│                                               │                        │
│                                               ▼                        │
│                                    ┌─────────────────────┐             │
│                                    │   Context Builder    │             │
│                                    │  (max 8000 chars)    │             │
│                                    └─────────┬───────────┘             │
│                                              │                         │
│                                              ▼                         │
│                                   ┌──────────────────────┐             │
│                                   │ LLM (Groq/OpenRouter) │             │
│                                   │  System + User Prompt  │             │
│                                   └──────────┬───────────┘             │
│                                              │                         │
│                                              ▼                         │
│                                   ┌──────────────────────┐             │
│                                   │  Structured Response  │             │
│                                   │  reply + sources +    │             │
│                                   │  confidence + score   │             │
│                                   └──────────────────────┘             │
└─────────────────────────────────────────────────────────────────────────┘
```

### Offline Ingestion Pipeline (Separate Process)

```
┌──────────┐    ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐
│  PDF /   │───▶│  Text   │───▶│ Chunking │───▶│ Metadata │───▶│ Embedding │
│  DOCX /  │    │ Extract │    │ (1600ch) │    │ Enrich   │    │ Generate  │
│  TXT     │    └─────────┘    └──────────┘    └──────────┘    └─────┬─────┘
└──────────┘                                                         │
                                                                     ▼
                                                          ┌──────────────────┐
                                                          │  FAISS IndexFlatIP│
                                                          │  + metadata.json  │
                                                          └──────────────────┘
```

---

## 3. Ingestion Pipeline

The ingestion script (`scripts/ingest_documents.py`) is run **offline** to build the knowledge base.

### 3.1 Text Extraction

| Format | Library | Strategy |
|--------|---------|----------|
| PDF    | PyPDF2  | All pages extracted; text joined with `\n\n` |
| DOCX   | python-docx | Paragraphs joined with `\n\n` |
| TXT    | Built-in | UTF-8 read with error replacement |

### 3.2 Text Cleaning

Applied after extraction:

1. Replace tabs and non-breaking spaces with normal spaces
2. Collapse horizontal whitespace into single spaces
3. Remove trailing whitespace per line
4. Limit consecutive blank lines to one (`\n\n` maximum)
5. Strip leading/trailing whitespace

### 3.3 Character-Based Chunking

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Chunk size** | 1600 characters | ≈ 400 tokens (1 token ≈ 4 chars) |
| **Overlap** | 200 characters | ≈ 50 tokens — prevents context loss at boundaries |
| **Minimum chunk** | 300 characters | Discards noise fragments |

**Boundary respect:** Chunks split at paragraph boundaries first, then sentence boundaries, then word boundaries as fallback.

### 3.4 Metadata Enrichment

Every chunk receives:

```json
{
  "text": "...",
  "document_type": "regulation | syllabus | notice | website",
  "academic_year": "2025-26",
  "department": "CSE | ECE | General | ...",
  "authority_level": "high | medium | low",
  "source_file": "filename.pdf",
  "chunk_index": 0,
  "total_chunks": 12
}
```

- **`document_type`** and **`authority_level`** are inferred from the folder path
- **`academic_year`** is detected from path components or defaults to current year
- **`department`** is matched against known keywords (CSE, ECE, AI, etc.)
- **`chunk_index` / `total_chunks`** enable chunk-level provenance tracking

### 3.5 Embedding & Indexing

- Embeddings generated via OpenAI `text-embedding-3-large` (3072 dimensions)
- L2-normalised before insertion so Inner Product = Cosine Similarity
- Stored in **FAISS `IndexFlatIP`** (exact search, no approximation)
- Metadata persisted separately in `metadata.json`

---

## 4. Retrieval Logic

### 4.1 Query Pipeline

```
Query → Embed → FAISS Search (top-6) → Threshold Filter (≥0.65)
      → Deduplicate → Authority-Weighted Sort → Context Cap (8000 chars)
      → LLM Generation → Structured Response
```

### 4.2 Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `top_k_retrieval` | 6 | Number of candidate chunks from FAISS |
| `similarity_threshold` | 0.65 | Minimum cosine similarity to include a chunk |
| `max_context_chars` | 8000 | Maximum characters in the LLM context window |

### 4.3 Authority-Weighted Scoring

Raw similarity scores are adjusted by an authority multiplier:

| Authority Level | Multiplier | Example Documents |
|----------------|-----------|-------------------|
| `high` | ×1.10 | Official regulations, policies, academic calendar |
| `medium` | ×1.05 | Department syllabi, notices |
| `low` | ×1.00 | Website content, general information |

Sorting is by **adjusted score** (descending), ensuring official regulations rank above general content even at similar raw similarity.

### 4.4 Deduplication

Content-hash-based deduplication removes identical chunks before ranking. The first (highest-scored) occurrence is kept.

### 4.5 Confidence Scoring

| Tier | Condition | Meaning |
|------|-----------|---------|
| **High** | Top score ≥ 0.85 | Strong match — answer is highly reliable |
| **Medium** | Top score ≥ 0.75 | Reasonable match — answer is likely correct |
| **Low** | Top score < 0.75 | Weak match — answer may be incomplete |

If **no** chunk meets the 0.65 threshold, the system returns the standard refusal message.

---

## 5. API Endpoints

### 5.1 `POST /api/vait/chat`

Primary chat endpoint.

**Request:**
```json
{
  "message": "What is the exam schedule for CSE?",
  "department": "CSE",
  "academic_year": "2025-26"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | Yes | User question (1–2000 chars) |
| `department` | string | No | Optional department filter (future use) |
| `academic_year` | string | No | Optional year filter (future use) |

**Response:**
```json
{
  "reply": "**Examination Schedule — CSE Department**\n\n...",
  "sources": ["exam-schedule-2025-26.pdf", "academic-calendar.pdf"],
  "confidence": "High",
  "retrieval_score": 0.8921
}
```

| Field | Type | Description |
|-------|------|-------------|
| `reply` | string | LLM-generated answer or refusal message |
| `sources` | string[] | Unique source document names |
| `confidence` | string | `"High"` / `"Medium"` / `"Low"` |
| `retrieval_score` | float | Top cosine similarity score |

### 5.2 `GET /api/vait/debug?query=...`

Engineering diagnostics endpoint — returns raw retrieval results without LLM generation.

**Response:**
```json
{
  "query": "exam schedule",
  "retrieved_chunks": [
    {
      "text_preview": "The end-semester examinations for...",
      "source_file": "exam-schedule-2025-26.pdf",
      "authority_level": "high",
      "similarity_score": 0.8921,
      "adjusted_score": 0.9813
    }
  ],
  "performance": {
    "embed_ms": 12.4,
    "retrieval_ms": 3.1,
    "total_ms": 15.5
  }
}
```

### 5.3 `POST /api/vait/chat/detailed`

Same as `/chat` but includes `is_refusal` flag.

### 5.4 `GET /api/vait/stats`

Returns knowledge base statistics (total chunks, index size, threshold, top_k).

---

## 6. Logging & Diagnostics

### 6.1 Query Log (`data/logs/query.log`)

Every query is appended as a JSON-lines entry:

```json
{
  "timestamp": "2026-02-20T14:30:00.000Z",
  "question": "What are the regulations for attendance?",
  "top_similarity_score": 0.8234,
  "confidence": "Medium",
  "sources": ["attendance-policy-2025-26.pdf"],
  "is_refusal": false,
  "threshold": 0.65
}
```

### 6.2 Field Definitions

| Field | Meaning |
|-------|---------|
| `top_similarity_score` | Highest cosine similarity among retrieved chunks |
| `retrieval_score` | Same value returned to the client — the raw top score |
| `confidence` | Derived tier (High/Medium/Low) based on top score |
| `threshold` | Active similarity threshold at query time |

### 6.3 Performance Metrics (Debug Mode)

| Metric | Meaning |
|--------|---------|
| `embed_ms` | Time to embed the user query |
| `retrieval_ms` | Time for FAISS search + filtering |
| `total_ms` | End-to-end pipeline time |
| `generation_ms` | LLM response generation time (chat pipeline only) |

### 6.4 Ingestion Log (`data/logs/ingestion.log`)

Detailed log of every ingestion run including per-document extraction length, chunk counts, and aggregate statistics (avg/min/max chunk size).

---

## 7. Design Decisions

### 7.1 Why FAISS over ChromaDB?

| Criterion | FAISS | ChromaDB |
|-----------|-------|----------|
| **Dependency** | Single C library (`faiss-cpu`) | Requires SQLite + multiple Python deps |
| **Exactness** | `IndexFlatIP` = exact cosine search | Approximate by default |
| **Offline** | Fully offline — binary index file | Needs persistent server or SQLite |
| **Transparency** | Raw vectors + metadata.json — fully inspectable | Opaque internal storage |
| **Performance** | Excellent for < 100K vectors (our scale) | Overhead for small datasets |
| **Portability** | Copy 2 files to deploy | Database migration required |

FAISS is the right choice for an institutional system with tens of thousands of chunks where **exactness, transparency, and offline operation** are priorities.

### 7.2 Why Offline-First?

- **Data sovereignty:** Institutional documents must not leave the university network
- **Zero recurring cost:** No API bills for embeddings or LLM after deployment
- **Reliability:** Works during internet outages
- **Compliance:** Simplifies data handling — no third-party data processing agreements

### 7.3 Why External APIs with Fallback?

- **Availability:** Multi-provider strategy reduces outage impact
- **Performance:** Managed inference endpoints provide lower operational overhead
- **Simplicity:** Avoids maintaining local LLM serving infrastructure
- **Resilience:** Automatic fallback protects user experience during provider failures

---

## 8. Future Improvements

### 8.1 Semantic Re-Ranking

After FAISS retrieval, apply a cross-encoder model (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) to re-rank the top-K chunks by query-passage relevance. This improves precision without increasing the FAISS index size.

### 8.2 Hybrid Keyword + Vector Search

Combine BM25 keyword matching with vector similarity for retrieval. This helps with queries containing proper nouns, regulation numbers, or exact policy names that pure semantic search may miss.

### 8.3 Admin Upload Dashboard

Build a web-based admin panel allowing authorised staff to:
- Upload new documents with metadata
- Trigger re-ingestion
- View ingestion logs and chunk previews
- Monitor query logs and confidence distributions

### 8.4 Multi-Document Authority Ranking

Implement cross-document authority resolution — when the same topic is covered by multiple documents (e.g., a notice overriding a regulation), the system should recognise temporal precedence and document hierarchy.

### 8.5 Threshold Tuning via Evaluation

Empirically determine the optimal similarity threshold by evaluating against a labelled query set:
- Test at 0.60, 0.65, 0.70, 0.72, 0.75
- Measure precision, recall, and refusal rate
- Select the threshold that maximises F1 while keeping false-positive rate below 5%

### 8.6 Streaming Responses

Implement Server-Sent Events (SSE) for real-time token streaming from provider APIs, reducing perceived latency for long answers.

### 8.7 Multi-Language Support

Add translation layer for queries in Telugu or Hindi, embedding in English, and optional response translation.

---

## Appendix: File Structure

```
vait-backend/
├── app/
│   ├── main.py                    # FastAPI application entry point
│   ├── controllers/
│   │   └── chat_controller.py     # Business logic for chat operations
│   ├── routes/
│   │   ├── chat.py                # Chat + debug API endpoints
│   │   └── admin.py               # Admin API endpoints
│   ├── services/
│   │   ├── rag_service.py         # Core RAG pipeline
│   │   ├── embedding_service.py   # Embedding generation
│   │   └── llm_service.py         # LLM interaction
│   └── utils/
│       ├── config.py              # Centralised settings
│       └── text_chunker.py        # Character-based chunking
├── scripts/
│   └── ingest_documents.py        # Offline ingestion pipeline
├── prompts/
│   └── vait_system_prompt.txt     # System prompt for LLM
├── data/
│   ├── raw_docs/                  # Input documents
│   ├── vector_store/
│   │   ├── vait.index             # FAISS binary index
│   │   └── metadata.json          # Chunk text + metadata
│   └── logs/
│       ├── query.log              # Per-query JSON-lines log
│       └── ingestion.log          # Ingestion run logs
├── knowledge/                     # Structured knowledge folders
│   ├── regulations/
│   ├── academic-calendar/
│   ├── examinations/
│   ├── syllabus/
│   ├── departments/
│   └── notices/
├── requirements.txt
└── README.md
```

---

*This document is maintained as part of the VAIT project repository and is intended for university review, internship portfolio presentation, and open-source publication.*
