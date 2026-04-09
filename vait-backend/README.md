# VAIT Backend

**Virtual Academic Intelligence Terminal** - Official University Intelligence Assistant

A strict RAG-based AI system that prioritizes **correctness, authority, and refusal over answering**.

---

## Overview

VAIT is an institutional AI assistant that provides accurate information exclusively from verified university records. It implements a strict Retrieval-Augmented Generation (RAG) system that:

- **NEVER answers without retrieved context**
- **REFUSES when similarity score is below threshold**
- **Does NOT learn from users** - only document re-ingestion improves knowledge
- **Prioritizes authority** - regulations > academic calendar > notices > website content

---

## Tech Stack (Non-Negotiable)

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| API Framework | FastAPI |
| Vector Store | **FAISS** (THE PRIMARY DATABASE) |
| Embeddings | Ollama nomic-embed-text (768 dimensions) |
| LLM | Groq llama-3.3-70b (primary) + Ollama phi3:mini (fallback) |

**Explicitly NOT used for vector/database:** JavaScript, Node.js, ChromaDB, Pinecone

---

## Project Structure

```
vait-backend/
├── app/
│   ├── main.py                    # FastAPI application entry point
│   ├── routes/
│   │   ├── chat.py                # Chat API endpoints
│   │   └── admin.py               # Admin/ingestion endpoints
│   ├── controllers/
│   │   └── chat_controller.py     # Business logic for chat
│   ├── services/
│   │   ├── rag_service.py         # Main RAG implementation
│   │   ├── llm_service.py         # Groq primary + Ollama fallback integration
│   │   └── embedding_service.py   # Embedding generation
│   └── utils/
│       ├── config.py              # Centralized configuration
│       └── text_chunker.py        # Document chunking (~400 tokens)
│
├── data/
│   ├── raw_docs/                  # Original PDFs/docs (input)
│   ├── vector_store/              # THE DATABASE
│   │   ├── vait.index             # FAISS index (vectors)
│   │   └── metadata.json          # text + metadata per vector
│   └── logs/                      # queries, refusals, confidence
│
├── knowledge/                     # Organized authoritative docs
│   ├── regulations/
│   ├── academic-calendar/
│   ├── examinations/
│   ├── syllabus/
│   ├── departments/
│   └── notices/
│
├── prompts/
│   └── vait_system_prompt.txt     # VAIT's system prompt
│
├── scripts/
│   └── ingest_documents.py        # Offline document ingestion
│
├── requirements.txt
└── README.md
```

---

## RAG Configuration

| Parameter | Value |
|-----------|-------|
| Chunk Size | ~400 tokens |
| Top-K Retrieval | 6 |
| Similarity Threshold | 0.65 (strict) |
| Below Threshold | Hard refusal |

---

## Metadata Structure

Each chunk stores:

| Field | Description | Values |
|-------|-------------|--------|
| `text` | The chunk content | string |
| `document_type` | Type of document | regulation, syllabus, notice, website |
| `academic_year` | Academic year | e.g., 2025-2026 |
| `department` | Responsible department | string |
| `authority_level` | Weight of the information | high, medium, low |
| `source_file` | Original filename | string |

### Authority Priority

1. **Regulations** (high)
2. **Academic Calendar** (high)
3. **Notices** (medium)
4. **Website Content** (low)

---

## Installation

### 1. Clone and Navigate

```powershell
cd vait/vait-backend
```

### 2. Create Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure Environment

Create a `.env` file in `vait-backend/`:

```env
# Groq primary LLM configuration
GROQ_API_KEY=your_api_key_here
GROQ_MODEL=llama-3.3-70b
GROQ_BASE_URL=https://api.groq.com/openai/v1

# Ollama fallback LLM + embeddings
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=phi3:mini
EMBEDDING_MODEL=nomic-embed-text
DEBUG=false
```

### 4b. Pull Ollama Models

```powershell
ollama pull phi3:mini
ollama pull nomic-embed-text
```

### 5. Run the Application

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Document Ingestion

The ingestion script runs **OFFLINE ONLY**. It:

1. Reads documents from `data/raw_docs/`
2. Extracts text (PDF/DOCX/TXT)
3. Chunks text (~400 tokens)
4. Attaches metadata based on folder + filename
5. Generates embeddings
6. Builds FAISS index
7. Saves to `data/vector_store/`

### Basic Usage

```powershell
# Ingest from default directory (data/raw_docs/)
python -m scripts.ingest_documents

# Rebuild index from scratch
python -m scripts.ingest_documents --rebuild

# Include knowledge directory
python -m scripts.ingest_documents --include-knowledge

# Override metadata
python -m scripts.ingest_documents --document-type regulation --authority-level high
```

### Metadata Inference

The script infers metadata from folder structure:

- `knowledge/regulations/` → document_type=regulation, authority_level=high
- `knowledge/syllabus/` → document_type=syllabus, authority_level=medium
- `knowledge/notices/` → document_type=notice, authority_level=medium

You can also create companion `.json` files with metadata:

```json
// document.json (next to document.pdf)
{
  "document_type": "regulation",
  "academic_year": "2025-2026",
  "department": "Computer Science",
  "authority_level": "high"
}
```

---

## API Endpoints

### POST /api/vait/chat

Process a chat message.

**Request:**
```json
{
  "message": "What are the examination rules?"
}
```

**Response (with context):**
```json
{
  "reply": "According to the University Examination Regulations...",
  "sources": ["Examination Regulations 2025-2026"]
}
```

**Response (refusal - mandatory message):**
```json
{
"reply": "I'm unable to find sufficient verified information in the VVIT knowledge base to answer this query.",
"sources": []
}
```

### POST /api/vait/chat/detailed

Same as above with additional metadata:

```json
{
  "reply": "...",
  "sources": ["..."],
  "question_type": "policy",
  "is_refusal": false
}
```

### GET /health

Health check endpoint.

### Admin Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/vait/admin/ingest` | POST | Ingest single document |
| `/api/vait/admin/ingest/bulk` | POST | Ingest multiple documents |
| `/api/vait/admin/knowledge-base/stats` | GET | Get KB statistics |
| `/api/vait/admin/knowledge-base/clear` | DELETE | Clear knowledge base |

---

## Runtime Query Flow

```
POST /api/vait/chat
     ↓
1. Embed question (Ollama nomic-embed-text, 768-dim)
     ↓
2. Search FAISS (top-k=6, cosine similarity)
     ↓
3. Fetch metadata + text
     ↓
4. Check similarity threshold (≥0.65)
     ↓
5. Authority-weighted scoring + deduplication
     ↓
6. Build grounded prompt
     ↓
7. Call LLM (Groq llama-3.3-70b, fallback to Ollama phi3:mini)
     ↓
8. Hallucination guard + response polish
     ↓
9. Return answer + sources + confidence OR refusal
```

---

## Query Logging

All queries are logged to `data/logs/queries_YYYY-MM-DD.jsonl`:

```json
{
  "timestamp": "2026-02-08T12:34:56.789Z",
  "query": "What are the library hours?",
  "question_type": "factual",
  "is_refusal": false,
  "top_score": 0.89,
  "scores": [0.89, 0.82, 0.78, 0.71],
  "threshold": 0.75
}
```

---

## System Prompt

The system prompt is located at `prompts/vait_system_prompt.txt` and defines:

- VAIT as an official university intelligence system
- Formal, calm, academic tone
- Answer ONLY from provided context
- Never guess, never hallucinate
- Mandatory refusal message when context is insufficient

---

## Design Principles

1. **FAISS is the primary database** - stores vectors only, metadata in JSON
2. **VAIT NEVER answers without context** - strict retrieval requirement
3. **No online learning** - knowledge improves only through document re-ingestion
4. **Authority-aware retrieval** - prioritizes official regulations
5. **Correctness over helpfulness** - better to refuse than to guess

---

## Development

### API Documentation

Once running, access interactive API docs at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Adding Documents

1. Place documents in `data/raw_docs/` or `knowledge/{category}/`
2. Run ingestion: `python -m scripts.ingest_documents`
3. Restart server if needed

### Testing

```powershell
# Test chat endpoint
curl -X POST "http://localhost:8000/api/vait/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the admission requirements?"}'
```

---

## Refusal Message (Mandatory)

When context is missing or similarity is below threshold:

> **"I'm unable to find sufficient verified information in the VVIT knowledge base to answer this query."**

This message is non-negotiable and ensures institutional consistency.

---

## License

University Internal Use Only
