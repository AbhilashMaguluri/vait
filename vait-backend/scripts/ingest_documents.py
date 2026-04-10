"""
VAIT Document Ingestion Script
================================
Production-grade offline utility for building the VAIT knowledge base.

This script runs OFFLINE ONLY. It does NOT handle chat queries.

Pipeline:
  1. Scan data/raw_docs/ (and optionally knowledge/) for PDF, DOCX, TXT files
  2. Extract clean text from each document
  3. Clean excessive whitespace / normalise line breaks
  4. Chunk text (~400 tokens ≈ 1600 characters, 200-char overlap)
  5. Derive metadata from folder name + file name
  6. Generate embeddings via Ollama nomic-embed-text
  7. Build FAISS index (cosine similarity via IndexFlatIP + L2-norm)
  8. Persist:
     - data/vector_store/vait.index   (FAISS binary)
     - data/vector_store/metadata.json (chunk text + metadata)
  9. Print detailed summary with chunk statistics

Usage:
    cd vait-backend
    python scripts/ingest_documents.py
    python scripts/ingest_documents.py --include-knowledge
    python scripts/ingest_documents.py --rebuild

Errors are logged to: data/logs/ingestion.log
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv


# =============================================================================
# PATH SETUP
# =============================================================================

# Project root = vait-backend/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_docs"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
LOGS_DIR = DATA_DIR / "logs"
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"

# Output artifacts
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "vait.index"
METADATA_JSON_PATH = VECTOR_STORE_DIR / "metadata.json"
INGESTION_LOG_PATH = LOGS_DIR / "ingestion.log"

# Embedding config
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIMENSION = 768  # nomic-embed-text output dimension
EMBEDDING_BATCH_SIZE = 100  # max texts per batch

# Chunking config (character-based: 1 token ≈ 4 characters)
CHUNK_CHARS = 1600      # ~400 tokens
OVERLAP_CHARS = 200     # ~50 tokens
MIN_CHUNK_CHARS = 300   # discard tiny trailing chunks

# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging() -> logging.Logger:
    """Configure dual logging: console (INFO) + file (DEBUG)."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("vait_ingestion")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler — all messages
    fh = logging.FileHandler(INGESTION_LOG_PATH, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)

    # Console handler — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return logger


logger = setup_logging()


# =============================================================================
# TEXT EXTRACTION
# =============================================================================

def extract_text_from_file(file_path: Path) -> Optional[str]:
    """
    Extract text content from a supported file.

    Supports: .txt, .pdf, .docx
    Returns None on failure (logged, not raised).
    """
    suffix = file_path.suffix.lower()

    try:
        if suffix == ".txt":
            return file_path.read_text(encoding="utf-8", errors="replace")

        elif suffix == ".pdf":
            return _extract_pdf(file_path)

        elif suffix == ".docx":
            return _extract_docx(file_path)

        else:
            logger.warning("Unsupported file type '%s': %s", suffix, file_path.name)
            return None

    except Exception as e:
        logger.error("Failed to extract text from %s: %s", file_path.name, e)
        return None


def _extract_pdf(file_path: Path) -> Optional[str]:
    """
    Extract full text from all pages of a PDF using PyPDF2.

    - Iterates every page and collects text.
    - Preserves paragraph structure with double-newline separators.
    """
    import PyPDF2

    text_parts: list[str] = []
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text_parts.append(page_text.strip())

    if not text_parts:
        return None

    raw = "\n\n".join(text_parts)
    return _clean_extracted_text(raw)


def _extract_docx(file_path: Path) -> Optional[str]:
    """Extract text from DOCX using python-docx, preserving paragraph breaks."""
    import docx

    doc = docx.Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    if not paragraphs:
        return None

    raw = "\n\n".join(paragraphs)
    return _clean_extracted_text(raw)


def _clean_extracted_text(text: str) -> str:
    """
    Normalise whitespace and line breaks in extracted text.

    1. Replace tabs and non-breaking spaces with normal spaces.
    2. Collapse multiple spaces on the same line into one.
    3. Limit consecutive blank lines to one (two newlines max).
    4. Remove trailing whitespace on each line.
    5. Strip leading/trailing whitespace from the full text.
    """
    # Replace tabs/NBSP with spaces
    text = text.replace("\t", " ").replace("\u00a0", " ")
    # Collapse horizontal whitespace (not newlines) into single space
    text = re.sub(r"[^\S\n]+", " ", text)
    # Remove trailing spaces per line
    text = re.sub(r" +\n", "\n", text)
    # Limit consecutive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# =============================================================================
# TEXT CLEANING & CHUNKING (character-based: 1 token ≈ 4 characters)
# =============================================================================

class TextChunker:
    """
    Splits text into chunks of ~CHUNK_CHARS characters with ~OVERLAP_CHARS overlap.
    Uses character count as a proxy for tokens (1 token ≈ 4 characters).
    Discards chunks shorter than MIN_CHUNK_CHARS.
    """

    def __init__(
        self,
        chunk_chars: int = CHUNK_CHARS,
        overlap_chars: int = OVERLAP_CHARS,
        min_chunk_chars: int = MIN_CHUNK_CHARS,
    ):
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars
        self.min_chunk_chars = min_chunk_chars

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------
    def chunk_text(self, text: str) -> List[str]:
        """Return a list of text chunks from *text*."""
        text = self._clean(text)
        if not text:
            return []

        paragraphs = self._split_paragraphs(text)
        raw_chunks = self._merge_paragraphs(paragraphs)

        # Filter out extremely small trailing chunks
        return [c for c in raw_chunks if len(c) >= self.min_chunk_chars]

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _clean(self, text: str) -> str:
        """Remove excessive whitespace while preserving paragraph breaks."""
        text = re.sub(r"[^\S\n]+", " ", text)       # collapse spaces/tabs
        text = re.sub(r"\n{3,}", "\n\n", text)      # max 2 consecutive newlines
        return text.strip()

    def _split_paragraphs(self, text: str) -> List[str]:
        parts = re.split(r"\n\n+", text)
        return [p.strip() for p in parts if p.strip()]

    def _merge_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """Merge paragraphs into chunks respecting character limits."""
        chunks: List[str] = []
        buf: List[str] = []
        buf_len = 0

        for para in paragraphs:
            para_len = len(para)

            # Single paragraph exceeds chunk size → split it by sentences
            if para_len > self.chunk_chars:
                if buf:
                    chunks.append("\n\n".join(buf))
                    buf, buf_len = [], 0
                chunks.extend(self._split_large(para))
                continue

            # Adding this paragraph would exceed limit → finalize current chunk
            if buf_len + para_len > self.chunk_chars and buf:
                chunks.append("\n\n".join(buf))
                # Carry overlap from end of previous chunk
                overlap = self._overlap_tail("\n\n".join(buf))
                if overlap:
                    buf = [overlap, para]
                    buf_len = len("\n\n".join(buf))
                else:
                    buf = [para]
                    buf_len = para_len
            else:
                buf.append(para)
                buf_len += para_len

        if buf:
            chunks.append("\n\n".join(buf))

        return chunks

    def _split_large(self, text: str) -> List[str]:
        """Split oversized text by sentences, then by words as fallback."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: List[str] = []
        buf: List[str] = []
        buf_len = 0

        for sent in sentences:
            sent_len = len(sent)

            if sent_len > self.chunk_chars:
                # Sentence itself too long — fall back to word-level split
                if buf:
                    chunks.append(" ".join(buf))
                    buf, buf_len = [], 0
                chunks.extend(self._split_by_words(sent))
                continue

            if buf_len + sent_len > self.chunk_chars and buf:
                chunks.append(" ".join(buf))
                buf = [sent]
                buf_len = sent_len
            else:
                buf.append(sent)
                buf_len += sent_len

        if buf:
            chunks.append(" ".join(buf))
        return chunks

    def _split_by_words(self, text: str) -> List[str]:
        words = text.split()
        chunks: List[str] = []
        buf: List[str] = []
        buf_len = 0

        for w in words:
            wl = len(w)
            if buf_len + wl > self.chunk_chars and buf:
                chunks.append(" ".join(buf))
                buf = [w]
                buf_len = wl
            else:
                buf.append(w)
                buf_len += wl

        if buf:
            chunks.append(" ".join(buf))
        return chunks

    def _overlap_tail(self, text: str) -> str:
        """Return the last ~overlap_chars characters, breaking at a space."""
        if len(text) <= self.overlap_chars:
            return text
        tail = text[-self.overlap_chars:]
        # Try to break at a word boundary
        space_idx = tail.find(" ")
        if space_idx != -1 and space_idx < len(tail) - 1:
            tail = tail[space_idx + 1:]
        return tail


# =============================================================================
# METADATA INFERENCE
# =============================================================================

# Maps folder names (lowercased) → (document_type, authority_level)
FOLDER_METADATA_MAP: Dict[str, Tuple[str, str]] = {
    "regulations":       ("regulation", "high"),
    "academic-calendar": ("regulation", "high"),
    "examinations":      ("regulation", "high"),
    "syllabus":          ("syllabus",   "medium"),
    "departments":       ("syllabus",   "medium"),
    "notices":           ("notice",     "medium"),
    "website":           ("website",    "low"),
    "raw_docs":          ("website",    "low"),
}

# Known department keywords → canonical names
DEPARTMENT_KEYWORDS: Dict[str, str] = {
    "cse":  "CSE",
    "cs":   "CSE",
    "computer science": "CSE",
    "ai":   "AI",
    "artificial intelligence": "AI",
    "ece":  "ECE",
    "eee":  "EEE",
    "mech": "MECH",
    "mechanical": "MECH",
    "civil": "CIVIL",
    "it":   "IT",
    "mba":  "MBA",
    "mca":  "MCA",
}

_YEAR_RE = re.compile(r"(20\d{2})[-_]((?:20)?\d{2})")


def _normalise_year(match: re.Match) -> str:
    """Normalise '2024-25' or '2024-2025' → '2024-25'."""
    start = match.group(1)
    end = match.group(2)
    if len(end) == 4:
        end = end[2:]
    return f"{start}-{end}"


def _infer_academic_year() -> str:
    """Current academic year (Aug boundary)."""
    now = datetime.now()
    y = now.year
    if now.month >= 8:
        return f"{y}-{str(y + 1)[2:]}"
    return f"{y - 1}-{str(y)[2:]}"


def _infer_department(text: str) -> Optional[str]:
    """Match department keywords in a string."""
    low = text.lower().replace("-", " ").replace("_", " ")
    for kw, dept in DEPARTMENT_KEYWORDS.items():
        if kw in low:
            return dept
    return None


def infer_metadata(file_path: Path, base_dir: Path) -> Dict:
    """
    Derive metadata for a document from its path components.

    Returns dict with keys:
        text, document_type, academic_year, department,
        authority_level, source_file
    """
    try:
        rel_parts = file_path.relative_to(base_dir).parts
    except ValueError:
        rel_parts = (file_path.name,)

    meta: Dict = {
        "text": "",               # placeholder — filled per chunk later
        "document_type": "website",
        "academic_year": _infer_academic_year(),
        "department": "General",
        "authority_level": "low",
        "source_tier": "related_web",
        "source_file": file_path.name,
    }

    for part in rel_parts:
        part_lower = part.lower()

        # folder → doc type + authority
        if part_lower in FOLDER_METADATA_MAP:
            dt, al = FOLDER_METADATA_MAP[part_lower]
            meta["document_type"] = dt
            meta["authority_level"] = al

        # year from path component
        ym = _YEAR_RE.search(part)
        if ym:
            meta["academic_year"] = _normalise_year(ym)

        # department from path component (skip known folder names & filename)
        if part_lower not in FOLDER_METADATA_MAP and part != file_path.name:
            dept = _infer_department(part)
            if dept:
                meta["department"] = dept
            else:
                # Use the folder name as-is if it's not a known category
                meta["department"] = part.replace("-", " ").replace("_", " ").title()

    # Also check filename itself for year / department
    ym = _YEAR_RE.search(file_path.stem)
    if ym:
        meta["academic_year"] = _normalise_year(ym)

    dept = _infer_department(file_path.stem)
    if dept:
        meta["department"] = dept

    # Default source tier assignment for document corpora.
    if meta["document_type"] in {"regulation", "syllabus", "notice", "website"}:
        meta["source_tier"] = "primary_official"

    return meta


# =============================================================================
# EMBEDDING GENERATION
# =============================================================================

def generate_embeddings(
    texts: List[str],
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> np.ndarray:
    """
    Generate embeddings for a list of texts using Ollama.

    Automatically batches for efficiency.
    Returns ndarray of shape (len(texts), EMBEDDING_DIMENSION), dtype float32.
    """
    # Import here to avoid circular dependency at module level
    sys.path.insert(0, str(PROJECT_ROOT))
    from app.services.ollama_service import OllamaService

    ollama = OllamaService()
    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        batch_label = f"{start + 1}–{min(start + batch_size, len(texts))} of {len(texts)}"
        logger.info("Embedding batch %s via Ollama", batch_label)

        emb = ollama.embed_batch(batch)
        all_embeddings.append(emb)

    arr = np.vstack(all_embeddings).astype(np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


# =============================================================================
# FAISS INDEX BUILDING
# =============================================================================

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS IndexFlatIP from embeddings.

    Vectors are L2-normalised first so Inner Product == Cosine Similarity.
    Dimension is auto-detected from the embedding matrix.
    """
    dimension = embeddings.shape[1]
    logger.info("Building FAISS index: dimension=%d, vectors=%d", dimension, embeddings.shape[0])

    # Normalise so IP == cosine similarity
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, metadata: List[Dict]) -> None:
    """
    Persist FAISS index and metadata to disk.

    Overwrites existing files safely by writing a temp file first,
    then atomically replacing.
    """
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    # Write FAISS index via temp file
    tmp_index = FAISS_INDEX_PATH.with_suffix(".index.tmp")
    faiss.write_index(index, str(tmp_index))
    # Atomic-ish replace (Windows: remove then rename)
    if FAISS_INDEX_PATH.exists():
        FAISS_INDEX_PATH.unlink()
    tmp_index.rename(FAISS_INDEX_PATH)
    logger.info("Saved FAISS index → %s (%d vectors)", FAISS_INDEX_PATH, index.ntotal)

    # Write metadata via temp file
    tmp_meta = METADATA_JSON_PATH.with_suffix(".json.tmp")
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    if METADATA_JSON_PATH.exists():
        METADATA_JSON_PATH.unlink()
    tmp_meta.rename(METADATA_JSON_PATH)
    logger.info("Saved metadata  → %s (%d entries)", METADATA_JSON_PATH, len(metadata))


# =============================================================================
# DIRECTORY SCANNING
# =============================================================================

def collect_files(directories: List[Path]) -> List[Path]:
    """Recursively collect all supported files from the given directories."""
    files: List[Path] = []
    for directory in directories:
        if not directory.exists():
            logger.warning("Directory does not exist, creating: %s", directory)
            directory.mkdir(parents=True, exist_ok=True)
            continue
        for fpath in sorted(directory.rglob("*")):
            if fpath.is_file() and fpath.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(fpath)
    return files


# =============================================================================
# MAIN INGESTION PIPELINE
# =============================================================================

def run_ingestion(
    directories: List[Path],
    base_dir: Path,
    rebuild: bool = True,
) -> None:
    """
    Execute the full ingestion pipeline.

    Steps:
      1. Collect files
      2. Extract text
      3. Chunk
      4. Attach metadata
      5. Embed
      6. Build FAISS index
      7. Save to disk
      8. Print summary
    """
    t0 = time.time()

    # ------------------------------------------------------------------
    # Step 1: Collect files
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("VAIT DOCUMENT INGESTION")
    logger.info("=" * 60)
    logger.info("Directories      : %s", [str(d) for d in directories])
    logger.info("FAISS index path : %s", FAISS_INDEX_PATH)
    logger.info("Metadata path    : %s", METADATA_JSON_PATH)
    logger.info("Embedding model  : %s", EMBEDDING_MODEL)
    logger.info("Chunk target     : %d chars (~%d tokens), overlap %d chars",
                CHUNK_CHARS, CHUNK_CHARS // 4, OVERLAP_CHARS)
    logger.info("Min chunk length : %d chars", MIN_CHUNK_CHARS)
    logger.info("=" * 60)

    all_files = collect_files(directories)
    if not all_files:
        logger.info("No documents found. Place PDF/DOCX/TXT files in:")
        for d in directories:
            logger.info("  • %s", d)
        return

    logger.info("Found %d document(s) to process.", len(all_files))

    # ------------------------------------------------------------------
    # Step 2–4: Extract, chunk, attach metadata
    # ------------------------------------------------------------------
    chunker = TextChunker()
    all_chunks: List[Dict] = []          # each entry = one metadata dict
    docs_processed = 0
    docs_skipped = 0

    for file_path in all_files:
        rel = file_path.relative_to(PROJECT_ROOT) if file_path.is_relative_to(PROJECT_ROOT) else file_path
        logger.info("Processing: %s", rel)

        # Extract text
        raw_text = extract_text_from_file(file_path)
        if not raw_text or not raw_text.strip():
            logger.warning("  SKIPPED — no text extracted: %s", file_path.name)
            docs_skipped += 1
            continue

        # Report extracted text length
        logger.info("  Extracted text length: %d characters", len(raw_text))

        # Chunk
        chunks = chunker.chunk_text(raw_text)
        if not chunks:
            logger.warning("  SKIPPED — no valid chunks: %s", file_path.name)
            docs_skipped += 1
            continue

        total_chunks_for_doc = len(chunks)

        # Metadata
        base_meta = infer_metadata(file_path, base_dir)
        logger.info(
            "  type=%s  authority=%s  dept=%s  year=%s  chunks=%d",
            base_meta["document_type"],
            base_meta["authority_level"],
            base_meta["department"],
            base_meta["academic_year"],
            total_chunks_for_doc,
        )

        # Check for companion .json metadata override
        companion = file_path.with_suffix(".json")
        if companion.exists():
            try:
                with open(companion, "r", encoding="utf-8") as f:
                    override = json.load(f)
                base_meta.update(override)
                logger.info("  Applied companion metadata: %s", companion.name)
            except Exception as e:
                logger.warning("  Could not load companion metadata %s: %s", companion.name, e)

        for chunk_idx, chunk_text in enumerate(chunks):
            entry = dict(base_meta)  # shallow copy
            entry["text"] = chunk_text
            entry["chunk_index"] = chunk_idx
            entry["total_chunks"] = total_chunks_for_doc
            all_chunks.append(entry)

        docs_processed += 1

    if not all_chunks:
        logger.info("No chunks produced. Nothing to index.")
        return

    logger.info("-" * 60)
    logger.info("Total documents processed : %d", docs_processed)
    logger.info("Total documents skipped   : %d", docs_skipped)
    logger.info("Total chunks created      : %d", len(all_chunks))

    # Chunk size statistics
    chunk_lengths = [len(c["text"]) for c in all_chunks]
    avg_len = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
    min_len = min(chunk_lengths) if chunk_lengths else 0
    max_len = max(chunk_lengths) if chunk_lengths else 0
    logger.info("Chunk stats (chars)       : avg=%d  min=%d  max=%d", avg_len, min_len, max_len)
    logger.info("Chunk stats (~tokens)     : avg=%d  min=%d  max=%d",
                avg_len // 4, min_len // 4, max_len // 4)

    # ------------------------------------------------------------------
    # Step 5: Generate embeddings
    # ------------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("Generating embeddings with %s ...", EMBEDDING_MODEL)

    chunk_texts = [c["text"] for c in all_chunks]
    embeddings = generate_embeddings(chunk_texts)

    detected_dim = embeddings.shape[1]
    logger.info("Embeddings generated: %d vectors × %d dimensions", embeddings.shape[0], detected_dim)

    # ------------------------------------------------------------------
    # Step 6: Build FAISS index
    # ------------------------------------------------------------------
    if rebuild:
        # Remove old artifacts before building new ones
        for old in (FAISS_INDEX_PATH, METADATA_JSON_PATH):
            if old.exists():
                old.unlink()
                logger.info("Removed old artifact: %s", old.name)

    index = build_faiss_index(embeddings)

    # ------------------------------------------------------------------
    # Step 7: Save to disk
    # ------------------------------------------------------------------
    save_index(index, all_chunks)

    # ------------------------------------------------------------------
    # Step 8: Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0

    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 60)
    logger.info("Total documents processed : %d", docs_processed)
    logger.info("Total chunks created      : %d", len(all_chunks))
    logger.info("Total vectors stored      : %d", index.ntotal)
    logger.info("Index dimension           : %d", detected_dim)
    logger.info("Time elapsed              : %.1f seconds", elapsed)
    logger.info("")
    logger.info("Artifacts:")
    logger.info("  %s", FAISS_INDEX_PATH)
    logger.info("  %s", METADATA_JSON_PATH)
    logger.info("=" * 60)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VAIT Document Ingestion — build the knowledge-base vector index."
    )
    parser.add_argument(
        "--directory", "-d",
        type=str,
        default=None,
        help="Custom directory to ingest instead of data/raw_docs/",
    )
    parser.add_argument(
        "--include-knowledge",
        action="store_true",
        help="Also ingest documents from the knowledge/ directory tree",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        default=True,
        help="Overwrite existing index (default: True)",
    )
    args = parser.parse_args()

    # Load .env from project root
    dotenv_path = PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path)

    # Ensure output directories exist
    for d in (RAW_DOCS_DIR, VECTOR_STORE_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # Determine input directories
    directories: List[Path] = []

    if args.directory:
        custom = Path(args.directory).resolve()
        directories.append(custom)
    else:
        directories.append(RAW_DOCS_DIR)

    if args.include_knowledge:
        directories.append(KNOWLEDGE_DIR)

    # Choose base_dir for relative-path metadata inference
    base_dir = PROJECT_ROOT

    run_ingestion(
        directories=directories,
        base_dir=base_dir,
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
