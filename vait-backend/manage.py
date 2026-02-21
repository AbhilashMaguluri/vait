"""
VAIT Management CLI — Administrative Commands
==============================================

Unified command-line interface for VAIT maintenance tasks.

Commands:
  reindex_websites   Crawl allowed websites, chunk, embed, update FAISS
  ingest_social      Ingest social media JSON files from social/ directory
  reindex_all        Full reindex: documents + websites + social

Usage:
    cd vait-backend
    python manage.py reindex_websites
    python manage.py ingest_social
    python manage.py reindex_all
    python manage.py reindex_websites --depth 3 --max-pages 100
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

# ── Path setup ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# Ensure project root is on sys.path so `app.*` imports work
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import faiss
import numpy as np

from app.services.ollama_service import OllamaService, EMBEDDING_DIMENSION as _OLLAMA_EMBED_DIM

from app.utils.config import (
    Settings,
    get_settings,
    VECTOR_STORE_DIR,
    LOGS_DIR,
    SOCIAL_DIR,
    DATA_DIR,
)
from app.services.website_crawler import WebsiteCrawler
from app.services.social_ingestor import SocialIngestor

# ── Logging ──────────────────────────────────────────────────────────

LOGS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("vait.manage")
logger.setLevel(logging.DEBUG)
logger.handlers.clear()

_fmt = logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_ch = logging.StreamHandler(sys.stdout)
_ch.setLevel(logging.INFO)
_ch.setFormatter(_fmt)
logger.addHandler(_ch)

_fh = logging.FileHandler(LOGS_DIR / "manage.log", encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(_fmt)
logger.addHandler(_fh)


# ── Constants ────────────────────────────────────────────────────────
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "vait.index"
METADATA_JSON_PATH = VECTOR_STORE_DIR / "metadata.json"
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIMENSION = _OLLAMA_EMBED_DIM
EMBEDDING_BATCH_SIZE = 50

CHUNK_CHARS = 1600
OVERLAP_CHARS = 200
MIN_CHUNK_CHARS = 300


# =====================================================================
# SHARED HELPERS
# =====================================================================

def _load_existing_index():
    """Load existing FAISS index + metadata, or create empty."""
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    if FAISS_INDEX_PATH.exists() and METADATA_JSON_PATH.exists():
        index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(METADATA_JSON_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        logger.info("Loaded existing index: %d vectors, %d chunks", index.ntotal, len(chunks))
        return index, chunks

    index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    logger.info("Created new empty FAISS index")
    return index, []


def _get_existing_hashes(chunks: List[Dict]) -> set:
    """Extract content_hash values from existing chunk metadata."""
    hashes = set()
    for c in chunks:
        meta = c.get("metadata", c)
        h = meta.get("content_hash")
        if h:
            hashes.add(h)
    return hashes


def _save_index(index, chunks):
    """Persist FAISS index + metadata to disk."""
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    tmp_idx = FAISS_INDEX_PATH.with_suffix(".index.tmp")
    faiss.write_index(index, str(tmp_idx))
    if FAISS_INDEX_PATH.exists():
        FAISS_INDEX_PATH.unlink()
    tmp_idx.rename(FAISS_INDEX_PATH)

    tmp_meta = METADATA_JSON_PATH.with_suffix(".json.tmp")
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    if METADATA_JSON_PATH.exists():
        METADATA_JSON_PATH.unlink()
    tmp_meta.rename(METADATA_JSON_PATH)

    logger.info("Saved: %d vectors → %s", index.ntotal, FAISS_INDEX_PATH)
    logger.info("Saved: %d chunks  → %s", len(chunks), METADATA_JSON_PATH)


def _embed_texts(ollama: OllamaService, texts: List[str]) -> np.ndarray:
    """Generate embeddings in batches via Ollama."""
    all_embeddings = []
    total = len(texts)

    for start in range(0, total, EMBEDDING_BATCH_SIZE):
        batch = texts[start: start + EMBEDDING_BATCH_SIZE]
        label = f"{start + 1}–{min(start + EMBEDDING_BATCH_SIZE, total)}/{total}"
        logger.info("Embedding batch %s", label)
        emb = ollama.embed_batch(batch)
        all_embeddings.append(emb)

    return np.vstack(all_embeddings).astype(np.float32)


def _chunk_text(text: str) -> List[str]:
    """Chunk text using the shared character-based chunker."""
    from app.utils.text_chunker import TextChunker
    chunker = TextChunker(chunk_chars=CHUNK_CHARS, overlap_chars=OVERLAP_CHARS, min_chunk_chars=MIN_CHUNK_CHARS)
    return chunker.chunk_text(text)


# =====================================================================
# COMMAND: reindex_websites
# =====================================================================

def cmd_reindex_websites(args):
    """Crawl allowed websites, chunk, embed, update FAISS.

    With --force, removes all existing website chunks before re-crawling
    to ensure a clean reindex.
    """
    t0 = time.time()
    settings = get_settings()

    domains = settings.allowed_domains
    seed_urls = settings.crawl_seed_urls
    depth = args.depth if args.depth is not None else settings.crawl_depth_limit
    max_pages = args.max_pages if args.max_pages is not None else settings.crawl_max_pages
    force = getattr(args, 'force', False)

    if not domains or not seed_urls:
        logger.error("No allowed_domains or crawl_seed_urls configured. Aborting.")
        return

    logger.info("=" * 60)
    logger.info("VAIT WEBSITE REINDEX")
    logger.info("=" * 60)
    logger.info("Domains   : %s", domains)
    logger.info("Seed URLs : %s", seed_urls)
    logger.info("Depth     : %d", depth)
    logger.info("Max pages : %d", max_pages)
    logger.info("Force     : %s", force)
    # ── Crawl ────────────────────────────────────────────────────────
    crawler = WebsiteCrawler(
        allowed_domains=domains,
        depth_limit=depth,
        max_pages=max_pages,
        delay=settings.crawl_delay,
    )
    pages = crawler.crawl_as_dicts(seed_urls)

    if not pages:
        logger.info("No pages crawled. Nothing to index.")
        return

    logger.info("Crawled %d pages", len(pages))

    # ── Load existing index ──────────────────────────────────────────
    index, chunks = _load_existing_index()
    existing_hashes = _get_existing_hashes(chunks)
    # ── Force mode: remove ALL existing website chunks before re-crawling
    if force:
        original_count = len(chunks)
        non_website_chunks = [
            c for c in chunks
            if c.get("metadata", c).get("source_type") != "website"
        ]
        removed = original_count - len(non_website_chunks)
        if removed > 0:
            logger.info(
                "FORCE mode: removing %d existing website chunks", removed
            )
            chunks = non_website_chunks
            # Rebuild FAISS index without website embeddings
            # We need to re-embed remaining chunks
            if chunks:
                ollama = OllamaService()
                texts = [c["content"] for c in chunks]
                embeddings = _embed_texts(ollama, texts)
                faiss.normalize_L2(embeddings)
                index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
                index.add(embeddings)
            else:
                index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
            existing_hashes = _get_existing_hashes(chunks)
            logger.info(
                "Index rebuilt without website chunks: %d vectors remain",
                index.ntotal,
            )
    # ── Chunk + dedup ────────────────────────────────────────────────
    new_chunks: List[Dict] = []
    for page in pages:
        text = page.get("text", "")
        if not text:
            continue
        text_chunks = _chunk_text(text)
        total = len(text_chunks)
        for idx, chunk_text in enumerate(text_chunks):
            content_hash = hashlib.md5(chunk_text.encode("utf-8")).hexdigest()
            if content_hash in existing_hashes:
                continue
            existing_hashes.add(content_hash)

            new_chunks.append({
                "content": chunk_text,
                "metadata": {
                    "document_type": page.get("document_type", "official_website"),
                    "source_type": page.get("source_type", "website"),
                    "authority_level": page.get("authority_level", "medium"),
                    "url": page.get("url", ""),
                    "document_name": page.get("title", page.get("url", "website")),
                    "academic_year": "2025-26",
                    "department": "General",
                    "source_file": page.get("url", ""),
                    "chunk_index": idx,
                    "total_chunks": total,
                    "content_hash": content_hash,
                },
            })

    dupes = sum(1 for p in pages for _ in _chunk_text(p.get("text", ""))) - len(new_chunks)
    logger.info("New chunks: %d  (duplicates skipped: %d)", len(new_chunks), max(dupes, 0))

    if not new_chunks:
        logger.info("All content already indexed. Nothing to add.")
        return

    # ── Embed ────────────────────────────────────────────────────────
    ollama = OllamaService()
    texts = [c["content"] for c in new_chunks]
    embeddings = _embed_texts(ollama, texts)
    faiss.normalize_L2(embeddings)

    # ── Append to FAISS ──────────────────────────────────────────────
    index.add(embeddings)
    chunks.extend(new_chunks)

    _save_index(index, chunks)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("WEBSITE REINDEX COMPLETE")
    logger.info("New chunks added  : %d", len(new_chunks))
    logger.info("Total index size  : %d", index.ntotal)
    logger.info("Time elapsed      : %.1fs", elapsed)
    logger.info("=" * 60)


# =====================================================================
# COMMAND: ingest_social
# =====================================================================

def cmd_ingest_social(args):
    """Ingest social media JSON files from social/ directory."""
    t0 = time.time()
    settings = get_settings()

    social_dir = Path(args.directory) if args.directory else SOCIAL_DIR
    social_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("VAIT SOCIAL MEDIA INGESTION")
    logger.info("=" * 60)
    logger.info("Source directory: %s", social_dir)

    ingestor = SocialIngestor()
    entries = ingestor.from_directory(social_dir)

    if not entries:
        logger.info("No social entries found. Nothing to index.")
        return

    logger.info("Processed %d social entries", len(entries))

    # ── Load existing index ──────────────────────────────────────────
    index, chunks = _load_existing_index()
    existing_hashes = _get_existing_hashes(chunks)

    # ── Chunk + dedup ────────────────────────────────────────────────
    new_chunks: List[Dict] = []
    for entry in entries:
        text = entry.get("text", "")
        if not text:
            continue
        text_chunks = _chunk_text(text)
        total = len(text_chunks)
        for idx, chunk_text in enumerate(text_chunks):
            content_hash = hashlib.md5(chunk_text.encode("utf-8")).hexdigest()
            if content_hash in existing_hashes:
                continue
            existing_hashes.add(content_hash)

            new_chunks.append({
                "content": chunk_text,
                "metadata": {
                    "document_type": entry.get("document_type", "announcement"),
                    "source_type": entry.get("source_type", "social_media"),
                    "authority_level": entry.get("authority_level", "low"),
                    "platform": entry.get("platform", "unknown"),
                    "url": entry.get("url", ""),
                    "date": entry.get("date", ""),
                    "document_name": f"Social: {entry.get('platform', 'post')}",
                    "academic_year": "2025-26",
                    "department": "General",
                    "source_file": entry.get("url", "social_post"),
                    "chunk_index": idx,
                    "total_chunks": total,
                    "content_hash": content_hash,
                },
            })

    logger.info("New chunks: %d", len(new_chunks))

    if not new_chunks:
        logger.info("All social content already indexed.")
        return

    # ── Embed ────────────────────────────────────────────────────────
    ollama = OllamaService()
    texts = [c["content"] for c in new_chunks]
    embeddings = _embed_texts(ollama, texts)
    faiss.normalize_L2(embeddings)

    # ── Append to FAISS ──────────────────────────────────────────────
    index.add(embeddings)
    chunks.extend(new_chunks)

    _save_index(index, chunks)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("SOCIAL INGESTION COMPLETE")
    logger.info("New chunks added  : %d", len(new_chunks))
    logger.info("Total index size  : %d", index.ntotal)
    logger.info("Time elapsed      : %.1fs", elapsed)
    logger.info("=" * 60)


# =====================================================================
# COMMAND: reindex_all
# =====================================================================

def cmd_reindex_all(args):
    """Run full reindex: documents + websites + social."""
    logger.info("=" * 60)
    logger.info("VAIT FULL REINDEX")
    logger.info("=" * 60)

    # Step 1: Run the standard document ingestion
    logger.info("\n[1/3] Document ingestion...")
    from scripts.ingest_documents import main as doc_main
    try:
        sys.argv = ["ingest_documents.py", "--include-knowledge", "--rebuild"]
        doc_main()
    except SystemExit:
        pass

    # Step 2: Website reindex (append mode)
    logger.info("\n[2/3] Website reindex...")
    website_args = argparse.Namespace(depth=None, max_pages=None)
    cmd_reindex_websites(website_args)

    # Step 3: Social media ingestion (append mode)
    logger.info("\n[3/3] Social media ingestion...")
    social_args = argparse.Namespace(directory=None)
    cmd_ingest_social(social_args)

    logger.info("=" * 60)
    logger.info("FULL REINDEX COMPLETE")
    logger.info("=" * 60)


# =====================================================================
# CLI ENTRY POINT
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="VAIT Management CLI — administrative commands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Commands:\n"
            "  reindex_websites   Crawl websites, chunk, embed, update FAISS\n"
            "  ingest_social      Ingest social media JSON from social/ directory\n"
            "  reindex_all        Full reindex: documents + websites + social\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # reindex_websites
    p_web = subparsers.add_parser("reindex_websites", help="Crawl and index websites")
    p_web.add_argument("--depth", type=int, default=None, help="Override crawl depth limit")
    p_web.add_argument("--max-pages", type=int, default=None, help="Override max pages to crawl")
    p_web.add_argument("--force", action="store_true", default=False, help="Remove existing website chunks and re-crawl from scratch")
    p_web.set_defaults(func=cmd_reindex_websites)

    # ingest_social
    p_social = subparsers.add_parser("ingest_social", help="Ingest social media JSON")
    p_social.add_argument("--directory", "-d", type=str, default=None, help="Custom social JSON directory")
    p_social.set_defaults(func=cmd_ingest_social)

    # reindex_all
    p_all = subparsers.add_parser("reindex_all", help="Full reindex: docs + web + social")
    p_all.set_defaults(func=cmd_reindex_all)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
