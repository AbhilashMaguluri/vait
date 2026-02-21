"""
VAIT File Watcher — Automatic Incremental Ingestion
====================================================

Monitors configured directories for new/modified files and triggers
incremental FAISS ingestion WITHOUT rebuilding the entire index.

Monitored folders (configurable):
  - knowledge/notices/
  - uploads/
  - announcements/

Supported file types: .pdf, .docx, .txt, .json (social media)

Dependencies: watchdog
"""

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger("vait.file_watcher")

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    logger.warning(
        "watchdog library not installed. File watching is disabled. "
        "Install with: pip install watchdog"
    )

# File types the watcher will act on
WATCHED_EXTENSIONS = {".pdf", ".docx", ".txt", ".json"}

# Debounce window: ignore rapid duplicate events (seconds)
DEBOUNCE_SECONDS = 3.0


class _IngestionHandler(FileSystemEventHandler if WATCHDOG_AVAILABLE else object):
    """
    Watchdog event handler that queues new/modified files for ingestion.

    Uses a debounce mechanism so that rapid successive writes to the
    same file result in only a single ingestion.
    """

    def __init__(self, callback: Callable[[Path], None]):
        super().__init__()
        self._callback = callback
        self._recent: Dict[str, float] = {}
        self._lock = threading.Lock()

    # watchdog fires on_created and on_modified
    def on_created(self, event):
        if not event.is_directory:
            self._handle(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._handle(event.src_path)

    def _handle(self, src_path: str) -> None:
        path = Path(src_path)
        if path.suffix.lower() not in WATCHED_EXTENSIONS:
            return

        now = time.time()
        with self._lock:
            last = self._recent.get(str(path), 0.0)
            if now - last < DEBOUNCE_SECONDS:
                return
            self._recent[str(path)] = now

        logger.info("File watcher detected: %s", path.name)
        try:
            self._callback(path)
        except Exception as exc:
            logger.error("Ingestion callback failed for %s: %s", path.name, exc)


class FileWatcher:
    """
    Monitors directories for new document files and invokes a callback.

    Usage::

        def ingest(path: Path):
            # extract, chunk, embed, add to FAISS
            ...

        watcher = FileWatcher(
            directories=[Path("uploads"), Path("notices")],
            on_new_file=ingest,
        )
        watcher.start()   # non-blocking (background thread)
        ...
        watcher.stop()

    The watcher runs in a daemon thread and will not block application
    shutdown.
    """

    def __init__(
        self,
        directories: List[Path],
        on_new_file: Callable[[Path], None],
    ):
        if not WATCHDOG_AVAILABLE:
            raise RuntimeError(
                "watchdog is required for FileWatcher. "
                "Install with: pip install watchdog"
            )
        self.directories = directories
        self._handler = _IngestionHandler(callback=on_new_file)
        self._observer = Observer()
        self._observer.daemon = True  # allow clean shutdown

    def start(self) -> None:
        """Start monitoring all directories (non-blocking)."""
        for d in self.directories:
            d.mkdir(parents=True, exist_ok=True)
            self._observer.schedule(self._handler, str(d), recursive=True)
            logger.info("File watcher monitoring: %s", d)

        self._observer.start()
        logger.info("File watcher started (%d directories)", len(self.directories))

    def stop(self) -> None:
        """Stop the file watcher gracefully."""
        self._observer.stop()
        self._observer.join(timeout=5)
        logger.info("File watcher stopped")

    @property
    def is_alive(self) -> bool:
        """Whether the watcher thread is running."""
        return self._observer.is_alive()


# =====================================================================
# INCREMENTAL INGESTION HELPER
# =====================================================================

class IncrementalIngestor:
    """
    Helper that performs safe incremental FAISS ingestion for a single file.

    This is the callback target for FileWatcher.  It:
      1. Extracts text (PDF / DOCX / TXT / social JSON)
      2. Chunks using the shared TextChunker
      3. Computes content hashes to prevent duplicate vectors
      4. Embeds only new chunks
      5. Appends to FAISS + metadata
      6. Persists index to disk

    Requires a reference to the live RAGService instance.
    """

    def __init__(self, rag_service):
        """
        Args:
            rag_service: The running RAGService instance.
        """
        from app.services.rag_service import RAGService  # deferred import
        self.rag: RAGService = rag_service
        self._existing_hashes: Optional[Set[str]] = None

    # ── Public ───────────────────────────────────────────────────────

    def ingest_file(self, file_path: Path) -> int:
        """
        Synchronous incremental ingest of a single file.

        Returns the number of new chunks added.
        """
        import asyncio

        # Run the async pipeline in an event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an async context — schedule a task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self._async_ingest(file_path))
                    return future.result()
            else:
                return loop.run_until_complete(self._async_ingest(file_path))
        except RuntimeError:
            return asyncio.run(self._async_ingest(file_path))

    async def _async_ingest(self, file_path: Path) -> int:
        """Async implementation of single-file incremental ingestion."""
        import faiss
        import numpy as np

        suffix = file_path.suffix.lower()

        # ── Extract text entries ─────────────────────────────────────
        if suffix == ".json":
            entries = self._extract_social(file_path)
        else:
            entries = self._extract_document(file_path)

        if not entries:
            logger.info("No content extracted from %s", file_path.name)
            return 0

        # ── Chunk ────────────────────────────────────────────────────
        all_chunks: List[Dict] = []
        for entry in entries:
            text = entry.pop("text", "")
            if not text:
                continue
            chunks = self.rag.text_chunker.chunk_text(text)
            total = len(chunks)
            for idx, chunk_text in enumerate(chunks):
                content_hash = hashlib.md5(chunk_text.encode("utf-8")).hexdigest()
                all_chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        **entry,
                        "document_name": file_path.stem,
                        "source_file": file_path.name,
                        "chunk_index": idx,
                        "total_chunks": total,
                        "content_hash": content_hash,
                    },
                })

        # ── Deduplicate against existing index ───────────────────────
        existing = self._get_existing_hashes()
        new_chunks = [
            c for c in all_chunks
            if c["metadata"].get("content_hash") not in existing
        ]

        skipped = len(all_chunks) - len(new_chunks)
        if skipped:
            logger.info("Dedup: skipped %d existing chunks from %s", skipped, file_path.name)

        if not new_chunks:
            logger.info("No new content in %s (all duplicates)", file_path.name)
            return 0

        # ── Embed + append to FAISS ──────────────────────────────────
        chunk_texts = [c["content"] for c in new_chunks]
        embeddings = await self.rag.embedding_service.get_embeddings(chunk_texts)
        faiss.normalize_L2(embeddings)

        self.rag.index.add(embeddings)
        self.rag.chunks.extend(new_chunks)

        # Update hash cache
        for c in new_chunks:
            h = c["metadata"].get("content_hash")
            if h:
                existing.add(h)

        # Persist
        await self.rag._save_index()

        logger.info(
            "Incremental ingest: +%d chunks from %s (total index: %d)",
            len(new_chunks), file_path.name, self.rag.index.ntotal,
        )
        return len(new_chunks)

    # ── Text extraction helpers ──────────────────────────────────────

    @staticmethod
    def _extract_document(file_path: Path) -> List[Dict]:
        """Extract text from PDF/DOCX/TXT and wrap in metadata dict."""
        # Reuse the ingestion script's extraction logic
        from scripts.ingest_documents import extract_text_from_file, infer_metadata

        text = extract_text_from_file(file_path)
        if not text:
            return []

        meta = infer_metadata(file_path, file_path.parent)
        meta["text"] = text
        return [meta]

    @staticmethod
    def _extract_social(file_path: Path) -> List[Dict]:
        """Extract social-media entries from a JSON file."""
        from app.services.social_ingestor import SocialIngestor

        ingestor = SocialIngestor()
        return ingestor.from_file(file_path)

    def _get_existing_hashes(self) -> Set[str]:
        """Build a set of content_hash values from the current index."""
        if self._existing_hashes is not None:
            return self._existing_hashes

        hashes: Set[str] = set()
        for chunk in self.rag.chunks:
            meta = chunk.get("metadata", chunk)
            h = meta.get("content_hash")
            if h:
                hashes.add(h)

        self._existing_hashes = hashes
        logger.debug("Loaded %d existing content hashes", len(hashes))
        return hashes
