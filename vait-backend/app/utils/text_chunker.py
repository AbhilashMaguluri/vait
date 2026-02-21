"""
VAIT Text Chunking Utility
==========================
Production-grade character-based chunking for RAG.

Approximation: 1 token ≈ 4 characters.
  - Target chunk size : ~400 tokens → 1600 characters
  - Overlap           : ~50 tokens  → 200 characters
  - Minimum chunk     : 300 characters (smaller chunks are discarded)

Chunks respect paragraph and sentence boundaries where possible.
"""

import re
from typing import List

# ── Defaults (character-based, 1 token ≈ 4 chars) ───────────────────
DEFAULT_CHUNK_CHARS = 1600    # ~400 tokens
DEFAULT_OVERLAP_CHARS = 200   # ~50 tokens
MIN_CHUNK_CHARS = 300         # discard tiny trailing chunks


class TextChunker:
    """
    Splits text into overlapping chunks using character-length targets.

    Preserves paragraph / sentence boundaries where feasible,
    falls back to word-level splitting for oversized blocks.
    """

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 50,
        *,
        chunk_chars: int | None = None,
        overlap_chars: int | None = None,
        min_chunk_chars: int = MIN_CHUNK_CHARS,
    ):
        """
        Args:
            chunk_size:     Legacy token target (converted to chars via ×4).
            chunk_overlap:  Legacy token overlap (converted to chars via ×4).
            chunk_chars:    Explicit character target (overrides chunk_size).
            overlap_chars:  Explicit character overlap (overrides chunk_overlap).
            min_chunk_chars: Minimum chunk length in characters.
        """
        self.chunk_chars = chunk_chars or chunk_size * 4
        self.overlap_chars = overlap_chars or chunk_overlap * 4
        self.min_chunk_chars = min_chunk_chars

    # ── Public API ───────────────────────────────────────────────────

    def chunk_text(self, text: str) -> List[str]:
        """Return a list of text chunks from *text*."""
        text = self.clean_text(text)
        if not text:
            return []

        paragraphs = self._split_paragraphs(text)
        raw_chunks = self._merge_paragraphs(paragraphs)

        # Discard chunks shorter than minimum
        return [c for c in raw_chunks if len(c) >= self.min_chunk_chars]

    # ── Text Cleaning (also usable standalone) ───────────────────────

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Normalise whitespace and line breaks.

        - Collapse runs of spaces/tabs (not newlines) into a single space.
        - Limit consecutive blank lines to one (two newlines max).
        - Strip leading/trailing whitespace.
        """
        text = re.sub(r"[^\S\n]+", " ", text)       # spaces/tabs → single space
        text = re.sub(r"\n{3,}", "\n\n", text)       # max 2 consecutive newlines
        text = re.sub(r"[ \t]+\n", "\n", text)        # trailing spaces before \n
        return text.strip()

    # ── Internal helpers ─────────────────────────────────────────────

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

            # Single paragraph exceeds chunk size → split by sentences
            if para_len > self.chunk_chars:
                if buf:
                    chunks.append("\n\n".join(buf))
                    buf, buf_len = [], 0
                chunks.extend(self._split_large(para))
                continue

            # Adding paragraph would exceed limit → finalise current chunk
            if buf_len + para_len > self.chunk_chars and buf:
                chunks.append("\n\n".join(buf))
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
        """Split oversized text by sentences, then words as fallback."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: List[str] = []
        buf: List[str] = []
        buf_len = 0

        for sent in sentences:
            sent_len = len(sent)
            if sent_len > self.chunk_chars:
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
