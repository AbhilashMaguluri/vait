"""
VAIT Social Media Ingestor — Structured JSON → Knowledge Base
=============================================================

Converts structured social-media JSON entries (Instagram, LinkedIn, etc.)
into clean text + metadata ready for FAISS ingestion.

Expected input format (per entry)::

    {
        "title": "...",
        "content": "...",
        "date": "2025-02-15",
        "platform": "Instagram",
        "url": "https://..."
    }

Output per entry::

    {
        "text": "<cleaned text>",
        "source_type": "social_media",
        "source_tier": "secondary_linkedin",
        "authority_level": "low",
        "document_type": "announcement",
        "platform": "Instagram",
        "url": "https://...",
        "date": "2025-02-15"
    }

Supports loading from:
  - A single JSON file (list of entries)
  - A directory of .json files (each containing one or more entries)
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("vait.social_ingestor")


# =====================================================================
# TEXT CLEANING
# =====================================================================

def _clean_social_text(text: str) -> str:
    """
    Normalise social-media text.

    - Removes excess hashtags/emojis (keeps readable ones)
    - Collapses whitespace
    - Strips leading/trailing whitespace
    """
    # Remove zero-width characters
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    # Collapse whitespace
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _build_text_block(entry: Dict) -> str:
    """
    Compose a human-readable text block from a social-media entry.

    Format::

        [Platform] Title
        Date: ...

        Content body
    """
    parts: List[str] = []

    platform = entry.get("platform", "Social Media")
    title = entry.get("title", "").strip()
    date = entry.get("date", "").strip()
    content = entry.get("content", "").strip()

    if title:
        parts.append(f"[{platform}] {title}")
    else:
        parts.append(f"[{platform}] Post")

    if date:
        parts.append(f"Date: {date}")

    if content:
        parts.append("")
        parts.append(content)

    return "\n".join(parts)


def _infer_source_profile(entry: Dict) -> Dict[str, str]:
    """
    Infer source tier and default authority from social entry metadata.

    Returns dict with:
      - source_tier
      - authority_level
    """
    platform = str(entry.get("platform", "")).strip().lower()
    url = str(entry.get("url", "")).strip().lower()
    title = str(entry.get("title", "")).strip().lower()
    content = str(entry.get("content", "")).strip().lower()

    combined_text = f"{title} {content}"
    mentions_vvit = "vvit" in combined_text or "vvitu" in combined_text

    is_linkedin = platform == "linkedin" or "linkedin.com" in url
    if is_linkedin:
        return {
            "source_tier": "secondary_linkedin",
            "authority_level": "medium" if mentions_vvit else "low",
        }

    if platform in {"instagram", "twitter", "x", "facebook", "youtube"}:
        return {
            "source_tier": "tertiary_social",
            "authority_level": "low",
        }

    if mentions_vvit:
        return {
            "source_tier": "tertiary_social",
            "authority_level": "low",
        }

    return {
        "source_tier": "related_web",
        "authority_level": "low",
    }


# =====================================================================
# SOCIAL INGESTOR
# =====================================================================

class SocialIngestor:
    """
    Converts structured social-media JSON into ingestion-ready dicts.

    Usage::

        ingestor = SocialIngestor()
        entries = ingestor.from_file(Path("social_posts.json"))
        entries = ingestor.from_directory(Path("social/"))
    """

    def __init__(self):
        self._seen_hashes: set = set()

    # ── Public API ───────────────────────────────────────────────────

    def from_entries(self, entries: List[Dict]) -> List[Dict]:
        """
        Convert a list of raw social-media dicts to ingestion-ready dicts.

        Deduplicates by content hash.
        """
        results: List[Dict] = []

        for entry in entries:
            processed = self._process_entry(entry)
            if processed is not None:
                results.append(processed)

        logger.info(
            "Social ingestor: %d entries → %d unique docs (skipped %d dupes)",
            len(entries), len(results), len(entries) - len(results),
        )
        return results

    def from_file(self, file_path: Path) -> List[Dict]:
        """
        Load entries from a JSON file.

        The file must contain either:
          - A JSON array of entry objects, OR
          - A single entry object
        """
        if not file_path.exists():
            logger.warning("Social JSON file not found: %s", file_path)
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.error("Failed to parse %s: %s", file_path.name, exc)
            return []

        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            logger.warning("Unexpected JSON structure in %s", file_path.name)
            return []

        logger.info("Loaded %d social entries from %s", len(data), file_path.name)
        return self.from_entries(data)

    def from_directory(self, directory: Path) -> List[Dict]:
        """Load and merge entries from all .json files in *directory*."""
        if not directory.exists():
            logger.warning("Social directory not found: %s", directory)
            return []

        all_results: List[Dict] = []
        json_files = sorted(directory.glob("*.json"))

        if not json_files:
            logger.info("No .json files found in %s", directory)
            return []

        for fpath in json_files:
            all_results.extend(self.from_file(fpath))

        logger.info(
            "Social ingestor: loaded %d total docs from %d file(s) in %s",
            len(all_results), len(json_files), directory,
        )
        return all_results

    def reset(self) -> None:
        """Clear deduplication state for a fresh run."""
        self._seen_hashes.clear()

    # ── Internal ─────────────────────────────────────────────────────

    def _process_entry(self, entry: Dict) -> Optional[Dict]:
        """Convert a single social entry to an ingestion-ready dict."""
        text_block = _build_text_block(entry)
        cleaned = _clean_social_text(text_block)
        profile = _infer_source_profile(entry)

        if len(cleaned) < 30:
            logger.debug("Social entry too short, skipping: %.60s…", cleaned)
            return None

        # Deduplicate
        content_hash = hashlib.md5(cleaned.encode("utf-8")).hexdigest()
        if content_hash in self._seen_hashes:
            return None
        self._seen_hashes.add(content_hash)

        return {
            "text": cleaned,
            "source_type": "social_media",
            "source_tier": entry.get("source_tier", profile["source_tier"]),
            "authority_level": entry.get("authority_level", profile["authority_level"]),
            "document_type": entry.get("document_type", "announcement"),
            "platform": entry.get("platform", "unknown"),
            "url": entry.get("url", ""),
            "date": entry.get("date", ""),
            "content_hash": content_hash,
        }
