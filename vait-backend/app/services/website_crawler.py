"""
VAIT Website Crawler — Production-Grade Web Content Extraction
==============================================================

Fully offline-capable crawl engine for institutional websites.

Features:
  - Domain-restricted crawling (never leaves allowed domains)
  - Configurable depth limit (default 2)
  - Removes script, style, nav, header, footer, aside
  - Preserves heading structure (h1–h3), paragraphs, list items
  - Content deduplication via text hash
  - Structured output ready for FAISS ingestion

Dependencies: requests, beautifulsoup4, urllib3
"""

import hashlib
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Comment, Tag

logger = logging.getLogger("vait.crawler")

# ── Defaults ─────────────────────────────────────────────────────────
DEFAULT_DEPTH_LIMIT = 2
DEFAULT_REQUEST_TIMEOUT = 15  # seconds
DEFAULT_DELAY_BETWEEN_REQUESTS = 0.5  # seconds (polite crawl)
DEFAULT_MAX_PAGES = 200

# Tags to remove entirely (including children)
REMOVE_TAGS = {"script", "style", "nav", "header", "footer", "aside", "form", "iframe", "noscript"}

# Tags whose visible text we keep
KEEP_TAGS = {"h1", "h2", "h3", "h4", "p", "li", "td", "th", "blockquote", "pre", "caption"}

# Binary / non-HTML suffixes to skip
SKIP_EXTENSIONS = {
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico",
    ".mp3", ".mp4", ".avi", ".mov", ".wmv", ".flv",
    ".zip", ".rar", ".tar", ".gz", ".7z",
    ".exe", ".msi", ".dmg",
    ".css", ".js", ".json", ".xml",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".webp", ".bmp", ".tiff",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
}

# URL path patterns to skip (login, admin, downloads, media)
SKIP_URL_PATTERNS = [
    re.compile(r"/login", re.IGNORECASE),
    re.compile(r"/signin", re.IGNORECASE),
    re.compile(r"/signup", re.IGNORECASE),
    re.compile(r"/register", re.IGNORECASE),
    re.compile(r"/admin", re.IGNORECASE),
    re.compile(r"/wp-admin", re.IGNORECASE),
    re.compile(r"/dashboard", re.IGNORECASE),
    re.compile(r"/logout", re.IGNORECASE),
    re.compile(r"/download[s]?/", re.IGNORECASE),
    re.compile(r"/uploads?/", re.IGNORECASE),
    re.compile(r"/media/", re.IGNORECASE),
    re.compile(r"/cgi-bin/", re.IGNORECASE),
    re.compile(r"/api/", re.IGNORECASE),
    re.compile(r"\?.*action=login", re.IGNORECASE),
    re.compile(r"\?.*action=register", re.IGNORECASE),
]


# =====================================================================
# CRAWL STATISTICS
# =====================================================================

@dataclass
class CrawlStats:
    """Tracks crawl session statistics for logging."""
    pages_crawled: int = 0
    pages_skipped: int = 0
    duplicate_pages: int = 0
    errors: int = 0
    skipped_reasons: Dict[str, int] = field(default_factory=dict)

    def record_skip(self, reason: str) -> None:
        self.pages_skipped += 1
        self.skipped_reasons[reason] = self.skipped_reasons.get(reason, 0) + 1

    def record_error(self) -> None:
        self.errors += 1

    def record_duplicate(self) -> None:
        self.duplicate_pages += 1

    def record_crawled(self) -> None:
        self.pages_crawled += 1

    def summary(self) -> str:
        lines = [
            f"Pages crawled   : {self.pages_crawled}",
            f"Pages skipped   : {self.pages_skipped}",
            f"Duplicate pages : {self.duplicate_pages}",
            f"Errors          : {self.errors}",
        ]
        if self.skipped_reasons:
            lines.append("Skip breakdown  :")
            for reason, count in sorted(self.skipped_reasons.items()):
                lines.append(f"  {reason}: {count}")
        return "\n".join(lines)


# =====================================================================
# DATA STRUCTURES
# =====================================================================

class CrawledPage:
    """Represents a single crawled page."""

    __slots__ = ("url", "title", "text", "depth", "raw_html")

    def __init__(self, url: str, title: str, text: str, depth: int, raw_html: str = ""):
        self.url = url
        self.title = title
        self.text = text
        self.depth = depth
        self.raw_html = raw_html

    def to_dict(self) -> Dict:
        """Convert to metadata-ready dict for ingestion."""
        return {
            "text": self.text,
            "url": self.url,
            "title": self.title,
            "source_type": "website",
            "source_tier": "primary_official",
            "authority_level": "medium",
            "document_type": "official_website",
        }


# =====================================================================
# CONTENT EXTRACTOR
# =====================================================================

class ContentExtractor:
    """
    Extract clean visible text from raw HTML.

    Strategy:
      1. Remove unwanted tags (script, style, nav, …)
      2. Remove HTML comments
      3. Walk remaining tree and build structured text
      4. Collapse excessive whitespace
    """

    @staticmethod
    def extract(html: str, url: str = "") -> Optional[str]:
        """
        Parse *html* and return clean text, or None if too short.

        Args:
            html: Raw HTML string.
            url:  Page URL (used only for debug logging).

        Returns:
            Cleaned text string or None.
        """
        soup = BeautifulSoup(html, "html.parser")

        # Remove unwanted elements
        for tag_name in REMOVE_TAGS:
            for element in soup.find_all(tag_name):
                element.decompose()

        # Remove HTML comments
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            comment.extract()

        # Build structured text
        lines: List[str] = []
        body = soup.find("body") or soup

        for element in body.descendants:
            if not isinstance(element, Tag):
                continue

            tag = element.name

            # Headings → "## Heading" style
            if tag in ("h1", "h2", "h3", "h4"):
                text = element.get_text(separator=" ", strip=True)
                if text:
                    prefix = "#" * int(tag[1])
                    lines.append(f"\n{prefix} {text}\n")

            # Paragraphs
            elif tag == "p":
                text = element.get_text(separator=" ", strip=True)
                if text:
                    lines.append(text + "\n")

            # List items
            elif tag == "li":
                text = element.get_text(separator=" ", strip=True)
                if text:
                    lines.append(f"• {text}")

            # Table cells (inline)
            elif tag in ("td", "th"):
                text = element.get_text(separator=" ", strip=True)
                if text:
                    lines.append(text)

        raw_text = "\n".join(lines)
        cleaned = ContentExtractor._normalise(raw_text)

        if len(cleaned) < 80:
            logger.debug("Page too short after extraction (%d chars): %s", len(cleaned), url)
            return None

        return cleaned

    @staticmethod
    def _normalise(text: str) -> str:
        """Collapse whitespace and limit blank lines."""
        text = re.sub(r"[^\S\n]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+\n", "\n", text)
        return text.strip()


# =====================================================================
# WEBSITE CRAWLER
# =====================================================================

class WebsiteCrawler:
    """
    BFS website crawler restricted to allowed domains.

    Usage::

        crawler = WebsiteCrawler(
            allowed_domains=["www.vvitguntur.com"],
            depth_limit=2,
        )
        pages = crawler.crawl(["https://www.vvitguntur.com"])
    """

    def __init__(
        self,
        allowed_domains: List[str],
        depth_limit: int = DEFAULT_DEPTH_LIMIT,
        max_pages: int = DEFAULT_MAX_PAGES,
        request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
        delay: float = DEFAULT_DELAY_BETWEEN_REQUESTS,
    ):
        self.allowed_domains: Set[str] = {d.lower() for d in allowed_domains}
        self.depth_limit = depth_limit
        self.max_pages = max_pages
        self.request_timeout = request_timeout
        self.delay = delay

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "VAIT-Crawler/1.0 (Institutional Knowledge Bot)",
            "Accept": "text/html,application/xhtml+xml",
        })

        self._visited: Set[str] = set()
        self._content_hashes: Set[str] = set()
        self.extractor = ContentExtractor()
        self.stats = CrawlStats()

    # ── Public API ───────────────────────────────────────────────────

    def crawl(self, seed_urls: List[str]) -> List[CrawledPage]:
        """
        BFS crawl starting from *seed_urls*.

        Returns a list of CrawledPage objects with deduplicated content.
        """
        pages: List[CrawledPage] = []
        queue: deque = deque()

        for url in seed_urls:
            normalised = self._normalise_url(url)
            if normalised and normalised not in self._visited:
                queue.append((normalised, 0))
                self._visited.add(normalised)

        logger.info(
            "Starting crawl: seeds=%d, domains=%s, depth=%d, max=%d",
            len(queue), list(self.allowed_domains), self.depth_limit, self.max_pages,
        )

        while queue and len(pages) < self.max_pages:
            url, depth = queue.popleft()

            # Check URL patterns BEFORE fetching (skip login/admin/download)
            if self._should_skip_url(url):
                self.stats.record_skip("url_pattern")
                logger.debug("Skipped (pattern): %s", url)
                continue

            page = self._fetch_and_extract(url, depth)
            if page is None:
                continue

            # Content deduplication
            text_hash = hashlib.md5(page.text.encode("utf-8")).hexdigest()
            if text_hash in self._content_hashes:
                self.stats.record_duplicate()
                logger.debug("Duplicate content skipped: %s", url)
                continue
            self._content_hashes.add(text_hash)

            pages.append(page)
            self.stats.record_crawled()
            logger.info(
                "  [%d/%d] depth=%d  chars=%d  %s",
                len(pages), self.max_pages, depth, len(page.text), url,
            )

            # Discover child links (if under depth limit)
            if depth < self.depth_limit:
                child_urls = self._extract_links(url, page)
                for child in child_urls:
                    if child not in self._visited:
                        self._visited.add(child)
                        queue.append((child, depth + 1))

            # Polite delay
            if self.delay > 0:
                time.sleep(self.delay)

        # ── Log crawl statistics ─────────────────────────────────────
        logger.info("─" * 50)
        logger.info("CRAWL STATISTICS:")
        for line in self.stats.summary().split("\n"):
            logger.info("  %s", line)
        logger.info(
            "Crawl complete: %d pages collected, %d URLs visited",
            len(pages), len(self._visited),
        )
        logger.info("─" * 50)

        return pages

    def crawl_as_dicts(self, seed_urls: List[str]) -> List[Dict]:
        """Crawl and return list of metadata-ready dicts."""
        pages = self.crawl(seed_urls)
        return [p.to_dict() for p in pages]

    def get_stats(self) -> CrawlStats:
        """Return the current crawl statistics."""
        return self.stats

    # ── Internal helpers ─────────────────────────────────────────────

    def _should_skip_url(self, url: str) -> bool:
        """Check if URL matches any skip pattern (login, admin, downloads, etc.)."""
        parsed = urlparse(url)
        path_query = parsed.path + ("?" + parsed.query if parsed.query else "")
        for pattern in SKIP_URL_PATTERNS:
            if pattern.search(path_query):
                return True
        return False

    def _fetch_and_extract(self, url: str, depth: int) -> Optional[CrawledPage]:
        """Fetch a URL and extract clean text."""
        try:
            resp = self.session.get(url, timeout=self.request_timeout, allow_redirects=True)
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                self.stats.record_skip("non_html")
                logger.debug("Skipping non-HTML: %s (%s)", url, content_type)
                return None

            raw_html = resp.text
            text = self.extractor.extract(raw_html, url=url)
            if text is None:
                self.stats.record_skip("too_short")
                return None

            # Extract title
            soup = BeautifulSoup(raw_html, "html.parser")
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else url

            return CrawledPage(
                url=url, title=title, text=text, depth=depth, raw_html=raw_html,
            )

        except requests.HTTPError as exc:
            self.stats.record_error()
            logger.warning("HTTP error fetching %s: %s", url, exc)
            return None
        except requests.ConnectionError as exc:
            self.stats.record_error()
            logger.warning("Connection error fetching %s: %s", url, exc)
            return None
        except requests.Timeout as exc:
            self.stats.record_error()
            logger.warning("Timeout fetching %s: %s", url, exc)
            return None
        except requests.RequestException as exc:
            self.stats.record_error()
            logger.warning("Request error fetching %s: %s", url, exc)
            return None
        except Exception as exc:
            self.stats.record_error()
            logger.error("Unexpected error crawling %s: %s", url, exc)
            return None

    def _extract_links(self, base_url: str, page: CrawledPage) -> List[str]:
        """Extract and filter child links from a fetched page.

        Uses cached raw_html from the CrawledPage to avoid re-fetching.
        """
        if not page.raw_html:
            return []

        try:
            soup = BeautifulSoup(page.raw_html, "html.parser")
        except Exception:
            return []

        links: List[str] = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            absolute = urljoin(base_url, href)
            normalised = self._normalise_url(absolute)
            if normalised and self._is_allowed(normalised):
                links.append(normalised)

        return links

    def _is_allowed(self, url: str) -> bool:
        """Check if a URL belongs to an allowed domain and is not a binary."""
        parsed = urlparse(url)
        domain = parsed.hostname
        if domain is None:
            return False
        if domain.lower() not in self.allowed_domains:
            return False
        # Skip binary file extensions
        path_lower = parsed.path.lower()
        for ext in SKIP_EXTENSIONS:
            if path_lower.endswith(ext):
                return False
        return True

    @staticmethod
    def _normalise_url(url: str) -> Optional[str]:
        """Normalise a URL: strip fragment, lowercase scheme + host."""
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return None
            # Rebuild without fragment
            normalised = parsed._replace(fragment="").geturl()
            # Strip trailing slash for consistency
            return normalised.rstrip("/")
        except Exception:
            return None

    def reset(self) -> None:
        """Clear visited set, content hashes, and stats for a fresh crawl."""
        self._visited.clear()
        self._content_hashes.clear()
        self.stats = CrawlStats()
