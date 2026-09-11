"""
VAIT Browser Retrieval Service — Headless Browser & Vision Fallback
==================================================================

Provides Tier 3 (Headless Browser DOM extraction) and Tier 4 (Targeted Screenshot + Vision OCR)
for official institutional pages that require client-side JavaScript rendering,
client-side routing, or dynamic AJAX loading.

Priority:
  1. Existing RAG (Fastest, 0ms)
  2. Normal HTTP Fetch (Fast, ~200ms)
  3. Headless Browser Rendered DOM (~1.5s - 3s)
  4. Screenshot + Vision/OCR Fallback (~3s - 5s)

Security & Guardrails:
  - Enforces strict SSRF protection: strictly limited to allowed official domains (vvitu.ac.in, vvitguntur.com)
  - Rejects localhost, 127.0.0.1, private IP ranges (RFC 1918)
  - Headless-only: user never sees a browser window
  - In-memory screenshot buffers: zero temporary image files left on disk
  - Strict vision extraction prompt: prevents visual hallucination
"""

import asyncio
import base64
import ipaddress
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

import httpx

from app.services.website_crawler import ContentExtractor
from app.utils.config import Settings, get_settings

logger = logging.getLogger("vait.browser_retrieval")

# ── Allowed Official Domains ─────────────────────────────────────────
ALLOWED_DOMAINS: Set[str] = {
    "vvitu.ac.in",
    "www.vvitu.ac.in",
    "vvitguntur.com",
    "www.vvitguntur.com",
}

BLOCKED_IP_PATTERNS = [
    re.compile(r"^127\."),
    re.compile(r"^10\."),
    re.compile(r"^172\.(1[6-9]|2[0-9]|3[0-1])\."),
    re.compile(r"^192\.168\."),
    re.compile(r"^169\.254\."),
    re.compile(r"^0\.0\.0\.0"),
]


# =====================================================================
# SSRF & URL VALIDATOR
# =====================================================================

def is_safe_official_url(url: str) -> bool:
    """
    Validate that *url* belongs strictly to an allowed official domain
    and does not target internal or private network interfaces.
    """
    if not url or not isinstance(url, str):
        return False

    try:
        parsed = urlparse(url.strip())
        if parsed.scheme not in ("http", "https"):
            logger.warning("SSRF blocked: Invalid scheme '%s' in %s", parsed.scheme, url)
            return False

        hostname = (parsed.hostname or "").lower()
        if not hostname:
            return False

        # Reject direct IP targets or localhost
        if hostname == "localhost":
            return False

        for pattern in BLOCKED_IP_PATTERNS:
            if pattern.search(hostname):
                logger.warning("SSRF blocked: Private IP target '%s'", hostname)
                return False

        try:
            ip_obj = ipaddress.ip_address(hostname)
            if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_reserved or ip_obj.is_link_local:
                logger.warning("SSRF blocked: Private IP address '%s'", hostname)
                return False
        except ValueError:
            # Not a raw IP literal, normal domain name
            pass

        # Must match allowlisted official domains
        if hostname not in ALLOWED_DOMAINS:
            logger.warning("SSRF blocked: Domain '%s' not in allowed official domains %s", hostname, ALLOWED_DOMAINS)
            return False

        return True

    except Exception as exc:
        logger.warning("URL validation error for '%s': %s", url, exc)
        return False


# =====================================================================
# DATA STRUCTURES
# =====================================================================

@dataclass
class BrowserRenderResult:
    """Represents the outcome of headless browser retrieval."""
    url: str
    title: str
    text: str
    method: str  # "rendered_dom" | "screenshot_vision" | "direct_url_only" | "failed"
    success: bool
    error: Optional[str] = None
    screenshot_b64: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =====================================================================
# VISION EXTRACTOR
# =====================================================================

class VisionExtractor:
    """Extracts factual text from webpage screenshots using multimodal LLMs."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "google/gemini-2.5-flash"

    async def extract_from_image(self, image_bytes: bytes, query: str) -> Optional[str]:
        """
        Send image bytes to vision model and extract strictly facts answering *query*.
        """
        if not self.settings.openrouter_api_key:
            logger.warning("Vision extraction skipped: OPENROUTER_API_KEY is not configured")
            return None

        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        prompt = (
            f"You are the official institutional document extractor for VVITU / VVIT.\n"
            f"Extract only verified, factual information visible in this webpage screenshot that answers:\n"
            f"QUESTION: {query}\n\n"
            f"RULES:\n"
            f"1. Extract ONLY information explicitly visible in the image (e.g. name, designation, department, qualification, fees, dates, announcements).\n"
            f"2. Do NOT invent, assume, or extrapolate unverified details.\n"
            f"3. If the requested information is not visible in the image, output exactly: 'The requested information is not visible on this page.'"
        )

        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": 800,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                    ],
                }
            ],
        }

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(self.endpoint, headers=headers, json=payload)
                if response.status_code != 200:
                    logger.warning("Vision API non-200 (%d): %s", response.status_code, response.text[:200])
                    return None

                data = response.json()
                choices = data.get("choices") or []
                if not choices:
                    return None

                extracted = choices[0].get("message", {}).get("content", "").strip()
                if not extracted or "not visible" in extracted.lower():
                    return None

                logger.info("Vision extraction succeeded (%d chars)", len(extracted))
                return extracted

        except Exception as exc:
            logger.warning("Vision extraction failed: %s", exc)
            return None


# =====================================================================
# BROWSER RETRIEVAL SERVICE
# =====================================================================

class BrowserRetrievalService:
    """
    Singleton headless browser service for JavaScript-rendered official page retrieval.
    Reuses browser context across requests to minimize overhead.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.vision_extractor = VisionExtractor(self.settings)
        self._playwright = None
        self._browser = None
        self._lock = asyncio.Lock()
        self._available = True

    async def _ensure_browser(self):
        """Initialize Playwright and launch headless browser if not already active."""
        if self._browser is not None and self._browser.is_connected():
            return self._browser

        from playwright.async_api import async_playwright

        if self._playwright is None:
            self._playwright = await async_playwright().start()

        # Try system Google Chrome first, then system Microsoft Edge, then bundled Chromium
        launch_attempts = [
            {"channel": "chrome"},
            {"channel": "msedge"},
            {},  # default bundled chromium
        ]

        for opts in launch_attempts:
            try:
                browser = await self._playwright.chromium.launch(
                    headless=True,
                    timeout=10000,
                    **opts,
                )
                self._browser = browser
                logger.info("Headless browser launched successfully with opts=%s", opts)
                return self._browser
            except Exception as exc:
                logger.debug("Browser launch attempt failed with %s: %s", opts, exc)

        logger.error("Could not launch any headless browser. Headless fallback will be disabled.")
        self._available = False
        return None

    async def render_page(
        self,
        url: str,
        query: Optional[str] = None,
        wait_selector: Optional[str] = None,
        timeout_seconds: int = 15,
    ) -> BrowserRenderResult:
        """
        Headlessly navigate to an official page, execute JavaScript,
        and extract the rendered DOM text. If rendered DOM text is insufficient,
        captures a screenshot and uses vision OCR.
        """
        if not is_safe_official_url(url):
            return BrowserRenderResult(
                url=url,
                title="Untrusted URL",
                text="",
                method="failed",
                success=False,
                error="SSRF security violation: URL is not an approved official domain",
            )

        async with self._lock:
            try:
                browser = await self._ensure_browser()
                if browser is None:
                    return BrowserRenderResult(
                        url=url,
                        title="",
                        text="",
                        method="direct_url_only",
                        success=False,
                        error="Headless browser unavailable in deployment environment",
                    )

                page = await browser.new_page(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) VAIT-HeadlessBrowser/1.0",
                    viewport={"width": 1280, "height": 960},
                )

                try:
                    # Navigate with domcontentloaded
                    await page.goto(url, wait_until="domcontentloaded", timeout=timeout_seconds * 1000)

                    # Wait for SPA client hydration (e.g. #root has children or wait_selector)
                    if wait_selector:
                        try:
                            await page.wait_for_selector(wait_selector, timeout=4000)
                        except Exception:
                            pass
                    else:
                        # Wait briefly for React/Vite mount
                        try:
                            await page.wait_for_selector("#root:not(:empty)", timeout=3500)
                        except Exception:
                            pass

                    # Small tick for animations/renders to settle
                    await asyncio.sleep(0.3)

                    title = await page.title()
                    raw_html = await page.content()

                    # Extract rendered text from #root or body
                    rendered_text = ""
                    try:
                        root_elem = await page.query_selector("#root")
                        if root_elem:
                            rendered_text = await root_elem.inner_text()
                    except Exception:
                        pass

                    if not rendered_text or len(rendered_text.strip()) < 80:
                        try:
                            body_elem = await page.query_selector("body")
                            if body_elem:
                                rendered_text = await body_elem.inner_text()
                        except Exception:
                            pass

                    clean_text = ContentExtractor._normalise(rendered_text or "")

                    # ── Evaluate if DOM text is sufficient ────────────────
                    is_dom_sufficient = len(clean_text) >= 120

                    if is_dom_sufficient and query:
                        # Check if query keywords appear in rendered DOM
                        q_words = [
                            w for w in re.sub(r"[^a-zA-Z0-9\s]", " ", query.lower()).split()
                            if len(w) > 3 and w not in {"what", "tell", "about", "give", "show", "who", "with"}
                        ]
                        if q_words and not any(w in clean_text.lower() for w in q_words):
                            is_dom_sufficient = False

                    if is_dom_sufficient:
                        logger.info("Rendered DOM extraction succeeded for %s (%d chars)", url, len(clean_text))
                        return BrowserRenderResult(
                            url=url,
                            title=title or "Official Portal",
                            text=clean_text,
                            method="headless_browser_dom",
                            success=True,
                            metadata={"chars": len(clean_text)},
                        )

                    # ── Tier 4: Screenshot + Vision Fallback ─────────────
                    logger.info("DOM text insufficient (%d chars). Falling back to screenshot vision for %s", len(clean_text), url)
                    screenshot_bytes = None

                    # Try element screenshot first
                    target_selectors = [wait_selector, "#root", "main", "article", ".container"]
                    for sel in target_selectors:
                        if sel:
                            try:
                                elem = await page.query_selector(sel)
                                if elem:
                                    screenshot_bytes = await elem.screenshot(type="png")
                                    break
                            except Exception:
                                pass

                    # Fallback to viewport screenshot
                    if not screenshot_bytes:
                        screenshot_bytes = await page.screenshot(type="png", full_page=False)

                    b64_str = base64.b64encode(screenshot_bytes).decode("utf-8") if screenshot_bytes else None

                    if screenshot_bytes and query:
                        vision_text = await self.vision_extractor.extract_from_image(screenshot_bytes, query)
                        if vision_text:
                            logger.info("Screenshot vision extraction succeeded for %s (%d chars)", url, len(vision_text))
                            return BrowserRenderResult(
                                url=url,
                                title=title or "Official Portal",
                                text=vision_text,
                                method="headless_browser_screenshot",
                                success=True,
                                screenshot_b64=b64_str,
                                metadata={"chars": len(vision_text)},
                            )

                    # If both DOM and vision failed to extract sufficient content,
                    # return direct_url_only so VAIT provides the direct clickable page link
                    return BrowserRenderResult(
                        url=url,
                        title=title or "Official Portal",
                        text=clean_text if len(clean_text) > 40 else "",
                        method="direct_url_only",
                        success=False,
                        error="Content could not be reliably extracted; direct official page URL provided",
                    )

                finally:
                    await page.close()

            except Exception as exc:
                logger.warning("Browser retrieval failed for %s: %s", url, exc)
                return BrowserRenderResult(
                    url=url,
                    title="Official Portal",
                    text="",
                    method="direct_url_only",
                    success=False,
                    error=str(exc),
                )

    async def close(self):
        """Cleanly terminate the browser and Playwright process."""
        async with self._lock:
            if self._browser:
                try:
                    await self._browser.close()
                except Exception:
                    pass
                self._browser = None

            if self._playwright:
                try:
                    await self._playwright.stop()
                except Exception:
                    pass
                self._playwright = None


# Singleton instance
_browser_retrieval_service: Optional[BrowserRetrievalService] = None

def get_browser_retrieval_service() -> BrowserRetrievalService:
    """Get or create singleton instance of BrowserRetrievalService."""
    global _browser_retrieval_service
    if _browser_retrieval_service is None:
        _browser_retrieval_service = BrowserRetrievalService()
    return _browser_retrieval_service
