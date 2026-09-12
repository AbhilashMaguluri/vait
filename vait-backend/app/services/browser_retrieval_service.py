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
from datetime import datetime, timezone
import ipaddress
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse, urljoin

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
    """Represents the outcome of headless browser retrieval with full provenance."""
    url: str
    title: str
    text: str
    method: str  # "rendered_dom" | "screenshot_vision" | "direct_url_only" | "failed"
    success: bool
    entity_type: str = "general_page"  # "faculty_directory" | "leadership" | "course" | "exam" | "general_page"
    entities: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    screenshot_b64: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    screenshot_used: bool = False
    screenshot_scope: Optional[str] = None  # "element", "semantic_tiles", "viewport"
    vision_provider: Optional[str] = None
    vision_model: Optional[str] = None
    extraction_timestamp: Optional[str] = None


# =====================================================================
# VISION EXTRACTOR
# =====================================================================

class VisionExtractor:
    """Extracts factual text from webpage screenshots using multimodal LLMs with dual-model fallback."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
        self.primary_provider = self.settings.vision_primary_provider
        self.primary_model = self.settings.vision_primary_model
        self.secondary_provider = self.settings.vision_secondary_provider
        self.secondary_model = self.settings.vision_secondary_model
        self.timeout = float(self.settings.vision_timeout_seconds)
        self.retry_count = int(self.settings.vision_retry_count)

    async def _call_model(self, model: str, b64_image: str, query: str) -> Optional[str]:
        """Send base64 image to OpenRouter with query-answering prompt and retry."""
        if not self.settings.openrouter_api_key:
            logger.warning("Vision extraction skipped: OPENROUTER_API_KEY is not configured")
            return None

        prompt = (
            f"You are the official institutional document extractor for VVITU / VVIT.\n"
            f"Extract only verified, factual information visible in this webpage screenshot that answers:\n"
            f"QUESTION: {query}\n\n"
            f"RULES:\n"
            f"1. Extract ONLY information explicitly visible in the image (e.g. names, designations, departments, qualifications, fees, dates, announcements).\n"
            f"2. Do NOT invent, assume, or extrapolate unverified details.\n"
            f"3. If the requested information is not visible in the image, output exactly: 'The requested information is not visible on this page.'"
        )

        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
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

        for attempt in range(self.retry_count + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(self.endpoint, headers=headers, json=payload)
                    if response.status_code != 200:
                        logger.warning("Vision model %s attempt %d returned %d: %s", model, attempt + 1, response.status_code, response.text[:200])
                        continue

                    data = response.json()
                    choices = data.get("choices") or []
                    if not choices:
                        continue

                    extracted = choices[0].get("message", {}).get("content", "").strip()
                    if not extracted or "not visible" in extracted.lower():
                        return None

                    return extracted
            except Exception as exc:
                logger.warning("Vision model %s attempt %d failed: %s", model, attempt + 1, exc)

        return None

    async def extract_from_image(self, image_bytes: bytes, query: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Send image bytes to vision model and extract facts answering query.
        Returns: (extracted_text, provider, model_used)
        """
        if not self.settings.vision_enabled or not self.settings.openrouter_api_key:
            return None, None, None

        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Try primary model
        primary_text = await self._call_model(self.primary_model, b64_image, query)
        if primary_text:
            logger.info("Vision extraction succeeded with primary model %s (%d chars)", self.primary_model, len(primary_text))
            return primary_text, self.primary_provider, self.primary_model

        # Fallback to secondary model if configured
        if self.secondary_model and self.secondary_model != self.primary_model:
            logger.info("Falling back to secondary vision model: %s", self.secondary_model)
            secondary_text = await self._call_model(self.secondary_model, b64_image, query)
            if secondary_text:
                logger.info("Vision extraction succeeded with secondary model %s (%d chars)", self.secondary_model, len(secondary_text))
                return secondary_text, self.secondary_provider, self.secondary_model

        return None, None, None

    async def extract_from_tiles(self, tiles_bytes: List[bytes], query: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Extract facts across multiple semantic tiles and merge/deduplicate lines.
        Returns: (merged_text, provider, model_used)
        """
        if not tiles_bytes:
            return None, None, None

        extracted_parts = []
        model_used = None
        provider_used = None

        max_tiles = min(len(tiles_bytes), self.settings.vision_max_tiles)
        for i, tile_bytes in enumerate(tiles_bytes[:max_tiles]):
            text, provider, model = await self.extract_from_image(tile_bytes, query)
            if text:
                extracted_parts.append(text)
                model_used = model
                provider_used = provider

        if not extracted_parts:
            return None, None, None

        # Deduplicate lines across tiles
        seen_lines = set()
        merged_lines = []
        for part in extracted_parts:
            for line in part.splitlines():
                norm = line.strip().lower()
                if norm and norm not in seen_lines:
                    seen_lines.add(norm)
                    merged_lines.append(line.strip())

        merged_text = "\n".join(merged_lines)
        return merged_text, provider_used, model_used


# =====================================================================
# BROWSER RETRIEVAL SERVICE
# =====================================================================

class BrowserRetrievalService:
    """
    Singleton headless browser service for JavaScript-rendered official page retrieval.
    Reuses browser context across requests with semaphore-controlled concurrency.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.vision_extractor = VisionExtractor(self.settings)
        self._playwright = None
        self._browser = None
        self._init_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(self.settings.max_concurrent_browser_pages)
        self._available = True

    async def _ensure_browser(self):
        """Initialize Playwright and launch headless browser if not already active."""
        if self._browser is not None and self._browser.is_connected():
            return self._browser

        async with self._init_lock:
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

    def _score_element(self, text: str, tag: str, class_name: str, query_terms: List[str]) -> int:
        """Query-aware element scoring heuristic (generic across any institutional domain)."""
        score = 0
        text_lower = text.lower()
        for term in query_terms:
            if term in text_lower:
                score += 3
        if any(h in tag.lower() for h in ["h1", "h2", "h3", "h4"]):
            score += 2
        if any(k in class_name.lower() for k in ["card", "grid", "row", "item", "profile", "content"]):
            score += 2
        if len(text.strip()) > 50:
            score += 1
        return score

    async def render_page(
        self,
        url: str,
        query: Optional[str] = None,
        wait_selector: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
    ) -> BrowserRenderResult:
        """
        Headlessly navigate to an official page, execute JavaScript,
        and extract the rendered DOM text. If rendered DOM text is insufficient,
        captures targeted semantic tiles/screenshots and uses vision OCR.
        """
        if timeout_seconds is None:
            timeout_seconds = self.settings.browser_timeout_seconds

        if not is_safe_official_url(url):
            return BrowserRenderResult(
                url=url,
                title="Untrusted URL",
                text="",
                method="failed",
                success=False,
                error="SSRF security violation: URL is not an approved official domain",
            )

        async with self._semaphore:
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

                # SSRF Protection: Intercept client-side redirects / navigations
                async def _intercept_route(route):
                    req = route.request
                    if req.is_navigation_request():
                        dest_url = req.url
                        if not is_safe_official_url(dest_url):
                            logger.warning("SSRF blocked: intercepted navigation to off-domain %s", dest_url)
                            await route.abort("blockedbyclient")
                            return
                    await route.continue_()

                await page.route("**/*", _intercept_route)

                try:
                    # Navigate with domcontentloaded
                    await page.goto(url, wait_until="domcontentloaded", timeout=timeout_seconds * 1000)

                    # Robust hydration readiness: wait for meaningful elements
                    meaningful_selectors = wait_selector or "a[href*='/faculty/'], table, [class*='grid'], main, #root:not(:empty)"
                    try:
                        await page.wait_for_selector(meaningful_selectors, timeout=5000)
                    except Exception:
                        pass

                    # Small tick for React state & rendering to settle
                    await asyncio.sleep(0.4)

                    title = await page.title()

                    entity_type = "general_page"
                    entities = []
                    clean_text = ""

                    # ── Strategy A: Faculty Card Grid Extraction ─────────────────
                    faculty_links = await page.query_selector_all("a[href*='/faculty/']")
                    if faculty_links and len(faculty_links) >= 2:
                        extracted_faculty = []
                        for a in faculty_links:
                            try:
                                href = (await a.get_attribute("href")) or ""
                                full_url = urljoin(url, href)
                                card_text = await a.inner_text()
                                parts = [p.strip() for p in card_text.splitlines() if p.strip()]
                                if len(parts) >= 2:
                                    fname = parts[0]
                                    fdesig = parts[1]
                                    fqual = parts[2] if len(parts) >= 3 else ""
                                    extracted_faculty.append((fname, fdesig, fqual, full_url))
                            except Exception:
                                pass

                        if extracted_faculty:
                            entity_type = "faculty_directory"
                            entities = [
                                {
                                    "name": fn,
                                    "designation": fd,
                                    "qualification": fq,
                                    "profile_url": fp,
                                }
                                for (fn, fd, fq, fp) in extracted_faculty
                            ]

                            # Search for department title
                            dept_title = ""
                            for h_sel in ["h1", "h2", "h3", "h4", "p"]:
                                try:
                                    h_elem = await page.query_selector(h_sel)
                                    if h_elem:
                                        htxt = (await h_elem.inner_text()).strip()
                                        if any(k in htxt.lower() for k in ["department", "school", "faculty"]):
                                            dept_title = htxt
                                            break
                                except Exception:
                                    pass

                            md_lines = [
                                f"**VVITU Official Faculty Directory{(' — ' + dept_title) if dept_title else ''}**",
                                f"• **Official URL:** {url}",
                                f"• **Total Verified Faculty Listed:** {len(extracted_faculty)}\n",
                                "| S.No | Faculty Name | Designation | Qualification | Official Profile Link |",
                                "|---|---|---|---|---|",
                            ]
                            for idx, (fname, fdesig, fqual, fprofile) in enumerate(extracted_faculty, 1):
                                md_lines.append(f"| {idx} | **{fname}** | {fdesig} | {fqual} | [Profile]({fprofile}) |")

                            clean_text = "\n".join(md_lines)

                    # ── Strategy B: HTML Table Extraction ─────────────────────────
                    if not clean_text:
                        tables = await page.query_selector_all("table")
                        if tables:
                            md_tables = []
                            for t in tables:
                                rows = await t.query_selector_all("tr")
                                if not rows:
                                    continue
                                grid = []
                                for r in rows:
                                    cells = await r.query_selector_all("th, td")
                                    row_vals = [ContentExtractor._normalise(await c.inner_text()) for c in cells]
                                    if any(row_vals):
                                        grid.append(row_vals)
                                if grid:
                                    header = grid[0]
                                    t_lines = ["| " + " | ".join(header) + " |"]
                                    t_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
                                    for row in grid[1:]:
                                        padded = row + [""] * (len(header) - len(row))
                                        t_lines.append("| " + " | ".join(padded[:len(header)]) + " |")
                                    md_tables.append("\n".join(t_lines))
                            if md_tables:
                                clean_text = "\n\n".join(md_tables)

                    # ── Strategy C: Standard Clean Rendered DOM ────────────────────
                    if not clean_text or len(clean_text) < 80:
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
                    is_dom_sufficient = len(clean_text) >= 120 or len(entities) > 0

                    if is_dom_sufficient:
                        logger.info("Rendered DOM extraction succeeded for %s (%d chars, %d entities)", url, len(clean_text), len(entities))
                        return BrowserRenderResult(
                            url=url,
                            title=title or "Official Portal",
                            text=clean_text,
                            method="headless_browser_dom",
                            success=True,
                            entity_type=entity_type,
                            entities=entities,
                            metadata={"chars": len(clean_text), "entities_count": len(entities)},
                            screenshot_used=False,
                            extraction_timestamp=datetime.now(timezone.utc).isoformat(),
                        )

                    # ── Tier 4: Query-Aware Semantic Screenshot + Vision Fallback ──
                    logger.info("DOM text insufficient (%d chars). Triggering query-aware screenshot vision for %s", len(clean_text), url)

                    # Extract query terms for generic scoring
                    query_terms = [t.lower() for t in re.findall(r"\w+", query or "") if len(t) > 3]

                    # Discover candidate regions and score them
                    candidate_selectors = [
                        wait_selector,
                        "main",
                        "article",
                        "[class*='grid']",
                        "[class*='list']",
                        ".container",
                        "#root",
                    ]
                    best_elem = None
                    best_score = -1
                    best_selector = None

                    for sel in candidate_selectors:
                        if not sel:
                            continue
                        try:
                            elems = await page.query_selector_all(sel)
                            for elem in elems:
                                tag_name = await elem.evaluate("el => el.tagName.toLowerCase()")
                                cls_name = await elem.evaluate("el => el.className || ''")
                                el_text = (await elem.inner_text()).strip()[:400]
                                score = self._score_element(el_text, tag_name, cls_name, query_terms)
                                if score > best_score:
                                    best_score = score
                                    best_elem = elem
                                    best_selector = sel
                        except Exception:
                            pass

                    # Fallback element if none scored high
                    if not best_elem:
                        try:
                            best_elem = await page.query_selector("#root") or await page.query_selector("body")
                            best_selector = "#root"
                        except Exception:
                            pass

                    screenshot_scope = "element"
                    vision_text = None
                    provider_used = None
                    model_used = None

                    if best_elem and query:
                        try:
                            bbox = await best_elem.bounding_box()
                            elem_height = bbox["height"] if bbox else 0

                            # Semantic tiling if element height exceeds max dimension (e.g. 1600px)
                            max_dim = self.settings.vision_max_image_dimension
                            if elem_height > max_dim:
                                logger.info("Target element %s height is %.1fpx (> %dpx). Capturing semantic tiles.", best_selector, elem_height, max_dim)
                                tiles_bytes = []
                                # Attempt child items tiling
                                child_cards = await best_elem.query_selector_all("> div, > article, > section, [class*='card'], tr")
                                if child_cards and len(child_cards) >= 2:
                                    screenshot_scope = "semantic_tiles"
                                    # Collect tiles for child card groups up to max_tiles
                                    max_tiles = min(len(child_cards), self.settings.vision_max_tiles)
                                    for card in child_cards[:max_tiles]:
                                        try:
                                            c_bytes = await card.screenshot(type="png")
                                            if c_bytes:
                                                tiles_bytes.append(c_bytes)
                                        except Exception:
                                            pass

                                if tiles_bytes:
                                    vision_text, provider_used, model_used = await self.vision_extractor.extract_from_tiles(tiles_bytes, query)
                                else:
                                    # Viewport clip tiles fallback
                                    screenshot_scope = "viewport_tiles"
                                    num_slices = min(int(elem_height // max_dim) + 1, self.settings.vision_max_tiles)
                                    for slice_idx in range(num_slices):
                                        try:
                                            y_offset = slice_idx * 800
                                            s_bytes = await page.screenshot(
                                                type="png",
                                                clip={"x": 0, "y": y_offset, "width": 1280, "height": 800},
                                            )
                                            if s_bytes:
                                                tiles_bytes.append(s_bytes)
                                        except Exception:
                                            pass
                                    if tiles_bytes:
                                        vision_text, provider_used, model_used = await self.vision_extractor.extract_from_tiles(tiles_bytes, query)
                            else:
                                # Element fits in single bounding box
                                screenshot_scope = "element"
                                s_bytes = await best_elem.screenshot(type="png")
                                if s_bytes:
                                    vision_text, provider_used, model_used = await self.vision_extractor.extract_from_image(s_bytes, query)

                        except Exception as exc:
                            logger.warning("Targeted element screenshot failed: %s. Falling back to viewport.", exc)

                    # Viewport screenshot fallback if element screenshot failed
                    if not vision_text and query:
                        try:
                            screenshot_scope = "viewport"
                            s_bytes = await page.screenshot(type="png", full_page=False)
                            if s_bytes:
                                vision_text, provider_used, model_used = await self.vision_extractor.extract_from_image(s_bytes, query)
                        except Exception as exc:
                            logger.warning("Viewport screenshot fallback failed: %s", exc)

                    if vision_text:
                        logger.info("Screenshot vision extraction succeeded for %s (%d chars, model=%s)", url, len(vision_text), model_used)
                        return BrowserRenderResult(
                            url=url,
                            title=title or "Official Portal",
                            text=vision_text,
                            method="headless_browser_screenshot",
                            success=True,
                            screenshot_used=True,
                            screenshot_scope=screenshot_scope,
                            vision_provider=provider_used,
                            vision_model=model_used,
                            screenshot_b64=None,  # Discard in-memory image buffer immediately to prevent heap bloat
                            extraction_timestamp=datetime.now(timezone.utc).isoformat(),
                            metadata={
                                "chars": len(vision_text),
                                "scope": screenshot_scope,
                                "model": model_used,
                                "provider": provider_used,
                            },
                        )

                    # If both DOM and vision failed to extract sufficient content,
                    # return direct_url_only so VAIT provides the direct clickable page link
                    return BrowserRenderResult(
                        url=url,
                        title=title or "Official Portal",
                        text=clean_text if len(clean_text) > 40 else "",
                        method="direct_url_only",
                        success=False,
                        screenshot_used=False,
                        extraction_timestamp=datetime.now(timezone.utc).isoformat(),
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
                    screenshot_used=False,
                    extraction_timestamp=datetime.now(timezone.utc).isoformat(),
                    error=str(exc),
                )

    async def close(self):
        """Cleanly terminate the browser and Playwright process."""
        async with self._init_lock:
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

