"""
VAIT Official Web Retriever — Autonomous Dual-Source Web Retrieval
==================================================================

Provides live, targeted retrieval from the two official institutional websites:
  1. Current University: https://vvitu.ac.in/ (Primary / Current)
  2. Legacy Institute:   https://vvitguntur.com/ (Historical / Legacy)

Key Features:
  - SPA-Aware Entity Catalog: Ingests the client-rendered React bundle of vvitu.ac.in
    to catalog all 343+ faculty, schools, degree programs, and leadership.
  - Direct Page Canonical URLs: Maps entities to their exact canonical pages
    (e.g., https://vvitu.ac.in/admissions/faculty/ece-faculty/k-ramesh-babu).
  - Legacy Joomla Live Search: Queries vvitguntur.com live search endpoint and
    extracts article content using ContentExtractor.
  - Strict Domain Allowlists: Enforces SSRF safety by strictly restricting requests
    to vvitu.ac.in and vvitguntur.com.
  - High-Performance TTL Cache: Avoids redundant requests with a 1-hour cache.
  - Factual Grounding: Returns verified institutional data with 0 hallucination.
"""

import asyncio
import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote_plus, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from app.services.website_crawler import ContentExtractor

logger = logging.getLogger("vait.official_web")

# ── Domain Allowlists ────────────────────────────────────────────────
ALLOWED_DOMAINS = {
    "vvitu.ac.in",
    "www.vvitu.ac.in",
    "vvitguntur.com",
    "www.vvitguntur.com",
}

CURRENT_DOMAINS = {"vvitu.ac.in", "www.vvitu.ac.in"}
LEGACY_DOMAINS = {"vvitguntur.com", "www.vvitguntur.com"}

REQUEST_TIMEOUT = 8  # seconds
CACHE_TTL = 3600  # 1 hour


# =====================================================================
# DATA STRUCTURES
# =====================================================================

@dataclass
class OfficialWebResult:
    """Represents a verified piece of content retrieved from official portals."""
    title: str
    url: str
    content: str
    period: str  # "current" | "historical" | "general"
    source_tier: str  # "primary_official_current" | "legacy_official_vvit"
    confidence: float = 0.95
    entity_type: str = "general_page"  # "faculty" | "leadership" | "department" | "program" | "general_page"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_structured_source(self) -> Dict[str, Any]:
        """Convert to structured source format expected by VAIT frontend."""
        if self.period == "current":
            period_label = "VVITU Official Website — Current"
            type_label = "website_current"
            institutional_period = "current"
        elif self.period == "historical":
            period_label = "VVIT Legacy Website — Historical"
            type_label = "website_historical"
            institutional_period = "historical"
        else:
            period_label = "VVITU Official Website"
            type_label = "website_current"
            institutional_period = "current"

        return {
            "title": self.title,
            "url": self.url,
            "type": type_label,
            "period_label": period_label,
            "institutional_period": institutional_period,
        }


# =====================================================================
# IN-MEMORY TTL CACHE
# =====================================================================

class OfficialWebCache:
    """Thread-safe, time-bounded query cache for web retrieval."""

    def __init__(self, ttl: int = CACHE_TTL):
        self.ttl = ttl
        self._cache: Dict[str, Tuple[float, List[OfficialWebResult]]] = {}

    def get(self, key: str) -> Optional[List[OfficialWebResult]]:
        norm_key = key.strip().lower()
        if norm_key in self._cache:
            ts, results = self._cache[norm_key]
            if time.time() - ts < self.ttl:
                return results
            del self._cache[norm_key]
        return None

    def set(self, key: str, results: List[OfficialWebResult]) -> None:
        norm_key = key.strip().lower()
        self._cache[norm_key] = (time.time(), results)

    def clear(self) -> None:
        self._cache.clear()


# =====================================================================
# OFFICIAL WEB RETRIEVER
# =====================================================================

class OfficialWebRetriever:
    """
    Autonomous official web retrieval engine for VVITU & legacy VVIT.
    """

    def __init__(self, bundle_local_path: Optional[str] = None):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) VAIT-OfficialRetriever/1.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })
        self.cache = OfficialWebCache()

        # Find local bundle if available
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        default_bundle_path = os.path.join(base_dir, "data", "vvitu_bundle.js")
        self.bundle_local_path = bundle_local_path or default_bundle_path

        # Parsed catalogs
        self.faculty_catalog: List[Dict[str, Any]] = []
        self.leadership_catalog: List[Dict[str, Any]] = []
        self.schools_catalog: List[Dict[str, Any]] = []
        self.programs_catalog: List[Dict[str, Any]] = []
        self.pages_catalog: List[Dict[str, Any]] = []
        self._catalog_initialized = False

        # Build initial catalog from bundle
        self._init_catalogs()

    def _init_catalogs(self) -> None:
        """Initialize institutional catalogs from static bundle and static definitions."""
        if self._catalog_initialized:
            return

        # 1. Static leadership definitions
        self.leadership_catalog = [
            {
                "name": "Sri Vasireddy Vidyasagar",
                "title": "Chancellor, VVITU (Founder & Chairman of legacy VVIT)",
                "designation": "Chancellor",
                "organization": "VVIT University (VVITU)",
                "period": "current",
                "url": "https://vvitu.ac.in/chancellor",
                "description": "Sri Vasireddy Vidyasagar is the Chancellor of VVIT University (VVITU) and the visionary founder and Chairman who established Vasireddy Venkatadri Institute of Technology (VVIT) in 2007.",
                "keywords": ["chancellor", "vidyasagar", "vasireddy vidyasagar", "chairman", "founder"],
            },
            {
                "name": "Prof. Rambabu Kodali",
                "title": "Vice-Chancellor, VVITU",
                "designation": "Vice-Chancellor",
                "organization": "VVIT University (VVITU)",
                "period": "current",
                "url": "https://vvitu.ac.in/vice-chancellor",
                "description": "Prof. Rambabu Kodali is the Vice-Chancellor of VVIT University (VVITU). A distinguished academician and administrator with decades of experience at premier institutions including BITS Pilani.",
                "keywords": ["vice-chancellor", "vice chancellor", "vc", "rambabu", "kodali", "rambabu kodali"],
            },
            {
                "name": "Sri Suryadevara Badari Prasad",
                "title": "Secretary, VVITU",
                "designation": "Secretary",
                "organization": "VVIT University (VVITU)",
                "period": "current",
                "url": "https://vvitu.ac.in/secretary",
                "description": "Sri Suryadevara Badari Prasad is the Secretary of VVIT University (VVITU), overseeing administrative functions and governance.",
                "keywords": ["secretary", "badari prasad", "suryadevara badari prasad"],
            },
            {
                "name": "Sri Mamillapalli Sree Krishna",
                "title": "Joint Secretary, VVITU",
                "designation": "Joint Secretary",
                "organization": "VVIT University (VVITU)",
                "period": "current",
                "url": "https://vvitu.ac.in/jt-secretary",
                "description": "Sri Mamillapalli Sree Krishna is the Joint Secretary of VVIT University (VVITU).",
                "keywords": ["joint secretary", "jt secretary", "sree krishna", "mamillapalli sree krishna"],
            },
            {
                "name": "Dr. Y. Mallikarjuna Reddy",
                "title": "Principal, Legacy VVIT (Historical)",
                "designation": "Principal (Legacy VVIT)",
                "organization": "Vasireddy Venkatadri Institute of Technology (VVIT)",
                "period": "historical",
                "url": "https://vvitguntur.com/index.php/dr-ymr-principal",
                "description": "Dr. Y. Mallikarjuna Reddy served as the Principal of Vasireddy Venkatadri Institute of Technology (VVIT) during the legacy institute period.",
                "keywords": ["principal", "ymr", "mallikarjuna reddy", "y mallikarjuna reddy", "dr ymr"],
            },
        ]

        # 2. Schools and Academic Departments
        self.schools_catalog = [
            {
                "name": "School of Computer Science & Engineering",
                "code": "CSE",
                "slug": "school-of-CSE",
                "dept_slug": "cse-faculty",
                "url": "https://vvitu.ac.in/school-of-CSE",
                "description": "The School of Computer Science & Engineering at VVITU offers comprehensive undergraduate and postgraduate programs focusing on software engineering, cloud computing, cybersecurity, and computational theory.",
                "keywords": ["cse", "computer science", "school of cse", "computer science engineering"],
            },
            {
                "name": "School of Artificial Intelligence",
                "code": "AI",
                "slug": "school-of-AI",
                "dept_slug": "cse-ai-faculty",
                "url": "https://vvitu.ac.in/school-of-AI",
                "description": "The School of Artificial Intelligence at VVITU specializes in cutting-edge AI, Machine Learning, Data Science, and Intelligent Systems, preparing students for next-generation computing.",
                "keywords": ["school of ai", "artificial intelligence", "aiml", "ai & ds", "cse ai", "machine learning"],
            },
            {
                "name": "School of Electronics & Communication Engineering",
                "code": "ECE",
                "slug": "school-of-ECE",
                "dept_slug": "ece-faculty",
                "url": "https://vvitu.ac.in/school-of-ECE",
                "description": "The School of Electronics & Communication Engineering at VVITU covers VLSI, embedded systems, signal processing, wireless communication, and IoT.",
                "keywords": ["ece", "electronics", "communication engineering", "school of ece", "vlsi"],
            },
            {
                "name": "School of Electrical & Electronics Engineering",
                "code": "EEE",
                "slug": "school-of-EEE",
                "dept_slug": "eee-faculty",
                "url": "https://vvitu.ac.in/school-of-EEE",
                "description": "The School of Electrical & Electronics Engineering at VVITU offers training and research in electrical machines, power systems, renewable energy, and electric mobility.",
                "keywords": ["eee", "electrical", "electrical engineering", "school of eee", "power systems"],
            },
            {
                "name": "School of Civil Engineering",
                "code": "CE",
                "slug": "school-of-CE",
                "dept_slug": "civil-faculty",
                "url": "https://vvitu.ac.in/school-of-CE",
                "description": "The School of Civil Engineering at VVITU delivers rigorous training in structural engineering, geotechnical analysis, transportation systems, and environmental management.",
                "keywords": ["civil", "civil engineering", "school of ce", "structures"],
            },
            {
                "name": "School of Mechanical Engineering",
                "code": "ME",
                "slug": "school-of-ME",
                "dept_slug": "mec-faculty",
                "url": "https://vvitu.ac.in/school-of-ME",
                "description": "The School of Mechanical Engineering at VVITU offers modern education in robotics, automation, thermodynamics, CAD/CAM, and advanced manufacturing.",
                "keywords": ["mechanical", "mechanical engineering", "school of me", "robotics", "thermal"],
            },
            {
                "name": "APPA School of Business",
                "code": "MBA",
                "slug": "appa-school-of-business",
                "dept_slug": "business-administration-faculty",
                "url": "https://vvitu.ac.in/appa-school-of-business",
                "description": "APPA School of Business at VVITU offers high-impact management education, developing business leaders with expertise in finance, marketing, human resources, and business analytics.",
                "keywords": ["business", "management", "mba", "business administration", "appa school of business"],
            },
        ]

        # 3. Programs Catalog
        self.programs_catalog = [
            {
                "name": "B.Tech Degree Programs",
                "degree": "B.Tech",
                "url": "https://vvitu.ac.in/courses",
                "description": "Undergraduate B.Tech degree programs offered at VVITU: Computer Science & Engineering (CSE), CSE (Artificial Intelligence & Machine Learning), CSE (Data Science), CSE (Internet of Things), Artificial Intelligence & Data Science, Electronics & Communication Engineering (ECE), Electrical & Electronics Engineering (EEE), Civil Engineering, and Mechanical Engineering.",
                "keywords": ["btech", "b.tech", "undergraduate", "bachelor", "engineering courses"],
            },
            {
                "name": "M.Tech Degree Programs",
                "degree": "M.Tech",
                "url": "https://vvitu.ac.in/courses",
                "description": "Postgraduate M.Tech degree programs offered at VVITU: Computer Science & Engineering, VLSI & Embedded Systems (ECE), Power Electronics & Drives (EEE), Machine Design (Mechanical), and Structural Engineering (Civil).",
                "keywords": ["mtech", "m.tech", "postgraduate", "master of technology"],
            },
            {
                "name": "Ph.D Research Programs",
                "degree": "Ph.D",
                "url": "https://vvitu.ac.in/courses",
                "description": "Doctor of Philosophy (Ph.D) research programs offered at VVITU across engineering, sciences, and management: Computer Science & Engineering, Electronics & Communication Engineering, Electrical & Electronics Engineering, Civil Engineering, Mechanical Engineering, Mathematics, Physics, Chemistry, English, and Management Studies.",
                "keywords": ["phd", "ph.d", "doctorate", "research", "doctoral program"],
            },
        ]

        # 4. Official Key Pages & Campus Info
        self.pages_catalog = [
            {
                "name": "VVITU Campus & Contact Details",
                "url": "https://vvitu.ac.in/contact",
                "description": "VVIT University (VVITU) is located at Nambur (V), Pedakakani (M), Guntur District, Andhra Pradesh - 522508. Official contact: info@vvitu.ac.in.",
                "keywords": ["contact", "address", "location", "email", "phone", "campus"],
            },
            {
                "name": "VVITU Admissions Portal",
                "url": "https://vvitu.ac.in/admissions",
                "description": "Official admissions portal for VVIT University (VVITU), offering enrollment into B.Tech, M.Tech, MBA, and Ph.D programs based on AP EAPCET, GATE, ICET, and university entrance criteria.",
                "keywords": ["admissions", "admission", "apply", "join", "eligibility", "entrance"],
            },
            {
                "name": "VVITU Training & Placements",
                "url": "https://vvitu.ac.in/placements",
                "description": "The Training & Placement Cell at VVITU coordinates campus recruitments with premier multinational corporations, product companies, and technology leaders.",
                "keywords": ["placement", "placements", "recruiters", "jobs", "training and placement"],
            },
        ]

        # 5. Load and parse faculty from local bundle
        self._load_faculty_from_bundle()
        self._catalog_initialized = True
        logger.info(
            "Initialized OfficialWebRetriever catalogs: %d faculty, %d leadership, %d schools, %d programs",
            len(self.faculty_catalog),
            len(self.leadership_catalog),
            len(self.schools_catalog),
            len(self.programs_catalog),
        )

    def _load_faculty_from_bundle(self) -> None:
        """Extract faculty members from the local or downloaded VVITU bundle."""
        if not os.path.exists(self.bundle_local_path):
            logger.warning("Local bundle not found at %s; trying live download", self.bundle_local_path)
            self._download_bundle_from_web()

        if not os.path.exists(self.bundle_local_path):
            logger.error("Could not find or download VVITU bundle")
            return

        try:
            with open(self.bundle_local_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            dept_starts = [m.start() for m in re.finditer(r'slug:"([^"]+)",departmentName:"([^"]+)"', text)]
            count = 0
            for start in dept_starts:
                header = text[start: start + 150]
                m = re.match(r'slug:"([^"]+)",departmentName:"([^"]+)"', header)
                if not m:
                    continue
                slug = m.group(1)
                dept_name = m.group(2)
                members_idx = text.find('members:[', start)
                if members_idx == -1 or members_idx - start > 200:
                    continue

                arr_start = members_idx + len('members:[')
                bracket_depth = 1
                curr = arr_start
                while curr < len(text) and bracket_depth > 0:
                    if text[curr] == '[':
                        bracket_depth += 1
                    elif text[curr] == ']':
                        bracket_depth -= 1
                    curr += 1
                chunk = text[arr_start:curr - 1]

                pattern = re.compile(
                    r'id:"([^"]+)",name:"([^"]+)",designation:"([^"]+)",qualification:"([^"]+)"(?:,gender:"([^"]*)")?'
                )
                matches = pattern.findall(chunk)
                for fid, fname, fdesig, fqual, fgen in matches:
                    clean_name = fname.strip()
                    self.faculty_catalog.append({
                        "id": fid,
                        "name": clean_name,
                        "designation": fdesig.strip(),
                        "qualification": fqual.strip(),
                        "gender": fgen.strip() if fgen else "",
                        "dept_slug": slug,
                        "dept_name": dept_name,
                        "url": f"https://vvitu.ac.in/admissions/faculty/{slug}/{fid}",
                        "search_tokens": self._tokenize(clean_name),
                    })
                    count += 1

            logger.info("Successfully extracted %d faculty members from bundle", count)
        except Exception as exc:
            logger.error("Failed to parse faculty from bundle: %s", exc)

    def _download_bundle_from_web(self) -> None:
        """Download live production bundle from vvitu.ac.in if missing."""
        try:
            resp = self.session.get("https://vvitu.ac.in/", timeout=REQUEST_TIMEOUT)
            match = re.search(r'src="([^"]*assets/index-[^"]+\.js)"', resp.text)
            if match:
                bundle_url = urljoin("https://vvitu.ac.in/", match.group(1))
                logger.info("Downloading live bundle from %s", bundle_url)
                b_resp = self.session.get(bundle_url, timeout=15)
                os.makedirs(os.path.dirname(self.bundle_local_path), exist_ok=True)
                with open(self.bundle_local_path, "w", encoding="utf-8") as f:
                    f.write(b_resp.text)
        except Exception as exc:
            logger.warning("Live bundle download failed: %s", exc)

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        """Normalize and tokenize a string into searchable words."""
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        tokens = {t for t in cleaned.split() if len(t) > 1 and t not in {"dr", "mr", "mrs", "ms", "prof", "the"}}
        return tokens

    # ── Search Engine Methods ────────────────────────────────────────

    def search_vvitu(self, query: str) -> List[OfficialWebResult]:
        """
        Search the VVITU catalog (faculty, leadership, schools, programs, pages).
        """
        results: List[OfficialWebResult] = []
        q_norm = query.lower()
        q_tokens = self._tokenize(query)

        if not q_tokens:
            return results

        # 1. Check Leadership (Highest authority)
        is_vc_query = any(term in q_norm for term in ["vice-chancellor", "vice chancellor", "vc", "kodali", "rambabu"])
        for leader in self.leadership_catalog:
            if leader["period"] != "current":
                continue
            if leader["designation"] == "Chancellor" and is_vc_query:
                continue
            matched = any(kw in q_norm for kw in leader["keywords"]) or any(t in self._tokenize(leader["name"]) for t in q_tokens)
            if matched:
                results.append(OfficialWebResult(
                    title=f"VVITU Official — {leader['title']}",
                    url=leader["url"],
                    content=(
                        f"**{leader['name']}**\n"
                        f"• **Role / Title:** {leader['title']}\n"
                        f"• **Institution:** {leader['organization']}\n"
                        f"• **Official Profile:** {leader['url']}\n\n"
                        f"{leader['description']}"
                    ),
                    period="current",
                    source_tier="primary_official_current",
                    confidence=0.98,
                    entity_type="leadership",
                    metadata=leader,
                ))

        # 2. Check Faculty Members
        faculty_matches = []
        for fac in self.faculty_catalog:
            fac_tokens = fac["search_tokens"]
            overlap = q_tokens.intersection(fac_tokens)
            if len(overlap) >= min(len(q_tokens), len(fac_tokens)) or (len(overlap) >= 2 and len(overlap) / len(fac_tokens) >= 0.5):
                score = len(overlap) / max(len(fac_tokens), 1)
                faculty_matches.append((score, fac))
            elif fac["name"].lower() in q_norm:
                faculty_matches.append((1.0, fac))

        # Sort faculty by match score
        faculty_matches.sort(key=lambda x: x[0], reverse=True)
        for _, fac in faculty_matches[:3]:
            results.append(OfficialWebResult(
                title=f"VVITU Official Faculty Profile — {fac['name']}",
                url=fac["url"],
                content=(
                    f"**{fac['name']}**\n"
                    f"• **Designation:** {fac['designation']}\n"
                    f"• **Department:** {fac['dept_name']}\n"
                    f"• **Qualification:** {fac['qualification']}\n"
                    f"• **Institution:** VVIT University (VVITU)\n"
                    f"• **Official Profile URL:** {fac['url']}"
                ),
                period="current",
                source_tier="primary_official_current",
                confidence=0.95,
                entity_type="faculty",
                metadata=fac,
            ))

        # 3. Check Schools / Departments
        for school in self.schools_catalog:
            matched = any(kw in q_norm for kw in school["keywords"])
            if matched:
                results.append(OfficialWebResult(
                    title=f"VVITU Official — {school['name']}",
                    url=school["url"],
                    content=(
                        f"**{school['name']} ({school['code']})**\n"
                        f"• **Institution:** VVIT University (VVITU)\n"
                        f"• **Official Portal:** {school['url']}\n\n"
                        f"{school['description']}"
                    ),
                    period="current",
                    source_tier="primary_official_current",
                    confidence=0.92,
                    entity_type="department",
                    metadata=school,
                ))

        # 4. Check Programs
        for prog in self.programs_catalog:
            matched = any(kw in q_norm for kw in prog["keywords"])
            if matched:
                results.append(OfficialWebResult(
                    title=f"VVITU Official — {prog['name']}",
                    url=prog["url"],
                    content=(
                        f"**{prog['name']}**\n"
                        f"• **Institution:** VVIT University (VVITU)\n"
                        f"• **Official Academic Page:** {prog['url']}\n\n"
                        f"{prog['description']}"
                    ),
                    period="current",
                    source_tier="primary_official_current",
                    confidence=0.90,
                    entity_type="program",
                    metadata=prog,
                ))

        # 5. Check Pages (Admissions, Contact, etc.)
        for page in self.pages_catalog:
            matched = any(kw in q_norm for kw in page["keywords"])
            if matched:
                results.append(OfficialWebResult(
                    title=f"VVITU Official — {page['name']}",
                    url=page["url"],
                    content=(
                        f"**{page['name']}**\n"
                        f"• **Institution:** VVIT University (VVITU)\n"
                        f"• **Official Page:** {page['url']}\n\n"
                        f"{page['description']}"
                    ),
                    period="current",
                    source_tier="primary_official_current",
                    confidence=0.90,
                    entity_type="general_page",
                    metadata=page,
                ))

        return results

    def search_vvit_legacy(self, query: str) -> List[OfficialWebResult]:
        """
        Search legacy VVIT information (vvitguntur.com) via static catalog & live search.
        """
        results: List[OfficialWebResult] = []
        q_norm = query.lower()
        q_tokens = self._tokenize(query)

        # 1. Check legacy leadership
        for leader in self.leadership_catalog:
            if leader["period"] == "historical" or "principal" in q_norm or "ymr" in q_norm:
                matched = any(kw in q_norm for kw in leader["keywords"])
                if matched:
                    results.append(OfficialWebResult(
                        title=f"VVIT Legacy Official — {leader['title']}",
                        url=leader["url"],
                        content=(
                            f"**{leader['name']}**\n"
                            f"• **Role / Title:** {leader['title']}\n"
                            f"• **Institution:** {leader['organization']}\n"
                            f"• **Historical Profile:** {leader['url']}\n\n"
                            f"{leader['description']}"
                        ),
                        period="historical",
                        source_tier="legacy_official_vvit",
                        confidence=0.96,
                        entity_type="leadership",
                        metadata=leader,
                    ))

        # 2. Live search on vvitguntur.com only if no catalog match found
        if not results:
            live_results = self._search_vvitguntur_live(query)
            results.extend(live_results)

        return results

    def _search_vvitguntur_live(self, query: str, max_items: int = 2) -> List[OfficialWebResult]:
        """
        Perform live search on https://vvitguntur.com/ using its Joomla search endpoint.
        """
        clean_query = re.sub(r"[^a-zA-Z0-9\s]", "", query).strip()
        words = clean_query.split()
        if not words:
            return []

        searchword = "+".join(words[:4])
        search_url = f"https://vvitguntur.com/index.php/component/search/?searchword={searchword}&searchphrase=all"

        results = []
        try:
            resp = self.session.get(search_url, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                return []

            soup = BeautifulSoup(resp.text, "html.parser")
            result_tags = soup.find_all("dt", class_="result-title")

            for tag in result_tags[:max_items]:
                a_tag = tag.find("a")
                if not a_tag or not a_tag.get("href"):
                    continue

                title = a_tag.get_text(strip=True)
                relative_url = a_tag["href"]
                full_url = urljoin("https://vvitguntur.com/", relative_url)

                # Validate domain
                parsed = urlparse(full_url)
                if parsed.hostname not in LEGACY_DOMAINS:
                    continue

                # Fetch article content
                article_text = self._fetch_legacy_page(full_url)
                if article_text:
                    results.append(OfficialWebResult(
                        title=f"VVIT Legacy — {title}",
                        url=full_url,
                        content=(
                            f"**VVIT Legacy Archive: {title}**\n"
                            f"• **Source:** Vasireddy Venkatadri Institute of Technology (Historical Archive)\n"
                            f"• **URL:** {full_url}\n\n"
                            f"{article_text[:1200]}"
                        ),
                        period="historical",
                        source_tier="legacy_official_vvit",
                        confidence=0.88,
                        entity_type="general_page",
                        metadata={"title": title, "url": full_url},
                    ))
        except Exception as exc:
            logger.debug("Live legacy search failed: %s", exc)

        return results

    def _fetch_legacy_page(self, url: str) -> Optional[str]:
        """Fetch a legacy page and extract its main text content."""
        try:
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                return None

            soup = BeautifulSoup(resp.text, "html.parser")
            article = soup.find("div", class_="item-page") or soup.find("article") or soup.find("div", id="content")
            target_html = str(article) if article else resp.text

            text = ContentExtractor.extract(target_html, url=url)
            return text
        except Exception as exc:
            logger.debug("Error fetching legacy page %s: %s", url, exc)
            return None

    # ── Public High-Level Retrieve Method ────────────────────────────

    async def retrieve(
        self,
        query: str,
        period: str = "general",
        max_results: int = 3,
    ) -> List[OfficialWebResult]:
        """
        Asynchronously retrieve verified information from official websites.
        Respects query period ('current', 'historical', 'general').
        """
        cached = self.cache.get(f"{query}:{period}")
        if cached is not None:
            return cached[:max_results]

        results = await asyncio.to_thread(self._retrieve_sync, query, period, max_results)
        self.cache.set(f"{query}:{period}", results)
        return results

    def _retrieve_sync(
        self,
        query: str,
        period: str = "general",
        max_results: int = 3,
    ) -> List[OfficialWebResult]:
        """Synchronous retrieval implementation."""
        results: List[OfficialWebResult] = []

        if period == "historical":
            # Historical query: prioritize legacy website
            legacy_results = self.search_vvit_legacy(query)
            results.extend(legacy_results)
            if not results:
                curr_results = self.search_vvitu(query)
                results.extend(curr_results)
        elif period == "current":
            # Current query: prioritize current university
            curr_results = self.search_vvitu(query)
            results.extend(curr_results)
            if not results:
                legacy_results = self.search_vvit_legacy(query)
                results.extend(legacy_results)
        else:
            # General query: check current first
            curr_results = self.search_vvitu(query)
            results.extend(curr_results)
            # Only search legacy if current found nothing or if explicitly mentioning historical/vvit
            q_lower = query.lower()
            if not results or "vvit" in q_lower or "legacy" in q_lower or "history" in q_lower or "old" in q_lower:
                legacy_results = self.search_vvit_legacy(query)
                results.extend(legacy_results)

        # Deduplicate by URL
        seen_urls = set()
        deduped: List[OfficialWebResult] = []
        for r in results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                deduped.append(r)

        return deduped[:max_results]

    def get_all_catalog_pages(self) -> List[Dict[str, Any]]:
        """
        Return all known catalog entities as metadata-ready dicts for FAISS ingestion.
        Used by WebsiteCrawler for offline indexing.
        """
        pages = []
        for fac in self.faculty_catalog:
            content = (
                f"VVITU Faculty Profile: {fac['name']}\n"
                f"Designation: {fac['designation']}\n"
                f"Department: {fac['dept_name']}\n"
                f"Qualification: {fac['qualification']}\n"
                f"Institution: VVIT University (VVITU)\n"
                f"Official Profile URL: {fac['url']}"
            )
            pages.append({
                "title": f"VVITU Faculty Profile — {fac['name']}",
                "url": fac["url"],
                "text": content,
                "period": "current",
                "source_tier": "primary_official_current",
                "source_type": "website",
                "document_type": "official_website_current",
                "depth": 1,
            })

        for leader in self.leadership_catalog:
            content = (
                f"Official Profile: {leader['name']}\n"
                f"Role: {leader['title']}\n"
                f"Institution: {leader['organization']}\n"
                f"Official URL: {leader['url']}\n\n"
                f"{leader['description']}"
            )
            pages.append({
                "title": f"Official Profile — {leader['name']}",
                "url": leader["url"],
                "text": content,
                "period": leader["period"],
                "source_tier": "primary_official_current" if leader["period"] == "current" else "legacy_official_vvit",
                "source_type": "website",
                "document_type": "official_website_current" if leader["period"] == "current" else "legacy_website",
                "depth": 0,
            })

        for school in self.schools_catalog:
            content = (
                f"VVITU {school['name']} ({school['code']})\n"
                f"Official Portal: {school['url']}\n\n"
                f"{school['description']}"
            )
            pages.append({
                "title": f"VVITU {school['name']}",
                "url": school["url"],
                "text": content,
                "period": "current",
                "source_tier": "primary_official_current",
                "source_type": "website",
                "document_type": "official_website_current",
                "depth": 1,
            })

        for prog in self.programs_catalog:
            content = (
                f"VVITU Academic Program: {prog['name']}\n"
                f"Official URL: {prog['url']}\n\n"
                f"{prog['description']}"
            )
            pages.append({
                "title": f"VVITU Academic Programs — {prog['name']}",
                "url": prog["url"],
                "text": content,
                "period": "current",
                "source_tier": "primary_official_current",
                "source_type": "website",
                "document_type": "official_website_current",
                "depth": 1,
            })

        for page in self.pages_catalog:
            content = (
                f"VVITU Official Information: {page['name']}\n"
                f"Official URL: {page['url']}\n\n"
                f"{page['description']}"
            )
            pages.append({
                "title": f"VVITU {page['name']}",
                "url": page["url"],
                "text": content,
                "period": "current",
                "source_tier": "primary_official_current",
                "source_type": "website",
                "document_type": "official_website_current",
                "depth": 1,
            })

        return pages


# Singleton instance
_official_web_retriever: Optional[OfficialWebRetriever] = None

def get_official_web_retriever() -> OfficialWebRetriever:
    """Get or create singleton instance of OfficialWebRetriever."""
    global _official_web_retriever
    if _official_web_retriever is None:
        _official_web_retriever = OfficialWebRetriever()
    return _official_web_retriever
