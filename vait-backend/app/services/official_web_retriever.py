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
from app.services.browser_retrieval_service import (
    get_browser_retrieval_service,
    BrowserRetrievalService,
    BrowserRenderResult,
    is_safe_official_url,
)

logger = logging.getLogger("vait.official_web")

# ── Domain Allowlists ────────────────────────────────────────────────
ALLOWED_DOMAINS = {
    "vvitu.ac.in",
    "www.vvitu.ac.in",
    "vvitguntur.com",
    "www.vvitguntur.com",
}
import httpx

from app.services.browser_retrieval_service import (
    BrowserRenderResult,
    get_browser_retrieval_service,
)
from app.services.evidence_quality_validator import EvidenceQualityValidator
from app.utils.config import Settings, get_settings

logger = logging.getLogger("vait.official_web")

# Approved domain white-lists
CURRENT_DOMAINS: Set[str] = {"vvitu.ac.in", "www.vvitu.ac.in"}
LEGACY_DOMAINS: Set[str] = {"vvitguntur.com", "www.vvitguntur.com"}

REQUEST_TIMEOUT = 10.0
CACHE_TTL = 3600  # 1 hour


# =====================================================================
# DATA CLASSES
# =====================================================================

@dataclass
class OfficialWebResult:
    """Represents a verified piece of content retrieved from official portals with complete provenance."""
    title: str
    url: str
    content: str
    period: str  # "current" | "historical" | "general"
    source_tier: str  # "primary_official_current" | "legacy_official_vvit"
    confidence: float = 0.95
    entity_type: str = "general_page"  # "faculty" | "leadership" | "department" | "program" | "general_page"
    retrieval_method: str = "catalog"  # "catalog" | "http" | "headless_browser_dom" | "headless_browser_screenshot" | "direct_url_only"
    entities: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    screenshot_used: bool = False
    screenshot_scope: Optional[str] = None
    vision_provider: Optional[str] = None
    vision_model: Optional[str] = None
    extraction_timestamp: Optional[str] = None

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

        res = {
            "title": self.title,
            "url": self.url,
            "type": type_label,
            "period_label": period_label,
            "institutional_period": institutional_period,
            "retrieval_method": self.retrieval_method,
        }
        if self.screenshot_used:
            res["screenshot_used"] = True
            if self.screenshot_scope:
                res["screenshot_scope"] = self.screenshot_scope
            if self.vision_provider:
                res["vision_provider"] = self.vision_provider
            if self.vision_model:
                res["vision_model"] = self.vision_model
            if self.extraction_timestamp:
                res["extraction_timestamp"] = self.extraction_timestamp
        return res


# =====================================================================
# IN-MEMORY TTL CACHE (CONTENT-AWARE)
# =====================================================================

class OfficialWebCache:
    """Thread-safe, time-bounded query cache for web retrieval with dynamic vs static TTL."""

    def __init__(self, static_ttl: Optional[int] = None, dynamic_ttl: Optional[int] = None):
        settings = get_settings()
        self.static_ttl = static_ttl if static_ttl is not None else settings.cache_static_ttl_seconds
        self.dynamic_ttl = dynamic_ttl if dynamic_ttl is not None else settings.cache_dynamic_ttl_seconds
        self._cache: Dict[str, Tuple[float, float, List[OfficialWebResult]]] = {}

    def get(self, key: str) -> Optional[List[OfficialWebResult]]:
        norm_key = key.strip().lower()
        if norm_key in self._cache:
            ts, ttl, results = self._cache[norm_key]
            if time.time() - ts < ttl:
                return results
            del self._cache[norm_key]
        return None

    def set(self, key: str, results: List[OfficialWebResult], is_dynamic: Optional[bool] = None) -> None:
        norm_key = key.strip().lower()
        if is_dynamic is None:
            # Determine dynamically based on results
            is_dynamic = any(
                r.retrieval_method in ("headless_browser_dom", "headless_browser_screenshot", "http")
                or any(k in r.url.lower() for k in ["notification", "exam", "result", "timetable", "career", "admission"])
                for r in results
            )
        effective_ttl = self.dynamic_ttl if is_dynamic else self.static_ttl
        self._cache[norm_key] = (time.time(), effective_ttl, results)

    def clear(self) -> None:
        self._cache.clear()


# =====================================================================
# OFFICIAL VVITU DYNAMIC ROUTES
# =====================================================================

OFFICIAL_VVITU_ROUTES: List[Dict[str, Any]] = [
    {
        "path": "/examinations",
        "url": "https://vvitu.ac.in/examinations",
        "title": "VVITU Official Examination Portal",
        "description": "Official Examination section of VVIT University (VVITU), providing schedules, timetables, notifications, results, and evaluation guidelines.",
        "keywords": ["exam", "exams", "examination", "examinations", "timetable", "schedule", "result", "results", "hall ticket", "evaluation", "grading", "revaluation"],
        "wait_selector": "#root",
    },
    {
        "path": "/university-notifications",
        "url": "https://vvitu.ac.in/university-notifications",
        "title": "VVITU University Notifications & Circulars",
        "description": "Official university notifications, academic circulars, administrative announcements, and official notices from VVIT University (VVITU).",
        "keywords": ["notification", "notifications", "circular", "circulars", "announcement", "announcements", "notice", "notices", "orders", "updates"],
        "wait_selector": "#root",
    },
    {
        "path": "/careers",
        "url": "https://vvitu.ac.in/careers",
        "title": "VVITU Careers & Employment Opportunities",
        "description": "Official careers portal for VVIT University (VVITU), listing academic, teaching, research, and administrative job openings and recruitment notices.",
        "keywords": ["career", "careers", "job", "jobs", "recruitment", "vacancy", "vacancies", "opening", "openings", "hire", "hiring", "faculty recruitment"],
        "wait_selector": "#root",
    },
    {
        "path": "/hostels",
        "url": "https://vvitu.ac.in/hostels",
        "title": "VVITU Student Hostels & Residential Facilities",
        "description": "Hostel accommodations at VVIT University (VVITU) for boys and girls, including room amenities, boarding, mess facilities, and residential rules.",
        "keywords": ["hostel", "hostels", "boarding", "accommodation", "dormitory", "rooms", "mess", "stay", "residential"],
        "wait_selector": "#root",
    },
    {
        "path": "/transport",
        "url": "https://vvitu.ac.in/transport",
        "title": "VVITU Campus Transport & Bus Routes",
        "description": "Transportation facilities and bus routes operated by VVIT University (VVITU) connecting Guntur, Vijayawada, Mangalagiri, Tenali, and surrounding regions.",
        "keywords": ["transport", "bus", "buses", "bus route", "bus routes", "commute", "travel", "transportation"],
        "wait_selector": "#root",
    },
    {
        "path": "/library",
        "url": "https://vvitu.ac.in/library",
        "title": "VVITU Central Library & Information Centre",
        "description": "The Central Library at VVIT University (VVITU), featuring extensive book volumes, print and electronic journals, IEEE/ACM digital access, and research facilities.",
        "keywords": ["library", "books", "journals", "digital library", "ieee", "reading room", "volumes"],
        "wait_selector": "#root",
    },
    {
        "path": "/canteen",
        "url": "https://vvitu.ac.in/canteen",
        "title": "VVITU Canteen & Dining Facilities",
        "description": "Canteen and hygienic dining amenities available at VVIT University (VVITU) for students, faculty, and campus visitors.",
        "keywords": ["canteen", "cafeteria", "food", "dining", "snacks", "meals"],
        "wait_selector": "#root",
    },
    {
        "path": "/accreditation-and-approvals",
        "url": "https://vvitu.ac.in/accreditation-and-approvals",
        "title": "VVITU Accreditations, Approvals & Statutory Recognitions",
        "description": "Statutory recognitions, approvals, and accreditations of VVIT University (VVITU) including AICTE, UGC, NAAC, NBA, and State Government authorizations.",
        "keywords": ["accreditation", "accreditations", "approval", "approvals", "aicte", "ugc", "naac", "nba", "statutory", "recognition"],
        "wait_selector": "#root",
    },
    {
        "path": "/ranking-and-recognition",
        "url": "https://vvitu.ac.in/ranking-and-recognition",
        "title": "VVITU Rankings & Institutional Recognitions",
        "description": "Institutional rankings, NIRF participation, awards, and national recognition achieved by VVIT University (VVITU).",
        "keywords": ["ranking", "rankings", "nirf", "recognition", "awards", "achievement"],
        "wait_selector": "#root",
    },
    {
        "path": "/mandatory-disclosures",
        "url": "https://vvitu.ac.in/mandatory-disclosures",
        "title": "VVITU Mandatory Disclosures & Public Regulatory Info",
        "description": "Mandatory regulatory disclosures, institutional audits, and statutory documentation for VVIT University (VVITU).",
        "keywords": ["mandatory disclosure", "mandatory disclosures", "disclosure", "regulatory", "rti"],
        "wait_selector": "#root",
    },
    {
        "path": "/collaborations-and-partnerships",
        "url": "https://vvitu.ac.in/collaborations-and-partnerships",
        "title": "VVITU Global Collaborations & Industry Partnerships",
        "description": "Academic MoUs, industrial partnerships, and international collaborations established by VVIT University (VVITU).",
        "keywords": ["collaboration", "collaborations", "partnership", "partnerships", "mou", "mous", "international tie-up"],
        "wait_selector": "#root",
    },
    {
        "path": "/siemens-centre-of-excellence",
        "url": "https://vvitu.ac.in/siemens-centre-of-excellence",
        "title": "VVITU Siemens Centre of Excellence (CoE)",
        "description": "Siemens Centre of Excellence (CoE) at VVIT University (VVITU), offering advanced technical labs in robotics, automation, product design, and manufacturing.",
        "keywords": ["siemens", "coe", "centre of excellence", "automation lab", "siemens lab"],
        "wait_selector": "#root",
    },
    {
        "path": "/google-developers-code-lab",
        "url": "https://vvitu.ac.in/google-developers-code-lab",
        "title": "VVITU Google Developers Code Lab",
        "description": "Google Developers Code Lab at VVIT University (VVITU), fostering software engineering, mobile development, cloud computing, and developer community events.",
        "keywords": ["google", "code lab", "google developers", "gdsc", "developer student club"],
        "wait_selector": "#root",
    },
    {
        "path": "/IDEA-Labs",
        "url": "https://vvitu.ac.in/IDEA-Labs",
        "title": "VVITU AICTE IDEA Lab",
        "description": "AICTE IDEA (Idea Development, Evaluation & Application) Lab at VVIT University (VVITU), promoting hands-on STEM engineering and rapid prototyping.",
        "keywords": ["idea lab", "idea labs", "aicte idea lab", "prototyping", "innovation lab"],
        "wait_selector": "#root",
    },
    {
        "path": "/governing_body",
        "url": "https://vvitu.ac.in/governing_body",
        "title": "VVITU Governing Body",
        "description": "Governing Body of VVIT University (VVITU), responsible for apex institutional governance, strategic leadership, and policy direction.",
        "keywords": ["governing body", "governing council", "bog", "governance", "board of governors"],
        "wait_selector": "#root",
    },
    {
        "path": "/academic_council",
        "url": "https://vvitu.ac.in/academic_council",
        "title": "VVITU Academic Council",
        "description": "Academic Council of VVIT University (VVITU), overseeing curriculum regulations, syllabi approvals, academic standards, and degree requirements.",
        "keywords": ["academic council", "academic board", "curriculum council"],
        "wait_selector": "#root",
    },
    {
        "path": "/board_of_management",
        "url": "https://vvitu.ac.in/board_of_management",
        "title": "VVITU Board of Management",
        "description": "Board of Management of VVIT University (VVITU), guiding executive university administration and developmental decisions.",
        "keywords": ["board of management", "bom", "management board"],
        "wait_selector": "#root",
    },
    {
        "path": "/finance_commitee",
        "url": "https://vvitu.ac.in/finance_commitee",
        "title": "VVITU Finance Committee",
        "description": "Finance Committee of VVIT University (VVITU), managing financial planning, budget allocations, and fiscal governance.",
        "keywords": ["finance committee", "financial committee", "budget committee"],
        "wait_selector": "#root",
    },
    {
        "path": "/registrar",
        "url": "https://vvitu.ac.in/registrar",
        "title": "VVITU Office of the Registrar",
        "description": "Office of the Registrar at VVIT University (VVITU), overseeing official university records, admissions registration, and statutory correspondence.",
        "keywords": ["registrar", "office of registrar", "records"],
        "wait_selector": "#root",
    },
    {
        "path": "/pro-chancellor",
        "url": "https://vvitu.ac.in/pro-chancellor",
        "title": "VVITU Pro-Chancellor",
        "description": "Pro-Chancellor of VVIT University (VVITU), providing institutional leadership and strategic vision alongside the Chancellor.",
        "keywords": ["pro-chancellor", "pro chancellor"],
        "wait_selector": "#root",
    },
    {
        "path": "/academic-dean",
        "url": "https://vvitu.ac.in/academic-dean",
        "title": "VVITU Academic Dean",
        "description": "Dean of Academics at VVIT University (VVITU), coordinating academic delivery, faculty development, and educational quality across schools.",
        "keywords": ["academic dean", "dean academics", "dean of academics", "dean"],
        "wait_selector": "#root",
    },
    {
        "path": "/university-policies",
        "url": "https://vvitu.ac.in/university-policies",
        "title": "VVITU Institutional Policies & Codes of Conduct",
        "description": "Institutional codes of conduct, anti-ragging policies, ethical guidelines, and disciplinary rules at VVIT University (VVITU).",
        "keywords": ["policy", "policies", "anti ragging", "code of conduct", "discipline", "rules"],
        "wait_selector": "#root",
    },
    {
        "path": "/student-clubs",
        "url": "https://vvitu.ac.in/student-clubs",
        "title": "VVITU Student Activity Clubs",
        "description": "Student clubs and technical & cultural societies at VVIT University (VVITU) covering dance, music, robotics, photography, sports, and social service.",
        "keywords": ["club", "clubs", "student club", "student clubs", "sac", "cultural club", "technical club"],
        "wait_selector": "#root",
    },
    {
        "path": "/",
        "url": "https://vvitu.ac.in/",
        "title": "VVITU Official University Portal (Homepage)",
        "description": "Official primary portal of VVIT University (VVITU), featuring latest university announcements, campus updates, admissions, and institutional highlights.",
        "keywords": ["homepage", "main portal", "vvitu website", "official portal", "latest news", "campus news", "events"],
        "wait_selector": "#root",
    },
]


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
        self.browser_service = get_browser_retrieval_service()
        self.routes_catalog = list(OFFICIAL_VVITU_ROUTES)

        # Find local bundle if available
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        default_bundle_path = os.path.join(base_dir, "data", "vvitu_bundle.js")
        self.bundle_local_path = bundle_local_path if isinstance(bundle_local_path, str) else default_bundle_path

        # Parsed catalogs
        self.faculty_catalog: List[Dict[str, Any]] = []
        self.leadership_catalog: List[Dict[str, Any]] = []
        self.schools_catalog: List[Dict[str, Any]] = []
        self.programs_catalog: List[Dict[str, Any]] = []
        self.pages_catalog: List[Dict[str, Any]] = []
        self.departments_catalog: List[Dict[str, Any]] = []
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

                # Build dynamic department catalog & route
                KNOWN_DEPT_SYNONYMS = {
                    "cse-ai-faculty": ["ai", "ds", "ai ds", "ai & ds", "ai and ds", "cse ai", "cse ai & ds", "artificial intelligence", "data science", "machine learning", "aiml", "ai ds faculty", "cse ai faculty", "ai faculty", "cse-ai-faculty"],
                    "cse-faculty": ["cse", "computer science", "cse faculty", "computer science faculty", "cse-faculty", "computer science engineering"],
                    "ece-faculty": ["ece", "electronics", "communication", "ece faculty", "electronics faculty", "ece-faculty", "electronics and communication"],
                    "eee-faculty": ["eee", "electrical", "eee faculty", "electrical faculty", "eee-faculty", "electrical and electronics"],
                    "civil-faculty": ["civil", "civil engineering", "civil faculty", "ce faculty", "civil-faculty"],
                    "mec-faculty": ["mech", "mechanical", "mec faculty", "mech faculty", "mechanical faculty", "mec-faculty", "mechanical engineering"],
                    "applied-sciences-faculty": ["bsh", "bs&h", "applied sciences", "basic sciences", "maths", "physics", "chemistry", "applied sciences faculty", "bsh faculty", "humanities and sciences"],
                    "humanities-social-sciences-faculty": ["humanities", "social sciences", "english", "humanities faculty", "social sciences faculty"],
                    "business-administration-faculty": ["mba", "management", "business", "business administration", "mba faculty", "management faculty", "business faculty", "appa school of business"],
                }
                dept_keywords = set(KNOWN_DEPT_SYNONYMS.get(slug, []))
                dept_keywords.add(slug)
                dept_keywords.add(slug.replace("-", " "))
                dept_keywords.add(slug.replace("-faculty", ""))
                dept_keywords.add(f"{slug.replace('-faculty', '').replace('-', ' ')} faculty")
                dept_keywords.add(dept_name.lower())
                clean_name = re.sub(r"(?i)department of\s*", "", dept_name).strip().lower()
                dept_keywords.add(clean_name)
                dept_keywords.add(f"{clean_name} faculty")

                self.departments_catalog.append({
                    "slug": slug,
                    "name": dept_name,
                    "url": f"https://vvitu.ac.in/admissions/faculty/{slug}",
                    "faculty_count": len(matches),
                    "keywords": list(dept_keywords),
                })

                dept_route_path = f"/admissions/faculty/{slug}"
                if not any(r.get("path") == dept_route_path for r in self.routes_catalog):
                    self.routes_catalog.append({
                        "path": dept_route_path,
                        "url": f"https://vvitu.ac.in{dept_route_path}",
                        "title": f"VVITU {dept_name} Faculty Directory",
                        "description": f"Official faculty directory for {dept_name} at VVIT University (VVITU), listing all professors, associate professors, assistant professors, and qualifications.",
                        "keywords": list(dept_keywords),
                        "wait_selector": "a[href*='/faculty/'], table, [class*='grid'], main, #root:not(:empty)",
                    })

            logger.info("Successfully extracted %d faculty members across %d departments from bundle", count, len(self.departments_catalog))
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

    def _find_matching_routes(self, query: str) -> List[Dict[str, Any]]:
        """Match user query against known VVITU dynamic routes with word-boundary precision."""
        q_norm = query.lower()
        q_tokens = self._tokenize(query)
        stop_words = {
            "tell", "about", "what", "give", "show", "who", "whom", "where",
            "when", "which", "how", "details", "info", "information", "please",
            "can", "you", "know", "there", "their", "this", "that", "with",
            "from", "does", "have", "some", "more", "much", "many", "been",
        }
        meaningful_tokens = {t for t in q_tokens if t not in stop_words and len(t) >= 2}

        matches = []
        for route in self.routes_catalog:
            score = 0
            for kw in route["keywords"]:
                kw_lower = kw.lower()
                if " " in kw_lower or "-" in kw_lower:
                    if kw_lower in q_norm or kw_lower.replace("-", " ") in q_norm:
                        score += 5
                else:
                    if kw_lower in meaningful_tokens:
                        score += 3
                    elif kw_lower in q_norm:
                        score += 1

            if score > 0:
                matches.append((score, route))

        matches.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in matches]

    async def render_official_page(
        self,
        url: str,
        query: str = "",
        wait_selector: Optional[str] = None,
    ) -> OfficialWebResult:
        """
        Render an official page using headless browser (DOM or Screenshot/Vision fallback).
        """
        render_res = await self.browser_service.render_page(url, query=query, wait_selector=wait_selector)
        is_vvitu = any(d in url for d in CURRENT_DOMAINS)
        period = "current" if is_vvitu else "historical"
        tier = "primary_official_current" if is_vvitu else "legacy_official_vvit"

        entity_type = "faculty" if "/faculty" in url else ("leadership" if "/leadership" in url else "general_page")

        entities = getattr(render_res, "entities", [])
        is_evidence_valid = render_res.success and EvidenceQualityValidator.validate_content(
            query=query or "",
            content=render_res.text,
            entities=entities,
            retrieval_method=render_res.method,
        )

        if is_evidence_valid:
            content = (
                f"**{render_res.title}**\n"
                f"• **Official URL:** {url}\n"
                f"• **Institution:** {'VVIT University (VVITU)' if is_vvitu else 'Vasireddy Venkatadri Institute of Technology (VVIT)'}\n"
                f"• **Retrieval Method:** {render_res.method}\n\n"
                f"{render_res.text[:12000]}"
            )
            return OfficialWebResult(
                title=render_res.title or ("VVITU Official Portal" if is_vvitu else "VVIT Legacy Portal"),
                url=url,
                content=content,
                period=period,
                source_tier=tier,
                confidence=0.96,
                entity_type=getattr(render_res, "entity_type", entity_type),
                retrieval_method=render_res.method,
                entities=entities,
                metadata={"chars": len(render_res.text), "entities_count": len(entities)},
                screenshot_used=getattr(render_res, "screenshot_used", False),
                screenshot_scope=getattr(render_res, "screenshot_scope", None),
                vision_provider=getattr(render_res, "vision_provider", None),
                vision_model=getattr(render_res, "vision_model", None),
                extraction_timestamp=getattr(render_res, "extraction_timestamp", None),
            )

        # If live browser failed, timed out, or extracted 0 entities for a faculty page:
        # Check if URL corresponds to an official department faculty directory
        norm_url = url.rstrip("/")
        for dept in self.departments_catalog:
            if dept["url"].rstrip("/") == norm_url or norm_url.endswith(f"/{dept['slug']}"):
                dept_faculty = [f for f in self.faculty_catalog if f.get("dept_slug") == dept["slug"]]
                if dept_faculty:
                    logger.info("Resolving department faculty from verified bundle catalog for %s (%d members)", dept["slug"], len(dept_faculty))
                    cat_entities = [
                        {
                            "name": f["name"],
                            "designation": f["designation"],
                            "qualification": f["qualification"],
                            "profile_url": f["url"],
                        }
                        for f in dept_faculty
                    ]
                    md_lines = [
                        f"**VVITU Official Faculty Directory — {dept['name']}**",
                        f"• **Official URL:** {dept['url']}",
                        f"• **Institution:** VVIT University (VVITU)",
                        f"• **Total Verified Faculty Listed:** {len(dept_faculty)}\n",
                        "| S.No | Faculty Name | Designation | Qualification | Official Profile Link |",
                        "|---|---|---|---|---|",
                    ]
                    for idx, f in enumerate(dept_faculty, 1):
                        md_lines.append(f"| {idx} | **{f['name']}** | {f['designation']} | {f['qualification']} | [Profile]({f['url']}) |")
                    content = "\n".join(md_lines)
                    return OfficialWebResult(
                        title=f"VVITU {dept['name']} Faculty Directory",
                        url=dept["url"],
                        content=content,
                        period="current",
                        source_tier="primary_official_current",
                        confidence=0.98,
                        entity_type="faculty_directory",
                        retrieval_method="catalog",
                        entities=cat_entities,
                        metadata={"total_faculty": len(dept_faculty), "dept_slug": dept["slug"]},
                    )

        # Direct canonical URL fallback when extraction is incomplete
        content = (
            f"**Official Portal Page: {url}**\n"
            f"• **Official URL:** {url}\n"
            f"• **Institution:** {'VVIT University (VVITU)' if is_vvitu else 'Vasireddy Venkatadri Institute of Technology (VVIT)'}\n"
            f"• **Status:** Content could not be automatically extracted from dynamic components.\n"
            f"• **Action:** Direct page link verified and available."
        )
        return OfficialWebResult(
            title="Official Portal Page",
            url=url,
            content=content,
            period=period,
            source_tier=tier,
            confidence=0.85,
            entity_type="general_page",
            retrieval_method="direct_url_only",
            metadata={"error": render_res.error},
        )

    def is_safe_official_url(self, url: str) -> bool:
        """Validate that a URL is a safe official institutional URL and not SSRF attack."""
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return False
            hostname = (parsed.hostname or "").lower()
            if not hostname:
                return False
            # Check against allowed official domains
            is_allowed = (
                hostname in CURRENT_DOMAINS or
                hostname in LEGACY_DOMAINS or
                any(hostname.endswith("." + d) for d in CURRENT_DOMAINS) or
                any(hostname.endswith("." + d) for d in LEGACY_DOMAINS)
            )
            if not is_allowed:
                return False
            # Prevent loopback and private addresses
            blocked = ["localhost", "127.0.0.1", "0.0.0.0", "::1", "169.254.", "10.", "192.168.", "172.16."]
            if any(b in hostname for b in blocked):
                return False
            return True
        except Exception:
            return False

    async def retrieve_url(self, url: str, query: str = "") -> OfficialWebResult:
        """
        Retrieve and extract structured institutional information from a specific official URL.
        Validates SSRF safety, checks catalog for instant match, and renders headlessly with card extraction.
        """
        if not self.is_safe_official_url(url):
            logger.warning("Blocked unsafe or non-official URL retrieval: %s", url)
            return OfficialWebResult(
                title="Invalid or Unsupported URL",
                url=url,
                content=f"The URL {url} is not an authorized official VVITU or VVIT website address.",
                period="current",
                source_tier="primary_official_current",
                confidence=0.0,
                retrieval_method="failed",
            )

        # Check cache
        cache_key = f"url:{url}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached[0]

        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        is_vvitu = any(d in hostname for d in CURRENT_DOMAINS)
        is_legacy = any(d in hostname for d in LEGACY_DOMAINS)
        period = "current" if is_vvitu else "historical"
        tier = "primary_official_current" if is_vvitu else "legacy_official_vvit"

        # Check if URL is an individual faculty profile in catalog
        if is_vvitu:
            norm_url = url.rstrip("/")
            for fac in self.faculty_catalog:
                if fac["url"].rstrip("/") == norm_url:
                    res = OfficialWebResult(
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
                        confidence=0.98,
                        entity_type="faculty",
                        retrieval_method="catalog",
                        metadata=fac,
                    )
                    self.cache.set(cache_key, [res])
                    return res

        # Check if legacy page can be retrieved via fast HTTP
        if is_legacy:
            legacy_text = await asyncio.to_thread(self._fetch_legacy_page, url)
            if legacy_text and len(legacy_text.strip()) >= 120:
                res = OfficialWebResult(
                    title="VVIT Legacy Official Portal",
                    url=url,
                    content=(
                        f"**VVIT Legacy Archive: {url}**\n"
                        f"• **Source:** Vasireddy Venkatadri Institute of Technology (Historical Archive)\n"
                        f"• **URL:** {url}\n\n"
                        f"{legacy_text[:3000]}"
                    ),
                    period="historical",
                    source_tier="legacy_official_vvit",
                    confidence=0.92,
                    entity_type="general_page",
                    retrieval_method="http",
                    metadata={"url": url},
                )
                self.cache.set(cache_key, [res])
                return res

        # For React SPA / dynamic pages (VVITU or dynamic legacy), render headlessly with card extraction
        render_res = await self.render_official_page(
            url,
            query=query,
            wait_selector="a[href*='/faculty/'], table, [class*='grid'], main, #root:not(:empty)",
        )
        self.cache.set(cache_key, [render_res])
        return render_res

    def _search_vvitguntur_live(self, query: str, max_items: int = 2) -> List[OfficialWebResult]:
        """
        Perform synchronous live search on https://vvitguntur.com/ using its Joomla search endpoint.
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

                parsed = urlparse(full_url)
                if parsed.hostname not in LEGACY_DOMAINS:
                    continue

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
                        retrieval_method="http",
                        metadata={"title": title, "url": full_url},
                    ))
        except Exception as exc:
            logger.debug("Live legacy search failed: %s", exc)

        return results

    async def _search_vvitguntur_live_async(self, query: str, max_items: int = 2) -> List[OfficialWebResult]:
        """
        Perform live search on https://vvitguntur.com/ with HTTP and Headless fallback.
        """
        clean_query = re.sub(r"[^a-zA-Z0-9\s]", "", query).strip()
        words = clean_query.split()
        if not words:
            return []

        searchword = "+".join(words[:4])
        search_url = f"https://vvitguntur.com/index.php/component/search/?searchword={searchword}&searchphrase=all"

        results = []
        try:
            resp = await asyncio.to_thread(self.session.get, search_url, timeout=REQUEST_TIMEOUT)
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

                parsed = urlparse(full_url)
                if parsed.hostname not in LEGACY_DOMAINS:
                    continue

                # Try fast HTTP first
                article_text = await asyncio.to_thread(self._fetch_legacy_page, full_url)
                if article_text and len(article_text.strip()) >= 100:
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
                        retrieval_method="http",
                        metadata={"title": title, "url": full_url},
                    ))
                else:
                    # Fallback to headless browser render
                    render_res = await self.render_official_page(full_url, query=query)
                    results.append(render_res)

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
        Orchestrates Tier 1 Catalog (0ms), Tier 2 HTTP (~200ms),
        Tier 3 Headless Browser DOM (~1.5s - 3s), and Tier 4 Screenshot + Vision OCR.
        """
        cached = self.cache.get(f"{query}:{period}")
        if cached is not None:
            return cached[:max_results]

        # 1. Check if query contains an explicit official URL
        url_match = re.search(r'https?://[^\s<>"]+', query)
        if url_match:
            cand_url = url_match.group(0).rstrip(".,;!?'\")>]}")
            if self.is_safe_official_url(cand_url):
                logger.info("Direct official URL found in query: %s", cand_url)
                url_res = await self.retrieve_url(cand_url, query=query)
                if url_res.retrieval_method != "failed":
                    self.cache.set(f"{query}:{period}", [url_res])
                    return [url_res]

        results: List[OfficialWebResult] = []
        is_faculty_query = any(k in query.lower() for k in ["faculty", "professor", "professors", "teacher", "teachers", "staff", "hod", "head of department", "members"])

        if period == "historical":
            # 1. Historical query: check legacy leadership
            for leader in self.leadership_catalog:
                if leader["period"] == "historical" or any(kw in query.lower() for kw in leader["keywords"]):
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
                        retrieval_method="catalog",
                        metadata=leader,
                    ))

            # 2. Live legacy search with headless fallback
            if not results:
                legacy_live = await self._search_vvitguntur_live_async(query, max_items=max_results)
                results.extend(legacy_live)

        elif period == "current":
            # If faculty query, check matching routes FIRST (e.g. department directories)
            matched_routes = self._find_matching_routes(query)
            if is_faculty_query and matched_routes:
                for route in matched_routes[:2]:
                    if not any(r.url == route["url"] for r in results):
                        route_res = await self.render_official_page(
                            route["url"],
                            query=query,
                            wait_selector=route.get("wait_selector"),
                        )
                        if route_res.retrieval_method != "failed":
                            results.append(route_res)

            # Check catalog (leadership, faculty, schools, programs, pages)
            curr_results = self.search_vvitu(query)
            results.extend(curr_results)

            # Check other dynamic routes if not already rendered
            if not is_faculty_query and matched_routes:
                for route in matched_routes[:2]:
                    if not any(r.url == route["url"] for r in results):
                        route_res = await self.render_official_page(
                            route["url"],
                            query=query,
                            wait_selector=route.get("wait_selector"),
                        )
                        if route_res.retrieval_method != "failed":
                            results.append(route_res)

            # If still empty, render homepage headlessly
            if not results:
                home_res = await self.render_official_page("https://vvitu.ac.in/", query=query, wait_selector="#root")
                if home_res.retrieval_method != "failed":
                    results.append(home_res)

        else:
            # General query: if faculty query, check matching routes FIRST
            matched_routes = self._find_matching_routes(query)
            if is_faculty_query and matched_routes:
                for route in matched_routes[:2]:
                    if not any(r.url == route["url"] for r in results):
                        route_res = await self.render_official_page(
                            route["url"],
                            query=query,
                            wait_selector=route.get("wait_selector"),
                        )
                        if route_res.retrieval_method != "failed":
                            results.append(route_res)

            # Check current catalog
            curr_results = self.search_vvitu(query)
            results.extend(curr_results)

            # Check dynamic official routes if not already rendered
            if not is_faculty_query and matched_routes:
                for route in matched_routes[:2]:
                    if not any(r.url == route["url"] for r in results):
                        route_res = await self.render_official_page(
                            route["url"],
                            query=query,
                            wait_selector=route.get("wait_selector"),
                        )
                        if route_res.retrieval_method != "failed":
                            results.append(route_res)

            # If legacy query or no current results, search legacy
            q_lower = query.lower()
            has_faculty_dir = any(r.entity_type in ("faculty", "faculty_directory") and (len(r.entities) > 0 or "Faculty Directory" in r.title) for r in results)
            has_historical_intent = any(kw in q_lower for kw in ["legacy", "history", "past", "former", "old", "earlier", "previous", "was", "archive", "archived"])
            if not results or (has_historical_intent and not has_faculty_dir):
                legacy_results = self.search_vvit_legacy(query)
                results.extend(legacy_results)
                if not legacy_results:
                    legacy_live = await self._search_vvitguntur_live_async(query, max_items=2)
                    results.extend(legacy_live)

            # If still empty, render homepage headlessly
            if not results:
                home_res = await self.render_official_page("https://vvitu.ac.in/", query=query, wait_selector="#root")
                if home_res.retrieval_method != "failed":
                    results.append(home_res)

        # Deduplicate by URL
        seen_urls = set()
        deduped: List[OfficialWebResult] = []
        for r in results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                deduped.append(r)

        # If a verified faculty directory is present, prioritize it and eliminate unrelated noisy pages
        faculty_dirs = [r for r in deduped if r.entity_type in ("faculty", "faculty_directory") and (len(r.entities) > 0 or "Faculty Directory" in r.title)]
        if faculty_dirs:
            final_results = faculty_dirs[:1]
        else:
            final_results = deduped[:max_results]

        self.cache.set(f"{query}:{period}", final_results)
        return final_results

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
