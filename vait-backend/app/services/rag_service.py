"""VAIT RAG Service — Retrieval-Augmented Generation pipeline."""

import hashlib
import asyncio
import json
import logging
import re
import time
import faiss
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import datetime
from urllib.parse import urlparse

from app.utils.config import (
    Settings,
    REQUIRED_METADATA_FIELDS,
    AUTHORITY_LEVELS,
    LOGS_DIR,
)
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.response_planner import response_planner, ResponsePlan
from app.services.official_web_retriever import get_official_web_retriever, OfficialWebResult
from app.utils.text_chunker import TextChunker
from app.services.conversation_context_resolver import (
    ConversationContextResolver,
    ConversationContext,
    ExtractedEntity,
    FollowUpIntent,
    IntentType,
)
from app.utils.timezone import get_ist_grounding_context, format_ist, now_ist

logger = logging.getLogger("vait.rag")

# ── Query constraints ────────────────────────────────────────────────
MAX_QUERY_LENGTH = 2000  # characters

# ── Authority multipliers for adjusted scoring ───────────────────────
AUTHORITY_MULTIPLIER = {
    "high": 1.15,
    "medium": 1.05,
    "low": 1.00,
}

SOURCE_TYPE_BOOST = {
    "official_website": 0.03,
    "website": 0.03,
    "pdf": 0.05,
}

SOURCE_TIER_BOOST = {
    "primary_official_current": 0.08,
    "primary_official": 0.06,
    "legacy_official_vvit": 0.05,
    "secondary_linkedin": 0.03,
    "tertiary_social": 0.01,
    "related_web": 0.00,
}

CURRENT_UNIVERSITY_DOMAINS = ("vvitu.ac.in", "www.vvitu.ac.in")
LEGACY_INSTITUTE_DOMAINS = ("vvitguntur.com", "www.vvitguntur.com")
OFFICIAL_DOMAINS = CURRENT_UNIVERSITY_DOMAINS + LEGACY_INSTITUTE_DOMAINS

CURRENT_UNIVERSITY_URL = "https://vvitu.ac.in/"
LEGACY_INSTITUTE_URL = "https://vvitguntur.com/"

DYNAMIC_QUERY_KEYWORDS = {
    "latest", "recent", "new", "happening", "happen", "happens",
    "update", "updates", "news", "today", "currently", "ongoing",
    "upcoming", "this week", "this month", "announcements",
}

RECENT_SOCIAL_DAYS = 60

# ── Context optimization constants ──────────────────────────────────
MAX_CONTEXT_CHUNKS = 4                 # Top-N highest quality chunks
NEAR_DUPLICATE_SIMILARITY = 0.90       # Jaccard word-overlap threshold
MIN_VALID_RESPONSE_LENGTH = 40         # Characters — shorter = invalid

CONFIDENCE_HIGH = 0.85
CONFIDENCE_MEDIUM = 0.75
CONFIDENCE_LOW = 0.65
REFUSAL_ADJUSTED_THRESHOLD = 0.55

MAX_RESPONSE_WORDS = 600

CACHE_CAPACITY = 100
CACHE_TTL_SECONDS = 600

INTENT_KEYWORDS: Dict[str, List[str]] = {
    "academic": [
        "syllabus", "curriculum", "course", "subject", "semester",
        "credit", "grade", "gpa", "cgpa", "marks", "internal",
        "external", "mid", "lab", "theory", "elective", "regulation",
        "r20", "r23", "academic", "calendar", "class", "timetable",
        "schedule", "lecture", "faculty", "professor", "hod",
    ],
    "admissions": [
        "admission", "admissions", "fee", "fees", "tuition",
        "scholarship", "eamcet", "ecet", "counseling", "counselling",
        "seat", "intake", "eligibility", "cutoff", "cut-off",
        "application", "apply", "joining", "lateral", "management",
    ],
    "examinations": [
        "exam", "examination", "result", "results", "supply",
        "supplementary", "revaluation", "hall ticket", "seating",
        "question paper", "model paper", "backlog", "arrear",
        "pass", "fail", "detained", "attendance",
    ],
    "placements": [
        "placement", "placements", "recruit", "recruiting",
        "company", "companies", "package", "salary", "ctc",
        "lpa", "offer", "interview", "internship", "training",
        "campus", "off-campus", "drive", "hire", "hiring",
    ],
    "events": [
        "event", "events", "fest", "festival", "cultural",
        "technical", "hackathon", "workshop", "seminar",
        "webinar", "conference", "guest lecture", "sports",
        "nss", "ncc", "club", "clubs", "activity", "celebration",
        "sac", "student activity council", "latest", "recent",
        "update", "updates", "ongoing", "upcoming", "happening",
        "community", "communities",
    ],
    "infrastructure": [
        "hostel", "library", "lab", "laboratory", "bus",
        "transport", "canteen", "wifi", "internet", "campus",
        "building", "facility", "facilities", "gym", "ground",
        "auditorium", "parking",
    ],
}


@dataclass
class RetrievedChunk:
    """Represents a retrieved chunk with its metadata and score."""
    content: str
    metadata: Dict
    similarity_score: float
    document_name: str
    authority_level: str = "low"
    adjusted_score: float = 0.0
    url: str = ""
    source_type: str = ""
    document_type: str = ""
    source_tier: str = ""


@dataclass
class SourceInfo:
    """Structured source object for response."""
    title: str
    url: str
    type: str  # e.g. "website", "pdf", "social_media"


@dataclass
class RAGResponse:
    """Response from the RAG system."""
    reply: str
    sources: List[str]
    structured_sources: List[Dict] = field(default_factory=list)
    confidence: str = "Low"  # "High" | "Medium" | "Low"
    source_visibility: str = "none"  # "none" | "compact" | "full"
    retrieval_score: float = 0.0
    question_type: str = ""
    intent: str = "general"  # classified intent category
    response_type: str = "informational"  # classified presentation format
    retrieval_scores: List[float] = field(default_factory=list)
    is_refusal: bool = False
    performance: Optional[Dict] = None
    context_state: Optional[Dict] = None


@dataclass
class _CacheEntry:
    """Internal cache entry with TTL."""
    response: RAGResponse
    timestamp: float  # time.time() when stored


# Query normalization patterns
_QUERY_NORMALIZE_PATTERNS = [
    (re.compile(r"(?i)^full\s+form\s+of\s+(.+)$"), r"What is the full name of \1?"),
    (re.compile(r"(?i)^expand\s+(.+)$"), r"What is the full name of \1?"),
    (re.compile(r"(?i)^what\s+is\s+([A-Za-z]{2,6})\??$"), r"What is the full name of \1?"),
]


class RAGService:
    """Production-grade RAG service for institutional knowledge retrieval."""

    def __init__(self, settings: Settings):
        """Initialize the RAG service."""
        self.settings = settings
        self.embedding_service = EmbeddingService(settings)
        self.llm_service = LLMService(settings)
        self.text_chunker = TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        # FAISS index and metadata storage
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Dict] = []

        # Load system prompt
        self.system_prompt = self._load_system_prompt()

        # ── Response cache (LRU with TTL) ────────────────────────────
        from collections import OrderedDict
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._start_time: float = time.time()
        self._official_bootstrap_attempted: bool = False
        self.web_retriever = get_official_web_retriever()
        self.context_resolver = ConversationContextResolver()
        self._conversation_contexts: Dict[str, ConversationContext] = {}

    async def initialize(self) -> None:
        """Initialize the RAG service and load existing index if available."""
        index_path = Path(self.settings.faiss_index_path)
        metadata_path = Path(self.settings.metadata_path)

        if index_path.exists() and metadata_path.exists():
            try:
                await self._load_index()
                logger.info(
                    "Loaded existing FAISS index with %d chunks", len(self.chunks)
                )
            except Exception as exc:
                logger.error("Failed to load FAISS index: %s", exc)
                self._create_empty_index()
        else:
            self._create_empty_index()
            if not index_path.exists():
                logger.warning("FAISS index not found at %s", index_path)
            if not metadata_path.exists():
                logger.warning("Metadata file not found at %s", metadata_path)
            logger.info("Initialized empty FAISS index — run ingest_documents.py first")

    def _create_empty_index(self) -> None:
        """Create a fresh empty FAISS index."""
        dimension = self.embedding_service.embedding_dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks = []

    @staticmethod
    def _normalize_query(query: str) -> str:
        """Normalize query for better embedding matches.

        - Strips whitespace
        - Lowercases before pattern matching
        - Rewrites shorthand forms (e.g. 'full form of X' -> 'What is the full name of X?')
        """
        query = query.strip()
        if not query:
            return query
        for pattern, replacement in _QUERY_NORMALIZE_PATTERNS:
            m = pattern.match(query)
            if m:
                query = pattern.sub(replacement, query)
                break
        return query

    @staticmethod
    def classify_intent(query: str) -> Tuple[str, List[str]]:
        """
        Classify query intent using rule-based keyword matching.

        Returns:
            (intent_category, matched_keywords)
            Categories: academic, admissions, examinations, placements,
                        events, infrastructure, general
        """
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))

        scores: Dict[str, List[str]] = {}
        for intent, keywords in INTENT_KEYWORDS.items():
            matched = []
            for kw in keywords:
                # Support multi-word keywords
                if " " in kw:
                    if kw in query_lower:
                        matched.append(kw)
                elif kw in query_words:
                    matched.append(kw)
            if matched:
                scores[intent] = matched

        if not scores:
            return "general", []

        # Pick the intent with the most keyword matches
        best_intent = max(scores, key=lambda k: len(scores[k]))
        return best_intent, scores[best_intent]

    def _cache_key(self, query: str) -> str:
        """Normalize query into a cache key."""
        normalized = query.strip().lower()
        normalized = re.sub(r'\s+', ' ', normalized)
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    def _cache_get(self, query: str) -> Optional[RAGResponse]:
        """Check cache for a valid (non-expired) entry."""
        key = self._cache_key(query)
        entry = self._cache.get(key)
        if entry is None:
            self._cache_misses += 1
            return None

        # Check TTL
        if (time.time() - entry.timestamp) > CACHE_TTL_SECONDS:
            del self._cache[key]
            self._cache_misses += 1
            return None

        # Move to end (LRU refresh)
        self._cache.move_to_end(key)
        self._cache_hits += 1
        logger.debug("Cache HIT for query: %.80s", query)
        return entry.response

    def _cache_put(self, query: str, response: RAGResponse) -> None:
        """Store a response in the cache, evicting oldest if at capacity."""
        key = self._cache_key(query)
        self._cache[key] = _CacheEntry(
            response=response,
            timestamp=time.time(),
        )
        self._cache.move_to_end(key)

        # Evict oldest entries if over capacity
        while len(self._cache) > CACHE_CAPACITY:
            self._cache.popitem(last=False)

    @classmethod
    def _is_conversational_query(cls, query: str) -> Optional[str]:
        """
        Detect casual conversational queries, greetings, thanks, and pleasantries.
        Returns a natural, friendly reply string if matched, else None.
        """
        q = query.strip().lower()
        q_clean = re.sub(r"[?!.,;:]+$", "", q).strip()

        # Greetings
        if re.search(r"^(hi|hello|hey|hey there|hi vait|hello vait|good morning|good afternoon|good evening|greetings)\b", q_clean):
            if re.search(r"\b(how are you|how're you|how are you doing|how's it going|what's up|sup)\b", q_clean):
                return (
                    "Hello! I'm doing well, thank you for asking. "
                    "I am VAIT, your AI assistant for VVIT University (VVITU). "
                    "How can I help you today?"
                )
            return (
                "Hello! I am VAIT (VVIT's Artificial Intelligence Technology). "
                "How can I assist you with VVITU or academic information today?"
            )

        # Well-being check-ins
        if re.search(r"^(how are you|how are you doing|how are you dude|how are you bro|how's it going|how do you do|what's up|whats up|sup)$", q_clean):
            return (
                "I'm doing great, thank you! I'm ready to assist you with anything related to "
                "VVIT University (VVITU), courses, regulations, exams, or campus resources. "
                "What would you like to know?"
            )

        # Gratitude
        if re.search(r"^(thank you|thanks|thanks a lot|thank you so much|thx|appreciate it|cool thanks|ok thanks|okay thanks)$", q_clean):
            return "You're welcome! Feel free to ask whenever you need any information."

        # Farewells
        if re.search(r"^(bye|goodbye|see you|see ya|have a good day|talk to you later)$", q_clean):
            return "Goodbye! Have a great day ahead. Feel free to reach out anytime."

        # Ping / test
        if re.search(r"^(test|testing|are you there|are you online|ping)$", q_clean):
            return "VAIT is online and ready to assist you. How can I help you today?"

        return None

    @classmethod
    def _is_identity_query(cls, query: str) -> bool:
        """Check if the query is asking about VAIT's identity, name, expansion, or capabilities."""
        q = query.strip().lower().rstrip("?.! ")
        patterns = [
            r"^who are you$",
            r"^what is your name$",
            r"^what('s| is) vait$",
            r"^what does vait stand for$",
            r"^what is the full form of vait$",
            r"^what is the full name of vait$",
            r"^what is the meaning of vait$",
            r"^what is vait stand for$",
            r"^vait stand for$",
            r"^expand vait$",
            r"^tell me about yourself$",
            r"^tell me about vait$",
            r"^introduce yourself$",
            r"^what are you$",
            r"^who created you$",
            r"^what can you do$",
            r"^what are your capabilities$",
            r"^how can you help me$",
            r"^what do you do$",
        ]
        return any(re.search(p, q, re.IGNORECASE) for p in patterns)

    def _is_institutional_query(self, query: str) -> bool:
        """
        Determine if the query is specifically about VVIT, VVITU, campus,
        faculty, leadership, departments, or institutional operations,
        rather than general knowledge/coding/casual chat.
        """
        q = query.lower()

        # 1. Direct college / institute / university name mentions or official URLs
        institutional_names = {
            "vvit", "vvitu", "vasireddy", "venkatadri", "nambur",
            "vvitian", "vvitians", "vvitguntur",
        }
        if any(re.search(rf"\b{re.escape(name)}\b", q) for name in institutional_names):
            return True
        if re.search(r"https?://[^\s]*(?:vvitu\.ac\.in|vvitguntur\.com)", q):
            return True

        # 2. Check if official web retriever catalog matches an entity (faculty, leadership, school, program)
        if hasattr(self, "web_retriever") and self.web_retriever:
            vvitu_matches = self.web_retriever.search_vvitu(query)
            if vvitu_matches:
                return True

        # 3. Person title patterns (e.g. Dr., Prof., faculty)
        if re.search(r"\b(dr\b|prof\b|professor|faculty|lecturer|teacher|dean|chancellor|principal|hod)\b", q):
            return True

        # 4. College / campus roles, bodies, facilities, and administration
        college_entities = [
            r"\b(principal|vice principal|director|dean|chairman|hod|chancellor|vice chancellor)\b",
            r"\b(faculty|professor|prof|assistant professor|associate professor|dr\b|lecturer|teacher)\b",
            r"\b(our college|our campus|our university|this college|this campus|this university)\b",
            r"\b(college|campus|university|institute)\b",
            r"\b(hostel|mess|canteen|library|bus|transport|bus route|bus fees?)\b",
            r"\b(fee|fees|fee structure|tuition|scholarship)\b",
            r"\b(admission|admissions|eamcet|ecet|counseling|counselling|cutoff|intake)\b",
            r"\b(syllabus|curriculum|regulation|regulations|r20|r23|jntuk|autonomous)\b",
            r"\b(sac|student activity council|nss|ncc|club|clubs)\b",
            r"\b(placement|placements|placement cell|highest package|average package|recruiters?)\b",
            r"\b(attendance|hall ticket|mid exam|semester exam|revaluation|supply exam|backlog)\b",
            r"\b(academic calendar|class schedule|timetable)\b",
        ]
        return any(re.search(pattern, q) for pattern in college_entities)

    @staticmethod
    def _extract_official_urls(text: str) -> List[str]:
        """Extract and validate official URLs from user message."""
        url_pattern = re.compile(r'https?://[^\s<>"]+')
        matches = url_pattern.findall(text)
        official_urls = []
        for u in matches:
            u_clean = u.rstrip(".,;!?'\")>]}")
            parsed = urlparse(u_clean)
            hostname = (parsed.hostname or "").lower()
            if any(d in hostname for d in ["vvitu.ac.in", "vvitguntur.com"]):
                official_urls.append(u_clean)
        return official_urls

    def _get_or_rehydrate_context(
        self,
        history: Optional[list],
        context_state: Optional[dict] = None,
        conversation_id: Optional[str] = None,
    ) -> ConversationContext:
        """Get or rehydrate ConversationContext from explicit state, conversation_id, or history."""
        if context_state:
            try:
                return ConversationContext.model_validate(context_state)
            except Exception as exc:
                logger.warning("Could not deserialize context_state: %s", exc)

        if conversation_id and conversation_id in self._conversation_contexts:
            return self._conversation_contexts[conversation_id]

        ctx = ConversationContext()
        if not history:
            return ctx

        # 1. Look backwards for user turns to infer active subject, concept, and filters
        for turn in reversed(history):
            role = turn.get("role") if isinstance(turn, dict) else getattr(turn, "role", "")
            content = turn.get("content") or turn.get("text") if isinstance(turn, dict) else (getattr(turn, "content", None) or getattr(turn, "text", ""))
            if role == "user" and content:
                concept = self.context_resolver.detect_concept(content)
                subject = self.context_resolver.detect_subject(content)
                filters = self.context_resolver.detect_filters(content)
                if concept:
                    ctx.active_concept = concept
                    ctx.active_entity_type = concept
                if subject:
                    ctx.active_subject = subject
                if filters:
                    ctx.active_filters.update(filters)
                if concept or subject:
                    break

        # 2. Look backwards for the last assistant turn to extract active entities and evidence
        for turn in reversed(history):
            role = turn.get("role") if isinstance(turn, dict) else getattr(turn, "role", "")
            content = turn.get("content") or turn.get("text") if isinstance(turn, dict) else (getattr(turn, "content", None) or getattr(turn, "text", ""))
            if role == "assistant" and content:
                extracted = self.context_resolver.extract_entities_from_evidence(content, concept=ctx.active_concept)
                if extracted:
                    ctx.active_entities = extracted
                    if not ctx.active_entity_type and extracted[0].entity_type:
                        ctx.active_entity_type = extracted[0].entity_type
                    ctx.last_evidence_text = content
                    ctx.last_answer = content
                    break

        # 3. Look for official URLs in history to preserve source continuity
        for turn in reversed(history):
            content = turn.get("content") or turn.get("text") if isinstance(turn, dict) else (getattr(turn, "content", None) or getattr(turn, "text", ""))
            if content:
                urls = re.findall(r"https?://(?:www\.)?(?:vvitu\.ac\.in|vvitguntur\.com)[^\s)\"'>]+", content)
                if urls:
                    ctx.active_source_urls = list(set(urls))
                    if any("vvitu.ac.in" in u for u in urls):
                        ctx.institutional_period = "current_vvitu"
                    elif any("vvitguntur.com" in u for u in urls):
                        ctx.institutional_period = "historical_vvit"
                    break

        return ctx

    def _detect_followup_official_url_context(self, message: str, history: Optional[list]) -> Optional[str]:
        """
        Generic institutional follow-up URL resolver: looks for official URLs or matching routes
        across any institutional domain in conversation history.
        """
        if not history:
            return None

        msg_clean = message.lower().strip()
        followup_cues = [
            "them", "they", "their", "these", "those", "him", "her", "it", "this", "that",
            "details about", "fetch details", "more details", "what are the", "who is the",
            "qualifications", "designation", "hod", "head of department", "who are they",
            "fees", "hostel", "transport", "bus", "schedule", "syllabus", "curriculum",
        ]
        has_followup_cue = any(cue in msg_clean for cue in followup_cues)
        if not has_followup_cue:
            return None

        official_url_pattern = re.compile(
            r'https?://(?:www\.)?(?:vvitu\.ac\.in|vvitguntur\.com)/[a-zA-Z0-9_/-]+'
        )

        for turn in reversed(history):
            content = ""
            if isinstance(turn, dict):
                content = turn.get("content", "") or ""
            elif hasattr(turn, "content"):
                content = getattr(turn, "content", "") or ""

            match = official_url_pattern.search(content)
            if match:
                resolved = match.group(0)
                logger.info("Resolved follow-up query '%s' to official URL from history: %s", message, resolved)
                return resolved

            if hasattr(self, "web_retriever") and self.web_retriever:
                routes = self.web_retriever._find_matching_routes(content)
                if routes:
                    logger.info("Resolved follow-up query '%s' to matching route: %s", message, routes[0]["url"])
                    return routes[0]["url"]

        return None

    def _detect_followup_faculty_context(self, message: str, history: Optional[list]) -> Optional[str]:
        """
        Detect if user message is an anaphoric follow-up (e.g. 'can u please fetch details about them',
        'who is the hod?', 'tell me more about them') referring to a faculty department
        previously discussed in conversation history.
        """
        if not history:
            return None

        msg_clean = message.lower().strip()
        followup_cues = [
            "them", "they", "their", "these", "those", "him", "her",
            "details about them", "fetch details", "more details",
            "qualifications", "designation", "who is the hod", "hod",
            "head of department", "who are they", "tell me about them",
            "give details", "fetch more", "what are their", "faculty list",
            "about them",
        ]
        has_followup_cue = any(cue in msg_clean for cue in followup_cues)
        if not has_followup_cue:
            return None

        faculty_url_pattern = re.compile(
            r'https?://(?:www\.)?(?:vvitu\.ac\.in|vvitguntur\.com)/admissions/faculty/([a-zA-Z0-9_-]+)'
        )

        for turn in reversed(history):
            content = ""
            if isinstance(turn, dict):
                content = turn.get("content", "") or ""
            elif hasattr(turn, "content"):
                content = getattr(turn, "content", "") or ""

            match = faculty_url_pattern.search(content)
            if match:
                slug = match.group(1)
                resolved = f"https://vvitu.ac.in/admissions/faculty/{slug}"
                logger.info("Resolved follow-up query '%s' to faculty URL from history: %s", message, resolved)
                return resolved

            if hasattr(self, "web_retriever") and self.web_retriever:
                routes = self.web_retriever._find_matching_routes(content)
                for r in routes:
                    if "/faculty/" in r.get("url", ""):
                        logger.info("Resolved follow-up query '%s' to department route: %s", message, r["url"])
                        return r["url"]

        return None

    def _build_grounded_evidence(
        self,
        query: str,
        results: List[OfficialWebResult],
    ) -> Tuple[str, str, List[str], List[Dict], str, str, str]:
        """
        Single unified evidence builder for official web retrieval results.
        Enforces grounded facts, structured entity extraction, and authoritative formatting directives.
        Returns:
            (context, formatting_directive, sources, structured_sources, confidence, intent, response_type)
        """
        sources = [r.url for r in results]
        structured_sources = [r.to_structured_source() for r in results]
        confidence = "High"

        has_faculty = any(
            r.entity_type in ("faculty", "faculty_directory") or
            len(getattr(r, "entities", [])) > 0 or
            "faculty" in r.url.lower()
            for r in results
        )

        all_direct_urls = all(r.retrieval_method == "direct_url_only" for r in results)

        context_parts = []
        total_faculty_entities = 0
        for r in results:
            entities = getattr(r, "entities", []) or []
            if entities and r.entity_type in ("faculty", "faculty_directory"):
                total_faculty_entities += len(entities)
            context_parts.append(r.content)

        context = "\n\n".join(context_parts)
        max_evidence_chars = getattr(self.settings, "max_context_chars", 9500)
        if len(context) > max_evidence_chars:
            context = context[:max_evidence_chars] + "\n\n[... content truncated to fit model token limit ...]"

        if has_faculty and (total_faculty_entities > 0 or not all_direct_urls):
            intent = "faculty_lookup"
            response_type = "tabular"
            directive = (
                "GROUNDED OFFICIAL FACULTY DIRECTIVE:\n"
                f"• The official institutional faculty directory was verified and loaded with verified faculty records.\n"
                "• Answer the user's question directly, factually, and completely using the verified faculty data above.\n"
                "• CITE actual faculty names, designations, qualifications, and official profile links directly from the table/list.\n"
                "• NEVER claim the faculty names, roles, or qualifications are unavailable or tell the user to manually browse the site; present the verified details directly.\n"
                "• Format the answer cleanly using a structured table or organized list with profile links."
            )
        elif all_direct_urls:
            intent = "general"
            response_type = "factual"
            directive = (
                "GROUNDED OFFICIAL ANSWER DIRECTIVE:\n"
                "• The relevant official institutional page was identified.\n"
                "• Provide the direct canonical URL and explain the official resources available at this destination.\n"
                "• Do NOT invent details that were not extracted."
            )
        else:
            intent = "general"
            response_type = "factual"
            directive = (
                "GROUNDED OFFICIAL ANSWER DIRECTIVE:\n"
                "• The above information was retrieved directly from the official institutional portal.\n"
                "• Answer the user's question directly, factually, and completely using this verified information.\n"
                "• Cite key details (e.g. designation, department, qualification, role, portal link).\n"
                "• NEVER tell the user to visit or search the website for information given above; provide the verified facts directly."
            )

        return context, directive, sources, structured_sources, confidence, intent, response_type

    def _is_rag_sufficient(self, query: str, qualified_chunks: List[RetrievedChunk]) -> bool:
        """
        Evaluate whether retrieved RAG chunks actually contain the specific named entity
        or core subject of the query.
        """
        if not qualified_chunks:
            return False

        q_lower = query.lower()
        name_indicators = [
            "dr.", "dr ", "prof.", "prof ", "mr.", "mr ", "faculty", "professor",
            "who is", "tell me about", "details of", "profile of", "members", "hod"
        ]
        has_name_indicator = any(ind in q_lower for ind in name_indicators)

        if has_name_indicator:
            stop_words = {
                "tell", "about", "who", "what", "where", "when", "how", "the", "for",
                "with", "from", "and", "our", "are", "you", "details", "give", "info",
                "information", "does", "profile", "vvit", "vvitu", "college", "university",
                "please", "know", "name", "can", "faculty", "professors", "teachers",
                "department", "school", "branch", "at", "in", "on", "of", "is", "to",
                "me", "us", "by", "as", "an", "a", "or", "so", "if", "be", "do", "we",
                "he", "it", "my", "any", "all", "some", "this", "that", "these", "those",
            }
            # Include 2+ char tokens to capture acronyms like ai, ds, cs, ce, me, it
            tokens = [
                w for w in re.sub(r"[^a-zA-Z0-9\s]", " ", q_lower).split()
                if len(w) >= 2 and w not in stop_words
            ]
            if tokens:
                combined_text = " ".join(c.content.lower() for c in qualified_chunks)
                found = any(bool(re.search(rf"\b{re.escape(t)}\b", combined_text)) for t in tokens)
                if not found:
                    logger.info("RAG insufficient: Query entity tokens %s not found in retrieved chunks", tokens)
                    return False

        return True

    @staticmethod
    def _extract_searched_entity_name(query: str) -> Optional[str]:
        """Extract person or specific entity name from query."""
        m = re.search(r'((?:dr\.?|prof\.?|mr\.?|mrs\.?|ms\.?)\s+[a-zA-Z\.\s]+)', query, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip('?.,')
        m = re.search(r'(?:tell me about|who is|details of|profile of|faculty)\s+([a-zA-Z\.\s]+)', query, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip().rstrip('?.,')
            if candidate.lower() not in {"vvit", "vvitu", "the college", "the campus", "the university", "admissions", "placements", "our college", "our campus"}:
                return candidate
        return None

    @staticmethod
    def _detect_query_period(query: str) -> str:
        """
        Determine whether a query is asking for CURRENT, HISTORICAL, or BOTH/GENERAL information.

        Returns:
            "current", "historical", or "both"
        """
        q = query.lower()
        historical_terms = {
            "old", "legacy", "historical", "history", "past", "former",
            "earlier", "previous", "was", "archive", "archived", "before transition",
        }
        current_terms = {
            "current", "currently", "now", "present", "latest", "today",
            "active", "new", "vvitu", "university",
        }

        has_historical = any(re.search(rf"\b{re.escape(w)}\b", q) for w in historical_terms)
        has_current = any(re.search(rf"\b{re.escape(w)}\b", q) for w in current_terms)

        if has_historical and has_current:
            return "both"
        if has_historical:
            return "historical"
        if has_current:
            return "current"

        if "vvitu" in q:
            return "current"

        return "both"

    def _get_fallback_sources_for_query(self, query: str) -> Tuple[List[str], List[Dict]]:
        """Return period-appropriate default sources when local index yields no chunks."""
        period = self._detect_query_period(query)
        curr_url = self.settings.current_university_url
        leg_url = self.settings.legacy_institute_url
        if period == "current":
            sources = [curr_url]
            structured = [{
                "title": "VVITU Official Website (Current)",
                "url": curr_url,
                "type": "website_current",
                "period_label": "VVITU Official Website — Current",
            }]
        elif period == "historical":
            sources = [leg_url]
            structured = [{
                "title": "VVIT Legacy Website (Historical)",
                "url": leg_url,
                "type": "website_historical",
                "period_label": "VVIT Legacy Website — Historical",
            }]
        else:
            sources = [curr_url, leg_url]
            structured = [
                {
                    "title": "VVITU Official Website (Current)",
                    "url": curr_url,
                    "type": "website_current",
                    "period_label": "VVITU Official Website — Current",
                },
                {
                    "title": "VVIT Legacy Website (Historical)",
                    "url": leg_url,
                    "type": "website_historical",
                    "period_label": "VVIT Legacy Website — Historical",
                },
            ]
        return sources, structured

    @classmethod
    def _handle_direct_institutional_query(cls, query: str) -> Optional[RAGResponse]:
        """
        Direct deterministic response handler for core identity, official website, and transition questions.
        Guarantees 100% accurate, authoritative institutional era routing with zero hallucination.
        """
        q = query.strip().lower().rstrip("?.! ")

        # 1. VAIT Identity Questions
        if cls._is_identity_query(query):
            reply = (
                "I am **VAIT** (**VVIT's Artificial Intelligence Technology**), the official "
                "institutional AI assistant for **VVIT University (VVITU)** and the legacy **Vasireddy Venkatadri Institute of Technology (VVIT)**.\n\n"
                "I can assist you with:\n"
                "• **Academic Regulations & Curriculum:** R20/R23 regulations, syllabus, course structures, and academic calendars.\n"
                "• **Admissions & Programs:** Degree offerings, eligibility, intake details, and fee structures.\n"
                "• **Examinations & Results:** Schedules, grading systems, hall tickets, and revaluation procedures.\n"
                "• **Campus & Facilities:** Infrastructure, hostel guidelines, transport routes, placements, and student activities.\n\n"
                "How can I help you today?"
            )
            return RAGResponse(
                reply=reply,
                sources=[],
                structured_sources=[],
                source_visibility="none",
                confidence="High",
                retrieval_score=1.0,
                is_refusal=False,
                intent="general",
                response_type="simple",
            )

        # 2. Current Official Website Questions
        current_site_patterns = [
            r"^what is the current official website\??$",
            r"^what is the official website\??$",
            r"^what is the current website\??$",
            r"^what is the official site\??$",
            r"^what is the official website of vvitu\??$",
            r"^what is the official website of vvit\??$",
            r"^what is the website of vvitu\??$",
            r"^what is vvitu('s)? website\??$",
            r"^current official website\??$",
            r"^official website\??$",
            r"^vvitu official website\??$",
            r"^current university website\??$",
            r"^what is the current university website\??$",
        ]
        if any(re.search(p, q, re.IGNORECASE) for p in current_site_patterns):
            reply = (
                "**Current Official Website**\n\n"
                "The current official website of **VVIT University (VVITU)** is:\n"
                "**https://vvitu.ac.in/**\n\n"
                "This is the primary and authoritative portal for all current university affairs, admissions, active academic calendars, examinations, and official notices."
            )
            return RAGResponse(
                reply=reply,
                sources=["https://vvitu.ac.in/"],
                structured_sources=[
                    {
                        "title": "VVITU Official Website (Current)",
                        "url": "https://vvitu.ac.in/",
                        "type": "website_current",
                        "period_label": "VVITU Official Website — Current",
                    },
                ],
                source_visibility="compact",
                confidence="High",
                retrieval_score=1.0,
                is_refusal=False,
                intent="general",
                response_type="factual",
            )

        # 3. Old / Legacy VVIT Website Questions
        legacy_site_patterns = [
            r"^what was the old vvit website\??$",
            r"^what was the old website\??$",
            r"^what is the old website\??$",
            r"^what is the old vvit website\??$",
            r"^what is the legacy website\??$",
            r"^what was the legacy website\??$",
            r"^what was the legacy vvit website\??$",
            r"^what is the legacy vvit website\??$",
            r"^old vvit website\??$",
            r"^legacy vvit website\??$",
            r"^old website\??$",
            r"^legacy website\??$",
        ]
        if any(re.search(p, q, re.IGNORECASE) for p in legacy_site_patterns):
            reply = (
                "**Legacy VVIT Website**\n\n"
                "The legacy website for **Vasireddy Venkatadri Institute of Technology (VVIT)** is:\n"
                "**https://vvitguntur.com/**\n\n"
                "This portal represents the older VVIT institute period and serves as a historical reference for older academic records and regulations. "
                "For current university information, please refer to the official VVITU portal at **https://vvitu.ac.in/**."
            )
            return RAGResponse(
                reply=reply,
                sources=["https://vvitguntur.com/"],
                structured_sources=[
                    {
                        "title": "VVIT Legacy Website (Historical)",
                        "url": "https://vvitguntur.com/",
                        "type": "website_historical",
                        "period_label": "VVIT Legacy Website — Historical",
                    },
                ],
                source_visibility="compact",
                confidence="High",
                retrieval_score=1.0,
                is_refusal=False,
                intent="general",
                response_type="factual",
            )

        # 4. Institutional Transition / Evolution Questions
        transition_patterns = [
            r"why did vvit become vvitu",
            r"how did vvit become vvitu",
            r"transition from vvit to vvitu",
            r"difference between vvit and vvitu",
            r"compare old vvit and current vvitu",
            r"compare vvit and vvitu",
            r"is vvit now vvitu",
            r"did vvit change to vvitu",
            r"vvit to vvitu transition",
            r"history of vvit and vvitu",
        ]
        if any(re.search(p, q, re.IGNORECASE) for p in transition_patterns):
            reply = (
                "**Institutional Evolution: VVIT to VVIT University (VVITU)**\n\n"
                "**Vasireddy Venkatadri Institute of Technology (VVIT)** was established in 2007 as an engineering college affiliated with JNTUK in Nambur, Guntur, later attaining autonomous status.\n\n"
                "Recognizing its academic excellence, infrastructure, and research achievements, the institution transitioned into a full-fledged State Private University named **VVIT University (VVITU)** under the Andhra Pradesh Private Universities Act.\n\n"
                "**Dual Source Reference:**\n"
                "• **Current University (VVITU):** https://vvitu.ac.in/ — Primary official source for current academic programs, degree awards, university administration, and ongoing regulations.\n"
                "• **Legacy Institute (VVIT):** https://vvitguntur.com/ — Historical reference for archival academic records, pre-transition regulations, and past institutional history."
            )
            return RAGResponse(
                reply=reply,
                sources=["https://vvitu.ac.in/", "https://vvitguntur.com/"],
                structured_sources=[
                    {
                        "title": "VVITU Official Website (Current)",
                        "url": "https://vvitu.ac.in/",
                        "type": "website_current",
                        "period_label": "VVITU Official Website — Current",
                    },
                    {
                        "title": "VVIT Legacy Website (Historical)",
                        "url": "https://vvitguntur.com/",
                        "type": "website_historical",
                        "period_label": "VVIT Legacy Website — Historical",
                    },
                ],
                source_visibility="full",
                confidence="High",
                retrieval_score=1.0,
                is_refusal=False,
                intent="general",
                response_type="current_historical",
            )

        return None

    @staticmethod
    def _is_dynamic_query(query: str) -> bool:
        """Return True when a query asks for latest/current updates."""
        q = query.lower()
        return any(kw in q for kw in DYNAMIC_QUERY_KEYWORDS)

    @staticmethod
    def _infer_source_tier(metadata: Dict) -> str:
        """
        Infer source tier when explicit metadata is missing.

        Priority order:
          1. primary_official_current (current university website: vvitu.ac.in)
          2. legacy_official_vvit (legacy institute website: vvitguntur.com)
          3. primary_official (official documents: regulations/syllabi)
          4. secondary_linkedin (LinkedIn sources)
          5. tertiary_social (Instagram/Twitter/Facebook/YouTube and other social)
          6. related_web (extended web sources)
        """
        explicit = str(metadata.get("source_tier", "")).strip().lower()
        if explicit in SOURCE_TIER_BOOST:
            return explicit

        url = str(metadata.get("url", "")).lower()
        source_type = str(metadata.get("source_type", "")).lower()
        document_type = str(metadata.get("document_type", "")).lower()
        platform = str(metadata.get("platform", "")).lower()

        if any(domain in url for domain in CURRENT_UNIVERSITY_DOMAINS):
            return "primary_official_current"

        if any(domain in url for domain in LEGACY_INSTITUTE_DOMAINS):
            return "legacy_official_vvit"

        if any(domain in url for domain in OFFICIAL_DOMAINS):
            return "primary_official"

        if document_type in {"regulation", "syllabus", "notice", "official_website"}:
            return "primary_official"

        if "linkedin.com" in url or platform == "linkedin":
            return "secondary_linkedin"

        if source_type == "social_media" or platform in {
            "instagram", "twitter", "x", "facebook", "youtube"
        }:
            return "tertiary_social"

        return "related_web"

    @staticmethod
    def _recency_boost_for_social(metadata: Dict) -> float:
        """Add a freshness boost for recent social/linkedin posts."""
        raw_date = str(metadata.get("date", "")).strip()
        if not raw_date:
            return 0.0

        parsed_date = None
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y"):
            try:
                parsed_date = datetime.datetime.strptime(raw_date, fmt).date()
                break
            except ValueError:
                continue

        if parsed_date is None:
            return 0.0

        age_days = (datetime.date.today() - parsed_date).days
        if age_days < 0:
            return 0.0
        if age_days <= RECENT_SOCIAL_DAYS:
            return 0.03
        if age_days <= 120:
            return 0.01
        return 0.0

    async def process_query(self, message: str, history: list = None, context_state: dict = None, conversation_id: str = None) -> RAGResponse:
        """
        Process a user query through the strict RAG pipeline.

        Flow:
        1. Validate input
        2. Check response cache (LRU + TTL)
        3. Classify intent (rule-based)
        4. Embed question       (timed → embed_ms)
        5. Search FAISS (Top-K) (timed → retrieval_ms)
        6. Apply similarity threshold (0.65)
        7. Remove duplicate source chunks
        8. Authority-weighted adjusted scoring + intent boost + sort
        9. Cap context length
        10. Build grounded prompt
        11. Call LLM              (timed → generation_ms)
        12. Polish response (clean spacing, dedup sentences, word cap)
        13. Format response + confidence + sources + performance
        14. Store in cache
        """
        t_start = time.perf_counter()

        # Normalize query for better embedding matches
        message = self._normalize_query(message)
        logger.info("Query: %s", message)

        if not message or not message.strip():
            return self._make_error_response(
                "Please provide a question to proceed."
            )

        if len(message) > MAX_QUERY_LENGTH:
            return self._make_error_response(
                "Your query exceeds the maximum length. "
                "Please shorten it and try again."
            )

        # ── Check cache ──────────────────────────────────────────────
        cached = self._cache_get(message)
        if cached is not None:
            # Inject cache indicator into performance
            if cached.performance:
                cached.performance["cache_hit"] = True
            return cached

        # ── Conversational / Greeting Handler ────────────────────────
        conv_reply = self._is_conversational_query(message)
        if conv_reply is not None:
            resp = RAGResponse(
                reply=conv_reply,
                sources=[],
                structured_sources=[],
                source_visibility="none",
                confidence="High",
                retrieval_score=1.0,
                is_refusal=False,
                intent="general",
                response_type="simple",
            )
            self._cache_put(message, resp)
            return resp

        # ── Direct Institutional / Website / Identity Handler ────────
        direct_resp = self._handle_direct_institutional_query(message)
        if direct_resp is not None:
            self._cache_put(message, direct_resp)
            return direct_resp

        # ── Context Resolution & Anaphora Handling ────────────────────
        current_context = self._get_or_rehydrate_context(history, context_state, conversation_id)
        followup_intent = self.context_resolver.resolve_turn(
            current_query=message,
            history=history,
            context=current_context,
        )
        logger.info(
            "Context Resolver: Intent=%s | Resolved='%s' | NeedsDeep=%s | TargetURLs=%d",
            followup_intent.intent_type,
            followup_intent.resolved_query,
            followup_intent.needs_deeper_retrieval,
            len(followup_intent.target_profile_urls),
        )

        # A0. Ambiguous Reference Clarification
        if followup_intent.intent_type == IntentType.AMBIGUOUS_CLARIFICATION:
            clarification_text = followup_intent.clarification_prompt or "Could you please clarify which entity or topic you are referring to?"
            current_context.last_answer = clarification_text
            current_context.last_query = message
            current_context.last_updated_ist = format_ist()
            if conversation_id:
                self._conversation_contexts[conversation_id] = current_context
            return RAGResponse(
                reply=clarification_text,
                sources=[],
                structured_sources=[],
                source_visibility="none",
                confidence="High",
                retrieval_score=1.0,
                is_refusal=False,
                intent="clarification",
                response_type="simple",
                context_state=current_context.model_dump(),
            )

        # A. Evidence Reuse or Single Entity Filter
        if followup_intent.intent_type in (IntentType.FOLLOWUP_EVIDENCE_REUSE, IntentType.FOLLOWUP_ENTITY_FILTER) and followup_intent.reusable_evidence:
            logger.info("Resolving query via conversational evidence reuse: %s", followup_intent.intent_type)
            context = followup_intent.reusable_evidence
            sources = current_context.active_source_urls or [self.settings.current_university_url]
            concept_label = (current_context.active_concept or current_context.active_entity_type or "institutional").capitalize()
            subject_label = (current_context.active_subject or "VVITU").title()
            title_text = f"VVITU Official {subject_label} {concept_label} Records" if current_context.active_subject else f"VVITU Official {concept_label} Directory & Records"
            structured_sources = [
                {
                    "title": title_text,
                    "url": u,
                    "type": "website_current",
                    "period_label": "VVITU Official Portal — Grounded Record",
                }
                for u in sources
            ]
            directive = (
                "GROUNDED CONVERSATIONAL FOLLOW-UP DIRECTIVE:\n"
                "• The user is asking a follow-up question referring to entities verified in earlier turns.\n"
                "• Answer directly, concisely, and factually using the verified details in the context above.\n"
                "• Cite specific names, designations, qualifications, roles, fees, or links given in context.\n"
                "• NEVER reset to generic greeting, and NEVER claim the information is unavailable if present in context."
            )
            final_prompt = self._build_generation_prompt(
                context=context,
                query=message,
                formatting_instructions=directive,
            )
            t_gen_start = time.perf_counter()
            response_text = self.llm_service.generate(
                system_prompt=self.system_prompt,
                user_prompt=final_prompt,
                history=history,
            )
            t_gen_end = time.perf_counter()
            response_text, confidence = self._hallucination_guard(
                response_text, "High", sources, is_institutional=True,
            )
            response_text = self._polish_response(response_text)

            current_context.last_answer = response_text
            current_context.last_query = message
            current_context.last_updated_ist = format_ist()
            if conversation_id:
                self._conversation_contexts[conversation_id] = current_context

            intent_val = f"{current_context.active_concept}_lookup" if current_context.active_concept else ("faculty_lookup" if current_context.active_entity_type == "faculty" else "academic")
            resp = RAGResponse(
                reply=response_text,
                sources=sources,
                structured_sources=structured_sources,
                source_visibility="compact",
                confidence=confidence,
                retrieval_score=0.98,
                is_refusal=False,
                intent=intent_val,
                response_type="factual",
                performance={
                    "retrieval_ms": 0,
                    "generation_ms": int((t_gen_end - t_gen_start) * 1000),
                    "total_ms": int((time.perf_counter() - t_start) * 1000),
                    "cache_hit": False,
                },
                context_state=current_context.model_dump(),
            )
            return resp

        # B. Deeper Profile Retrieval
        if followup_intent.intent_type == IntentType.FOLLOWUP_DEEPER_RETRIEVAL and followup_intent.target_profile_urls:
            logger.info("Executing deeper profile retrieval for %d URLs: %s",
                        len(followup_intent.target_profile_urls), followup_intent.target_profile_urls)
            deeper_results: List[OfficialWebResult] = []
            max_crawl = getattr(self.settings, "max_deeper_profiles_fetch", 3)
            # Parallel subpage retrieval
            crawl_tasks = [self.web_retriever.retrieve_url(purl, query=message) for purl in followup_intent.target_profile_urls[:max_crawl]]
            crawl_res = await asyncio.gather(*crawl_tasks, return_exceptions=True)
            for p_res in crawl_res:
                if isinstance(p_res, OfficialWebResult) and p_res.retrieval_method != "failed":
                    deeper_results.append(p_res)

            if deeper_results:
                (
                    context,
                    official_directive,
                    sources,
                    structured_sources,
                    confidence,
                    intent_val,
                    response_type_val,
                ) = self._build_grounded_evidence(message, deeper_results)

                if current_context.last_evidence_text:
                    prev_summary = current_context.last_evidence_text[:1200]
                    context = f"=== PREVIOUS TURN SUMMARY ===\n{prev_summary}\n\n=== DEEP PROFILE PAGES RETRIEVED ===\n{context}"

                final_prompt = self._build_generation_prompt(
                    context=context,
                    query=message,
                    formatting_instructions=official_directive,
                )
                t_gen_start = time.perf_counter()
                response_text = self.llm_service.generate(
                    system_prompt=self.system_prompt,
                    user_prompt=final_prompt,
                    history=history,
                )
                t_gen_end = time.perf_counter()
                response_text, confidence = self._hallucination_guard(
                    response_text, confidence, sources, is_institutional=True,
                )
                response_text = self._polish_response(response_text)

                current_context.last_answer = response_text
                current_context.last_query = message
                current_context.last_evidence_text = context
                current_context.last_updated_ist = format_ist()
                for d_res in deeper_results:
                    if d_res.url not in current_context.active_source_urls:
                        current_context.active_source_urls.append(d_res.url)
                if conversation_id:
                    self._conversation_contexts[conversation_id] = current_context

                resp = RAGResponse(
                    reply=response_text,
                    sources=sources,
                    structured_sources=structured_sources,
                    source_visibility="full" if len(structured_sources) > 1 else "compact",
                    confidence=confidence,
                    retrieval_score=0.98,
                    is_refusal=False,
                    intent=intent_val,
                    response_type=response_type_val,
                    performance={
                        "retrieval_ms": int((time.perf_counter() - t_start) * 1000),
                        "generation_ms": int((t_gen_end - t_gen_start) * 1000),
                        "total_ms": int((time.perf_counter() - t_start) * 1000),
                        "cache_hit": False,
                    },
                    context_state=current_context.model_dump(),
                )
                return resp

        # C. Topic Expansion / Concept Transition
        if followup_intent.intent_type == IntentType.FOLLOWUP_TOPIC_EXPANSION:
            message = followup_intent.resolved_query
            current_context.active_entities = []
            current_context.last_evidence_text = None
            current_context.active_subject = followup_intent.resolved_query
            if followup_intent.active_concept:
                current_context.active_concept = followup_intent.active_concept

        if followup_intent.intent_type == IntentType.FOLLOWUP_CONCEPT_TRANSITION:
            logger.info("Follow-up concept transition detected: '%s' -> resolved='%s'", message, followup_intent.resolved_query)
            message = followup_intent.resolved_query
            current_context.active_entities = []
            current_context.last_evidence_text = None
            if followup_intent.active_subject:
                current_context.active_subject = followup_intent.active_subject
            if followup_intent.active_concept:
                current_context.active_concept = followup_intent.active_concept
                current_context.active_entity_type = followup_intent.active_concept

        # ── Direct Official URL Handler ──────────────────────────────
        explicit_urls = self._extract_official_urls(message)
        if not explicit_urls and history:
            followup_url = self._detect_followup_faculty_context(message, history)
            if followup_url:
                logger.info("Direct official URL resolved from follow-up context: %s", followup_url)
                explicit_urls = [followup_url]

        if explicit_urls:
            logger.info("Direct official URL detected in query: %s", explicit_urls)
            url_results: List[OfficialWebResult] = []
            for u in explicit_urls[:2]:
                u_res = await self.web_retriever.retrieve_url(u, query=message)
                if u_res.retrieval_method != "failed":
                    url_results.append(u_res)

            if url_results:
                (
                    context,
                    official_directive,
                    sources,
                    structured_sources,
                    confidence,
                    intent_val,
                    response_type_val,
                ) = self._build_grounded_evidence(message, url_results)
                source_visibility = "compact" if len(structured_sources) <= 1 else "full"

                final_prompt = self._build_generation_prompt(
                    context=context,
                    query=message,
                    formatting_instructions=f"{official_directive}",
                )

                t_gen_start = time.perf_counter()
                response_text = self.llm_service.generate(
                    system_prompt=self.system_prompt,
                    user_prompt=final_prompt,
                    history=history,
                )
                t_gen_end = time.perf_counter()
                response_text, confidence = self._hallucination_guard(
                    response_text, confidence, sources, is_institutional=True,
                )
                response_text = self._polish_response(response_text)
                resp = RAGResponse(
                    reply=response_text,
                    sources=sources,
                    structured_sources=structured_sources,
                    source_visibility=source_visibility,
                    confidence=confidence,
                    retrieval_score=0.98,
                    is_refusal=False,
                    intent=intent_val,
                    response_type=response_type_val,
                    performance={
                        "retrieval_ms": int((time.perf_counter() - t_start) * 1000),
                        "generation_ms": int((t_gen_end - t_gen_start) * 1000),
                        "total_ms": int((time.perf_counter() - t_start) * 1000),
                        "cache_hit": False,
                    },
                )
                # Update conversational context state with discovered entities
                extracted_entities = []
                for r in url_results:
                    ext = self.context_resolver.extract_entities_from_evidence(
                        r.content, [r.url], entity_type=r.entity_type, raw_entities=getattr(r, "entities", None)
                    )
                    extracted_entities.extend(ext)
                if extracted_entities:
                    current_context.active_entities = extracted_entities
                    current_context.active_entity_type = extracted_entities[0].entity_type
                    current_context.active_subject = message
                current_context.active_source_urls = sources
                current_context.last_evidence_text = context
                current_context.last_answer = response_text
                current_context.last_query = message
                current_context.last_updated_ist = format_ist()
                if conversation_id:
                    self._conversation_contexts[conversation_id] = current_context

                resp.context_state = current_context.model_dump()
                self._cache_put(message, resp)
                return resp

        # ── Classify intent & Plan response format ───────────────────
        intent, matched_keywords = self.classify_intent(message)
        logger.debug("Intent: %s (keywords: %s)", intent, matched_keywords)
        query_period = self._detect_query_period(message)
        is_institutional = self._is_institutional_query(message) or (followup_intent and followup_intent.is_institutional)
        response_plan = response_planner.plan_response(message, intent=intent, period=query_period)

        # ── Check index readiness ────────────────────────────────────
        retrieval_empty = False
        t_embed_start = t_embed_end = t_retrieval_start = t_retrieval_end = time.perf_counter()
        retrieved_chunks = []
        retrieval_scores = []
        
        if self.index is None or self.index.ntotal == 0:
            await self._bootstrap_official_sites_if_empty()

        if self.index is None or self.index.ntotal == 0:
            logger.info("Knowledge base is empty. Falling back to LLM-only mode.")
            retrieval_empty = True
        else:
            # ── Step 1: Retrieve relevant chunks from FAISS ──────────────
            try:
                t_embed_start = time.perf_counter()
                query_embedding = await self.embedding_service.get_embedding(message)
                t_embed_end = time.perf_counter()

                t_retrieval_start = time.perf_counter()
                retrieved_chunks = self._search_index(query_embedding)
                t_retrieval_end = time.perf_counter()
            except Exception as exc:
                logger.error("Retrieval failed: %s. Falling back to LLM-only mode.", exc)
                retrieval_empty = True

        if not retrieval_empty:
            retrieval_scores = [c.similarity_score for c in retrieved_chunks]
            if not retrieved_chunks:
                logger.info("No documents retrieved. Falling back to LLM-only mode.")
                retrieval_empty = True

        # ── Step 3: Filter, dedup, authority-adjust, sort ────────────
        qualified = []
        top_adjusted = 0.0
        if not retrieval_empty:
            qualified = [
                c for c in retrieved_chunks
                if c.similarity_score >= self.settings.similarity_threshold
            ]
            qualified = self._deduplicate_chunks(qualified)
            self._apply_authority_multiplier(qualified, intent=intent, query=message)
            qualified = self._sort_by_adjusted_score(qualified)

            top_adjusted = qualified[0].adjusted_score if qualified else 0.0
            if top_adjusted < REFUSAL_ADJUSTED_THRESHOLD or not qualified:
                logger.info("Low similarity score (top adjusted: %.2f). Falling back to LLM-only mode.", top_adjusted)
                retrieval_empty = True
                qualified = []

        # Check if RAG chunks actually contain the query entity
        if not retrieval_empty and not self._is_rag_sufficient(message, qualified):
            logger.info("RAG context is insufficient for query entity; falling back to official web retrieval")
            retrieval_empty = True
            qualified = []

        # ── Step 4: Official Web Retrieval Fallback (if RAG is empty/insufficient) ──
        official_web_results: List[OfficialWebResult] = []
        if retrieval_empty and is_institutional:
            try:
                official_web_results = await self.web_retriever.retrieve(message, period=query_period)
            except Exception as exc:
                logger.warning("Official web fallback retrieval failed: %s", exc)

        if official_web_results:
            (
                context,
                official_directive,
                sources,
                structured_sources,
                confidence,
                intent_val,
                response_type_val,
            ) = self._build_grounded_evidence(message, official_web_results)
            source_visibility = "compact" if len(structured_sources) <= 1 else "full"
            top_adjusted = 0.95
            intent = intent_val
            response_plan.response_type = response_type_val

            final_prompt = self._build_generation_prompt(
                context=context,
                query=message,
                formatting_instructions=f"{response_plan.formatting_instructions}\n\n{official_directive}",
            )
        elif not retrieval_empty:
            # 1. Remove near-duplicate chunks
            qualified = self._remove_near_duplicates(qualified)
            # 2. Merge adjacent chunks from same source
            qualified = self._merge_adjacent_chunks(qualified)
            # 3. Already sorted by adjusted_score
            # 4. Limit to top MAX_CONTEXT_CHUNKS
            qualified = qualified[:MAX_CONTEXT_CHUNKS]

            # Build context and sources
            context = self._build_context(qualified)
            sources = list(dict.fromkeys(c.document_name for c in qualified))
            structured_sources = self._build_structured_sources(qualified)
            source_visibility = "compact" if len(structured_sources) <= 1 else "full"
            final_prompt = self._build_generation_prompt(
                context=context,
                query=message,
                formatting_instructions=response_plan.formatting_instructions,
            )

            # ── Compute confidence (adjusted_score-based) ────────────────
            confidence = self._compute_confidence_adjusted(top_adjusted)
        else:
            # Check if query was searching for a specific named entity/person that does not exist
            entity_name = self._extract_searched_entity_name(message)
            if entity_name and is_institutional:
                reply = (
                    f"Based on official records across the VVITU official portal (https://vvitu.ac.in/) "
                    f"and the legacy VVIT portal (https://vvitguntur.com/), no official record was found for **{entity_name}**.\n\n"
                    f"If you are looking for faculty or staff in a specific school or department (such as CSE, AI, ECE, EEE, Civil, Mechanical, or Business Administration), "
                    f"please let me know and I will be happy to look up the official faculty directory for you."
                )
                sources, structured_sources = self._get_fallback_sources_for_query(message)
                resp = RAGResponse(
                    reply=reply,
                    sources=sources,
                    structured_sources=structured_sources,
                    source_visibility="compact",
                    confidence="High",
                    retrieval_score=1.0,
                    is_refusal=False,
                    intent="faculty_lookup",
                    response_type="factual",
                )
                self._cache_put(message, resp)
                return resp

            context = ""
            if is_institutional:
                sources, structured_sources = self._get_fallback_sources_for_query(message)
                source_visibility = "compact" if len(structured_sources) <= 1 else "full"
                confidence = "Low"
            else:
                sources, structured_sources = [], []
                source_visibility = "none"
                confidence = "High"

            final_prompt = self._build_llm_only_prompt(
                query=message,
                formatting_instructions=response_plan.formatting_instructions,
                is_institutional=is_institutional,
            )

        # ── Generate response ────────────────────────────────────────
        try:
            t_gen_start = time.perf_counter()
            response_text = self.llm_service.generate(
                system_prompt=self.system_prompt,
                user_prompt=final_prompt,
                history=history,
            )
            t_gen_end = time.perf_counter()
        except MemoryError:
            logger.error("LLM out of memory — model too large for available RAM")
            return RAGResponse(
                reply=(
                    "The local model exceeded available memory. "
                    "Please close other applications or use a smaller model."
                ),
                sources=[],
                confidence="Low",
                retrieval_score=0.0,
                is_refusal=True,
                response_type="no_answer",
                source_visibility="none",
            )
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            return self._make_error_response(
                "An error occurred while generating the response. "
                "Please try again later."
            )

        # ── HALLUCINATION DEFENSE ────────────────────────────────────
        response_text, confidence = self._hallucination_guard(
            response_text, confidence, sources, is_institutional=is_institutional,
        )

        # ── RESPONSE POLISH MODE ─────────────────────────────────────
        response_text = self._polish_response(response_text)

        # ── CITATION FORMATTER — append sources block for rich answers ──
        if response_plan.response_type not in {"simple"}:
            response_text = self._append_citations(
                response_text, structured_sources, source_visibility=source_visibility,
            )

        t_total = time.perf_counter() - t_start

        # ── Build performance dict ───────────────────────────────────
        perf = {
            "embed_ms": round((t_embed_end - t_embed_start) * 1000, 1),
            "retrieval_ms": round((t_retrieval_end - t_retrieval_start) * 1000, 1),
            "generation_ms": round((t_gen_end - t_gen_start) * 1000, 1),
            "total_ms": round(t_total * 1000, 1),
            "intent": intent,
            "intent_keywords": matched_keywords,
            "cache_hit": False,
        }

        # ── Format response ──────────────────────────────────────────
        top_score = max(retrieval_scores) if retrieval_scores else 0.0
        formatted = self._format_response(
            text=response_text,
            sources=sources,
            structured_sources=structured_sources,
            confidence=confidence,
            top_score=top_score,
            retrieval_scores=retrieval_scores,
            performance=perf,
            intent=intent,
            response_type=response_plan.response_type,
            source_visibility=source_visibility,
        )

        # ── Store in cache ───────────────────────────────────────────
        self._cache_put(message, formatted)

        # ── Log & return ─────────────────────────────────────────────
        self._log_query(
            message=message,
            confidence=confidence,
            sources=sources,
            is_refusal=False,
            retrieval_scores=retrieval_scores,
            intent=intent,
            top_adjusted_score=top_adjusted,
        )

        current_context.last_answer = response_text
        current_context.last_query = message
        current_context.last_evidence_text = context
        current_context.active_source_urls = sources
        detected_concept = self.context_resolver.detect_concept(message)
        if detected_concept:
            current_context.active_concept = detected_concept
            current_context.active_entity_type = detected_concept
        detected_subject = self.context_resolver.detect_subject(message)
        if detected_subject:
            current_context.active_subject = detected_subject
        detected_filters = self.context_resolver.detect_filters(message)
        if detected_filters:
            current_context.active_filters.update(detected_filters)
        current_context.last_updated_ist = format_ist()
        if context:
            extracted_entities = self.context_resolver.extract_entities_from_evidence(
                context, sources, concept=current_context.active_concept
            )
            if extracted_entities:
                current_context.active_entities = extracted_entities
        if conversation_id:
            self._conversation_contexts[conversation_id] = current_context

        formatted.context_state = current_context.model_dump()
        return formatted

    async def process_query_stream(self, message: str, history: list = None, context_state: dict = None, conversation_id: str = None):
        """
        Stream a user query through the strict RAG pipeline using SSE format.
        Yields JSON strings prefixed with 'data: '.
        """
        import json
        t_start = time.perf_counter()

        message = self._normalize_query(message)
        logger.info("Stream Query: %s", message)

        if not message or not message.strip():
            yield f'data: {json.dumps({"type": "error", "error": "Please provide a question to proceed."})}\n\n'
            return

        if len(message) > MAX_QUERY_LENGTH:
            yield f'data: {json.dumps({"type": "error", "error": "Your query exceeds the maximum length."})}\n\n'
            return

        # ── Conversational / Greeting Handler ────────────────────────
        conv_reply = self._is_conversational_query(message)
        if conv_reply is not None:
            yield f'data: {json.dumps({"type": "metadata", "sources": [], "structured_sources": [], "source_visibility": "none", "confidence": "High", "retrieval_score": 1.0, "intent": "general", "response_type": "simple"})}\n\n'
            yield f'data: {json.dumps({"type": "content", "content": conv_reply})}\n\n'
            yield f'data: {json.dumps({"type": "done", "reply": conv_reply, "sources": [], "structured_sources": [], "source_visibility": "none", "confidence": "High", "response_type": "simple"})}\n\n'
            return

        # ── Direct Institutional / Website / Identity Handler ────────
        direct_resp = self._handle_direct_institutional_query(message)
        if direct_resp is not None:
            yield f'data: {json.dumps({"type": "metadata", "sources": direct_resp.sources, "structured_sources": direct_resp.structured_sources, "source_visibility": direct_resp.source_visibility, "confidence": direct_resp.confidence, "retrieval_score": 1.0, "intent": direct_resp.intent, "response_type": direct_resp.response_type})}\n\n'
            yield f'data: {json.dumps({"type": "content", "content": direct_resp.reply})}\n\n'
            yield f'data: {json.dumps({"type": "done", "reply": direct_resp.reply, "sources": direct_resp.sources, "structured_sources": direct_resp.structured_sources, "source_visibility": direct_resp.source_visibility, "confidence": direct_resp.confidence, "response_type": direct_resp.response_type})}\n\n'
            return

        # ── Context Resolution & Anaphora Handling (Streaming) ────────
        current_context = self._get_or_rehydrate_context(history, context_state, conversation_id)
        followup_intent = self.context_resolver.resolve_turn(
            current_query=message,
            history=history,
            context=current_context,
        )
        logger.info(
            "Stream Context Resolver: Intent=%s | Resolved='%s' | NeedsDeep=%s | TargetURLs=%d",
            followup_intent.intent_type,
            followup_intent.resolved_query,
            followup_intent.needs_deeper_retrieval,
            len(followup_intent.target_profile_urls),
        )

        # A0. Ambiguous Reference Clarification (Streaming)
        if followup_intent.intent_type == IntentType.AMBIGUOUS_CLARIFICATION:
            clarification_text = followup_intent.clarification_prompt or "Could you please clarify which entity or topic you are referring to?"
            current_context.last_answer = clarification_text
            current_context.last_query = message
            current_context.last_updated_ist = format_ist()
            if conversation_id:
                self._conversation_contexts[conversation_id] = current_context
            yield f'data: {json.dumps({"type": "metadata", "sources": [], "structured_sources": [], "source_visibility": "none", "confidence": "High", "retrieval_score": 1.0, "intent": "clarification", "response_type": "simple", "context_state": current_context.model_dump()})}\n\n'
            yield f'data: {json.dumps({"type": "content", "content": clarification_text})}\n\n'
            yield f'data: {json.dumps({"type": "done", "reply": clarification_text, "sources": [], "structured_sources": [], "source_visibility": "none", "confidence": "High", "response_type": "simple", "context_state": current_context.model_dump()})}\n\n'
            return

        # A. Evidence Reuse or Single Entity Filter (Streaming)
        if followup_intent.intent_type in (IntentType.FOLLOWUP_EVIDENCE_REUSE, IntentType.FOLLOWUP_ENTITY_FILTER) and followup_intent.reusable_evidence:
            logger.info("Stream resolving query via conversational evidence reuse: %s", followup_intent.intent_type)
            context = followup_intent.reusable_evidence
            sources = current_context.active_source_urls or [self.settings.current_university_url]
            concept_label = (current_context.active_concept or current_context.active_entity_type or "institutional").capitalize()
            subject_label = (current_context.active_subject or "VVITU").title()
            title_text = f"VVITU Official {subject_label} {concept_label} Records" if current_context.active_subject else f"VVITU Official {concept_label} Directory & Records"
            structured_sources = [
                {
                    "title": title_text,
                    "url": u,
                    "type": "website_current",
                    "period_label": "VVITU Official Portal — Grounded Record",
                }
                for u in sources
            ]
            directive = (
                "GROUNDED CONVERSATIONAL FOLLOW-UP DIRECTIVE:\n"
                "• The user is asking a follow-up question referring to entities verified in earlier turns.\n"
                "• Answer directly, concisely, and factually using the verified details in the context above.\n"
                "• Cite specific names, designations, qualifications, roles, fees, or links given in context.\n"
                "• NEVER reset to generic greeting, and NEVER claim the information is unavailable if present in context."
            )
            final_prompt = self._build_generation_prompt(
                context=context,
                query=message,
                formatting_instructions=directive,
            )

            intent_val = f"{current_context.active_concept}_lookup" if current_context.active_concept else ("faculty_lookup" if current_context.active_entity_type == "faculty" else "academic")
            yield f'data: {json.dumps({"type": "metadata", "sources": sources, "structured_sources": structured_sources, "source_visibility": "compact", "confidence": "High", "retrieval_score": 0.98, "intent": intent_val, "response_type": "factual", "context_state": current_context.model_dump()})}\n\n'

            accumulated_chunks = []
            try:
                async for text_chunk in self.llm_service.generate_stream(
                    system_prompt=self.system_prompt,
                    user_prompt=final_prompt,
                    history=history,
                ):
                    accumulated_chunks.append(text_chunk)
                    yield f'data: {json.dumps({"type": "content", "content": text_chunk})}\n\n'
            except Exception as exc:
                logger.error("LLM streaming failed for evidence reuse: %s", exc)
                fallback_reply = self.llm_service.generate(
                    system_prompt=self.system_prompt,
                    user_prompt=final_prompt,
                    history=history,
                )
                accumulated_chunks = [fallback_reply]
                yield f'data: {json.dumps({"type": "content", "content": fallback_reply})}\n\n'

            complete_reply = "".join(accumulated_chunks)
            polished_reply = self._polish_response(complete_reply)

            current_context.last_answer = polished_reply
            current_context.last_query = message
            current_context.last_updated_ist = format_ist()
            if conversation_id:
                self._conversation_contexts[conversation_id] = current_context

            yield f'data: {json.dumps({"type": "done", "reply": polished_reply, "sources": sources, "structured_sources": structured_sources, "source_visibility": "compact", "confidence": "High", "response_type": "factual", "context_state": current_context.model_dump()})}\n\n'
            return

        # B. Deeper Profile Retrieval (Streaming)
        if followup_intent.intent_type == IntentType.FOLLOWUP_DEEPER_RETRIEVAL and followup_intent.target_profile_urls:
            logger.info("Stream executing deeper profile retrieval for %d URLs: %s",
                        len(followup_intent.target_profile_urls), followup_intent.target_profile_urls)
            deeper_results: List[OfficialWebResult] = []
            max_crawl = getattr(self.settings, "max_deeper_profiles_fetch", 3)
            # Parallel subpage retrieval
            crawl_tasks = [self.web_retriever.retrieve_url(purl, query=message) for purl in followup_intent.target_profile_urls[:max_crawl]]
            crawl_res = await asyncio.gather(*crawl_tasks, return_exceptions=True)
            for p_res in crawl_res:
                if isinstance(p_res, OfficialWebResult) and p_res.retrieval_method != "failed":
                    deeper_results.append(p_res)

            if deeper_results:
                (
                    context,
                    official_directive,
                    sources,
                    structured_sources,
                    confidence,
                    intent_val,
                    response_type_val,
                ) = self._build_grounded_evidence(message, deeper_results)

                if current_context.last_evidence_text:
                    prev_summary = current_context.last_evidence_text[:1200]
                    context = f"=== PREVIOUS TURN SUMMARY ===\n{prev_summary}\n\n=== DEEP PROFILE PAGES RETRIEVED ===\n{context}"

                final_prompt = self._build_generation_prompt(
                    context=context,
                    query=message,
                    formatting_instructions=official_directive,
                )

                yield f'data: {json.dumps({"type": "metadata", "sources": sources, "structured_sources": structured_sources, "source_visibility": "full" if len(structured_sources) > 1 else "compact", "confidence": confidence, "retrieval_score": 0.98, "intent": intent_val, "response_type": response_type_val, "context_state": current_context.model_dump()})}\n\n'

                accumulated_chunks = []
                try:
                    async for text_chunk in self.llm_service.generate_stream(
                        system_prompt=self.system_prompt,
                        user_prompt=final_prompt,
                        history=history,
                    ):
                        accumulated_chunks.append(text_chunk)
                        yield f'data: {json.dumps({"type": "content", "content": text_chunk})}\n\n'
                except Exception as exc:
                    logger.error("LLM streaming failed for deep profiles: %s", exc)
                    fallback_reply = self.llm_service.generate(
                        system_prompt=self.system_prompt,
                        user_prompt=final_prompt,
                        history=history,
                    )
                    accumulated_chunks = [fallback_reply]
                    yield f'data: {json.dumps({"type": "content", "content": fallback_reply})}\n\n'

                complete_reply = "".join(accumulated_chunks)
                polished_reply = self._polish_response(complete_reply)

                current_context.last_answer = polished_reply
                current_context.last_query = message
                current_context.last_evidence_text = context
                current_context.last_updated_ist = format_ist()
                for d_res in deeper_results:
                    if d_res.url not in current_context.active_source_urls:
                        current_context.active_source_urls.append(d_res.url)
                if conversation_id:
                    self._conversation_contexts[conversation_id] = current_context

                yield f'data: {json.dumps({"type": "done", "reply": polished_reply, "sources": sources, "structured_sources": structured_sources, "source_visibility": "full" if len(structured_sources) > 1 else "compact", "confidence": confidence, "response_type": response_type_val, "context_state": current_context.model_dump()})}\n\n'
                return

        # C. Topic Expansion / Concept Transition (Streaming)
        if followup_intent.intent_type == IntentType.FOLLOWUP_TOPIC_EXPANSION:
            message = followup_intent.resolved_query
            current_context.active_entities = []
            current_context.last_evidence_text = None
            current_context.active_subject = followup_intent.resolved_query
            if followup_intent.active_concept:
                current_context.active_concept = followup_intent.active_concept

        if followup_intent.intent_type == IntentType.FOLLOWUP_CONCEPT_TRANSITION:
            logger.info("Stream follow-up concept transition detected: '%s' -> resolved='%s'", message, followup_intent.resolved_query)
            message = followup_intent.resolved_query
            current_context.active_entities = []
            current_context.last_evidence_text = None
            if followup_intent.active_subject:
                current_context.active_subject = followup_intent.active_subject
            if followup_intent.active_concept:
                current_context.active_concept = followup_intent.active_concept
                current_context.active_entity_type = followup_intent.active_concept

        # ── Direct Official URL Handler (Streaming) ───────────────────
        explicit_urls = self._extract_official_urls(message)
        if not explicit_urls and history:
            followup_url = self._detect_followup_faculty_context(message, history)
            if followup_url:
                logger.info("Direct official URL resolved from stream follow-up context: %s", followup_url)
                explicit_urls = [followup_url]

        if explicit_urls:
            logger.info("Direct official URL detected in stream query: %s", explicit_urls)
            url_results: List[OfficialWebResult] = []
            for u in explicit_urls[:2]:
                u_res = await self.web_retriever.retrieve_url(u, query=message)
                if u_res.retrieval_method != "failed":
                    url_results.append(u_res)

            if url_results:
                (
                    context,
                    official_directive,
                    sources,
                    structured_sources,
                    confidence,
                    intent_val,
                    format_val,
                ) = self._build_grounded_evidence(message, url_results)
                source_visibility = "compact" if len(structured_sources) <= 1 else "full"

                final_prompt = self._build_generation_prompt(
                    context=context,
                    query=message,
                    formatting_instructions=f"{official_directive}",
                )

                yield f'data: {json.dumps({"type": "metadata", "sources": sources, "structured_sources": structured_sources, "source_visibility": source_visibility, "confidence": confidence, "retrieval_score": 0.98, "intent": intent_val, "response_type": format_val})}\n\n'

                accumulated_chunks = []
                try:
                    async for text_chunk in self.llm_service.generate_stream(
                        system_prompt=self.system_prompt,
                        user_prompt=final_prompt,
                        history=history,
                    ):
                        if text_chunk == "[FALLBACK_TRIGGERED]":
                            yield f'data: {json.dumps({"type": "fallback_triggered"})}\n\n'
                        else:
                            accumulated_chunks.append(text_chunk)
                            yield f'data: {json.dumps({"type": "content", "content": text_chunk})}\n\n'
                except Exception as exc:
                    logger.error("LLM streaming failed for official URL: %s", exc)
                    fallback_reply = self.llm_service.generate(
                        system_prompt=self.system_prompt,
                        user_prompt=final_prompt,
                        history=history,
                    )
                    accumulated_chunks = [fallback_reply]
                    yield f'data: {json.dumps({"type": "content", "content": fallback_reply})}\n\n'

                complete_reply = "".join(accumulated_chunks)
                polished_reply = self._polish_response(complete_reply)

                # Update stream conversational context state
                extracted_entities = []
                for r in url_results:
                    ext = self.context_resolver.extract_entities_from_evidence(
                        r.content, [r.url], entity_type=r.entity_type, raw_entities=getattr(r, "entities", None)
                    )
                    extracted_entities.extend(ext)
                if extracted_entities:
                    current_context.active_entities = extracted_entities
                    current_context.active_entity_type = extracted_entities[0].entity_type
                    current_context.active_subject = message
                current_context.active_source_urls = sources
                current_context.last_evidence_text = context
                current_context.last_answer = polished_reply
                current_context.last_query = message
                current_context.last_updated_ist = format_ist()
                if conversation_id:
                    self._conversation_contexts[conversation_id] = current_context

                yield f'data: {json.dumps({"type": "done", "reply": polished_reply, "sources": sources, "structured_sources": structured_sources, "source_visibility": source_visibility, "confidence": confidence, "response_type": format_val, "context_state": current_context.model_dump()})}\n\n'
                return

        intent, matched_keywords = self.classify_intent(message)
        query_period = self._detect_query_period(message)
        is_institutional = self._is_institutional_query(message)
        response_plan = response_planner.plan_response(message, intent=intent, period=query_period)

        retrieval_empty = False
        retrieved_chunks = []
        retrieval_scores = []

        if self.index is None or self.index.ntotal == 0:
            await self._bootstrap_official_sites_if_empty()

        if self.index is None or self.index.ntotal == 0:
            logger.info("Knowledge base is empty. Falling back to LLM-only mode.")
            retrieval_empty = True
        else:
            try:
                query_embedding = await self.embedding_service.get_embedding(message)
                retrieved_chunks = self._search_index(query_embedding)
            except Exception as exc:
                logger.error("Retrieval failed: %s. Falling back to LLM-only mode.", exc)
                retrieval_empty = True

        if not retrieval_empty:
            retrieval_scores = [c.similarity_score for c in retrieved_chunks]
            if not retrieved_chunks:
                retrieval_empty = True

        qualified = []
        top_adjusted = 0.0
        if not retrieval_empty:
            qualified = [
                c for c in retrieved_chunks
                if c.similarity_score >= self.settings.similarity_threshold
            ]
            qualified = self._deduplicate_chunks(qualified)
            self._apply_authority_multiplier(qualified, intent=intent, query=message)
            qualified = self._sort_by_adjusted_score(qualified)

            top_adjusted = qualified[0].adjusted_score if qualified else 0.0
            if top_adjusted < REFUSAL_ADJUSTED_THRESHOLD or not qualified:
                retrieval_empty = True
                qualified = []

        # Check if RAG chunks actually contain the query entity
        if not retrieval_empty and not self._is_rag_sufficient(message, qualified):
            logger.info("RAG context is insufficient for query entity; falling back to official web retrieval")
            retrieval_empty = True
            qualified = []

        # ── Step 4: Official Web Retrieval Fallback (if RAG is empty/insufficient) ──
        official_web_results: List[OfficialWebResult] = []
        if retrieval_empty and is_institutional:
            try:
                official_web_results = await self.web_retriever.retrieve(message, period=query_period)
            except Exception as exc:
                logger.warning("Official web fallback retrieval failed: %s", exc)

        # ── CONTEXT OPTIMIZATION & PROMPT PREPARATION ─────────────────
        if official_web_results:
            (
                context,
                official_directive,
                sources,
                structured_sources,
                confidence,
                intent_val,
                response_type_val,
            ) = self._build_grounded_evidence(message, official_web_results)
            source_visibility = "compact" if len(structured_sources) <= 1 else "full"
            top_adjusted = 0.95
            intent = intent_val
            response_plan.response_type = response_type_val

            final_prompt = self._build_generation_prompt(
                context=context,
                query=message,
                formatting_instructions=f"{response_plan.formatting_instructions}\n\n{official_directive}",
            )
        elif not retrieval_empty:
            qualified = self._remove_near_duplicates(qualified)
            qualified = self._merge_adjacent_chunks(qualified)
            qualified = qualified[:MAX_CONTEXT_CHUNKS]

            context = self._build_context(qualified)
            sources = list(dict.fromkeys(c.document_name for c in qualified))
            structured_sources = self._build_structured_sources(qualified)
            source_visibility = "compact" if len(structured_sources) <= 1 else "full"
            final_prompt = self._build_generation_prompt(
                context=context,
                query=message,
                formatting_instructions=response_plan.formatting_instructions,
            )
            confidence = self._compute_confidence_adjusted(top_adjusted)
        else:
            # Check if query was searching for a specific named entity/person that does not exist
            entity_name = self._extract_searched_entity_name(message)
            if entity_name and is_institutional:
                reply = (
                    f"Based on official records across the VVITU official portal (https://vvitu.ac.in/) "
                    f"and the legacy VVIT portal (https://vvitguntur.com/), no official record was found for **{entity_name}**.\n\n"
                    f"If you are looking for faculty or staff in a specific school or department (such as CSE, AI, ECE, EEE, Civil, Mechanical, or Business Administration), "
                    f"please let me know and I will be happy to look up the official faculty directory for you."
                )
                sources, structured_sources = self._get_fallback_sources_for_query(message)
                yield f'data: {json.dumps({"type": "metadata", "sources": sources, "structured_sources": structured_sources, "source_visibility": "compact", "confidence": "High", "retrieval_score": 1.0, "intent": "faculty_lookup", "response_type": "factual"})}\n\n'
                yield f'data: {json.dumps({"type": "content", "content": reply})}\n\n'
                yield f'data: {json.dumps({"type": "done", "response_type": "factual", "source_visibility": "compact", "sources": sources, "structured_sources": structured_sources, "confidence": "High"})}\n\n'
                return

            context = ""
            if is_institutional:
                sources, structured_sources = self._get_fallback_sources_for_query(message)
                source_visibility = "compact" if len(structured_sources) <= 1 else "full"
                confidence = "Low"
            else:
                sources, structured_sources = [], []
                source_visibility = "none"
                confidence = "High"

            final_prompt = self._build_llm_only_prompt(
                query=message,
                formatting_instructions=response_plan.formatting_instructions,
                is_institutional=is_institutional,
            )

        # Yield metadata first
        yield f'data: {json.dumps({"type": "metadata", "sources": sources, "structured_sources": structured_sources, "source_visibility": source_visibility, "confidence": confidence, "retrieval_score": round(top_adjusted, 4), "intent": intent, "response_type": response_plan.response_type, "context_state": current_context.model_dump()})}\n\n'

        accumulated_chunks = []
        try:
            async for chunk in self.llm_service.generate_stream(
                system_prompt=self.system_prompt,
                user_prompt=final_prompt,
                history=history
            ):
                if chunk == "[FALLBACK_TRIGGERED]":
                    yield f'data: {json.dumps({"type": "fallback_triggered"})}\n\n'
                else:
                    accumulated_chunks.append(chunk)
                    yield f'data: {json.dumps({"type": "content", "content": chunk})}\n\n'
        except Exception as exc:
            logger.error("LLM stream failed completely: %s", exc)
            yield f'data: {json.dumps({"type": "error", "error": "An error occurred during response generation."})}\n\n'

        complete_reply = "".join(accumulated_chunks)
        polished_reply = self._polish_response(complete_reply)

        current_context.last_answer = polished_reply
        current_context.last_query = message
        current_context.last_evidence_text = context
        current_context.active_source_urls = sources
        detected_concept = self.context_resolver.detect_concept(message)
        if detected_concept:
            current_context.active_concept = detected_concept
            current_context.active_entity_type = detected_concept
        detected_subject = self.context_resolver.detect_subject(message)
        if detected_subject:
            current_context.active_subject = detected_subject
        detected_filters = self.context_resolver.detect_filters(message)
        if detected_filters:
            current_context.active_filters.update(detected_filters)
        current_context.last_updated_ist = format_ist()
        if context:
            extracted_entities = self.context_resolver.extract_entities_from_evidence(
                context, sources, concept=current_context.active_concept
            )
            if extracted_entities:
                current_context.active_entities = extracted_entities
        if conversation_id:
            self._conversation_contexts[conversation_id] = current_context

        self._log_query(
            message=message,
            confidence=confidence,
            sources=sources,
            is_refusal=False,
            retrieval_scores=retrieval_scores,
            intent=intent,
            top_adjusted_score=top_adjusted,
        )

        yield f'data: {json.dumps({"type": "done", "reply": polished_reply, "response_type": response_plan.response_type, "source_visibility": source_visibility, "sources": sources, "structured_sources": structured_sources, "confidence": confidence, "context_state": current_context.model_dump()})}\n\n'

    async def _bootstrap_official_sites_if_empty(self) -> bool:
        """
        Populate an empty knowledge base from configured official college sites.

        This is intentionally bounded and domain-restricted. It runs at most once
        per process and persists the resulting FAISS index for later requests.
        """
        if self._official_bootstrap_attempted:
            return False
        if not self.settings.auto_ingest_official_sites_on_empty:
            return False
        if self.index is not None and self.index.ntotal > 0:
            return False
        if not self.settings.crawl_seed_urls:
            logger.warning("Official-site bootstrap skipped: no crawl_seed_urls configured")
            return False
        if not self.settings.openrouter_api_key:
            logger.warning("Official-site bootstrap skipped: OPENROUTER_API_KEY is required for embeddings")
            return False

        self._official_bootstrap_attempted = True
        logger.info(
            "Knowledge base empty; bootstrapping from official sites: %s",
            self.settings.crawl_seed_urls,
        )

        try:
            from app.services.website_crawler import WebsiteCrawler

            def crawl_pages():
                crawler = WebsiteCrawler(
                    allowed_domains=self.settings.allowed_domains,
                    depth_limit=self.settings.official_bootstrap_depth_limit,
                    max_pages=self.settings.official_bootstrap_max_pages,
                    delay=self.settings.crawl_delay,
                )
                return crawler.crawl_as_dicts(self.settings.crawl_seed_urls)

            pages = await asyncio.to_thread(crawl_pages)
            if not pages:
                logger.warning("Official-site bootstrap found no crawlable pages")
                return False

            documents = []
            metadata = []
            for page in pages:
                text = (page.get("text") or "").strip()
                if not text:
                    continue

                url = page.get("url", "")
                title = page.get("title") or url or "Official website"
                documents.append({"content": text, "name": title})
                url_lower = url.lower()
                is_vvitu = any(d in url_lower for d in CURRENT_UNIVERSITY_DOMAINS)
                is_vvit_legacy = any(d in url_lower for d in LEGACY_INSTITUTE_DOMAINS)

                if is_vvitu:
                    period = "current"
                    tier = "primary_official_current"
                    doc_type = "official_website_current"
                    source_label = "VVITU Official Website (Current)"
                    identity = "VVITU (Current University)"
                elif is_vvit_legacy:
                    period = "historical"
                    tier = "legacy_official_vvit"
                    doc_type = "legacy_website"
                    source_label = "VVIT Legacy Website (Historical)"
                    identity = "VVIT (Legacy Institute)"
                else:
                    period = "general"
                    tier = "primary_official"
                    doc_type = "official_website"
                    source_label = "Official Website"
                    identity = "Institutional Source"

                metadata.append(
                    {
                        "document_type": doc_type,
                        "academic_year": "current" if period == "current" else "historical",
                        "department": "General",
                        "authority_level": page.get("authority_level", "high" if is_vvitu else "medium"),
                        "source_type": "website",
                        "source_tier": tier,
                        "institutional_period": period,
                        "institutional_identity": identity,
                        "source_label": source_label,
                        "url": url,
                        "source_file": url,
                    }
                )

            if not documents:
                logger.warning("Official-site bootstrap had pages but no usable text")
                return False

            added = await self.add_documents(documents, metadata)
            logger.info("Official-site bootstrap complete: added %d chunks", added)
            return added > 0
        except Exception as exc:
            logger.error("Official-site bootstrap failed: %s", exc, exc_info=True)
            return False

    def _search_index(
        self, query_embedding: np.ndarray
    ) -> List[RetrievedChunk]:
        """Run FAISS nearest-neighbour search on a pre-embedded query."""
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        k = min(self.settings.top_k_retrieval, self.index.ntotal)
        scores, indices = self.index.search(query_embedding.reshape(1, -1), k)

        retrieved: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.chunks):
                chunk_data = self.chunks[idx]
                meta = chunk_data.get("metadata", chunk_data)
                content = chunk_data.get("content", meta.get("text", ""))
                retrieved.append(
                    RetrievedChunk(
                        content=content,
                        metadata=meta,
                        similarity_score=float(score),
                        document_name=meta.get(
                            "document_name",
                            meta.get("source_file", "Unknown"),
                        ),
                        authority_level=meta.get("authority_level", "low"),
                        url=meta.get("url", ""),
                        source_type=meta.get("source_type", ""),
                        document_type=meta.get("document_type", ""),
                        source_tier=meta.get("source_tier", ""),
                    )
                )
        return retrieved

    async def _retrieve(self, query: str) -> List[RetrievedChunk]:
        """Embed a query and retrieve the most relevant chunks (convenience wrapper)."""
        if self.index is None or self.index.ntotal == 0:
            return []
        query_embedding = await self.embedding_service.get_embedding(query)
        return self._search_index(query_embedding)

    async def debug_retrieve(self, query: str) -> Dict:
        """
        Run retrieval and return raw diagnostics (no LLM call).

        Returns a dict with 'chunks' (list of scored chunk dicts),
        'performance' (embed_ms, retrieval_ms), and 'intent'.
        """
        if self.index is None or self.index.ntotal == 0:
            return {"chunks": [], "performance": {}, "intent": "general", "intent_keywords": []}

        # Classify intent
        intent, intent_keywords = self.classify_intent(query)

        t0 = time.perf_counter()
        query_embedding = await self.embedding_service.get_embedding(query)
        t_embed = time.perf_counter()

        chunks = self._search_index(query_embedding)
        t_retrieve = time.perf_counter()

        self._apply_authority_multiplier(chunks, intent=intent, query=query)

        results = []
        for c in chunks:
            results.append({
                "text_preview": c.content[:300],
                "source_file": c.metadata.get("source_file", "Unknown"),
                "authority_level": c.authority_level,
                "similarity_score": round(c.similarity_score, 4),
                "adjusted_score": round(c.adjusted_score, 4),
            })

        perf = {
            "embed_ms": round((t_embed - t0) * 1000, 1),
            "retrieval_ms": round((t_retrieve - t_embed) * 1000, 1),
        }

        return {
            "chunks": results,
            "performance": perf,
            "intent": intent,
            "intent_keywords": intent_keywords,
        }

    def _has_sufficient_context(self, chunks: List[RetrievedChunk]) -> bool:
        """At least one chunk must meet the similarity threshold."""
        if not chunks:
            return False
        threshold = self.settings.similarity_threshold
        return any(c.similarity_score >= threshold for c in chunks)

    @staticmethod
    def _compute_confidence_adjusted(top_adjusted: float) -> str:
        """
        Determine confidence level from top adjusted_score.

        - High   : adjusted_score >= 0.85
        - Medium : 0.75–0.85
        - Low    : 0.65–0.75
        - Refusal: < 0.65 (handled before this is called)
        """
        if top_adjusted >= CONFIDENCE_HIGH:
            return "High"
        if top_adjusted >= CONFIDENCE_MEDIUM:
            return "Medium"
        return "Low"

    # ── Intent-based source priority boosts ─────────────────────────
    INTENT_SOURCE_BOOST: Dict[str, Dict[str, float]] = {
        "academic": {
            "primary_official": 0.08,
            "pdf": 0.04,
            "official_website": 0.04,
        },
        "admissions": {
            "primary_official": 0.10,
            "official_website": 0.06,
        },
        "examinations": {
            "primary_official": 0.08,
            "pdf": 0.04,
            "official_website": 0.04,
        },
        "placements": {
            "primary_official": 0.05,
            "secondary_linkedin": 0.05,
            "tertiary_social": 0.02,
            "official_website": 0.03,
        },
        "events": {
            "secondary_linkedin": 0.07,
            "tertiary_social": 0.05,
            "primary_official": 0.03,
            "social_media": 0.03,
        },
        "infrastructure": {
            "primary_official": 0.08,
            "official_website": 0.05,
        },
    }

    @classmethod
    def _apply_authority_multiplier(
        cls,
        chunks: List[RetrievedChunk],
        intent: str = "general",
        query: str = "",
    ) -> None:
        """
        Compute adjusted score with source-priority-aware ranking.

        Policy:
          - Prefer official VVIT/VVITU sources by default.
          - Prefer LinkedIn/social for dynamic event/update queries.
          - Keep external related sources as supporting evidence only.

        Mutates each chunk in-place.
        """
        intent_boosts = cls.INTENT_SOURCE_BOOST.get(intent, {})
        dynamic_query = cls._is_dynamic_query(query)
        query_period = cls._detect_query_period(query)

        for c in chunks:
            multiplier = AUTHORITY_MULTIPLIER.get(c.authority_level, 1.0)
            base = c.similarity_score * multiplier

            # Source metadata
            boost = 0.0
            doc_type = c.metadata.get("document_type", c.document_type)
            src_type = c.metadata.get("source_type", c.source_type)
            source_tier = cls._infer_source_tier(c.metadata)
            c.source_tier = source_tier

            # Base source weighting
            boost += SOURCE_TIER_BOOST.get(source_tier, 0.0)

            url = (c.url or c.metadata.get("url", "")).lower()
            is_vvitu = any(d in url for d in CURRENT_UNIVERSITY_DOMAINS)
            is_vvit_legacy = any(d in url for d in LEGACY_INSTITUTE_DOMAINS)

            # Period-aware institutional weighting
            if query_period == "current":
                if is_vvitu:
                    boost += 0.08
                elif is_vvit_legacy:
                    boost += 0.01
            elif query_period == "historical":
                if is_vvit_legacy:
                    boost += 0.08
                elif is_vvitu:
                    boost += 0.02
            else:  # "both" or general
                if is_vvitu:
                    boost += 0.06
                elif is_vvit_legacy:
                    boost += 0.04

            if doc_type in {"official_website", "official_website_current"} or source_tier in {"primary_official", "primary_official_current"}:
                boost += 0.03
            if src_type == "pdf":
                boost += 0.05
            if source_tier in {"secondary_linkedin", "tertiary_social"}:
                boost += cls._recency_boost_for_social(c.metadata)

            # If the query asks for latest updates, emphasize social freshness.
            if dynamic_query and source_tier in {"secondary_linkedin", "tertiary_social"}:
                boost += 0.04

            # Intent-based dynamic boost
            if intent_boosts:
                if doc_type in intent_boosts:
                    boost += intent_boosts[doc_type]
                if src_type in intent_boosts:
                    boost += intent_boosts[src_type]
                if source_tier in intent_boosts:
                    boost += intent_boosts[source_tier]

            c.adjusted_score = base + boost

    @staticmethod
    def _sort_by_adjusted_score(
        chunks: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        """
        Sort chunks by adjusted_score descending.

        adjusted_score already factors in authority weighting.
        """
        return sorted(chunks, key=lambda c: c.adjusted_score, reverse=True)

    @staticmethod
    def _deduplicate_chunks(
        chunks: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        """
        Remove duplicate chunks based on content hash.

        Keeps the first (highest-scored) occurrence.
        """
        seen: set = set()
        unique: List[RetrievedChunk] = []
        for c in chunks:
            key = hashlib.md5(c.content.strip().encode("utf-8")).hexdigest()
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique

    @staticmethod
    def _word_set(text: str) -> set:
        """Return lowercase word set for Jaccard comparison."""
        return set(text.lower().split())

    @staticmethod
    def _jaccard_similarity(a: set, b: set) -> float:
        """Jaccard similarity between two word sets."""
        if not a and not b:
            return 1.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union else 0.0

    @classmethod
    def _remove_near_duplicates(
        cls, chunks: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        """
        Remove near-duplicate chunks based on Jaccard word-overlap.

        Keeps the higher-scored chunk when overlap >= NEAR_DUPLICATE_SIMILARITY.
        Already sorted by adjusted_score descending.
        """
        kept: List[RetrievedChunk] = []
        kept_word_sets: List[set] = []

        for c in chunks:
            ws = cls._word_set(c.content)
            is_near_dup = False
            for existing_ws in kept_word_sets:
                if cls._jaccard_similarity(ws, existing_ws) >= NEAR_DUPLICATE_SIMILARITY:
                    is_near_dup = True
                    break
            if not is_near_dup:
                kept.append(c)
                kept_word_sets.append(ws)

        if len(chunks) != len(kept):
            logger.debug(
                "Near-dedup: %d → %d chunks", len(chunks), len(kept),
            )
        return kept

    @staticmethod
    def _merge_adjacent_chunks(
        chunks: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        """
        Merge adjacent (sequential) chunks from the same source document.

        Two chunks are adjacent if:
          - Same document_name (source_file)
          - chunk_index values are consecutive

        The merged chunk keeps the higher adjusted_score and concatenated content.
        Preserves heading hierarchy by keeping natural order.
        """
        if len(chunks) <= 1:
            return chunks

        # Group by source and find mergeable pairs
        merged: List[RetrievedChunk] = []
        skip_indices: set = set()

        for i, c in enumerate(chunks):
            if i in skip_indices:
                continue

            c_idx = c.metadata.get("chunk_index", -1)
            c_source = c.metadata.get("source_file", c.document_name)

            # Look for the next chunk from the same source that is adjacent
            for j in range(i + 1, len(chunks)):
                if j in skip_indices:
                    continue
                other = chunks[j]
                o_idx = other.metadata.get("chunk_index", -2)
                o_source = other.metadata.get("source_file", other.document_name)

                if c_source == o_source and abs(c_idx - o_idx) == 1:
                    # Merge: put the earlier chunk's content first
                    if c_idx < o_idx:
                        merged_content = c.content + "\n\n" + other.content
                    else:
                        merged_content = other.content + "\n\n" + c.content

                    c.content = merged_content
                    c.adjusted_score = max(c.adjusted_score, other.adjusted_score)
                    skip_indices.add(j)
                    break  # Only merge one adjacent pair per chunk

            merged.append(c)

        if skip_indices:
            logger.debug(
                "Merged %d adjacent chunk pairs", len(skip_indices),
            )
        return merged

    def _build_context(self, chunks: List[RetrievedChunk]) -> str:
        """
        Build a formatted context string from retrieved chunks.

        Caps total length to settings.max_context_chars to prevent
        LLM token overflow.
        """
        max_chars = getattr(self.settings, "max_context_chars", 8000)
        parts: List[str] = []
        total_len = 0

        for i, chunk in enumerate(chunks, 1):
            meta_str = self._format_metadata(chunk.metadata)
            block = (
                f"[Document {i}: {chunk.document_name}]\n"
                f"Metadata: {meta_str}\n"
                f"Content: {chunk.content}\n"
            )
            if total_len + len(block) > max_chars:
                break
            parts.append(block)
            total_len += len(block)

        return "\n---\n".join(parts)

    @staticmethod
    def _build_generation_prompt(context: str, query: str, formatting_instructions: str = "") -> str:
        """Build the final user prompt payload for the generation step with IST grounding."""
        temporal_context = get_ist_grounding_context()
        base_instructions = (
            "INSTRUCTIONS & DUAL-SOURCE RECONCILIATION RULES:\n"
            "1. Answer ONLY from CONTEXT.\n"
            "2. Understand the institutional transition:\n"
            "   - VVITU (vvitu.ac.in) is the CURRENT university identity.\n"
            "   - VVIT (vvitguntur.com) is the LEGACY institute identity.\n"
            "3. Source Priority & Conflict Resolution:\n"
            "   - For questions about CURRENT facts (current programs, current principal/leadership, current calendar, current fees/regulations): Current VVITU sources (vvitu.ac.in) ALWAYS take precedence.\n"
            "   - For questions about HISTORICAL facts (past syllabus, older regulations, legacy records): Legacy VVIT sources (vvitguntur.com) represent the older period.\n"
            "   - If both sources are present, do not randomly choose between them: explain the transition from VVIT to VVITU and specify which era each fact belongs to.\n"
            "4. NEVER describe vvitguntur.com as the current official website of VVITU.\n"
            "5. If information is unclear or unavailable, state clearly that the specific institutional details are not available."
        )
        if formatting_instructions:
            format_block = f"\n\nDYNAMIC FORMATTING DIRECTIVE:\n{formatting_instructions}"
        else:
            format_block = "\n6. Keep the response concise, student-friendly, and cleanly structured."

        return (
            f"{temporal_context}\n\n"
            "CONTEXT:\n"
            f"{context}\n\n"
            "USER QUESTION:\n"
            f"{query}\n\n"
            f"{base_instructions}{format_block}"
        )

    @staticmethod
    def _build_llm_only_prompt(query: str, formatting_instructions: str = "", is_institutional: bool = True) -> str:
        """Build the final user prompt payload when no context is available with IST grounding."""
        temporal_context = get_ist_grounding_context()
        if not is_institutional:
            format_block = (
                f"\n\nDYNAMIC FORMATTING DIRECTIVE:\n{formatting_instructions}"
                if formatting_instructions
                else "\nKeep the response clear, helpful, educational, and cleanly structured."
            )
            return (
                f"{temporal_context}\n\n"
                "USER QUESTION:\n"
                f"{query}\n\n"
                "INSTRUCTIONS:\n"
                "1. Answer the user's question directly, accurately, and thoroughly based on your knowledge.\n"
                "2. Maintain a natural, helpful, and polite tone.\n"
                "3. Do NOT mention VVIT, VVITU, university transitions, or institutional website links unless the question explicitly asks about them."
                f"{format_block}"
            )

        format_block = (
            f"\n\nDYNAMIC FORMATTING DIRECTIVE:\n{formatting_instructions}"
            if formatting_instructions
            else "\n5. Keep the response concise, student-friendly, and cleanly structured."
        )
        return (
            f"{temporal_context}\n\n"
            "USER QUESTION:\n"
            f"{query}\n\n"
            "INSTRUCTIONS & DUAL-SOURCE RULES:\n"
            "1. Answer the user's question directly based on your knowledge and institutional transition rules:\n"
            "   - VVITU (https://vvitu.ac.in/) is the CURRENT university and primary source for current information.\n"
            "   - VVIT (https://vvitguntur.com/) is the LEGACY institute and source for historical information.\n"
            "   - VAIT stands for VVIT's Artificial Intelligence Technology.\n"
            "2. If asked about the current official website, cite https://vvitu.ac.in/.\n"
            "3. If asked about the old VVIT website, cite https://vvitguntur.com/.\n"
            "4. For current matters, prioritize VVITU. For historical matters, use legacy VVIT.\n"
            "5. Only mention the institutional transition when relevant to the question.\n"
            "6. NEVER tell the user to manually visit or search the official website (e.g. 'You can check the faculty directory on vvitu.ac.in'). If details for a specific person or record are not known, state directly and plainly that the details are not available in current records."
            f"{format_block}"
        )

    @staticmethod
    def _format_metadata(metadata: Dict) -> str:
        """Format metadata fields for context display."""
        fields = [
            "source_label",
            "institutional_period",
            "institutional_identity",
            "document_type",
            "academic_year",
            "department",
            "authority_level",
            "source_tier",
            "url",
        ]
        parts = [f"{f}={metadata[f]}" for f in fields if f in metadata]
        return ", ".join(parts)

    @staticmethod
    def _format_response(
        text: str,
        sources: List[str],
        structured_sources: List[Dict],
        confidence: str,
        top_score: float,
        retrieval_scores: List[float],
        performance: Optional[Dict] = None,
        intent: str = "general",
        response_type: str = "informational",
        source_visibility: str = "none",
    ) -> RAGResponse:
        """
        Post-process LLM output into a structured RAGResponse.

        - Trims trailing whitespace
        - Attaches unique sources, structured_sources, confidence, retrieval_score, performance, response_type, source_visibility
        """
        cleaned = text.rstrip() if text else ""
        return RAGResponse(
            reply=cleaned,
            sources=list(dict.fromkeys(sources)),
            structured_sources=structured_sources,
            confidence=confidence,
            retrieval_score=round(top_score, 4),
            retrieval_scores=retrieval_scores,
            is_refusal=False,
            performance=performance,
            intent=intent,
            response_type=response_type,
            source_visibility=source_visibility,
        )

    @staticmethod
    def _hallucination_guard(
        response_text: str,
        confidence: str,
        sources: List[str],
        is_institutional: bool = True,
    ) -> Tuple[str, str]:
        """Post-LLM hallucination defence: reject only empty / too-short responses."""
        if not response_text or len(response_text.strip()) < 10:
            logger.warning(
                "Hallucination guard: response too short (%d chars)",
                len(response_text.strip()) if response_text else 0,
            )
            return "I apologize, but I could not generate a proper response. Please try again.", "Low"

        if is_institutional and not sources:
            confidence = "Low"

        return response_text, confidence

    @staticmethod
    def _append_citations(
        response_text: str,
        structured_sources: List[Dict],
        source_visibility: str = "none",
    ) -> str:
        """
        Append a formatted citation block at the bottom of the reply only when source_visibility != "none".
        Always strips any accidental LLM-generated source block when source_visibility == "none".
        """
        # Always strip any existing source block the LLM may have added
        cleaned = re.sub(
            r"\n*Source(?:\(s\))?s?:\s*\n(?:\d+\..*\n?)*",
            "",
            response_text,
            flags=re.IGNORECASE,
        ).rstrip()

        if source_visibility == "none" or not structured_sources:
            return cleaned

        lines = ["\n\nSource(s):"]
        for i, src in enumerate(structured_sources, 1):
            title = src.get("title", "Unknown")
            url = src.get("url", "")
            period_label = src.get("period_label", "")

            label_suffix = f" [{period_label}]" if period_label and period_label.lower() not in title.lower() else ""
            if url:
                lines.append(f"{i}. {title}{label_suffix} — {url}")
            else:
                lines.append(f"{i}. {title}{label_suffix}")

        return cleaned + "\n".join(lines)

    @staticmethod
    def _enforce_answer_structure(response_text: str, question: str) -> str:
        """
        Ensure the LLM response contains:
          - A title line
          - An explanation paragraph
          - A bullet/list section

        If any is missing, auto-reformat with regex and a fallback title.
        """
        if not response_text or len(response_text.strip()) < MIN_VALID_RESPONSE_LENGTH:
            return response_text  # handled by hallucination guard

        text = response_text.strip()

        # Detect presence of title (markdown heading or bold first line)
        has_title = bool(
            re.match(r"^(#{1,3}\s|\*\*).+", text)
            or re.match(r"^[A-Z][^\n]{5,80}$", text.split("\n")[0])
        )

        # Detect bullet points / numbered list
        has_bullets = bool(
            re.search(r"^\s*[-•●▸\*]\s+.+", text, re.MULTILINE)
            or re.search(r"^\s*\d+[\.\)]\s+.+", text, re.MULTILINE)
        )

        # Detect explanation (at least one paragraph of 80+ chars)
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= 80]
        has_explanation = len(paragraphs) >= 1

        needs_reformat = not has_title or not has_bullets or not has_explanation

        if not needs_reformat:
            return response_text

        # ── Auto-reformat ────────────────────────────────────────────
        parts: List[str] = []

        # Add fallback title if missing
        if not has_title:
            # Derive title from the question
            fallback_title = question.strip().rstrip("?").strip()
            if len(fallback_title) > 80:
                fallback_title = fallback_title[:77] + "..."
            parts.append(f"**{fallback_title}**\n")

        # Add the original text
        parts.append(text)

        # If no bullets were found, try converting sentences to bullets
        if not has_bullets:
            # Find the last paragraph and split into bullet points
            lines = text.split("\n")
            bullet_candidates = [
                l.strip() for l in lines
                if len(l.strip()) > 20 and not l.strip().startswith(("#", "**", "Source"))
            ]
            if len(bullet_candidates) >= 2:
                # Don't double the text — just return with title added
                pass

        return "\n".join(parts)

    @staticmethod
    def _polish_response(response_text: str) -> str:
        """
        Post-LLM response polishing:
          1. Collapse multiple blank lines into one.
          2. Remove repeated sentences (exact duplicates).
          3. Strip trailing whitespace on every line.
          4. Cap response at MAX_RESPONSE_WORDS words.

        Returns:
            Cleaned response text.
        """
        if not response_text or len(response_text.strip()) < MIN_VALID_RESPONSE_LENGTH:
            return response_text

        text = response_text

        # 1. Strip trailing whitespace per line
        text = "\n".join(line.rstrip() for line in text.split("\n"))

        # 2. Collapse 3+ consecutive newlines → 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 3. Remove exact duplicate sentences
        #    Split on sentence-ending punctuation, dedup, rejoin
        sentences = re.split(r'(?<=[.!?])\s+', text)
        seen: set = set()
        deduped: List[str] = []
        for sent in sentences:
            normalized = sent.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                deduped.append(sent.strip())
            elif not normalized:
                deduped.append(sent)  # preserve blank separators

        text = " ".join(deduped)

        # Restore paragraph breaks that were lost in sentence join
        text = text.replace("\n \n", "\n\n")

        # 4. Word count cap
        words = text.split()
        if len(words) > MAX_RESPONSE_WORDS:
            text = " ".join(words[:MAX_RESPONSE_WORDS]) + "..."
            logger.debug(
                "Response polished: capped from %d to %d words",
                len(words), MAX_RESPONSE_WORDS,
            )

        return text.strip()

    @staticmethod
    def _make_error_response(message: str) -> RAGResponse:
        """Create a structured error/refusal response."""
        return RAGResponse(
            reply=message,
            sources=[],
            structured_sources=[],
            confidence="Low",
            retrieval_score=0.0,
            is_refusal=True,
        )

    @staticmethod
    def _build_structured_sources(chunks: List[RetrievedChunk]) -> List[Dict]:
        """
        Build structured source objects from qualified chunks with era labeling.

        Each source returned as:
          { "title": "...", "url": "...", "type": "...", "period_label": "..." }

        Deduplicates by URL (or title if URL missing).
        """
        seen_keys: set = set()
        sources: List[Dict] = []

        for c in chunks:
            url = c.url or c.metadata.get("url", "")
            title = c.document_name or c.metadata.get("document_name", "Unknown")
            url_lower = url.lower()

            is_vvitu = any(d in url_lower for d in CURRENT_UNIVERSITY_DOMAINS)
            is_vvit_legacy = any(d in url_lower for d in LEGACY_INSTITUTE_DOMAINS)

            if is_vvitu:
                period_label = "VVITU Official Website — Current"
                institutional_period = "current"
                type_label = "website_current"
            elif is_vvit_legacy:
                period_label = "VVIT Legacy Website — Historical"
                institutional_period = "historical"
                type_label = "website_historical"
            else:
                period_label = c.metadata.get("source_label", "")
                institutional_period = c.metadata.get("institutional_period", "general")
                source_type = c.source_type or c.metadata.get("source_type", "")
                doc_type = c.document_type or c.metadata.get("document_type", "")
                source_tier = c.source_tier or c.metadata.get("source_tier", "")

                if source_tier == "secondary_linkedin":
                    type_label = "linkedin"
                elif source_tier == "tertiary_social" or source_type == "social_media":
                    type_label = "social_media"
                elif source_type == "pdf" or doc_type in ("regulation", "syllabus"):
                    type_label = "pdf"
                else:
                    type_label = doc_type or source_type or "document"

            # Dedup key: prefer URL, fall back to title
            dedup_key = url if url else title
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            source_obj = {
                "title": title,
                "url": url,
                "type": type_label,
                "period_label": period_label,
                "institutional_period": institutional_period,
            }
            sources.append(source_obj)

        return sources

    def _log_query(
        self,
        message: str,
        confidence: str,
        sources: List[str],
        is_refusal: bool,
        retrieval_scores: List[float],
        intent: str = "general",
        top_adjusted_score: float = 0.0,
    ) -> None:
        """
        Append a structured JSON-lines entry to data/logs/query.log.

        Fields: timestamp, question, top_similarity_score, confidence,
                sources, is_refusal, threshold, intent
        """
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            log_file = LOGS_DIR / "query.log"

            entry = {
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "question": message[:500],
                "top_similarity_score": round(max(retrieval_scores), 4) if retrieval_scores else 0.0,
                "top_adjusted_score": round(top_adjusted_score, 4),
                "confidence": confidence,
                "sources": sources,
                "is_refusal": is_refusal,
                "threshold": self.settings.similarity_threshold,
                "intent": intent,
            }

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            logger.debug("Failed to write query log entry", exc_info=True)

    def _load_system_prompt(self) -> str:
        """Load the system prompt from the configured path."""
        prompt_path = Path(self.settings.system_prompt_path)
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")

        return (
            "You are VAIT (VVIT's Artificial Intelligence Technology), the official institutional AI "
            "assistant for VVIT University (VVITU) and the legacy Vasireddy Venkatadri Institute of Technology (VVIT).\n\n"
            "CORE IDENTITY:\n"
            "- Your name is VAIT, which stands for VVIT's Artificial Intelligence Technology.\n"
            "- When asked 'What is your name?', 'Who are you?', 'What does VAIT stand for?', 'What is VAIT?', or 'Tell me about yourself', you must state clearly:\n"
            "  'My name is VAIT, which stands for VVIT's Artificial Intelligence Technology. I am the official institutional AI assistant for VVIT University (VVITU) and the legacy Vasireddy Venkatadri Institute of Technology (VVIT).'\n"
            "- You must NEVER use or introduce any other expansion for VAIT. The one and only official expansion is 'VVIT's Artificial Intelligence Technology'.\n\n"
            "INSTITUTIONAL RELATIONSHIP & SOURCE STRATEGY:\n"
            "- CURRENT UNIVERSITY (VVITU): https://vvitu.ac.in/ is the primary source for all current information.\n"
            "- LEGACY INSTITUTE (VVIT): https://vvitguntur.com/ is a historical source for past records. Never call it the current website of VVITU.\n"
            "- If conflicting information appears, prefer current VVITU for current queries and legacy VVIT for historical queries.\n\n"
            "STRICT RULES:\n"
            "1. Provide helpful AI responses grounded in institutional context.\n"
            "2. If institutional context is provided, prioritize it in this order: current VVITU sources > legacy VVIT sources > LinkedIn > social > related web.\n"
            "3. If no institutional context is available, answer from your knowledge following institutional guidelines.\n"
            "4. Be concise, student-friendly, and cite sources with their time period.\n\n"
            "RESPONSE STRUCTURE:\n"
            "1. Title (short, clear)\n"
            "2. Explanation (2-4 sentences)\n"
            "3. Key Points (bullet points)\n"
            "4. Source(s) (with period indicated)\n"
        )

    async def add_documents(
        self,
        documents: List[Dict[str, str]],
        metadata: List[Dict],
    ) -> int:
        """
        Add documents to the knowledge base with content-hash deduplication.

        Args:
            documents: List of dicts with 'content' and 'name' keys.
            metadata: List of metadata dicts for each document.

        Returns:
            Number of NEW chunks added (duplicates are silently skipped).
        """
        import hashlib

        # Build set of existing content hashes for dedup
        existing_hashes = self._get_content_hashes()

        all_chunks: List[Dict] = []

        for doc, meta in zip(documents, metadata):
            self._validate_metadata(meta)
            chunks = self.text_chunker.chunk_text(doc["content"])
            total = len(chunks)
            for idx, chunk in enumerate(chunks):
                content_hash = hashlib.md5(chunk.encode("utf-8")).hexdigest()

                if content_hash in existing_hashes:
                    logger.debug(
                        "Skipping duplicate chunk (hash=%s) from %s",
                        content_hash[:8], doc.get("name", "?"),
                    )
                    continue

                existing_hashes.add(content_hash)
                all_chunks.append(
                    {
                        "content": chunk,
                        "metadata": {
                            **meta,
                            "document_name": doc.get("name", "Unknown Document"),
                            "chunk_index": idx,
                            "total_chunks": total,
                            "content_hash": content_hash,
                        },
                    }
                )

        if not all_chunks:
            logger.info("No new chunks to add (all duplicates or empty).")
            return 0

        chunk_texts = [c["content"] for c in all_chunks]
        embeddings = await self.embedding_service.get_embeddings(chunk_texts)

        faiss.normalize_L2(embeddings)
        if self.index is None:
            self._create_empty_index()
        self.index.add(embeddings)
        self.chunks.extend(all_chunks)

        await self._save_index()
        logger.info(
            "Added %d new chunks (index total: %d)",
            len(all_chunks), self.index.ntotal,
        )
        return len(all_chunks)

    def _get_content_hashes(self) -> set:
        """Return a set of content_hash values from all existing chunks."""
        hashes: set = set()
        for chunk in self.chunks:
            meta = chunk.get("metadata", chunk)
            h = meta.get("content_hash")
            if h:
                hashes.add(h)
        return hashes

    def _validate_metadata(self, metadata: Dict) -> None:
        """Validate that required metadata fields are present."""
        missing = [f for f in REQUIRED_METADATA_FIELDS if f not in metadata]
        if missing:
            raise ValueError(f"Missing required metadata fields: {missing}")

    async def _save_index(self) -> None:
        """Save the FAISS index and metadata to disk."""
        index_path = Path(self.settings.faiss_index_path)
        metadata_path = Path(self.settings.metadata_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)

    async def _load_index(self) -> None:
        """Load the FAISS index and metadata from disk."""
        index_path = Path(self.settings.faiss_index_path)
        metadata_path = Path(self.settings.metadata_path)

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    async def hybrid_retrieve_fallback(
        self, query: str, seed_urls: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        """
        Scaffold for future hybrid retrieval.

        If standard FAISS retrieval produces NO chunks above threshold:
          1. Attempt a website crawl for the query topic  (NOT YET ACTIVE)
          2. Temporarily ingest the crawled page(s)
          3. Retry retrieval

        Currently returns an empty list — the full implementation will be
        connected once the website crawler is integrated into the runtime
        pipeline with a configurable fallback flag.

        Args:
            query:     The user's question.
            seed_urls: Optional seed URLs to crawl if fallback is triggered.

        Returns:
            List of RetrievedChunk (empty until fully implemented).
        """
        # ── Phase 1: Standard retrieval ──────────────────────────────
        chunks = await self._retrieve(query)

        if self._has_sufficient_context(chunks):
            return chunks

        # ── Phase 2: Fallback (scaffold — NOT YET ACTIVE) ───────────
        #
        # Future implementation:
        #
        #   from app.services.website_crawler import WebsiteCrawler
        #   crawler = WebsiteCrawler(
        #       allowed_domains=self.settings.allowed_domains,
        #       depth_limit=1,
        #       max_pages=3,
        #   )
        #   pages = crawler.crawl(seed_urls or self.settings.crawl_seed_urls)
        #   for page_dict in pages:
        #       await self.add_documents(
        #           [{"content": page_dict["text"], "name": page_dict.get("url", "web")}],
        #           [{"document_type": "website_page", "authority_level": "medium",
        #             "academic_year": "...", "department": "General"}],
        #       )
        #   chunks = await self._retrieve(query)   # retry
        #

        logger.info(
            "Hybrid fallback triggered for query (no sufficient context). "
            "Scaffold only — returning empty. Query: %.100s",
            query,
        )
        return chunks  # return whatever we got (may be below threshold)

    def get_stats(self) -> Dict:
        """Get comprehensive statistics about the knowledge base."""
        source_types: Dict[str, int] = {}
        for chunk in self.chunks:
            meta = chunk.get("metadata", chunk)
            st = meta.get("source_type", meta.get("document_type", "unknown"))
            source_types[st] = source_types.get(st, 0) + 1

        # Count vectors by category
        website_count = 0
        social_count = 0
        pdf_count = 0
        for chunk in self.chunks:
            meta = chunk.get("metadata", chunk)
            st = meta.get("source_type", "")
            dt = meta.get("document_type", "")
            if st == "website" or dt == "official_website":
                website_count += 1
            elif st == "social_media":
                social_count += 1
            elif st == "pdf" or dt in ("regulation", "syllabus", "notice"):
                pdf_count += 1

        # Average adjusted score from last 10 queries
        avg_adjusted = self._avg_adjusted_score_last_n(10)

        # Last reindex time
        last_reindex = self._get_last_reindex_time()

        # Uptime
        uptime_seconds = round(time.time() - self._start_time, 1)

        return {
            "total_chunks": len(self.chunks),
            "total_vectors": self.index.ntotal if self.index else 0,
            "website_vectors": website_count,
            "social_vectors": social_count,
            "pdf_vectors": pdf_count,
            "similarity_threshold": self.settings.similarity_threshold,
            "top_k": self.settings.top_k_retrieval,
            "source_breakdown": source_types,
            "avg_adjusted_score_last_10_queries": avg_adjusted,
            "last_reindex_time": last_reindex,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_size": len(self._cache),
            "cache_capacity": CACHE_CAPACITY,
            "uptime_seconds": uptime_seconds,
        }

    def _avg_adjusted_score_last_n(self, n: int = 10) -> Optional[float]:
        """Read the last N entries from query.log and compute average top score."""
        try:
            log_file = LOGS_DIR / "query.log"
            if not log_file.exists():
                return None
            lines = log_file.read_text(encoding="utf-8").strip().split("\n")
            recent = lines[-n:] if len(lines) >= n else lines
            scores = []
            for line in recent:
                try:
                    entry = json.loads(line)
                    s = entry.get("top_adjusted_score", entry.get("top_similarity_score", 0))
                    if s and not entry.get("is_refusal", False):
                        scores.append(float(s))
                except (json.JSONDecodeError, ValueError):
                    continue
            return round(sum(scores) / len(scores), 4) if scores else None
        except Exception:
            return None

    @staticmethod
    def _get_last_reindex_time() -> Optional[str]:
        """Read the last reindex timestamp from manage.log if available."""
        try:
            manage_log = LOGS_DIR / "manage.log"
            if not manage_log.exists():
                return None
            lines = manage_log.read_text(encoding="utf-8").strip().split("\n")
            # Search backwards for a reindex completion line
            for line in reversed(lines):
                if "REINDEX COMPLETE" in line or "WEBSITE REINDEX" in line:
                    # Extract timestamp from log format: "2026-02-21 10:30:00 | ..."
                    parts = line.split(" | ")
                    if parts:
                        return parts[0].strip()
            return None
        except Exception:
            return None

    def get_system_summary(self) -> Dict:
        """
        Generate a comprehensive system self-summary for demo and monitoring.

        Returns a dict covering:
          - identity (name, version, engine)
          - knowledge coverage (total docs, per-type breakdown, domains)
          - model configuration (LLM, embedding, chunk settings)
          - intelligence features (active pipeline stages)
          - runtime metrics (uptime, cache stats, avg score)
          - ingestion status (last reindex, watched dirs)
        """
        stats = self.get_stats()
        uptime_sec = round(time.time() - self._start_time, 1)

        # Unique source files
        unique_sources: set = set()
        departments: set = set()
        for chunk in self.chunks:
            meta = chunk.get("metadata", chunk)
            sf = meta.get("source_file", meta.get("document_name", ""))
            if sf:
                unique_sources.add(sf)
            dept = meta.get("department", "")
            if dept and dept.lower() != "general":
                departments.add(dept)

        # Format uptime
        hours, remainder = divmod(int(uptime_sec), 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{hours}h {minutes}m {seconds}s"

        cache_total = self._cache_hits + self._cache_misses
        cache_hit_rate = (
            round(self._cache_hits / cache_total * 100, 1)
            if cache_total > 0 else 0.0
        )

        return {
            # Identity
            "system_name": "VAIT — VVIT's Artificial Intelligence Technology",
            "version": self.settings.app_version,
            "engine": "FAISS IndexFlatIP + Groq/OpenRouter RAG Pipeline",
            "status": "operational",

            # Knowledge coverage
            "knowledge": {
                "total_vectors": stats.get("total_vectors", 0),
                "total_chunks": stats.get("total_chunks", 0),
                "unique_source_documents": len(unique_sources),
                "website_vectors": stats.get("website_vectors", 0),
                "social_vectors": stats.get("social_vectors", 0),
                "pdf_vectors": stats.get("pdf_vectors", 0),
                "source_breakdown": stats.get("source_breakdown", {}),
                "departments_covered": sorted(departments),
                "allowed_domains": self.settings.allowed_domains,
            },

            # Model configuration
            "models": {
                "llm_primary": self.settings.groq_model,
                "llm_fallback": self.settings.openrouter_model,
                "embedding": self.settings.embedding_model,
                "embedding_dimension": self.embedding_service.embedding_dimension,
                "chunk_size_tokens": self.settings.chunk_size,
                "chunk_overlap": self.settings.chunk_overlap,
                "top_k_retrieval": self.settings.top_k_retrieval,
                "similarity_threshold": self.settings.similarity_threshold,
                "max_context_chars": self.settings.max_context_chars,
                "max_context_chunks": MAX_CONTEXT_CHUNKS,
            },

            # Intelligence pipeline
            "intelligence_features": [
                "Authority-weighted scoring (high=1.15, medium=1.05)",
                "Source-type boost (official_website +0.03, pdf +0.05)",
                "Query intent classifier (rule-based, 6 categories)",
                "Smart source prioritization (intent-based dynamic boosts)",
                "Near-duplicate removal (Jaccard >= 0.90)",
                "Adjacent chunk merging",
                "Strict refusal guard (adjusted_score < 0.60)",
                "Hallucination defense (min length + citation check)",
                "Answer structure enforcer (title + bullets + explanation)",
                "Response polish (dedup sentences, 600-word cap)",
                "Citation formatter (numbered, clickable)",
                "LRU response cache (100 entries, 10-min TTL)",
            ],

            # Runtime metrics
            "runtime": {
                "uptime": uptime_str,
                "uptime_seconds": uptime_sec,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "cache_hit_rate_pct": cache_hit_rate,
                "cache_size": len(self._cache),
                "cache_capacity": CACHE_CAPACITY,
                "avg_adjusted_score_last_10": stats.get(
                    "avg_adjusted_score_last_10_queries"
                ),
            },

            # Ingestion status
            "ingestion": {
                "last_reindex_time": stats.get("last_reindex_time"),
                "crawl_seed_urls": self.settings.crawl_seed_urls,
                "crawl_depth_limit": self.settings.crawl_depth_limit,
                "crawl_max_pages": self.settings.crawl_max_pages,
                "auto_watch_enabled": self.settings.auto_watch_enabled,
            },
        }
