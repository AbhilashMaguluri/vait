"""Generic Institutional Conversational Context & Entity Resolution Engine for VAIT.

Provides domain-agnostic conversational context tracking, multi-turn anaphora
resolution (pronouns, ordinals, demonstratives, relationship navigation),
structured evidence extraction, ambiguity detection, and evidence reuse across
ANY institutional concept (faculty, fees, departments, courses, hostels, transport,
examinations, admissions, notices, placements, facilities, leadership, etc.).

Architecture Principle:
Determines "What does this refer to?" before deciding "What retrieval should I perform?"
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from app.utils.config import get_settings
from app.utils.timezone import format_ist, now_ist

logger = logging.getLogger("vait.context_resolver")


class IntentType(str, Enum):
    """Classification of conversational follow-up intent."""
    STANDALONE_NEW_TOPIC = "standalone_new_topic"
    FOLLOWUP_EVIDENCE_REUSE = "followup_evidence_reuse"
    FOLLOWUP_DEEPER_RETRIEVAL = "followup_deeper_retrieval"
    FOLLOWUP_ENTITY_FILTER = "followup_entity_filter"
    FOLLOWUP_TOPIC_EXPANSION = "followup_topic_expansion"
    FOLLOWUP_CONCEPT_TRANSITION = "followup_concept_transition"
    AMBIGUOUS_CLARIFICATION = "ambiguous_clarification"
    CASUAL_NO_CONTEXT = "casual_no_context"


class InstitutionalEntity(BaseModel):
    """Generic institutional entity across any VVIT/VVITU concept."""
    entity_id: str = ""
    entity_type: str = "generic"  # faculty, fee, department, course, program, hostel, transport, exam, notice, placement, facility, leadership, etc.
    name: str
    attributes: Dict[str, Any] = Field(default_factory=dict)
    relationships: List[Any] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)
    source_period: Optional[str] = None  # "current" or "historical"
    order_index: int = 0
    confidence: float = 1.0

    @property
    def designation(self) -> Optional[str]:
        return self.attributes.get("designation") or self.attributes.get("role")

    @property
    def qualification(self) -> Optional[str]:
        return self.attributes.get("qualification") or self.attributes.get("degree")

    @property
    def department(self) -> Optional[str]:
        return self.attributes.get("department") or self.attributes.get("dept")

    @property
    def profile_url(self) -> Optional[str]:
        return (
            self.attributes.get("profile_url")
            or self.attributes.get("profile")
            or self.attributes.get("url")
            or (self.source_urls[0] if self.source_urls else None)
        )

    @property
    def details(self) -> Dict[str, Any]:
        return self.attributes


# Backward compatibility alias
ExtractedEntity = InstitutionalEntity


class EntityRelationship(BaseModel):
    """Generic relationship between institutional concepts or entities."""
    source_concept: str = "generic"
    relation: str = "related_to"  # has_fee, has_hod, teaches_in, has_facility, has_schedule, has_date, offered_by, belongs_to
    target_concept: str = "generic"
    target_name: Optional[str] = None
    description: Optional[str] = None

    def __init__(self, **data):
        if "relation_type" in data and "relation" not in data:
            data["relation"] = data.pop("relation_type")
        if "target_type" in data and "target_concept" not in data:
            data["target_concept"] = data.pop("target_type")
        if "source_type" in data and "source_concept" not in data:
            data["source_concept"] = data.pop("source_type")
        super().__init__(**data)

    @property
    def relation_type(self) -> str:
        return self.relation

    @property
    def target_type(self) -> str:
        return self.target_concept


class ConversationContext(BaseModel):
    """Generic persistent conversational state across multiple turns."""
    active_subject: Optional[str] = None  # e.g., "CSE", "AI & DS", "Hostel", "Transport", "Second Year"
    active_concept: Optional[str] = None  # e.g., "fee", "faculty", "department", "hostel", "transport", "examination", "notice", "program"
    active_entities: List[InstitutionalEntity] = Field(default_factory=list)
    active_relationships: List[EntityRelationship] = Field(default_factory=list)
    active_filters: Dict[str, Any] = Field(default_factory=dict)  # e.g., {"year": "2nd year", "category": "merit"}
    active_source_urls: List[str] = Field(default_factory=list)
    institutional_period: Optional[str] = None  # "current_vvitu" or "historical_vvit"
    last_evidence_text: Optional[str] = None
    last_query: Optional[str] = None
    last_answer: Optional[str] = None
    last_response_type: Optional[str] = None
    last_updated_ist: str = Field(default_factory=lambda: format_ist())

    @property
    def active_entity_type(self) -> Optional[str]:
        return self.active_concept or (self.active_entities[0].entity_type if self.active_entities else None)

    @active_entity_type.setter
    def active_entity_type(self, val: Optional[str]):
        self.active_concept = val

    def get_entity_by_index(self, index: int) -> Optional[InstitutionalEntity]:
        """Return entity at 0-based index, or None if out of range."""
        if not self.active_entities:
            return None
        if 0 <= index < len(self.active_entities):
            return self.active_entities[index]
        if index == -1 and len(self.active_entities) > 0:
            return self.active_entities[-1]
        return None

    def find_entity_by_name(self, name_snippet: str) -> Optional[InstitutionalEntity]:
        """Find entity by full or partial name match."""
        if not name_snippet or not self.active_entities:
            return None
        clean_snippet = re.sub(r"[^a-zA-Z0-9\s]", "", name_snippet).lower().strip()
        for entity in self.active_entities:
            clean_name = re.sub(r"[^a-zA-Z0-9\s]", "", entity.name).lower().strip()
            if clean_snippet in clean_name or clean_name in clean_snippet:
                return entity
        return None

    def get_entities_by_type(self, entity_type: str) -> List[InstitutionalEntity]:
        """Return all active entities matching an entity type."""
        return [e for e in self.active_entities if e.entity_type.lower() == entity_type.lower()]


class FollowUpIntent(BaseModel):
    """Resolved follow-up intent with rewritten query, targets, and action flags."""
    intent_type: IntentType
    resolved_query: str
    target_entities: List[InstitutionalEntity] = Field(default_factory=list)
    needs_deeper_retrieval: bool = False
    target_profile_urls: List[str] = Field(default_factory=list)
    reusable_evidence: Optional[str] = None
    clarification_prompt: Optional[str] = None
    is_ambiguous: bool = False
    active_concept: Optional[str] = None
    active_subject: Optional[str] = None
    active_filters: Dict[str, Any] = Field(default_factory=dict)
    is_institutional: bool = True
    explanation: str = ""


class ConversationContextResolver:
    """Generic resolver for conversational context, arbitrary entity tracking, and anaphora."""

    # Plural pronoun indicators
    PLURAL_PRONOUNS: Set[str] = {
        "them", "they", "their", "theirs", "these", "those",
    }
    PLURAL_PHRASES: List[str] = [
        "all of them", "both of them", "each of them", "everyone", "those people",
        "those courses", "those fees", "those departments", "those programs",
        "those subjects", "those notices",
    ]

    # Singular non-person pronouns & demonstratives
    SINGULAR_NON_PERSON: Set[str] = {
        "it", "its", "this", "that",
    }

    # Singular person pronouns
    SINGULAR_PERSON_MALE: Set[str] = {"he", "him", "his"}
    SINGULAR_PERSON_FEMALE: Set[str] = {"she", "her", "hers"}

    # Ordinal token mapping
    ORDINAL_TOKEN_MAP: Dict[str, int] = {
        "first": 0, "1st": 0,
        "second": 1, "2nd": 1,
        "third": 2, "3rd": 2,
        "fourth": 3, "4th": 3,
        "fifth": 4, "5th": 4,
        "sixth": 5, "6th": 5,
        "last": -1, "final": -1,
    }

    # Deep detail attribute keywords (requiring subpage crawl if links exist)
    DEEPER_DETAIL_TERMS: Set[str] = {
        "detail", "details", "profile", "profiles", "research", "publication", "publications",
        "experience", "qualification", "qualifications", "contact", "email", "phone",
        "fetch", "fecth", "fech", "cv", "resume", "biodata", "more info", "more information",
        "tell more", "background", "specialization", "area of interest", "syllabus pdf",
        "full circular", "complete details",
    }

    # Concept keyword lexicon (maps indicator keywords to concept types)
    CONCEPT_INDICATORS: Dict[str, List[str]] = {
        "fee": ["fee", "fees", "tuition", "cost", "payment", "expenses", "charges", "amount"],
        "faculty": ["faculty", "teacher", "teachers", "professor", "professors", "lecturer", "staff", "who teaches", "instructor"],
        "leadership": ["hod", "head", "principal", "dean", "director", "chancellor", "vice chancellor", "chairman"],
        "hostel": ["hostel", "mess", "accommodation", "room", "boarding", "hostels"],
        "transport": ["transport", "bus", "bus route", "commute", "shuttle", "travel", "buses", "routes"],
        "examination": ["exam", "exams", "examination", "timetable", "schedule", "hall ticket", "revaluation", "results"],
        "notice": ["notice", "circular", "announcement", "notification", "update", "notices", "circulars"],
        "placement": ["placement", "placements", "recruiter", "recruiters", "company", "companies", "package", "salary", "job"],
        "admission": ["admission", "admissions", "eligibility", "intake", "seat", "seats", "apply", "application"],
        "course": ["course", "courses", "program", "programs", "b.tech", "m.tech", "curriculum", "syllabus", "degree"],
        "department": ["department", "dept", "departments", "branch", "branches"],
        "facility": ["facility", "facilities", "library", "lab", "laboratory", "laboratories", "canteen", "sports", "gym"],
    }

    # Department & Subject keywords
    DEPARTMENT_KEYWORDS: Dict[str, str] = {
        "cse": "Computer Science and Engineering",
        "ai": "Artificial Intelligence & Data Science",
        "ai&ds": "Artificial Intelligence & Data Science",
        "ai ds": "Artificial Intelligence & Data Science",
        "aids": "Artificial Intelligence & Data Science",
        "ece": "Electronics and Communication Engineering",
        "eee": "Electrical and Electronics Engineering",
        "mech": "Mechanical Engineering",
        "civil": "Civil Engineering",
        "it": "Information Technology",
        "csd": "Computer Science and Design",
        "csm": "Computer Science (AI & ML)",
        "cbs": "Computer Science and Business Systems",
    }

    def __init__(self):
        self.settings = get_settings()

    def detect_concept(self, query: str) -> Optional[str]:
        """Detect the primary institutional concept present in a query."""
        q_lower = query.lower()
        tokens = set(re.findall(r"[a-z0-9&']+", q_lower))
        for concept, indicators in self.CONCEPT_INDICATORS.items():
            for ind in indicators:
                if " " in ind and ind in q_lower:
                    return concept
                elif ind in tokens:
                    return concept
        return None

    def detect_subject(self, query: str) -> Optional[str]:
        """Detect the primary subject or department in a query."""
        q_lower = query.lower()
        tokens = set(re.findall(r"[a-z0-9&']+", q_lower))
        for abbr, full_name in self.DEPARTMENT_KEYWORDS.items():
            if abbr == "it":
                # Only match IT if uppercase in original query or followed by branch/department/engineering
                if re.search(r"\bIT\b", query) or any(phrase in q_lower for phrase in ["it branch", "it dept", "it department", "btech in it", "b.tech it", "b.tech in it"]):
                    return full_name
                continue
            if " " in abbr or "&" in abbr:
                if abbr in q_lower:
                    return full_name
            elif abbr in tokens:
                return full_name
        for general_subj in ["hostel", "transport", "admissions", "admission", "placement", "placements", "examination", "library"]:
            if general_subj in tokens:
                return general_subj.capitalize()
        return None

    def detect_filters(self, query: str) -> Dict[str, Any]:
        """Extract explicit filters such as academic year, semester, or category."""
        filters: Dict[str, Any] = {}
        q_lower = query.lower()

        # Academic year or student year filter
        year_match = re.search(r"\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+year\b", q_lower)
        if year_match:
            filters["academic_year"] = year_match.group(0)
            filters["academic_year_level"] = year_match.group(0)

        # Semester filter
        sem_match = re.search(r"\b(sem(?:ester)?\s*[1-8]|1st\s+sem|2nd\s+sem)\b", q_lower)
        if sem_match:
            filters["semester"] = sem_match.group(0)

        # Category filter (e.g. convenor, management, nri, girls, boys)
        for cat in ["management", "convenor", "cq", "mq", "nri", "boys", "girls"]:
            if cat in q_lower.split():
                filters["category"] = cat

        return filters

    def extract_entities_from_evidence(
        self,
        evidence_text: str,
        source_urls: Optional[List[str]] = None,
        entity_type: Optional[str] = None,
        raw_entities: Optional[List[Dict[str, Any]]] = None,
        concept: Optional[str] = None,
    ) -> List[InstitutionalEntity]:
        """Generic entity extractor capable of parsing arbitrary structured tables, cards, and lists."""
        entity_type = entity_type or concept
        primary_url = source_urls[0] if source_urls else None
        entities: List[InstitutionalEntity] = []

        # 0. Pre-extracted raw structured entities
        if raw_entities:
            for idx, item in enumerate(raw_entities):
                if isinstance(item, dict) and item.get("name"):
                    clean_name = str(item.get("name", "")).strip().replace("**", "")
                    e_type = entity_type or item.get("entity_type") or "generic"
                    entities.append(
                        InstitutionalEntity(
                            entity_id=f"entity_{idx+1}",
                            entity_type=e_type,
                            name=clean_name,
                            attributes=item,
                            source_urls=[item["profile_url"]] if item.get("profile_url") else (source_urls or []),
                            order_index=idx,
                        )
                    )
            if entities:
                return entities

        if not evidence_text or not evidence_text.strip():
            return []

        # 1. Parse Markdown Tables (Faculty tables, fee tables, program tables, exam tables)
        table_lines = [
            l.strip() for l in evidence_text.splitlines()
            if "|" in l and not re.match(r"^\|[-:| ]+\|$", l.strip())
        ]
        if len(table_lines) >= 2:
            raw_headers = [h.strip() for h in table_lines[0].split("|")[1:-1]]
            headers = [h.lower() for h in raw_headers]
            idx = 0
            for row in table_lines[1:]:
                cols = [c.strip() for c in row.split("|")[1:-1]]
                if not cols or len(cols) < 2:
                    continue
                row_dict = {headers[i]: cols[i] for i in range(min(len(headers), len(cols)))}

                # Find candidate name column
                name_col = next(
                    (headers[i] for i in range(len(headers))
                     if any(k in headers[i] for k in ["name", "faculty", "program", "course", "title", "subject", "role", "item", "department"])),
                    None
                )
                if not name_col:
                    if any(k in headers[0] for k in ["s.no", "sl.no", "#"]) and len(headers) > 1:
                        name_col = headers[1]
                    else:
                        name_col = headers[0]

                raw_name = row_dict.get(name_col, "")
                clean_name = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", raw_name)
                clean_name = re.sub(r"[*_`]", "", clean_name).strip()

                if clean_name and not clean_name.isdigit() and len(clean_name) >= 2:
                    target_urls = []
                    for val in cols:
                        lm = re.search(r"\[([^\]]+)\]\((https?://[^\)]+)\)", val)
                        if lm:
                            target_urls.append(lm.group(2))

                    inferred_type = entity_type or ("faculty" if any("faculty" in h for h in headers)
                                                    else "fee" if any("fee" in h for h in headers)
                                                    else "course" if any("program" in h or "course" in h for h in headers)
                                                    else "item")
                    entities.append(
                        InstitutionalEntity(
                            entity_id=f"entity_{idx+1}",
                            entity_type=inferred_type,
                            name=clean_name,
                            attributes=row_dict,
                            source_urls=target_urls or (source_urls or []),
                            order_index=idx,
                        )
                    )
                    idx += 1
            if entities:
                return entities

        # 2. Parse Structured Card Blocks (e.g. CARD 1: Name: ..., Details: ..., DEPARTMENT: ...)
        card_blocks = re.split(r"^\s*(?:FACULTY MEMBER\s+\d+:|--- CARD \d+ ---|CARD \d+:|ITEM \d+:|DEPARTMENT\s+\d+:|FACILITY\s+\d+:)", evidence_text, flags=re.IGNORECASE | re.MULTILINE)
        if len(card_blocks) > 1:
            idx = 0
            for block in card_blocks[1:]:
                lines = [line.strip() for line in block.splitlines() if line.strip()]
                data: Dict[str, Any] = {}
                name = None
                for line_idx, line in enumerate(lines):
                    if ":" in line:
                        k, v = line.split(":", 1)
                        data[k.strip().lower()] = v.strip()
                    elif line_idx == 0:
                        name = line.strip()

                if not name:
                    name = data.get("name") or data.get("title") or data.get("program") or data.get("facility") or data.get("department")

                if name:
                    clean_name = re.sub(r"[*_`]", "", name).strip()
                    urls = [data[k] for k in ["profile url", "profile", "url", "link"] if k in data and data[k].startswith("http")]
                    rels = []
                    hod = data.get("head of department") or data.get("hod") or data.get("head")
                    if hod:
                        rels.append(EntityRelationship(source_concept="department", relation="hod_of", target_concept="faculty", target_name=hod))

                    entities.append(
                        InstitutionalEntity(
                            entity_id=f"entity_{idx+1}",
                            entity_type=entity_type or "item",
                            name=clean_name,
                            attributes=data,
                            relationships=rels,
                            source_urls=urls or (source_urls or []),
                            order_index=idx,
                        )
                    )
                    idx += 1
            if entities:
                return entities

        # 3. Parse Bulleted or Numbered Lists
        list_lines = [l.strip() for l in evidence_text.splitlines() if re.match(r"^[\s*\-•\d.]+\s+[A-Za-z0-9]", l)]
        if list_lines:
            idx = 0
            for raw_line in list_lines:
                clean_line = re.sub(r"^[\s*\-•\d.]+\s*", "", raw_line).strip()
                if not clean_line or len(clean_line) < 3:
                    continue

                link_match = re.search(r"\[([^\]]+)\]\((https?://[^\)]+)\)", clean_line)
                target_url = link_match.group(2) if link_match else None
                clean_text = re.sub(r"\[([^\]]+)\]\((https?://[^\)]+)\)", r"\1", clean_line)

                parts = [p.strip() for p in re.split(r"\s*[-|:,]\s*", clean_text) if p.strip()]
                if not parts:
                    continue

                entity_name = re.sub(r"[*_`]", "", parts[0]).strip()
                if any(w in entity_name.lower() for w in ["department", "faculty", "syllabus", "academics", "contact us", "overview", "retrieval method"]):
                    continue

                attrs: Dict[str, Any] = {"raw_line": raw_line}
                if len(parts) > 1:
                    attrs["description"] = parts[1]
                if len(parts) > 2:
                    attrs["detail"] = parts[2]

                entities.append(
                    InstitutionalEntity(
                        entity_id=f"entity_{idx+1}",
                        entity_type=entity_type or "item",
                        name=entity_name,
                        attributes=attrs,
                        source_urls=[target_url] if target_url else (source_urls or []),
                        order_index=idx,
                    )
                )
                idx += 1
            if entities:
                return entities

        return entities

    def is_evidence_sufficient(
        self,
        requested_concept: Optional[str],
        requested_subject: Optional[str],
        query: str,
        context: ConversationContext,
    ) -> bool:
        """Entity-agnostic check whether existing evidence contains the answer."""
        if not context.last_evidence_text or not context.last_evidence_text.strip():
            return False

        ev_lower = context.last_evidence_text.lower()
        q_lower = query.lower()
        tokens = set(re.findall(r"[a-z0-9&']+", q_lower))

        # Check requested attributes against evidence text or entity attributes
        attribute_indicators = {
            "qualification": ["ph.d", "m.tech", "b.tech", "qualification", "degree", "doctorate"],
            "designation": ["professor", "assistant professor", "associate professor", "hod", "lecturer"],
            "fee": ["fee", "tuition", "amount", "rs", "inr", "₹", "per year", "per annum"],
            "date": ["published", "date", "dated", "202", "notification"],
            "hod": ["head of the department", "hod", "head:"],
            "duration": ["years", "semesters", "duration", "4 years", "2 years"],
            "eligibility": ["eligibility", "intermediate", "eamcet", "cutoff", "marks"],
            "hostel": ["hostel", "mess", "boarding", "rooms"],
            "transport": ["bus", "route", "stops", "boarding point"],
        }

        # Deeper retrieval cues always require fetching if profile links exist
        if any(term in q_lower for term in ["fetch", "fecth", "profile", "profiles", "research", "publication", "publications", "biodata", "cv", "resume"]):
            return False

        # If user is asking for specific attribute, verify it exists in evidence
        for attr, keywords in attribute_indicators.items():
            query_has_attr = (attr in tokens) or (f"{attr}s" in tokens) or any(kw in tokens for kw in keywords) or any(kw in q_lower for kw in keywords)
            if query_has_attr:
                if any(kw in ev_lower for kw in keywords):
                    return True
                else:
                    return False

        # If user asks for general details/more info and evidence has tabular/card content
        if any(term in q_lower for term in ["details", "more about", "tell more", "information", "tell about"]):
            if len(context.last_evidence_text) >= 250:
                return True

        return False

    def resolve_turn(
        self,
        current_query: str,
        history: Optional[List[Any]] = None,
        context: Optional[ConversationContext] = None,
    ) -> FollowUpIntent:
        """Resolve user's current query against generic institutional context state."""
        raw_query = current_query.strip()
        query_lower = raw_query.lower()

        if context is None:
            context = ConversationContext()

        # Step 1: Casual queries
        casual_patterns = [
            r"^(?:hi|hello|hey|greetings|hola)(?:\s+vait)?(?:[!?.])?$",
            r"^(?:who\s+are\s+you|what\s+are\s+you|what\s+can\s+you\s+do)[?.]?$",
            r"^(?:tell\s+me\s+a\s+joke|how\s+are\s+you)[?.]?$",
        ]
        for cp in casual_patterns:
            if re.match(cp, query_lower):
                return FollowUpIntent(
                    intent_type=IntentType.CASUAL_NO_CONTEXT,
                    resolved_query=raw_query,
                    is_institutional=False,
                    explanation="Casual interaction query; bypasses institutional context."
                )

        query_tokens = set(re.findall(r"[a-z0-9&']+", query_lower))

        # Detect candidate concepts, subjects, and filters in the current query
        detected_concept = self.detect_concept(raw_query)
        detected_subject = self.detect_subject(raw_query)
        detected_filters = self.detect_filters(raw_query)

        # ── Step 2: Ambiguity Detection ───────────────────────────────
        # If the query contains an ambiguous reference across multiple active entities/departments
        # e.g., Context has [CSE, ECE] and user asks "Who is their HOD?"
        if context.active_entities and len(context.active_entities) > 1:
            multi_antecedent_subjects = [
                e.name for e in context.active_entities
                if e.entity_type in ("department", "program", "school")
            ]
            has_ambiguous_rel = any(rel in query_tokens for rel in ["hod", "head", "fee", "fees", "dean", "placement"])
            has_ambiguous_pronoun = any(p in query_tokens for p in ["their", "them", "both", "the department"])

            if len(multi_antecedent_subjects) >= 2 and has_ambiguous_rel and has_ambiguous_pronoun:
                subject_choices = ", ".join(multi_antecedent_subjects[:3])
                clarification = f"Could you please clarify whether you would like information for {subject_choices}, or both?"
                return FollowUpIntent(
                    intent_type=IntentType.AMBIGUOUS_CLARIFICATION,
                    resolved_query=raw_query,
                    clarification_prompt=clarification,
                    is_ambiguous=True,
                    is_institutional=True,
                    explanation="Multiple plausible antecedents detected; requesting user clarification."
                )

        # ── Step 3: Ordinal References ("first one", "second one", "3rd", "last one") ───
        # Resolves against ANY active entity list (programs, courses, fees, faculty, notices)
        if context.active_entities:
            for ord_token, ord_idx in self.ORDINAL_TOKEN_MAP.items():
                if ord_token in query_tokens:
                    target_entity = context.get_entity_by_index(ord_idx)
                    if target_entity:
                        needs_deep = any(term in query_lower for term in self.DEEPER_DETAIL_TERMS)
                        target_urls = target_entity.source_urls if (target_entity.source_urls and needs_deep) else []

                        subject_label = context.active_subject or "VVITU"
                        concept_label = target_entity.entity_type
                        resolved = f"{subject_label} {concept_label} details for {target_entity.name}"

                        # Check evidence sufficiency
                        reusable = self._build_single_entity_evidence(target_entity, context.last_evidence_text)
                        has_enough = self.is_evidence_sufficient(concept_label, target_entity.name, raw_query, context)

                        intent = (
                            IntentType.FOLLOWUP_DEEPER_RETRIEVAL if (target_urls and not has_enough)
                            else IntentType.FOLLOWUP_ENTITY_FILTER
                        )

                        return FollowUpIntent(
                            intent_type=intent,
                            resolved_query=resolved,
                            target_entities=[target_entity],
                            needs_deeper_retrieval=bool(target_urls and not has_enough),
                            target_profile_urls=target_urls if (target_urls and not has_enough) else [],
                            reusable_evidence=reusable,
                            active_concept=concept_label,
                            active_subject=context.active_subject,
                            is_institutional=True,
                            explanation=f"Resolved ordinal reference '{ord_token}' to '{target_entity.name}' ({concept_label})."
                        )

        # ── Step 4: Direct Entity Name Mention ─────────────────────────
        if context.active_entities:
            for entity in context.active_entities:
                name_tokens = [t.lower() for t in re.findall(r"[a-zA-Z0-9]{3,}", entity.name)]
                if len(name_tokens) >= 1 and any(t in query_tokens for t in name_tokens if t not in {"and", "for", "the", "b.tech", "m.tech", "vvit", "vvitu"}):
                    needs_deep = any(term in query_lower for term in self.DEEPER_DETAIL_TERMS)
                    target_urls = entity.source_urls if (entity.source_urls and needs_deep) else []
                    reusable = self._build_single_entity_evidence(entity, context.last_evidence_text)
                    resolved = f"{context.active_subject or 'VVITU'} details for {entity.name}"

                    has_enough = self.is_evidence_sufficient(entity.entity_type, entity.name, raw_query, context)
                    intent = (
                        IntentType.FOLLOWUP_DEEPER_RETRIEVAL if (target_urls and not has_enough)
                        else IntentType.FOLLOWUP_ENTITY_FILTER
                    )

                    return FollowUpIntent(
                        intent_type=intent,
                        resolved_query=resolved,
                        target_entities=[entity],
                        needs_deeper_retrieval=bool(target_urls and not has_enough),
                        target_profile_urls=target_urls if (target_urls and not has_enough) else [],
                        reusable_evidence=reusable,
                        active_concept=entity.entity_type,
                        active_subject=context.active_subject,
                        is_institutional=True,
                        explanation=f"Resolved direct entity mention to '{entity.name}'."
                    )

        # ── Step 5: Sibling Subject Transition under Same Concept ───────
        # e.g., Context: concept="fee", subject="CSE" -> Query: "What about AI & DS?"
        # Retains active_concept ("fee"), updates subject to "AI & DS"
        if (context.active_subject or context.active_concept) and detected_subject and detected_subject != context.active_subject:
            is_transition_trigger = any(trig in query_tokens for trig in ["about", "what", "how", "for", "and", "is"])
            if is_transition_trigger or len(raw_query.split()) <= 4:
                active_concept = context.active_concept or detected_concept or "general"
                resolved = f"VVITU {detected_subject} {active_concept}"

                # Update context incrementally
                context.active_subject = detected_subject
                context.active_concept = active_concept

                return FollowUpIntent(
                    intent_type=IntentType.FOLLOWUP_CONCEPT_TRANSITION,
                    resolved_query=resolved,
                    active_concept=active_concept,
                    active_subject=detected_subject,
                    is_institutional=True,
                    explanation=f"Sibling subject transition to '{detected_subject}' under active concept '{active_concept}'."
                )

        # ── Step 6: Singular Pronoun Resolution ("it", "this", "that", "its") ─
        # Resolves to non-person concept or specific entity attribute comparison
        # e.g., "What is the CSE fee?" -> "Is it the same for AI & DS?" / "Is it higher?"
        has_singular_non_person = bool(self.SINGULAR_NON_PERSON.intersection(query_tokens))
        if has_singular_non_person and context.active_concept:
            active_concept = context.active_concept
            target_subject = detected_subject or context.active_subject or "VVITU"

            # Check for comparison: "Is it the same for X?", "Is it higher?"
            is_comparison = any(c in query_tokens for c in ["same", "higher", "lower", "different", "more", "less", "equal", "compare"])
            if is_comparison:
                resolved = f"Compare VVITU {context.active_subject or 'CSE'} {active_concept} and {target_subject} {active_concept}"
            elif detected_subject:
                resolved = f"VVITU {target_subject} {active_concept}"
            else:
                resolved = f"VVITU {context.active_subject or ''} {active_concept} {raw_query}".strip()

            has_enough = self.is_evidence_sufficient(active_concept, target_subject, raw_query, context)
            intent = IntentType.FOLLOWUP_EVIDENCE_REUSE if has_enough else IntentType.FOLLOWUP_CONCEPT_TRANSITION

            return FollowUpIntent(
                intent_type=intent,
                resolved_query=resolved,
                target_entities=context.active_entities,
                reusable_evidence=context.last_evidence_text if has_enough else None,
                active_concept=active_concept,
                active_subject=target_subject,
                is_institutional=True,
                explanation=f"Resolved singular pronoun ('it/this') to active concept '{active_concept}' for '{target_subject}'."
            )

        # ── Step 7: Singular Person Pronoun ("he", "him", "his", "she", "her") ──
        # Resolves to active person/leadership/faculty entity
        # e.g., "Who is the CSE HOD?" -> "What are his qualifications?"
        has_person_pronoun = bool(
            self.SINGULAR_PERSON_MALE.intersection(query_tokens)
            or self.SINGULAR_PERSON_FEMALE.intersection(query_tokens)
        )
        if has_person_pronoun and (context.active_entities or context.active_subject):
            target_person = next(
                (e for e in context.active_entities if e.entity_type in ("faculty", "leadership", "person")),
                (context.active_entities[0] if context.active_entities else None)
            )
            person_name = target_person.name if target_person else f"{context.active_subject} HOD / Head"
            resolved = f"{person_name} qualifications, profile and details"

            has_enough = self.is_evidence_sufficient("faculty", person_name, raw_query, context)
            target_urls = target_person.source_urls if (target_person and target_person.source_urls and not has_enough) else []

            intent = (
                IntentType.FOLLOWUP_DEEPER_RETRIEVAL if target_urls
                else IntentType.FOLLOWUP_EVIDENCE_REUSE if has_enough
                else IntentType.FOLLOWUP_CONCEPT_TRANSITION
            )
            reusable = self._build_single_entity_evidence(target_person, context.last_evidence_text) if target_person else context.last_evidence_text

            return FollowUpIntent(
                intent_type=intent,
                resolved_query=resolved,
                target_entities=[target_person] if target_person else [],
                needs_deeper_retrieval=bool(target_urls),
                target_profile_urls=target_urls,
                reusable_evidence=reusable if has_enough else None,
                active_concept="leadership" if context.active_concept == "leadership" else "faculty",
                active_subject=context.active_subject,
                is_institutional=True,
                explanation=f"Resolved person pronoun ('he/his/she/her') to '{person_name}'."
            )

        # ── Step 8: Relationship Navigation on Active Subject ──────────
        # e.g., "Tell me about CSE" -> "Who is the HOD?" (department -> leadership)
        # e.g., "Tell me about AI & DS department" -> "Who teaches there?" (department -> faculty)
        # e.g., "Tell me about transport" -> "What are the fees?" (transport -> fee)
        # e.g., "What are hostel facilities?" -> "What are their fees?" (hostel -> fee)
        # e.g., "Show me latest admission notice" -> "When was it published?" (notice -> date)
        if context.active_subject:
            rel_query = query_lower

            # Relation: HOD / Leadership
            if any(k in query_tokens for k in ["hod", "head", "principal", "dean", "incharge"]):
                resolved = f"VVITU {context.active_subject} Head of Department / HOD"
                target_hod_entities = []
                for ent in context.active_entities:
                    for rel in ent.relationships:
                        rel_name = getattr(rel, "relation", "") or getattr(rel, "relation_type", "")
                        if rel_name == "hod_of" and getattr(rel, "target_name", None):
                            target_hod_entities.append(
                                InstitutionalEntity(
                                    name=rel.target_name,
                                    entity_type=getattr(rel, "target_concept", "faculty")
                                )
                            )
                has_enough = self.is_evidence_sufficient("leadership", context.active_subject, raw_query, context) or bool(target_hod_entities)
                intent = IntentType.FOLLOWUP_EVIDENCE_REUSE if has_enough else IntentType.FOLLOWUP_CONCEPT_TRANSITION
                return FollowUpIntent(
                    intent_type=intent,
                    resolved_query=resolved,
                    target_entities=target_hod_entities,
                    reusable_evidence=context.last_evidence_text if has_enough else None,
                    active_concept="leadership",
                    active_subject=context.active_subject,
                    is_institutional=True,
                    explanation=f"Relationship navigation: '{context.active_subject}' -> 'leadership/HOD'."
                )

            # Relation: Who teaches / Faculty
            if "who teaches" in rel_query or ("teaches" in query_tokens and "there" in query_tokens):
                resolved = f"VVITU {context.active_subject} faculty members list"
                has_enough = self.is_evidence_sufficient("faculty", context.active_subject, raw_query, context)
                intent = IntentType.FOLLOWUP_EVIDENCE_REUSE if has_enough else IntentType.FOLLOWUP_CONCEPT_TRANSITION
                return FollowUpIntent(
                    intent_type=intent,
                    resolved_query=resolved,
                    reusable_evidence=context.last_evidence_text if has_enough else None,
                    active_concept="faculty",
                    active_subject=context.active_subject,
                    is_institutional=True,
                    explanation=f"Relationship navigation: '{context.active_subject}' -> 'faculty'."
                )

            # Relation: Fees of active context (Hostel, Transport, Admissions)
            if any(k in query_tokens for k in ["fee", "fees", "charges", "cost", "tariff"]):
                resolved = f"VVITU {context.active_subject} fees structure and charges"
                has_enough = self.is_evidence_sufficient("fee", context.active_subject, raw_query, context)
                intent = IntentType.FOLLOWUP_EVIDENCE_REUSE if has_enough else IntentType.FOLLOWUP_CONCEPT_TRANSITION
                return FollowUpIntent(
                    intent_type=intent,
                    resolved_query=resolved,
                    reusable_evidence=context.last_evidence_text if has_enough else None,
                    active_concept="fee",
                    active_subject=context.active_subject,
                    is_institutional=True,
                    explanation=f"Relationship navigation: '{context.active_subject}' -> 'fees'."
                )

            # Relation: Date / Publication of Notice
            if any(k in query_tokens for k in ["published", "date", "when", "time", "deadline"]):
                resolved = f"VVITU {context.active_subject} publication date and deadlines"
                has_enough = self.is_evidence_sufficient("date", context.active_subject, raw_query, context)
                intent = IntentType.FOLLOWUP_EVIDENCE_REUSE if has_enough else IntentType.FOLLOWUP_CONCEPT_TRANSITION
                return FollowUpIntent(
                    intent_type=intent,
                    resolved_query=resolved,
                    reusable_evidence=context.last_evidence_text if has_enough else None,
                    active_concept=context.active_concept or "notice",
                    active_subject=context.active_subject,
                    is_institutional=True,
                    explanation=f"Relationship navigation: '{context.active_subject}' -> 'date/timeline'."
                )

        # ── Step 9: Plural Pronoun Resolution ("them", "they", "their", "these", "those") ─
        # Works across ANY entity type (faculty, programs, courses, fees, notices, hostels)
        has_plural_pronoun = bool(
            self.PLURAL_PRONOUNS.intersection(query_tokens)
            or any(phrase in query_lower for phrase in self.PLURAL_PHRASES)
        )

        if has_plural_pronoun and context.active_entities:
            target_entities = context.active_entities
            names_summary = ", ".join([e.name for e in target_entities[:5]])
            subject = context.active_subject or f"VVITU {context.active_concept or 'Department'}"
            concept = context.active_concept or target_entities[0].entity_type

            has_enough = self.is_evidence_sufficient(concept, subject, raw_query, context)
            needs_deep = any(term in query_lower for term in self.DEEPER_DETAIL_TERMS)

            # Collect available subpage URLs for entities
            crawl_urls: List[str] = []
            if needs_deep and not has_enough:
                for e in target_entities:
                    crawl_urls.extend([u for u in e.source_urls if u.startswith("http")])
                max_crawl = getattr(self.settings, "max_deeper_profiles_fetch", 3)
                crawl_urls = crawl_urls[:max_crawl]

            resolved = f"{subject} {concept} detailed information for: {names_summary}"
            intent = IntentType.FOLLOWUP_DEEPER_RETRIEVAL if crawl_urls else IntentType.FOLLOWUP_EVIDENCE_REUSE

            return FollowUpIntent(
                intent_type=intent,
                resolved_query=resolved,
                target_entities=target_entities,
                needs_deeper_retrieval=bool(crawl_urls),
                target_profile_urls=crawl_urls,
                reusable_evidence=context.last_evidence_text,
                active_concept=concept,
                active_subject=subject,
                is_institutional=True,
                explanation=f"Resolved plural pronoun ('them/they/these') to {len(target_entities)} active {concept} entities in '{subject}'."
            )

        # ── Step 10: Implicit Follow-up Grounded by Active Context ─────
        # e.g., "fees?", "timings?", "qualifications?", "second year?", "is there any update for 3rd year?"
        is_inst = self._is_query_institutional(raw_query)
        if (bool(detected_filters) or not is_inst or len(raw_query.split()) <= 10) and (context.active_subject or context.active_concept):
            subject = context.active_subject or "VVITU"
            concept = detected_concept or context.active_concept or "general"
            resolved = f"{subject} {concept} {raw_query}".strip()

            has_filter_in_ev = False
            if detected_filters and context.last_evidence_text:
                for fval in detected_filters.values():
                    if all(part in context.last_evidence_text.lower() for part in str(fval).lower().split()):
                        has_filter_in_ev = True
                        break

            has_enough = self.is_evidence_sufficient(concept, subject, raw_query, context) or has_filter_in_ev
            intent = IntentType.FOLLOWUP_EVIDENCE_REUSE if has_enough else IntentType.FOLLOWUP_CONCEPT_TRANSITION

            return FollowUpIntent(
                intent_type=intent,
                resolved_query=resolved,
                target_entities=context.active_entities,
                needs_deeper_retrieval=False,
                reusable_evidence=context.last_evidence_text if has_enough else None,
                active_concept=concept,
                active_subject=subject,
                active_filters=detected_filters,
                is_institutional=True,
                explanation=f"Implicit follow-up grounded by active context ({subject} / {concept})."
            )

        # ── Step 11: Standalone Query or New Topic ─────────────────────
        return FollowUpIntent(
            intent_type=IntentType.STANDALONE_NEW_TOPIC,
            resolved_query=raw_query,
            active_concept=detected_concept,
            active_subject=detected_subject,
            is_institutional=is_inst,
            explanation="Standalone new topic or primary query."
        )

    def _build_single_entity_evidence(self, entity: InstitutionalEntity, full_evidence: Optional[str]) -> str:
        """Construct a focused evidence snippet for a single selected entity."""
        lines = [
            f"SELECTED ENTITY DETAILS:",
            f"- Name: {entity.name}",
            f"- Entity Type: {entity.entity_type}",
        ]
        if entity.designation:
            lines.append(f"- Designation: {entity.designation}")
        if entity.qualification:
            lines.append(f"- Qualification: {entity.qualification}")
        if entity.department:
            lines.append(f"- Department: {entity.department}")
        if entity.profile_url:
            lines.append(f"- Official URL: {entity.profile_url}")

        if entity.attributes:
            for k, v in entity.attributes.items():
                if k.lower() not in {"name", "designation", "qualification", "department", "profile_url", "raw_line", "entity_type"}:
                    lines.append(f"- {k.replace('_', ' ').capitalize()}: {v}")

        if full_evidence and entity.name.lower() in full_evidence.lower():
            lines.append("\nRELEVANT CONTEXT FROM PREVIOUS EVIDENCE:")
            for chunk in full_evidence.splitlines():
                if entity.name.lower() in chunk.lower():
                    lines.append(chunk.strip())

        return "\n".join(lines)

    def _is_query_institutional(self, query: str) -> bool:
        """Determine if a query is related to the college/university."""
        q = query.lower().strip()
        college_terms = [
            "vvit", "vvitu", "college", "campus", "university", "institute",
            "faculty", "professor", "hod", "department", "cse", "ece", "eee", "mech",
            "civil", "fee", "fees", "hostel", "placement", "placements", "exam", "exams",
            "timetable", "schedule", "syllabus", "regulation", "admissions", "principal",
            "dean", "director", "chancellor", "vice chancellor", "library", "bus", "transport",
            "curriculum", "sem", "semester", "autonomous", "jntuk", "facility", "facilities",
            "scholarship", "scholarships", "laboratory", "labs",
        ]
        return any(term in q for term in college_terms)
