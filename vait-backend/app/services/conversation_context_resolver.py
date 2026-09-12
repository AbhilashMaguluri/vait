"""Conversational Context & Reference Resolver for VAIT.

Provides generic, domain-agnostic conversational context tracking, multi-turn
anaphora resolution (pronouns, ordinals, entity filtering), structured evidence
extraction, and deep profile crawl triggering.

Zero hardcoding: works uniformly across faculty, fees, courses, departments,
notices, and leadership.
"""

from datetime import datetime
from enum import Enum
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
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
    CASUAL_NO_CONTEXT = "casual_no_context"


class ExtractedEntity(BaseModel):
    """Structured entity extracted from retrieval evidence or assistant responses."""
    id: str
    name: str
    entity_type: str = "generic"  # faculty, fee, course, department, notice, leadership
    designation: Optional[str] = None
    qualification: Optional[str] = None
    department: Optional[str] = None
    profile_url: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    provenance_url: Optional[str] = None
    order_index: int = 0


class ConversationContext(BaseModel):
    """Persistent conversational state across multiple turns."""
    active_subject: Optional[str] = None
    active_entity_type: Optional[str] = None
    active_entities: List[ExtractedEntity] = Field(default_factory=list)
    active_source_urls: List[str] = Field(default_factory=list)
    institutional_period: Optional[str] = None  # "current_vvitu" or "historical_vvit"
    last_evidence_text: Optional[str] = None
    last_query: Optional[str] = None
    last_answer: Optional[str] = None
    last_updated_ist: str = Field(default_factory=lambda: format_ist())

    def get_entity_by_index(self, index: int) -> Optional[ExtractedEntity]:
        """Return entity at 0-based index, or None if index out of range."""
        if not self.active_entities:
            return None
        if 0 <= index < len(self.active_entities):
            return self.active_entities[index]
        if index == -1 and len(self.active_entities) > 0:
            return self.active_entities[-1]
        return None

    def find_entity_by_name(self, name_snippet: str) -> Optional[ExtractedEntity]:
        """Find entity by full or partial name match."""
        if not name_snippet or not self.active_entities:
            return None
        clean_snippet = re.sub(r"[^a-zA-Z0-9\s]", "", name_snippet).lower().strip()
        for entity in self.active_entities:
            clean_name = re.sub(r"[^a-zA-Z0-9\s]", "", entity.name).lower().strip()
            if clean_snippet in clean_name or clean_name in clean_snippet:
                return entity
        return None


class FollowUpIntent(BaseModel):
    """Resolved follow-up intent with rewritten query and action flags."""
    intent_type: IntentType
    resolved_query: str
    target_entities: List[ExtractedEntity] = Field(default_factory=list)
    needs_deeper_retrieval: bool = False
    target_profile_urls: List[str] = Field(default_factory=list)
    reusable_evidence: Optional[str] = None
    is_institutional: bool = True
    explanation: str = ""


class ConversationContextResolver:
    """Generic resolver for conversational context, entity tracking, and anaphora."""

    # Plural pronoun patterns
    PLURAL_PRONOUNS = [
        "them", "they", "their", "theirs", "these", "those",
        "all of them", "both of them", "each of them", "everyone",
    ]

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

    # Keywords requesting deep details (requiring profile crawl if profile links exist)
    # Includes common typos like 'fecth', 'fech'
    DEEPER_DETAIL_TERMS = [
        "detail", "details", "profile", "profiles", "research", "publication", "publications",
        "experience", "qualification", "qualifications", "contact", "email", "phone",
        "fetch", "fecth", "fech", "cv", "resume", "biodata", "more info", "more information",
        "tell more", "background", "specialization", "area of interest",
    ]

    # Department abbreviations for topic expansion
    DEPARTMENT_KEYWORDS = {
        "cse": "Computer Science and Engineering",
        "ai": "Artificial Intelligence & Data Science",
        "ai&ds": "Artificial Intelligence & Data Science",
        "ai ds": "Artificial Intelligence & Data Science",
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

    def extract_entities_from_evidence(
        self,
        evidence_text: str,
        source_urls: Optional[List[str]] = None,
        entity_type: Optional[str] = None,
        raw_entities: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ExtractedEntity]:
        """Extract structured entities from raw evidence text or rendered DOM blocks."""
        primary_url = source_urls[0] if source_urls else None
        entities: List[ExtractedEntity] = []

        # 0. If pre-extracted raw structured entities are provided, convert them directly
        if raw_entities:
            for idx, item in enumerate(raw_entities):
                if isinstance(item, dict) and item.get("name"):
                    entities.append(
                        ExtractedEntity(
                            id=f"entity_{idx+1}",
                            name=str(item.get("name", "")).strip().replace("**", ""),
                            entity_type=entity_type or "faculty",
                            designation=item.get("designation"),
                            qualification=item.get("qualification"),
                            department=item.get("department"),
                            profile_url=item.get("profile_url"),
                            details=item,
                            provenance_url=primary_url,
                            order_index=idx,
                        )
                    )
            if entities:
                return entities

        if not evidence_text or not evidence_text.strip():
            return []

        # 1. Try Markdown Tables (Faculty tables, fee tables, etc.)
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

                # Find name column
                name_col = next((headers[i] for i in range(len(headers)) if any(k in headers[i] for k in ["name", "faculty", "person", "faculty name"])), None)
                if not name_col:
                    if any(k in headers[0] for k in ["s.no", "sl.no", "#"]) and len(headers) > 1:
                        name_col = headers[1]
                    else:
                        name_col = headers[0]

                raw_name = row_dict.get(name_col, "")
                clean_name = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", raw_name)
                clean_name = re.sub(r"[*_`]", "", clean_name).strip()

                if clean_name and not clean_name.isdigit() and len(clean_name) >= 3:
                    # Look for profile URL link in any column
                    profile_url = None
                    for val in cols:
                        lm = re.search(r"\[([^\]]+)\]\((https?://[^\)]+)\)", val)
                        if lm:
                            profile_url = lm.group(2)
                            break

                    desig_col = next((headers[i] for i in range(len(headers)) if any(k in headers[i] for k in ["designation", "role", "title"])), None)
                    designation = row_dict.get(desig_col) if desig_col else None
                    if designation:
                        designation = re.sub(r"[*_`]", "", designation).strip()

                    qual_col = next((headers[i] for i in range(len(headers)) if any(k in headers[i] for k in ["qualification", "degree"])), None)
                    qualification = row_dict.get(qual_col) if qual_col else None
                    if qualification:
                        qualification = re.sub(r"[*_`]", "", qualification).strip()

                    entities.append(
                        ExtractedEntity(
                            id=f"entity_{idx+1}",
                            name=clean_name,
                            entity_type=entity_type or ("faculty" if any("faculty" in h for h in headers) else "table_item"),
                            designation=designation,
                            qualification=qualification,
                            profile_url=profile_url,
                            details=row_dict,
                            provenance_url=primary_url,
                            order_index=idx,
                        )
                    )
                    idx += 1
            if entities:
                return entities

        # 2. Try Structured Faculty Card Blocks
        card_blocks = re.split(r"(?:FACULTY MEMBER\s+\d+:|--- CARD \d+ ---|CARD \d+:)", evidence_text, flags=re.IGNORECASE)
        if len(card_blocks) > 1:
            idx = 0
            for block in card_blocks[1:]:
                lines = [line.strip() for line in block.splitlines() if line.strip()]
                data: Dict[str, str] = {}
                for line in lines:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        data[k.strip().lower()] = v.strip()

                name = data.get("name")
                if name:
                    desig = data.get("designation") or data.get("role")
                    qual = data.get("qualification") or data.get("qualifications")
                    dept = data.get("department")
                    profile = data.get("profile url") or data.get("profile") or data.get("url")
                    entities.append(
                        ExtractedEntity(
                            id=f"entity_{idx+1}",
                            name=name.replace("**", ""),
                            entity_type="faculty",
                            designation=desig,
                            qualification=qual,
                            department=dept,
                            profile_url=profile,
                            details=data,
                            provenance_url=primary_url,
                            order_index=idx,
                        )
                    )
                    idx += 1
            if entities:
                return entities

        # 3. Try Bullet / Numbered List lines
        list_lines = re.findall(
            r"^[\s*\-•\d.]+\s*(?:(?:Dr\.|Prof\.|Mr\.|Mrs\.|Ms\.)\s+[^\n|,\[\(]+|[A-Z][a-z]+(?:\s+[A-Z][a-z.]+)+)[^\n]*",
            evidence_text,
            flags=re.MULTILINE,
        )
        if list_lines:
            for idx, raw_line in enumerate(list_lines):
                clean_line = re.sub(r"^[\s*\-•\d.]+\s*", "", raw_line).strip()
                if not clean_line or len(clean_line) < 3:
                    continue

                # Look for profile URL link markdown: [Profile](url)
                link_match = re.search(r"\[([^\]]+)\]\((https?://[^\)]+)\)", clean_line)
                profile_url = link_match.group(2) if link_match else None
                clean_line = re.sub(r"\[([^\]]+)\]\((https?://[^\)]+)\)", r" ", clean_line)

                # Extract qualifications in parentheses: (Ph.D, M.Tech)
                qual_match = re.search(r"\(([^)]*(?:Ph\.?D|M\.?Tech|B\.?Tech|M\.E|M\.Sc|MBA|NET|SET)[^)]*)\)", clean_line, flags=re.IGNORECASE)
                qualification = qual_match.group(1).strip() if qual_match else None
                line_without_qual = re.sub(r"\([^)]+\)", "", clean_line).strip()

                # Delimiters: -, |, ,, :
                parts = [p.strip() for p in re.split(r"\s*[-|:,]\s*", line_without_qual) if p.strip()]
                if not parts:
                    continue

                name = parts[0].replace("**", "")
                designation = parts[1] if len(parts) > 1 else None

                # Disqualify false positive headers
                if any(w in name.lower() for w in ["department", "faculty", "syllabus", "academics", "contact us", "overview", "home", "vasireddy", "retrieval method"]):
                    continue

                entities.append(
                    ExtractedEntity(
                        id=f"entity_{idx+1}",
                        name=name,
                        entity_type=entity_type or "faculty",
                        designation=designation,
                        qualification=qualification,
                        profile_url=profile_url,
                        details={"raw_line": raw_line},
                        provenance_url=primary_url,
                        order_index=idx,
                    )
                )

            if entities:
                return entities

        return entities

    def resolve_turn(
        self,
        current_query: str,
        history: Optional[List[Any]] = None,
        context: Optional[ConversationContext] = None,
    ) -> FollowUpIntent:
        """Resolve user's current query against conversational context state."""
        raw_query = current_query.strip()
        query_lower = raw_query.lower()

        if context is None:
            context = ConversationContext()

        # Step 1: Check for casual conversational queries (hi, hello, who are you)
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

        # Step 2: Check for topic expansion / department change (e.g. "what about ECE faculty?", "how about mechanical?")
        for dept_abbr, dept_full in self.DEPARTMENT_KEYWORDS.items():
            if dept_abbr in query_tokens and any(trigger in query_tokens for trigger in ["about", "what", "how", "tell"]):
                entity_noun = context.active_entity_type or "faculty"
                if "faculty" in query_lower or entity_noun == "faculty":
                    resolved = f"VVITU {dept_full} faculty members list and details"
                else:
                    resolved = f"VVITU {dept_full} department information"

                return FollowUpIntent(
                    intent_type=IntentType.FOLLOWUP_TOPIC_EXPANSION,
                    resolved_query=resolved,
                    is_institutional=True,
                    explanation=f"Topic expansion to {dept_full} department."
                )

        # Step 3: Check for ordinal references ("first one", "second person", "last one")
        if context.active_entities:
            for ord_token, ord_idx in self.ORDINAL_TOKEN_MAP.items():
                if ord_token in query_tokens:
                    target_entity = context.get_entity_by_index(ord_idx)
                    if target_entity:
                        needs_deep = any(term in query_lower for term in self.DEEPER_DETAIL_TERMS)
                        target_urls = [target_entity.profile_url] if (target_entity.profile_url and needs_deep) else []

                        subject_label = context.active_subject or "Department"
                        resolved = f"{subject_label} details for {target_entity.name}"
                        if target_entity.designation:
                            resolved += f", {target_entity.designation}"

                        intent = IntentType.FOLLOWUP_DEEPER_RETRIEVAL if target_urls else IntentType.FOLLOWUP_ENTITY_FILTER
                        reusable = self._build_single_entity_evidence(target_entity, context.last_evidence_text)

                        return FollowUpIntent(
                            intent_type=intent,
                            resolved_query=resolved,
                            target_entities=[target_entity],
                            needs_deeper_retrieval=bool(target_urls),
                            target_profile_urls=target_urls,
                            reusable_evidence=reusable,
                            is_institutional=True,
                            explanation=f"Resolved ordinal reference '{ord_token}' to entity '{target_entity.name}' (index {ord_idx})."
                        )

        # Step 4: Check for direct entity name mention from active entities
        if context.active_entities:
            for entity in context.active_entities:
                name_tokens = [t.lower() for t in re.findall(r"[a-zA-Z]{4,}", entity.name)]
                if any(t in query_tokens for t in name_tokens):
                    needs_deep = any(term in query_lower for term in self.DEEPER_DETAIL_TERMS)
                    target_urls = [entity.profile_url] if (entity.profile_url and needs_deep) else []
                    reusable = self._build_single_entity_evidence(entity, context.last_evidence_text)
                    resolved = f"{context.active_subject or 'Department'} profile details for {entity.name}"

                    return FollowUpIntent(
                        intent_type=IntentType.FOLLOWUP_DEEPER_RETRIEVAL if target_urls else IntentType.FOLLOWUP_ENTITY_FILTER,
                        resolved_query=resolved,
                        target_entities=[entity],
                        needs_deeper_retrieval=bool(target_urls),
                        target_profile_urls=target_urls,
                        reusable_evidence=reusable,
                        is_institutional=True,
                        explanation=f"Resolved direct entity mention to '{entity.name}'."
                    )

        # Step 5: Check for plural pronouns ("them", "they", "these", "those", "their ...", "details about them")
        has_plural_pronoun = (
            any(p in query_tokens for p in ["them", "they", "their", "theirs", "these", "those"])
            or any(phrase in query_lower for phrase in ["all of them", "both of them", "each of them", "about them", "details about"])
        )

        if has_plural_pronoun and context.active_entities:
            target_entities = context.active_entities
            names_summary = ", ".join([e.name for e in target_entities[:5]])
            subject = context.active_subject or f"VVITU {context.active_entity_type or 'Department'}"

            # Determine whether deeper crawl is needed (typo-tolerant)
            needs_deep = any(term in query_lower for term in self.DEEPER_DETAIL_TERMS)
            available_profile_urls = [
                e.profile_url for e in target_entities
                if e.profile_url and e.profile_url.startswith("http")
            ]

            max_crawl = getattr(self.settings, "max_deeper_profiles_fetch", 3)
            crawl_urls = available_profile_urls[:max_crawl] if needs_deep else []

            resolved = f"{subject} detailed information and profiles for: {names_summary}"
            intent = IntentType.FOLLOWUP_DEEPER_RETRIEVAL if crawl_urls else IntentType.FOLLOWUP_EVIDENCE_REUSE

            return FollowUpIntent(
                intent_type=intent,
                resolved_query=resolved,
                target_entities=target_entities,
                needs_deeper_retrieval=bool(crawl_urls),
                target_profile_urls=crawl_urls,
                reusable_evidence=context.last_evidence_text,
                is_institutional=True,
                explanation=f"Resolved plural pronoun reference ('them/they') to {len(target_entities)} active entities in '{subject}'."
            )

        # Step 6: Standalone query or new topic
        is_inst = self._is_query_institutional(raw_query)

        # Short implicit follow-up query grounded by active context (e.g. "qualifications?")
        if not is_inst and context.active_entities and len(raw_query.split()) <= 6:
            subject = context.active_subject or "Institutional members"
            resolved = f"{subject} {raw_query}"
            return FollowUpIntent(
                intent_type=IntentType.FOLLOWUP_EVIDENCE_REUSE,
                resolved_query=resolved,
                target_entities=context.active_entities,
                needs_deeper_retrieval=False,
                reusable_evidence=context.last_evidence_text,
                is_institutional=True,
                explanation="Short implicit follow-up query grounded by active context."
            )

        return FollowUpIntent(
            intent_type=IntentType.STANDALONE_NEW_TOPIC,
            resolved_query=raw_query,
            is_institutional=is_inst,
            explanation="Standalone new topic or first turn in conversation."
        )

    def _build_single_entity_evidence(self, entity: ExtractedEntity, full_evidence: Optional[str]) -> str:
        """Construct a focused evidence snippet for a single selected entity."""
        lines = [
            f"SELECTED ENTITY DETAILS:",
            f"- Name: {entity.name}",
        ]
        if entity.designation:
            lines.append(f"- Designation: {entity.designation}")
        if entity.qualification:
            lines.append(f"- Qualification: {entity.qualification}")
        if entity.department:
            lines.append(f"- Department: {entity.department}")
        if entity.profile_url:
            lines.append(f"- Profile URL: {entity.profile_url}")
        if entity.details:
            for k, v in entity.details.items():
                if k.lower() not in {"name", "designation", "qualification", "department", "profile_url", "raw_line"}:
                    lines.append(f"- {k.capitalize()}: {v}")

        if full_evidence and entity.name.lower() in full_evidence.lower():
            lines.append("\nRELEVANT CONTEXT FROM EVIDENCE:")
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
            "curriculum", "sem", "semester", "autonomous", "jntuk",
        ]
        return any(term in q for term in college_terms)
