"""Evidence Quality Validator for VAIT Official Web Retrieval.

Verifies that retrieved web content or rendered DOM contains actual evidentiary
facts matching the user query's intent (e.g. faculty names, qualifications, course tables,
fee amounts, circular notices) rather than empty shells, navigation headers, or error notices.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


class EvidenceQualityValidator:
    """
    Validates that an OfficialWebResult or rendered content contains sufficient,
    high-quality evidence for the query intent before accepting it.
    """

    @staticmethod
    def classify_evidence_type(query: str, url: str = "") -> str:
        """Determine what kind of evidence the user query expects."""
        q = query.lower()
        u = url.lower()

        if any(k in q or k in u for k in ["faculty", "professor", "professors", "teacher", "teachers", "staff", "hod", "lecturer"]):
            return "faculty"
        if any(k in q or k in u for k in ["course", "curriculum", "syllabus", "subject", "subjects", "regulation", "r20", "r23", "b.tech", "m.tech"]):
            return "course"
        if any(k in q or k in u for k in ["fee", "fees", "tuition", "hostel fee", "payment", "scholarship"]):
            return "fee"
        if any(k in q or k in u for k in ["exam", "examination", "timetable", "schedule", "result", "results", "hall ticket", "revaluation"]):
            return "exam"
        if any(k in q or k in u for k in ["notice", "circular", "announcement", "notifications", "orders"]):
            return "notice"
        if any(k in q or k in u for k in ["admission", "intake", "eligibility", "eamcet", "counseling", "seats"]):
            return "admission"

        return "general"

    @classmethod
    def validate_content(
        cls,
        query: str,
        content: str,
        entities: Optional[List[Dict[str, Any]]] = None,
        retrieval_method: str = "catalog",
    ) -> bool:
        """
        Validate that the content / entities satisfy the query's evidentiary expectations.
        Returns True if evidence is sufficient, False if it is an empty shell or failed placeholder.
        """
        if not content or len(content.strip()) < 50:
            return False

        # Reject explicit placeholder / extraction failure strings
        failure_markers = [
            "content could not be automatically extracted",
            "extraction is incomplete",
            "direct page link verified and available",
            "navigation menu",
        ]
        content_lower = content.lower()
        if any(m in content_lower for m in failure_markers) and (not entities or len(entities) == 0):
            return False

        evidence_type = cls.classify_evidence_type(query)

        # ── FACULTY VALIDATION ───────────────────────────────────────
        if evidence_type == "faculty":
            # If structured entities are present with names, valid!
            if entities and len(entities) > 0:
                has_names = any(bool(e.get("name")) for e in entities)
                if has_names:
                    return True

            # In markdown text: look for faculty designations or titles
            faculty_indicators = [
                "professor", "assistant professor", "associate professor",
                "dr.", "dr ", "ph.d", "m.tech", "m.e", "b.tech", "hod",
                "head of department", "qualification", "designation"
            ]
            matched_indicators = [ind for ind in faculty_indicators if ind in content_lower]
            if len(matched_indicators) >= 2:
                # Check for table rows or bulleted person names
                has_table_rows = bool(re.search(r'\|\s*\d+\s*\|', content))
                has_person_markers = bool(re.search(r'\*\*(?:Dr\.?|Mr\.?|Mrs\.?|Ms\.?|[A-Z][a-z]+)\s+[A-Za-z\.\s]+\*\*', content))
                if has_table_rows or has_person_markers:
                    return True

            return False

        # ── EXAM VALIDATION ──────────────────────────────────────────
        elif evidence_type == "exam":
            exam_terms = ["exam", "examination", "timetable", "schedule", "controller", "hall ticket", "date", "semester"]
            matches = sum(1 for t in exam_terms if t in content_lower)
            return matches >= 2

        # ── FEE VALIDATION ───────────────────────────────────────────
        elif evidence_type == "fee":
            fee_terms = ["rs.", "inr", "/-", "fee", "tuition", "payment", "amount", "per annum", "per year"]
            matches = sum(1 for t in fee_terms if t in content_lower)
            return matches >= 2

        # ── GENERAL / OTHER ──────────────────────────────────────────
        return len(content.strip()) >= 120
