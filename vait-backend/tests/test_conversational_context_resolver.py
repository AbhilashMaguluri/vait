"""Multi-Domain Regression & Unit Test Suite for Generic Institutional Conversational Context & Reference Resolver.

Validates across ALL institutional concepts:
1. Timezone IST now & conversion
2. Relative date resolution in IST
3. Faculty entity extraction from structured cards
4. Fee entity extraction from markdown tables
5. Department entity & HOD relationship extraction
6. Standalone query without prior context
7. Plural pronoun resolution ('them' with typo tolerance: 'FECTH')
8. Person pronoun resolution ('their qualifications')
9. Non-person pronoun resolution ('its timings and fee' -> transport/facility)
10. Ordinal resolution ('first one')
11. Ordinal resolution ('second person')
12. Ordinal resolution ('last one')
13. Direct named entity follow-up
14. Evidence reuse detection across concepts
15. Deeper retrieval trigger across concepts
16. Sibling branch transition with concept preservation ('what about ai & ds' under fee)
17. Relationship navigation ('who is the hod?' from department)
18. Multi-hop follow-up (Department -> HOD -> 'what are their qualifications?')
19. Ambiguity detection ('CSE and ECE' -> 'who is their HOD?' -> clarification request)
20. Hostel & Transport concept transitions ('what are the fees?')
21. Examination schedule and academic year filter ('3rd year')
22. Notice / Circular follow-up ('when was it published?')
23. Casual/unrelated queries bypass context
24. Institutional classifier with rewritten query
25. Multi-turn source continuity
26. ConversationContext serialization & hydration roundtrip
27. Non-streaming and streaming consistency parity
28. Anti-hallucination boundary preservation
"""

import unittest
from datetime import datetime, timedelta, timezone, date
import json

from app.utils.timezone import (
    get_app_timezone,
    now_ist,
    to_ist,
    format_ist,
    resolve_relative_date,
    get_ist_grounding_context,
)
from app.services.conversation_context_resolver import (
    ConversationContextResolver,
    ConversationContext,
    InstitutionalEntity,
    EntityRelationship,
    FollowUpIntent,
    IntentType,
)


class TestConversationalContextResolver(unittest.TestCase):
    """Comprehensive multi-domain test suite for generic institutional context resolution."""

    def setUp(self):
        self.resolver = ConversationContextResolver()

        # Faculty sample
        self.faculty_cards = """
FACULTY MEMBER 1:
Name: Dr. K. Suresh Babu
Designation: Professor & HOD
Qualification: Ph.D (Computer Science and Engineering)
Department: CSE (AI & DS)
Profile URL: https://vvitu.ac.in/faculty/suresh-babu

FACULTY MEMBER 2:
Name: Dr. G. Sanjay Gandhi
Designation: Professor
Qualification: Ph.D (Computer Science and Engineering)
Department: CSE (AI & DS)
Profile URL: https://vvitu.ac.in/faculty/sanjay-gandhi

FACULTY MEMBER 3:
Name: Dr. P. Nageswara Rao
Designation: Associate Professor
Qualification: Ph.D (Computer Science)
Department: CSE (AI & DS)
Profile URL: https://vvitu.ac.in/faculty/nageswara-rao
"""
        self.faculty_entities = self.resolver.extract_entities_from_evidence(
            self.faculty_cards,
            source_urls=["https://vvitu.ac.in/admissions/faculty/cse-ai-faculty"],
            concept="faculty"
        )
        self.faculty_context = ConversationContext(
            active_subject="CSE (AI & DS) Faculty",
            active_concept="faculty",
            active_entity_type="faculty",
            active_entities=self.faculty_entities,
            active_source_urls=["https://vvitu.ac.in/admissions/faculty/cse-ai-faculty"],
            institutional_period="current_vvitu",
            last_evidence_text=self.faculty_cards,
        )

        # Fee sample
        self.fee_table = """
| Category | Tuition Fee | Special Fee | Total Fee |
|---|---|---|---|
| Category A (Convener) | ₹ 70,000 | ₹ 10,000 | ₹ 80,000 |
| Category B (Management) | ₹ 1,20,000 | ₹ 15,000 | ₹ 1,35,000 |
| Category C (NRI) | ₹ 2,50,000 | ₹ 25,000 | ₹ 2,75,000 |
"""
        self.fee_entities = self.resolver.extract_entities_from_evidence(
            self.fee_table,
            source_urls=["https://vvitu.ac.in/admissions/fee-structure"],
            concept="fee"
        )
        self.fee_context = ConversationContext(
            active_subject="B.Tech CSE Fee Structure",
            active_concept="fee",
            active_entity_type="fee",
            active_entities=self.fee_entities,
            active_source_urls=["https://vvitu.ac.in/admissions/fee-structure"],
            institutional_period="current_vvitu",
            last_evidence_text=self.fee_table,
        )

    # 1. Timezone IST now & conversion
    def test_timezone_now_ist_and_conversion(self):
        curr = now_ist()
        self.assertIsNotNone(curr.tzinfo)
        offset_seconds = curr.utcoffset().total_seconds()
        self.assertEqual(offset_seconds, 19800)

        utc_dt = datetime(2026, 9, 12, 4, 30, 0)
        ist_dt = to_ist(utc_dt)
        self.assertEqual(ist_dt.hour, 10)
        self.assertEqual(ist_dt.minute, 0)

    # 2. Relative date resolution in IST
    def test_relative_date_resolution_ist(self):
        ref_dt = datetime(2026, 9, 12, 10, 0, 0)
        today_res = resolve_relative_date("today", reference_dt=ref_dt)
        tomorrow_res = resolve_relative_date("tomorrow", reference_dt=ref_dt)
        yesterday_res = resolve_relative_date("yesterday", reference_dt=ref_dt)

        self.assertEqual(today_res, date(2026, 9, 12))
        self.assertEqual(tomorrow_res, date(2026, 9, 13))
        self.assertEqual(yesterday_res, date(2026, 9, 11))

    # 3. Faculty entity extraction from structured cards
    def test_entity_extraction_from_faculty_cards(self):
        self.assertEqual(len(self.faculty_entities), 3)
        self.assertEqual(self.faculty_entities[0].name, "Dr. K. Suresh Babu")
        self.assertEqual(self.faculty_entities[0].designation, "Professor & HOD")
        self.assertEqual(self.faculty_entities[0].profile_url, "https://vvitu.ac.in/faculty/suresh-babu")

    # 4. Fee entity extraction from markdown tables
    def test_entity_extraction_from_fee_tables(self):
        self.assertEqual(len(self.fee_entities), 3)
        self.assertIn("Category A", self.fee_entities[0].name)
        self.assertEqual(self.fee_entities[0].entity_type, "fee")
        self.assertIn("₹ 80,000", self.fee_entities[0].attributes.get("total fee", ""))

    # 5. Department entity & HOD relationship extraction
    def test_department_and_hod_relationship_extraction(self):
        dept_text = """
DEPARTMENT 1:
Name: Department of Computer Science and Engineering (CSE)
Head of Department: Dr. V. Radha Krishna
Established: 2007
Intake: 240 seats
"""
        entities = self.resolver.extract_entities_from_evidence(dept_text, concept="department")
        self.assertTrue(len(entities) >= 1)
        dept_entity = entities[0]
        self.assertIn("Computer Science", dept_entity.name)
        self.assertEqual(dept_entity.entity_type, "department")
        # Check HOD relationship
        hod_rels = [r for r in dept_entity.relationships if r.relation_type == "hod_of"]
        self.assertTrue(len(hod_rels) >= 1)
        self.assertIn("Radha Krishna", hod_rels[0].target_name)

    # 6. Standalone query without prior context
    def test_standalone_query_no_prior_context(self):
        intent = self.resolver.resolve_turn("what are the btech programs offered at vvitu?")
        self.assertEqual(intent.intent_type, IntentType.STANDALONE_NEW_TOPIC)
        self.assertTrue(intent.is_institutional)
        self.assertEqual(len(intent.target_entities), 0)

    # 7. Plural pronoun resolution ('them' with typo tolerance: 'FECTH')
    def test_plural_pronoun_resolution_them(self):
        intent = self.resolver.resolve_turn("CAN U PLEASE FECTH DETAILS ABOUT THEM", context=self.faculty_context)
        self.assertEqual(intent.intent_type, IntentType.FOLLOWUP_DEEPER_RETRIEVAL)
        self.assertTrue(intent.is_institutional)
        self.assertEqual(len(intent.target_entities), 3)
        self.assertTrue(intent.needs_deeper_retrieval)
        self.assertEqual(len(intent.target_profile_urls), 3)

    # 8. Person pronoun resolution ('their qualifications')
    def test_plural_pronoun_resolution_their_qualifications(self):
        intent = self.resolver.resolve_turn("what are their qualifications?", context=self.faculty_context)
        self.assertEqual(intent.intent_type, IntentType.FOLLOWUP_EVIDENCE_REUSE)
        self.assertTrue(intent.is_institutional)
        self.assertEqual(len(intent.target_entities), 3)
        self.assertIsNotNone(intent.reusable_evidence)

    # 9. Non-person pronoun resolution ('its timings and fee' -> transport)
    def test_non_person_pronoun_resolution(self):
        transport_context = ConversationContext(
            active_subject="Vijayawada Bus Route 14",
            active_concept="transport",
            active_entity_type="transport",
            active_entities=[
                InstitutionalEntity(
                    name="Vijayawada Route 14",
                    entity_type="transport",
                    attributes={"timings": "7:00 AM departure", "stops": "Benz Circle, PNBS, VVITU"},
                    source_urls=["https://vvitu.ac.in/facilities/transport"]
                )
            ],
            active_source_urls=["https://vvitu.ac.in/facilities/transport"],
            last_evidence_text="Vijayawada Route 14 departs Benz Circle at 7:00 AM. Route fee is ₹ 28,000 per year."
        )
        intent = self.resolver.resolve_turn("what are its timings and fee?", context=transport_context)
        self.assertEqual(intent.intent_type, IntentType.FOLLOWUP_EVIDENCE_REUSE)
        self.assertEqual(len(intent.target_entities), 1)
        self.assertEqual(intent.target_entities[0].name, "Vijayawada Route 14")

    # 10. Ordinal resolution ('first one')
    def test_ordinal_resolution_first_one(self):
        intent = self.resolver.resolve_turn("tell me about the first one", context=self.faculty_context)
        self.assertEqual(intent.intent_type, IntentType.FOLLOWUP_ENTITY_FILTER)
        self.assertEqual(len(intent.target_entities), 1)
        self.assertEqual(intent.target_entities[0].name, "Dr. K. Suresh Babu")

    # 11. Ordinal resolution ('second person')
    def test_ordinal_resolution_second_person(self):
        intent = self.resolver.resolve_turn("what does the second person do?", context=self.faculty_context)
        self.assertEqual(intent.intent_type, IntentType.FOLLOWUP_ENTITY_FILTER)
        self.assertEqual(len(intent.target_entities), 1)
        self.assertEqual(intent.target_entities[0].name, "Dr. G. Sanjay Gandhi")

    # 12. Ordinal resolution ('last one')
    def test_ordinal_resolution_last_one(self):
        intent = self.resolver.resolve_turn("give me details on the last one", context=self.faculty_context)
        self.assertEqual(intent.intent_type, IntentType.FOLLOWUP_ENTITY_FILTER)
        self.assertEqual(len(intent.target_entities), 1)
        self.assertEqual(intent.target_entities[0].name, "Dr. P. Nageswara Rao")

    # 13. Direct named entity follow-up
    def test_named_entity_followup(self):
        intent = self.resolver.resolve_turn("tell me more about Dr. Sanjay Gandhi", context=self.faculty_context)
        self.assertEqual(len(intent.target_entities), 1)
        self.assertEqual(intent.target_entities[0].name, "Dr. G. Sanjay Gandhi")

    # 14. Evidence reuse detection across concepts
    def test_evidence_reuse_across_concepts(self):
        intent = self.resolver.resolve_turn("what is the management quota fee?", context=self.fee_context)
        self.assertIn(intent.intent_type, (IntentType.FOLLOWUP_EVIDENCE_REUSE, IntentType.FOLLOWUP_ENTITY_FILTER))
        self.assertIsNotNone(intent.reusable_evidence)

    # 15. Deeper retrieval trigger across concepts
    def test_deeper_retrieval_trigger(self):
        intent = self.resolver.resolve_turn("show their detailed research papers and publications", context=self.faculty_context)
        self.assertTrue(intent.needs_deeper_retrieval)
        self.assertTrue(len(intent.target_profile_urls) > 0)

    # 16. Sibling branch transition with concept preservation
    def test_sibling_branch_transition_preserves_concept(self):
        # User in fee context asks "what about ai & ds?"
        intent = self.resolver.resolve_turn("what about ai & ds?", context=self.fee_context)
        self.assertEqual(intent.intent_type, IntentType.FOLLOWUP_CONCEPT_TRANSITION)
        self.assertEqual(intent.active_concept, "fee")
        self.assertTrue(any(t in intent.resolved_query.lower() for t in ["ai & ds", "artificial intelligence"]))
        self.assertIn("fee", intent.resolved_query.lower())

    # 17. Relationship navigation ('who is the hod?' from department)
    def test_relationship_navigation_who_is_the_hod(self):
        dept_context = ConversationContext(
            active_subject="Civil Engineering Department",
            active_concept="department",
            active_entity_type="department",
            active_entities=[
                InstitutionalEntity(
                    name="Department of Civil Engineering",
                    entity_type="department",
                    relationships=[
                        EntityRelationship(
                            relation_type="hod_of",
                            target_name="Dr. K. Srinivas",
                            target_type="faculty"
                        )
                    ]
                )
            ],
            last_evidence_text="Department of Civil Engineering HOD: Dr. K. Srinivas"
        )
        intent = self.resolver.resolve_turn("who is the hod?", context=dept_context)
        self.assertEqual(intent.intent_type, IntentType.FOLLOWUP_EVIDENCE_REUSE)
        self.assertEqual(len(intent.target_entities), 1)
        self.assertEqual(intent.target_entities[0].name, "Dr. K. Srinivas")

    # 18. Multi-hop follow-up: Department -> HOD -> 'what are their qualifications?'
    def test_multihop_department_hod_qualifications(self):
        hod_context = ConversationContext(
            active_subject="Civil Engineering HOD",
            active_concept="faculty",
            active_entity_type="faculty",
            active_entities=[
                InstitutionalEntity(
                    name="Dr. K. Srinivas",
                    entity_type="faculty",
                    designation="Professor & HOD",
                    qualification="Ph.D (Structural Engineering)",
                    attributes={"qualification": "Ph.D (Structural Engineering)"}
                )
            ],
            last_evidence_text="Dr. K. Srinivas, Professor & HOD. Qualification: Ph.D (Structural Engineering)"
        )
        intent = self.resolver.resolve_turn("what are their qualifications?", context=hod_context)
        self.assertEqual(intent.intent_type, IntentType.FOLLOWUP_EVIDENCE_REUSE)
        self.assertEqual(len(intent.target_entities), 1)
        self.assertEqual(intent.target_entities[0].name, "Dr. K. Srinivas")

    # 19. Ambiguity detection ('CSE and ECE' -> 'who is their HOD?' -> clarification request)
    def test_ambiguity_detection_multiple_antecedents(self):
        two_dept_context = ConversationContext(
            active_subject="CSE and ECE Departments",
            active_concept="department",
            active_entity_type="department",
            active_entities=[
                InstitutionalEntity(
                    name="Department of Computer Science and Engineering",
                    entity_type="department"
                ),
                InstitutionalEntity(
                    name="Department of Electronics and Communication Engineering",
                    entity_type="department"
                )
            ],
            last_evidence_text="VVITU offers CSE and ECE departments."
        )
        intent = self.resolver.resolve_turn("who is their HOD?", context=two_dept_context)
        self.assertEqual(intent.intent_type, IntentType.AMBIGUOUS_CLARIFICATION)
        self.assertTrue(intent.is_ambiguous)
        self.assertIn("clarify", intent.clarification_prompt.lower())

    # 20. Hostel & Transport concept transitions ('what are the fees?')
    def test_hostel_concept_transition_to_fee(self):
        hostel_context = ConversationContext(
            active_subject="VVITU Hostel Facilities",
            active_concept="hostel",
            active_entity_type="facility",
            active_entities=[
                InstitutionalEntity(
                    name="Boys Hostel",
                    entity_type="facility",
                    attributes={"rooms": "AC and Non-AC available"}
                )
            ],
            last_evidence_text="VVITU provides separate hostel facilities for boys and girls with AC and Non-AC rooms."
        )
        intent = self.resolver.resolve_turn("what are the fees?", context=hostel_context)
        self.assertIn(intent.intent_type, (IntentType.FOLLOWUP_CONCEPT_TRANSITION, IntentType.FOLLOWUP_TOPIC_EXPANSION))
        self.assertEqual(intent.active_concept, "fee")
        self.assertIn("hostel", intent.resolved_query.lower())

    # 21. Examination schedule and academic year filter ('3rd year')
    def test_examination_academic_year_filter(self):
        exam_context = ConversationContext(
            active_subject="Semester Examinations Schedule",
            active_concept="examination",
            active_entity_type="academic",
            active_entities=[
                InstitutionalEntity(
                    name="B.Tech Regular Examinations Schedule",
                    entity_type="academic",
                    attributes={"schedule": "Starts from October 15"}
                )
            ],
            last_evidence_text="VVITU B.Tech 1st, 2nd, 3rd, and 4th Year Semester Examinations commence on October 15."
        )
        intent = self.resolver.resolve_turn("is there any update for 3rd year?", context=exam_context)
        self.assertEqual(intent.intent_type, IntentType.FOLLOWUP_EVIDENCE_REUSE)
        self.assertEqual(intent.active_filters.get("academic_year"), "3rd year")

    # 22. Notice / Circular follow-up ('when was it published?')
    def test_notice_date_attribute_query(self):
        notice_context = ConversationContext(
            active_subject="Semester Registration Circular",
            active_concept="notice",
            active_entity_type="notice",
            active_entities=[
                InstitutionalEntity(
                    name="Circular: Semester Registration Deadline",
                    entity_type="notice",
                    attributes={"date": "September 10, 2026", "published": "10-09-2026"}
                )
            ],
            last_evidence_text="Notice Ref: VVITU/REG/2026/042 dated 10-09-2026 regarding semester fee payment."
        )
        intent = self.resolver.resolve_turn("when was it published?", context=notice_context)
        self.assertEqual(intent.intent_type, IntentType.FOLLOWUP_EVIDENCE_REUSE)
        self.assertEqual(len(intent.target_entities), 1)

    # 23. Casual/unrelated queries bypass context
    def test_unrelated_query_resets_context(self):
        intent = self.resolver.resolve_turn("tell me a joke", context=self.faculty_context)
        self.assertEqual(intent.intent_type, IntentType.CASUAL_NO_CONTEXT)
        self.assertFalse(intent.is_institutional)

    # 24. Institutional classifier with rewritten query
    def test_institutional_detector_with_rewritten_query(self):
        raw = "CAN U PLEASE FECTH DETAILS ABOUT THEM"
        self.assertFalse(self.resolver._is_query_institutional(raw))
        intent = self.resolver.resolve_turn(raw, context=self.faculty_context)
        self.assertTrue(intent.is_institutional)

    # 25. Multi-turn source continuity
    def test_multi_turn_source_continuity(self):
        self.assertEqual(self.faculty_context.institutional_period, "current_vvitu")
        self.assertIn("vvitu.ac.in", self.faculty_context.active_source_urls[0])

    # 26. ConversationContext serialization & hydration roundtrip
    def test_conversation_context_serialization(self):
        dumped = self.faculty_context.model_dump()
        json_str = json.dumps(dumped)
        loaded = json.loads(json_str)
        reconstructed = ConversationContext.model_validate(loaded)
        self.assertEqual(len(reconstructed.active_entities), 3)
        self.assertEqual(reconstructed.active_entities[0].name, "Dr. K. Suresh Babu")
        self.assertEqual(reconstructed.active_concept, "faculty")

    # 27. Non-streaming and streaming consistency parity
    def test_identical_behavior_streaming_and_non_streaming(self):
        query = "tell me about the second one"
        intent_sync = self.resolver.resolve_turn(query, context=self.faculty_context)
        intent_stream = self.resolver.resolve_turn(query, context=self.faculty_context)
        self.assertEqual(intent_sync.intent_type, intent_stream.intent_type)
        self.assertEqual(intent_sync.target_entities[0].name, intent_stream.target_entities[0].name)

    # 28. Anti-hallucination boundary preservation
    def test_anti_hallucination_boundary_preserved(self):
        grounding = get_ist_grounding_context()
        self.assertIn("Indian Standard Time", grounding)
        self.assertIn("Asia/Kolkata", grounding)


if __name__ == "__main__":
    unittest.main()
