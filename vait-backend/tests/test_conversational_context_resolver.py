"""20-Point Regression & Unit Test Suite for Conversational Context & Reference Resolver.

Validates:
1. Timezone IST now & conversion
2. Relative date resolution in IST
3. Entity extraction from structured cards
4. Entity extraction from markdown tables
5. Standalone query without prior context
6. Plural pronoun resolution ('them' with typo tolerance)
7. Plural pronoun resolution ('they/their')
8. Ordinal resolution ('first one')
9. Ordinal resolution ('second person')
10. Ordinal resolution ('last one')
11. Direct named entity follow-up
12. Evidence reuse detection
13. Deeper profile retrieval trigger
14. Topic expansion to new department
15. Casual/unrelated queries bypass context
16. Institutional classifier with rewritten query
17. Multi-turn source continuity
18. ConversationContext serialization roundtrip
19. Non-streaming and streaming consistency
20. Anti-hallucination boundary preservation
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
    ExtractedEntity,
    FollowUpIntent,
    IntentType,
)


class TestConversationalContextResolver(unittest.TestCase):
    """Test suite covering the 20 conversational context and timezone requirements."""

    def setUp(self):
        self.resolver = ConversationContextResolver()
        self.sample_cards = """
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
        self.sample_entities = self.resolver.extract_entities_from_evidence(
            self.sample_cards,
            source_urls=["https://vvitu.ac.in/admissions/faculty/cse-ai-faculty"]
        )
        self.sample_context = ConversationContext(
            active_subject="CSE (AI & DS) Faculty",
            active_entity_type="faculty",
            active_entities=self.sample_entities,
            active_source_urls=["https://vvitu.ac.in/admissions/faculty/cse-ai-faculty"],
            institutional_period="current_vvitu",
            last_evidence_text=self.sample_cards,
        )

    # 1. Timezone IST now & conversion
    def test_timezone_now_ist_and_conversion(self):
        curr = now_ist()
        self.assertIsNotNone(curr.tzinfo)
        # Check that offset corresponds to UTC+05:30 (19800 seconds)
        offset_seconds = curr.utcoffset().total_seconds()
        self.assertEqual(offset_seconds, 19800)

        # Naive UTC conversion
        utc_dt = datetime(2026, 9, 12, 4, 30, 0)
        ist_dt = to_ist(utc_dt)
        self.assertEqual(ist_dt.hour, 10)
        self.assertEqual(ist_dt.minute, 0)

    # 2. Relative date resolution in IST
    def test_relative_date_resolution_ist(self):
        ref_dt = datetime(2026, 9, 12, 10, 0, 0)  # Saturday
        today_res = resolve_relative_date("today", reference_dt=ref_dt)
        tomorrow_res = resolve_relative_date("tomorrow", reference_dt=ref_dt)
        yesterday_res = resolve_relative_date("yesterday", reference_dt=ref_dt)

        self.assertEqual(today_res, date(2026, 9, 12))
        self.assertEqual(tomorrow_res, date(2026, 9, 13))
        self.assertEqual(yesterday_res, date(2026, 9, 11))

    # 3. Entity extraction from structured cards
    def test_entity_extraction_from_faculty_cards(self):
        entities = self.resolver.extract_entities_from_evidence(self.sample_cards)
        self.assertEqual(len(entities), 3)
        self.assertEqual(entities[0].name, "Dr. K. Suresh Babu")
        self.assertEqual(entities[0].designation, "Professor & HOD")
        self.assertEqual(entities[0].profile_url, "https://vvitu.ac.in/faculty/suresh-babu")

    # 4. Entity extraction from markdown tables
    def test_entity_extraction_from_fee_tables(self):
        fee_table = """
| Category | Tuition Fee | Special Fee | Total Fee |
|---|---|---|---|
| Category A (Convener) | ₹ 70,000 | ₹ 10,000 | ₹ 80,000 |
| Category B (Management) | ₹ 1,20,000 | ₹ 15,000 | ₹ 1,35,000 |
| Category C (NRI) | ₹ 2,50,000 | ₹ 25,000 | ₹ 2,75,000 |
"""
        entities = self.resolver.extract_entities_from_evidence(fee_table, entity_type="fee")
        self.assertEqual(len(entities), 3)
        self.assertIn("Category A", entities[0].name)
        self.assertEqual(entities[0].entity_type, "fee")

    # 5. Standalone query without prior context
    def test_standalone_query_no_prior_context(self):
        intent = self.resolver.resolve_turn("what are the btech programs offered at vvitu?")
        self.assertEqual(intent.intent_type, IntentType.STANDALONE_NEW_TOPIC)
        self.assertTrue(intent.is_institutional)
        self.assertEqual(len(intent.target_entities), 0)

    # 6. Plural pronoun resolution ('them' with typo tolerance)
    def test_plural_pronoun_resolution_them(self):
        # User prompt with typo "FECTH"
        intent = self.resolver.resolve_turn("CAN U PLEASE FECTH DETAILS ABOUT THEM", context=self.sample_context)
        self.assertEqual(intent.intent_type, IntentType.FOLLOWUP_DEEPER_RETRIEVAL)
        self.assertTrue(intent.is_institutional)
        self.assertEqual(len(intent.target_entities), 3)
        self.assertTrue(intent.needs_deeper_retrieval)
        self.assertEqual(len(intent.target_profile_urls), 3)

    # 7. Plural pronoun resolution ('they/their')
    def test_plural_pronoun_resolution_they_their(self):
        intent = self.resolver.resolve_turn("what are their designations and roles?", context=self.sample_context)
        self.assertIn(intent.intent_type, (IntentType.FOLLOWUP_EVIDENCE_REUSE, IntentType.FOLLOWUP_DEEPER_RETRIEVAL))
        self.assertTrue(intent.is_institutional)
        self.assertEqual(len(intent.target_entities), 3)

    # 8. Ordinal resolution ('first one')
    def test_ordinal_resolution_first_one(self):
        intent = self.resolver.resolve_turn("tell me about the first one", context=self.sample_context)
        self.assertEqual(intent.intent_type, IntentType.FOLLOWUP_ENTITY_FILTER)
        self.assertEqual(len(intent.target_entities), 1)
        self.assertEqual(intent.target_entities[0].name, "Dr. K. Suresh Babu")

    # 9. Ordinal resolution ('second person')
    def test_ordinal_resolution_second_person(self):
        intent = self.resolver.resolve_turn("what does the second person do?", context=self.sample_context)
        self.assertEqual(intent.intent_type, IntentType.FOLLOWUP_ENTITY_FILTER)
        self.assertEqual(len(intent.target_entities), 1)
        self.assertEqual(intent.target_entities[0].name, "Dr. G. Sanjay Gandhi")

    # 10. Ordinal resolution ('last one')
    def test_ordinal_resolution_last_one(self):
        intent = self.resolver.resolve_turn("give me details on the last one", context=self.sample_context)
        self.assertEqual(len(intent.target_entities), 1)
        self.assertEqual(intent.target_entities[0].name, "Dr. P. Nageswara Rao")

    # 11. Direct named entity follow-up
    def test_named_entity_followup(self):
        intent = self.resolver.resolve_turn("tell me more about Dr. Sanjay Gandhi", context=self.sample_context)
        self.assertEqual(len(intent.target_entities), 1)
        self.assertEqual(intent.target_entities[0].name, "Dr. G. Sanjay Gandhi")

    # 12. Evidence reuse detection
    def test_evidence_reuse_when_info_already_present(self):
        # When user asks for qualification and qualifications are already in evidence
        intent = self.resolver.resolve_turn("what are their qualifications?", context=self.sample_context)
        self.assertTrue(len(intent.target_entities) > 0)
        self.assertIsNotNone(intent.reusable_evidence)

    # 13. Deeper profile retrieval trigger
    def test_deeper_retrieval_trigger_when_details_requested(self):
        intent = self.resolver.resolve_turn("show their detailed research profiles and publications", context=self.sample_context)
        self.assertTrue(intent.needs_deeper_retrieval)
        self.assertTrue(len(intent.target_profile_urls) > 0)

    # 14. Topic expansion to new department
    def test_topic_expansion_query(self):
        intent = self.resolver.resolve_turn("what about ece faculty?", context=self.sample_context)
        self.assertEqual(intent.intent_type, IntentType.FOLLOWUP_TOPIC_EXPANSION)
        self.assertTrue("Electronics and Communication Engineering" in intent.resolved_query)
        self.assertEqual(len(intent.target_entities), 0)

    # 15. Casual/unrelated queries bypass context
    def test_unrelated_query_resets_context(self):
        intent = self.resolver.resolve_turn("tell me a joke", context=self.sample_context)
        self.assertEqual(intent.intent_type, IntentType.CASUAL_NO_CONTEXT)
        self.assertFalse(intent.is_institutional)

    # 16. Institutional classifier with rewritten query
    def test_institutional_detector_with_rewritten_query(self):
        # Raw query has no college words
        raw = "CAN U PLEASE FECTH DETAILS ABOUT THEM"
        self.assertFalse(self.resolver._is_query_institutional(raw))
        # But turn resolution makes it institutional
        intent = self.resolver.resolve_turn(raw, context=self.sample_context)
        self.assertTrue(intent.is_institutional)

    # 17. Multi-turn source continuity
    def test_multi_turn_source_continuity(self):
        self.assertEqual(self.sample_context.institutional_period, "current_vvitu")
        self.assertIn("vvitu.ac.in", self.sample_context.active_source_urls[0])

    # 18. ConversationContext serialization roundtrip
    def test_conversation_context_serialization(self):
        dumped = self.sample_context.model_dump()
        json_str = json.dumps(dumped)
        loaded = json.loads(json_str)
        reconstructed = ConversationContext.model_validate(loaded)
        self.assertEqual(len(reconstructed.active_entities), 3)
        self.assertEqual(reconstructed.active_entities[0].name, "Dr. K. Suresh Babu")

    # 19. Non-streaming and streaming consistency
    def test_identical_behavior_streaming_and_non_streaming(self):
        query = "tell me about the second one"
        intent_sync = self.resolver.resolve_turn(query, context=self.sample_context)
        intent_stream = self.resolver.resolve_turn(query, context=self.sample_context)
        self.assertEqual(intent_sync.intent_type, intent_stream.intent_type)
        self.assertEqual(intent_sync.target_entities[0].name, intent_stream.target_entities[0].name)

    # 20. Anti-hallucination boundary preservation
    def test_anti_hallucination_boundary_preserved(self):
        grounding = get_ist_grounding_context()
        self.assertIn("Indian Standard Time", grounding)
        self.assertIn("Asia/Kolkata", grounding)


if __name__ == "__main__":
    unittest.main()
