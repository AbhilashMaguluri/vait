"""
Comprehensive regression test suite for VAIT Official Web Retrieval and Faculty Directory Resolution.
Covers:
  1. SSRF Safety and Domain Whitelisting
  2. Official URL extraction from user messages
  3. Dynamic department and route discovery from bundle
  4. Query-to-route semantic and acronym matching (AI, DS, ECE, Civil, etc.)
  5. Direct official URL retrieval and metadata verification
  6. RAG sufficiency discriminator logic (avoids false-positive chunks)
"""

import asyncio
import unittest
from unittest.mock import MagicMock

from app.utils.config import get_settings
from app.services.official_web_retriever import (
    OfficialWebRetriever,
    is_safe_official_url,
    CURRENT_DOMAINS,
    LEGACY_DOMAINS,
)
from app.services.rag_service import RAGService


class TestOfficialFacultyRetrieval(unittest.IsolatedAsyncioTestCase):

    @classmethod
    def setUpClass(cls):
        cls.settings = get_settings()
        cls.retriever = OfficialWebRetriever(cls.settings)

    def test_ssrf_and_domain_safety(self):
        """Verify strict SSRF safety checks for allowed domains and IP blocking."""
        # Whitelisted official domains
        self.assertTrue(is_safe_official_url("https://vvitu.ac.in/admissions/faculty/cse-ai-faculty"))
        self.assertTrue(is_safe_official_url("https://www.vvitu.ac.in/about-us/leadership"))
        self.assertTrue(is_safe_official_url("https://vvitguntur.com/index.php/departments/ai-ds"))
        self.assertTrue(is_safe_official_url("http://vvitguntur.com/departments/cse"))

        # Blocked domains and SSRF attempts
        self.assertFalse(is_safe_official_url("https://google.com"))
        self.assertFalse(is_safe_official_url("http://127.0.0.1:8000/api/vait/chat"))
        self.assertFalse(is_safe_official_url("http://localhost/admin"))
        self.assertFalse(is_safe_official_url("http://169.254.169.254/latest/meta-data/"))
        self.assertFalse(is_safe_official_url("http://192.168.1.1/router"))
        self.assertFalse(is_safe_official_url("ftp://vvitu.ac.in/file"))
        self.assertFalse(is_safe_official_url("javascript:alert(1)"))
        self.assertFalse(is_safe_official_url(""))

    def test_official_url_extraction(self):
        """Verify RAGService extracts only valid official URLs from user prompts."""
        # Embedded official URL
        msg1 = "Please check this link: https://vvitu.ac.in/admissions/faculty/cse-ai-faculty"
        urls1 = RAGService._extract_official_urls(msg1)
        self.assertEqual(urls1, ["https://vvitu.ac.in/admissions/faculty/cse-ai-faculty"])

        # Multiple URLs (one official, one malicious/external)
        msg2 = "Check https://evil.com and https://vvitguntur.com/departments/ece"
        urls2 = RAGService._extract_official_urls(msg2)
        self.assertEqual(urls2, ["https://vvitguntur.com/departments/ece"])

        # No URL
        msg3 = "Can you please tell about ai ds faculty at vvit"
        urls3 = RAGService._extract_official_urls(msg3)
        self.assertEqual(urls3, [])

    def test_dynamic_bundle_department_discovery(self):
        """Verify dynamic discovery of all departments from the bundle without hardcoding."""
        self.assertGreaterEqual(len(self.retriever.departments_catalog), 7)
        dept_slugs = [d["slug"] for d in self.retriever.departments_catalog]
        self.assertIn("cse-ai-faculty", dept_slugs)
        self.assertIn("ece-faculty", dept_slugs)
        self.assertIn("civil-faculty", dept_slugs)
        self.assertIn("mec-faculty", dept_slugs)

        # Verify faculty count in catalog
        self.assertGreaterEqual(len(self.retriever.faculty_catalog), 300)

        # Verify dynamic route registration
        dept_routes = [r for r in self.retriever.routes_catalog if "/admissions/faculty/" in r.get("path", "")]
        self.assertGreaterEqual(len(dept_routes), 7)

    def test_query_route_matching(self):
        """Verify generic matching for department faculty queries."""
        # AI & DS query
        routes_aids = self.retriever._find_matching_routes("can you please tell about ai ds faculty at vvit")
        self.assertTrue(len(routes_aids) > 0)
        top_route = routes_aids[0]
        self.assertEqual(top_route["path"], "/admissions/faculty/cse-ai-faculty")
        self.assertIn("vvitu.ac.in", top_route["url"])

        # Civil query
        routes_civil = self.retriever._find_matching_routes("tell me about civil engineering faculty")
        self.assertTrue(len(routes_civil) > 0)
        self.assertEqual(routes_civil[0]["path"], "/admissions/faculty/civil-faculty")

        # ECE query
        routes_ece = self.retriever._find_matching_routes("who are the ece faculty members")
        self.assertTrue(len(routes_ece) > 0)
        self.assertEqual(routes_ece[0]["path"], "/admissions/faculty/ece-faculty")

    async def test_retrieve_url_direct(self):
        """Verify direct URL retrieval for official faculty directory."""
        url = "https://vvitu.ac.in/admissions/faculty/cse-ai-faculty"
        result = await self.retriever.retrieve_url(url, "ai ds faculty")

        self.assertGreater(len(result.content), 2000)
        self.assertEqual(result.url, url)
        self.assertEqual(result.period, "current")
        self.assertEqual(result.source_tier, "primary_official_current")
        self.assertEqual(result.entity_type, "faculty")
        self.assertIn(result.retrieval_method, ["headless_browser_dom", "catalog"])
        self.assertIn("Suresh Babu", result.content)
        self.assertIn("Professor", result.content)

    def test_rag_sufficiency_token_discrimination(self):
        """Verify that short department acronyms (AI, DS) prevent false positive chunk matching."""
        settings = get_settings()
        rag = RAGService(settings)

        class MockChunk:
            def __init__(self, content):
                self.content = content

        # Chunks mentioning only generic faculty without AI or DS
        generic_chunks = [
            MockChunk("The faculty at VVIT comprises qualified professors in various disciplines like Civil and Mechanical Engineering."),
            MockChunk("Faculty development programs are conducted every semester for teaching staff.")
        ]

        # A specific AI DS query should NOT be considered sufficient against purely generic chunks
        is_suff = rag._is_rag_sufficient("can you please tell about ai ds faculty at vvit", generic_chunks)
        self.assertFalse(is_suff, "Generic non-AI chunks must NOT be marked sufficient for an AI/DS query")


if __name__ == "__main__":
    unittest.main()
