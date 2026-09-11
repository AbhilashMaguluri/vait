"""VAIT Chat Controller — business logic for chat operations."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from app.services.rag_service import RAGResponse

logger = logging.getLogger("vait.chat_controller")


def _get_rag_service():
    """Lazy import to break circular dependency with app.main."""
    from app.main import get_rag_service
    return get_rag_service()


@dataclass
class ChatResult:
    """Result from chat processing."""
    reply: str
    sources: list
    structured_sources: list
    confidence: str  # "High" | "Medium" | "Low"
    retrieval_score: float
    is_refusal: bool
    performance: Optional[Dict] = None
    intent: str = "general"
    response_type: str = "informational"


class ChatController:
    """Controller for handling chat operations."""

    async def process_message(
        self,
        message: str,
        department: Optional[str] = None,
        academic_year: Optional[str] = None,
        history: Optional[List[Dict]] = None,
    ) -> ChatResult:
        """Process a user message through the RAG pipeline."""
        rag_service = _get_rag_service()

        if rag_service is None:
            return ChatResult(
                reply=(
                    "I apologize, but the VAIT system is currently initializing. "
                    "Please try again in a moment."
                ),
                sources=[],
                structured_sources=[],
                confidence="Low",
                retrieval_score=0.0,
                is_refusal=True,
            )

        try:
            response: RAGResponse = await rag_service.process_query(message, history=history)
        except Exception as exc:
            logger.error("RAG/LLM processing failed: %s", exc, exc_info=True)
            return ChatResult(
                reply=(
                    "I apologize, but I couldn't generate a response at the moment. "
                    "Please try again shortly."
                ),
                sources=[],
                structured_sources=[],
                confidence="Low",
                retrieval_score=0.0,
                is_refusal=True,
            )

        return ChatResult(
            reply=response.reply,
            sources=response.sources,
            structured_sources=response.structured_sources,
            confidence=response.confidence,
            retrieval_score=response.retrieval_score,
            is_refusal=response.is_refusal,
            performance=response.performance,
            intent=response.intent,
            response_type=getattr(response, "response_type", "informational"),
        )

    async def process_message_stream(
        self,
        message: str,
        department: Optional[str] = None,
        academic_year: Optional[str] = None,
        history: Optional[List[Dict]] = None,
    ):
        """Process a user message through the RAG pipeline with streaming and memory."""
        import json

        rag_service = _get_rag_service()
        if rag_service is None:
            yield f'data: {json.dumps({"type": "error", "error": "The VAIT system is currently initializing. Please try again in a moment."})}\n\n'
            return

        try:
            async for chunk in rag_service.process_query_stream(message, history=history):
                yield chunk
        except Exception as exc:
            logger.error("Streaming RAG/LLM processing failed: %s", exc, exc_info=True)
            error_payload = {
                "type": "error",
                "error": "I apologize, but I couldn't generate a response at the moment. Please try again shortly.",
            }
            yield "data: " + json.dumps(error_payload) + "\n\n"

    async def debug_retrieve(self, query: str) -> Dict:
        """
        Run retrieval diagnostics (no LLM call).

        Returns dict with 'chunks' (list of scored chunk dicts) and
        'performance' (embed_ms, retrieval_ms).
        """
        rag_service = _get_rag_service()

        if rag_service is None:
            return {"chunks": [], "performance": {}}

        return await rag_service.debug_retrieve(query)

    async def get_stats(self) -> Dict:
        """Get knowledge base statistics."""
        rag_service = _get_rag_service()
        if rag_service is None:
            return {"status": "initializing"}
        return rag_service.get_stats()

    async def get_system_summary(self) -> Dict:
        """Get comprehensive system self-summary."""
        rag_service = _get_rag_service()
        if rag_service is None:
            return {"status": "initializing", "system_name": "VAIT"}
        return rag_service.get_system_summary()
