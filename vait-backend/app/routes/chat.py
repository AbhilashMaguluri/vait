"""VAIT Chat Routes — API endpoints for the chat interface."""

import logging
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

from app.controllers.chat_controller import ChatController

logger = logging.getLogger("vait.routes.chat")

router = APIRouter(tags=["chat"])


# ── Request / Response models ──

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(
        ...,
        description="User's question or message",
        min_length=1,
        max_length=2000,
    )
    department: Optional[str] = Field(None)
    academic_year: Optional[str] = Field(None)


class SourceItem(BaseModel):
    """Structured source object."""
    title: str = Field("", description="Source document title")
    url: str = Field("", description="Source URL if available")
    type: str = Field("", description="Source type: website, pdf, social_media")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    reply: str = Field(..., description="VAIT's response or refusal")
    sources: List[str] = Field(
        default_factory=list,
        description="List of source document names used",
    )
    structured_sources: List[SourceItem] = Field(
        default_factory=list,
        description="Structured source objects with title, URL, and type",
    )
    confidence: str = Field(
        "Low",
        description="Confidence level: High, Medium, or Low",
    )
    retrieval_score: float = Field(
        0.0,
        description="Top similarity score from retrieval",
    )
    intent: str = Field(
        "general",
        description="Classified query intent: academic, admissions, examinations, placements, events, infrastructure, general",
    )
    performance: Optional[Dict] = Field(
        None,
        description="Timing breakdown: embed_ms, retrieval_ms, generation_ms, total_ms, intent, cache_hit",
    )


class ChatDetailedResponse(ChatResponse):
    """Detailed response with additional metadata."""
    is_refusal: bool = Field(
        False,
        description="Whether the response is a refusal",
    )


class DebugChunk(BaseModel):
    """Single retrieved chunk in debug output."""
    text_preview: str
    source_file: str
    authority_level: str
    similarity_score: float
    adjusted_score: float = Field(
        0.0,
        description="similarity_score × authority multiplier",
    )


class DebugResponse(BaseModel):
    """Response model for /debug endpoint."""
    query: str
    retrieved_chunks: List[DebugChunk]
    performance: Optional[Dict] = Field(
        None,
        description="Timing breakdown: embed_ms, retrieval_ms",
    )


# ── Dependency ──

def get_chat_controller() -> ChatController:
    """Dependency to get chat controller instance."""
    return ChatController()


# ── Endpoints ──

from fastapi.responses import StreamingResponse

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    controller: ChatController = Depends(get_chat_controller),
) -> ChatResponse:
    """
    Process a chat message and return VAIT's response.

    Implements strict RAG-based question answering:
    - Only answers with retrieved context
    - Refuses if similarity score is below threshold
    - Maintains institutional academic tone

    Response includes reply, sources, confidence, and retrieval_score.
    """
    try:
        logger.info("Incoming chat request: %s", request.message)
        response = await controller.process_message(
            message=request.message,
            department=request.department,
            academic_year=request.academic_year,
        )
        logger.info("Chat response confidence: %s, score: %.3f", response.confidence, response.retrieval_score)
        return ChatResponse(
            reply=response.reply,
            sources=response.sources,
            structured_sources=[
                SourceItem(**s) for s in (response.structured_sources or [])
            ],
            confidence=response.confidence,
            retrieval_score=response.retrieval_score,
            intent=response.intent,
            performance=response.performance,
        )
    except Exception as e:
        logger.error("Chat endpoint error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred processing your request. Please try again.",
        )


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    controller: ChatController = Depends(get_chat_controller),
):
    """
    Process a chat message with streaming response (SSE).
    """
    logger.info("Incoming stream chat request: %s", request.message)
    return StreamingResponse(
        controller.process_message_stream(
            message=request.message,
            department=request.department,
            academic_year=request.academic_year,
        ),
        media_type="text/event-stream"
    )

@router.post("/chat/detailed", response_model=ChatDetailedResponse)
async def chat_detailed(
    request: ChatRequest,
    controller: ChatController = Depends(get_chat_controller),
) -> ChatDetailedResponse:
    """
    Process a chat message with detailed response metadata.
    """
    try:
        response = await controller.process_message(
            message=request.message,
            department=request.department,
            academic_year=request.academic_year,
        )
        return ChatDetailedResponse(
            reply=response.reply,
            sources=response.sources,
            structured_sources=[
                SourceItem(**s) for s in (response.structured_sources or [])
            ],
            confidence=response.confidence,
            retrieval_score=response.retrieval_score,
            is_refusal=response.is_refusal,
            intent=response.intent,
            performance=response.performance,
        )
    except Exception as e:
        logger.error("Chat detailed endpoint error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred processing your request. Please try again.",
        )


@router.get("/debug", response_model=DebugResponse)
async def debug_retrieval(
    query: str = Query(..., min_length=1, max_length=2000, description="Search query"),
    controller: ChatController = Depends(get_chat_controller),
) -> DebugResponse:
    """
    Engineering diagnostics endpoint.

    Returns raw retrieval results for the given query
    (no LLM call, no prompt building).
    """
    try:
        result = await controller.debug_retrieve(query)
        chunks = result.get("chunks", [])
        perf = result.get("performance", None)
        return DebugResponse(
            query=query,
            retrieved_chunks=[DebugChunk(**c) for c in chunks],
            performance=perf,
        )
    except Exception as e:
        logger.error("Debug endpoint error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Debug retrieval failed. Please try again.",
        )


@router.get("/stats")
async def get_stats(
    controller: ChatController = Depends(get_chat_controller),
):
    """
    Get statistics about the VAIT knowledge base.
    """
    return await controller.get_stats()


@router.get("/system_summary")
async def system_summary(
    controller: ChatController = Depends(get_chat_controller),
):
    """
    VAIT System Self-Summary — comprehensive system overview for demo and monitoring.

    Returns:
        - Identity (name, version, engine)
        - Knowledge coverage (total docs, per-type breakdown, domains, departments)
        - Model configuration (LLM, embedding, chunk settings)
        - Intelligence features (all active pipeline stages)
        - Runtime metrics (uptime, cache stats, avg adjusted score)
        - Ingestion status (last reindex, crawl config, watch status)
    """
    try:
        return await controller.get_system_summary()
    except Exception as e:
        logger.error("System summary endpoint error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate system summary.",
        )
