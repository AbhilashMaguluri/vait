"""VAIT Chat Routes — API endpoints for the chat interface."""

import json
import logging
from fastapi import APIRouter, HTTPException, Depends, Query, status
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

from app.controllers.chat_controller import ChatController, ChatResult
from app.db import get_database
from app.dependencies import get_current_user
from app.services.activity_service import log_activity
from app.services.conversation_service import (
    clear_user_conversation,
    delete_user_conversation,
    list_user_conversations,
    record_chat_exchange,
    rename_user_conversation,
)

logger = logging.getLogger("vait.routes.chat")

router = APIRouter(tags=["chat"])


# ── Request / Response models ──

class MessageItem(BaseModel):
    """A single message in the conversation history."""
    role: str = Field(..., description="Role of the sender: 'user' or 'assistant'")
    content: str = Field(..., description="Content of the message")

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
    conversation_id: Optional[str] = Field(
        default=None,
        max_length=80,
        description="Client or database conversation id for persistence",
    )
    history: Optional[List[MessageItem]] = Field(
        default_factory=list,
        description="Previous conversation history"
    )


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
    source_visibility: str = Field(
        "none",
        description="Source visibility tier: none, compact, full",
    )
    retrieval_score: float = Field(
        0.0,
        description="Top similarity score from retrieval",
    )
    intent: str = Field(
        "general",
        description="Classified query intent: academic, admissions, examinations, placements, events, infrastructure, general",
    )
    response_type: str = Field(
        "informational",
        description="Classified adaptive presentation format",
    )
    performance: Optional[Dict] = Field(
        None,
        description="Timing breakdown: embed_ms, retrieval_ms, generation_ms, total_ms, intent, cache_hit",
    )
    conversation_id: Optional[str] = Field(
        None,
        description="Persisted conversation id owned by the authenticated user",
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


class ConversationUpdateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)


# ── Dependency ──

def get_chat_controller() -> ChatController:
    """Dependency to get chat controller instance."""
    return ChatController()


def _history_for_controller(history: Optional[List[MessageItem]]) -> list[dict]:
    return [item.model_dump() for item in (history or [])]


async def _persist_chat_response(database, current_user: dict, request: ChatRequest, response: ChatResult) -> str | None:
    try:
        conversation = await record_chat_exchange(
            database,
            user=current_user,
            conversation_id=request.conversation_id,
            user_message=request.message,
            assistant_reply=response.reply,
            intent=response.intent,
            confidence=response.confidence,
            sources=response.sources,
            structured_sources=response.structured_sources,
            retrieval_score=response.retrieval_score,
            department=request.department,
            academic_year=request.academic_year,
            performance=response.performance,
            source_visibility=getattr(response, "source_visibility", "none"),
        )
        await log_activity(
            database,
            current_user["_id"],
            "chat.message",
            {
                "conversation_id": conversation["id"],
                "intent": response.intent,
                "confidence": response.confidence,
            },
        )
        return conversation["id"]
    except Exception as exc:
        logger.error("Failed to persist chat exchange: %s", exc, exc_info=True)
        return request.conversation_id


# ── Endpoints ──

from fastapi.responses import StreamingResponse

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    controller: ChatController = Depends(get_chat_controller),
    current_user: dict = Depends(get_current_user),
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
        database = get_database()
        logger.info("Incoming chat request: %s", request.message)
        response = await controller.process_message(
            message=request.message,
            department=request.department,
            academic_year=request.academic_year,
            history=_history_for_controller(request.history),
        )
        conversation_id = await _persist_chat_response(database, current_user, request, response)
        logger.info(
            "Outgoing chat response: reply_len=%d sources=%d confidence=%s score=%.3f intent=%s",
            len(response.reply or ""),
            len(response.sources or []),
            response.confidence,
            response.retrieval_score,
            response.intent,
        )
        return ChatResponse(
            reply=response.reply,
            sources=response.sources,
            structured_sources=[
                SourceItem(**s) for s in (response.structured_sources or [])
            ],
            confidence=response.confidence,
            source_visibility=getattr(response, "source_visibility", "none"),
            retrieval_score=response.retrieval_score,
            intent=response.intent,
            response_type=getattr(response, "response_type", "informational"),
            performance=response.performance,
            conversation_id=conversation_id,
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
    current_user: dict = Depends(get_current_user),
):
    """
    Process a chat message with streaming response (SSE).
    """
    database = get_database()
    logger.info("Incoming stream chat request: %s", request.message)
    logger.info("Outgoing stream chat response: stream initialized")

    async def persisted_stream():
        reply_parts: list[str] = []
        metadata: dict = {
            "sources": [],
            "structured_sources": [],
            "source_visibility": "none",
            "confidence": "Low",
            "retrieval_score": 0.0,
            "intent": "general",
        }

        try:
            async for chunk in controller.process_message_stream(
                message=request.message,
                department=request.department,
                academic_year=request.academic_year,
                history=_history_for_controller(request.history),
            ):
                for event in [part for part in chunk.split("\n\n") if part.strip().startswith("data:")]:
                    try:
                        payload = json.loads(event.replace("data:", "", 1).strip())
                    except json.JSONDecodeError:
                        continue
                    if payload.get("type") == "metadata":
                        metadata.update(payload)
                    elif payload.get("type") in ("content", "token"):
                        reply_parts.append(payload.get("content", "") or payload.get("token", ""))
                    elif payload.get("type") == "done" and payload.get("reply") and not reply_parts:
                        reply_parts.append(payload.get("reply", ""))
                    elif payload.get("type") == "error":
                        reply_parts.append(payload.get("error", ""))
                yield chunk
        finally:
            assistant_reply = "".join(reply_parts).strip()
            if assistant_reply:
                try:
                    conversation = await record_chat_exchange(
                        database,
                        user=current_user,
                        conversation_id=request.conversation_id,
                        user_message=request.message,
                        assistant_reply=assistant_reply,
                        intent=metadata.get("intent", "general"),
                        confidence=metadata.get("confidence", "Low"),
                        sources=metadata.get("sources", []),
                        structured_sources=metadata.get("structured_sources", []),
                        retrieval_score=float(metadata.get("retrieval_score", 0.0) or 0.0),
                        department=request.department,
                        academic_year=request.academic_year,
                        source_visibility=metadata.get("source_visibility", "none"),
                    )
                    await log_activity(
                        database,
                        current_user["_id"],
                        "chat.stream_message",
                        {"conversation_id": conversation["id"], "intent": metadata.get("intent", "general")},
                    )
                except Exception as exc:
                    logger.error("Failed to persist stream chat exchange: %s", exc, exc_info=True)

    return StreamingResponse(
        persisted_stream(),
        media_type="text/event-stream"
    )

@router.post("/chat/detailed", response_model=ChatDetailedResponse)
async def chat_detailed(
    request: ChatRequest,
    controller: ChatController = Depends(get_chat_controller),
    current_user: dict = Depends(get_current_user),
) -> ChatDetailedResponse:
    """
    Process a chat message with detailed response metadata.
    """
    try:
        database = get_database()
        response = await controller.process_message(
            message=request.message,
            department=request.department,
            academic_year=request.academic_year,
            history=_history_for_controller(request.history),
        )
        conversation_id = await _persist_chat_response(database, current_user, request, response)
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
            conversation_id=conversation_id,
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


@router.get("/chat/conversations")
async def get_conversations(current_user: dict = Depends(get_current_user)):
    """Return persisted chat history for the authenticated user."""

    database = get_database()
    return {"conversations": await list_user_conversations(database, current_user["_id"])}


@router.put("/chat/conversations/{conversation_id}")
async def rename_conversation(
    conversation_id: str,
    request: ConversationUpdateRequest,
    current_user: dict = Depends(get_current_user),
):
    """Rename one persisted conversation owned by the current user."""

    database = get_database()
    try:
        conversation = await rename_user_conversation(database, current_user["_id"], conversation_id, request.title)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    await log_activity(database, current_user["_id"], "chat.conversation_renamed", {"conversation_id": conversation_id})
    return {"conversation": conversation}


@router.post("/chat/conversations/{conversation_id}/clear")
async def clear_conversation(
    conversation_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Clear messages from one persisted conversation owned by the current user."""

    database = get_database()
    try:
        conversation = await clear_user_conversation(database, current_user["_id"], conversation_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    await log_activity(database, current_user["_id"], "chat.conversation_cleared", {"conversation_id": conversation_id})
    return {"conversation": conversation}


@router.delete("/chat/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete one persisted conversation owned by the current user."""

    database = get_database()
    try:
        await delete_user_conversation(database, current_user["_id"], conversation_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    await log_activity(database, current_user["_id"], "chat.conversation_deleted", {"conversation_id": conversation_id})
    return {"success": True}


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
