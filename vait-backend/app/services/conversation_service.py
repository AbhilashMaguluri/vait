"""Per-user chat conversation persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from app.core.constants import CONVERSATIONS_COLLECTION, USERS_COLLECTION


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


def _as_object_id(value: str | ObjectId | None) -> ObjectId | None:
    if isinstance(value, ObjectId):
        return value
    if value and ObjectId.is_valid(str(value)):
        return ObjectId(str(value))
    return None


def _iso(value: Any) -> Any:
    return value.isoformat() if hasattr(value, "isoformat") else value


def _conversation_title(message: str) -> str:
    clean = " ".join(message.strip().split())
    if not clean:
        return "New Conversation"
    return clean[:50] + ("..." if len(clean) > 50 else "")


def _source_titles(structured_sources: list[dict[str, Any]] | None, fallback: list[str] | None) -> list[str]:
    if fallback:
        return fallback
    return [item.get("title", "") for item in structured_sources or [] if item.get("title")]


def _serialize_message(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": message.get("id", ""),
        "role": message.get("role", ""),
        "text": message.get("text", ""),
        "timestamp": _iso(message.get("timestamp")),
        "sources": message.get("sources", []),
        "structured_sources": message.get("structured_sources", []),
        "confidence": message.get("confidence"),
        "category": message.get("category"),
        "retrieval_score": message.get("retrieval_score", 0.0),
        "performance": message.get("performance"),
    }


def serialize_conversation(conversation: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe conversation for the frontend."""

    messages = conversation.get("messages", [])
    return {
        "id": conversation.get("client_conversation_id") or str(conversation["_id"]),
        "database_id": str(conversation["_id"]),
        "title": conversation.get("title", "New Conversation"),
        "messages": [_serialize_message(message) for message in messages],
        "category": conversation.get("category", "Academic"),
        "confidence": conversation.get("confidence"),
        "sources": conversation.get("sources", []),
        "structured_sources": conversation.get("structured_sources", []),
        "department": conversation.get("department"),
        "academic_year": conversation.get("academic_year"),
        "createdAt": _iso(conversation.get("created_at")),
        "updatedAt": _iso(conversation.get("updated_at")),
        "message_count": conversation.get("message_count", len(messages)),
    }


async def find_user_conversation(database, user_id: ObjectId, conversation_id: str | None) -> dict[str, Any] | None:
    """Find a conversation by database id or client id scoped to one user."""

    if not conversation_id:
        return None

    filters: list[dict[str, Any]] = [{"client_conversation_id": str(conversation_id)}]
    object_id = _as_object_id(conversation_id)
    if object_id is not None:
        filters.append({"_id": object_id})

    return await database[CONVERSATIONS_COLLECTION].find_one(
        {
            "user_id": user_id,
            "$or": filters,
        }
    )


async def list_user_conversations(database, user_id: ObjectId, limit: int = 100) -> list[dict[str, Any]]:
    """List recent conversations for one authenticated user."""

    cursor = (
        database[CONVERSATIONS_COLLECTION]
        .find({"user_id": user_id})
        .sort("updated_at", -1)
        .limit(limit)
    )
    conversations = await cursor.to_list(length=limit)
    return [serialize_conversation(conversation) for conversation in conversations]


async def record_chat_exchange(
    database,
    *,
    user: dict[str, Any],
    conversation_id: str | None,
    user_message: str,
    assistant_reply: str,
    intent: str,
    confidence: str,
    sources: list[str] | None,
    structured_sources: list[dict[str, Any]] | None,
    retrieval_score: float,
    department: str | None = None,
    academic_year: str | None = None,
    performance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append a user/assistant exchange and update per-user usage counters."""

    now = utc_now()
    user_id = user["_id"]
    client_conversation_id = str(conversation_id or ObjectId())
    conversation = await find_user_conversation(database, user_id, client_conversation_id)
    created = False

    if conversation is None:
        conversation = {
            "user_id": user_id,
            "client_conversation_id": client_conversation_id,
            "title": _conversation_title(user_message),
            "category": intent or "general",
            "confidence": confidence,
            "sources": _source_titles(structured_sources, sources),
            "structured_sources": structured_sources or [],
            "department": department,
            "academic_year": academic_year,
            "message_count": 0,
            "messages": [],
            "created_at": now,
            "updated_at": now,
            "last_message_at": now,
        }
        try:
            result = await database[CONVERSATIONS_COLLECTION].insert_one(conversation)
            conversation["_id"] = result.inserted_id
            created = True
        except DuplicateKeyError:
            conversation = await find_user_conversation(database, user_id, client_conversation_id)
            created = False
    if conversation is None:
        raise RuntimeError("Could not create or load the conversation")

    user_message_doc = {
        "id": str(ObjectId()),
        "role": "user",
        "text": user_message,
        "timestamp": now,
        "department": department,
        "academic_year": academic_year,
    }
    assistant_message_doc = {
        "id": str(ObjectId()),
        "role": "assistant",
        "text": assistant_reply,
        "timestamp": now,
        "sources": _source_titles(structured_sources, sources),
        "structured_sources": structured_sources or [],
        "confidence": confidence,
        "category": intent or "general",
        "retrieval_score": retrieval_score,
        "performance": performance or {},
    }

    updates = {
        "category": intent or conversation.get("category", "general"),
        "confidence": confidence,
        "sources": _source_titles(structured_sources, sources),
        "structured_sources": structured_sources or [],
        "department": department,
        "academic_year": academic_year,
        "updated_at": now,
        "last_message_at": now,
    }
    if not conversation.get("title") or conversation.get("title") == "New Conversation":
        updates["title"] = _conversation_title(user_message)

    await database[CONVERSATIONS_COLLECTION].update_one(
        {"_id": conversation["_id"], "user_id": user_id},
        {
            "$set": updates,
            "$push": {"messages": {"$each": [user_message_doc, assistant_message_doc]}},
            "$inc": {"message_count": 2},
        },
    )

    usage_inc = {"usage.chat_messages": 1}
    if created:
        usage_inc["usage.chat_conversations"] = 1
    await database[USERS_COLLECTION].update_one(
        {"_id": user_id},
        {
            "$set": {"usage.last_chat_at": now, "updated_at": now},
            "$inc": usage_inc,
        },
    )

    refreshed = await database[CONVERSATIONS_COLLECTION].find_one({"_id": conversation["_id"], "user_id": user_id})
    return serialize_conversation(refreshed)


async def rename_user_conversation(database, user_id: ObjectId, conversation_id: str, title: str) -> dict[str, Any]:
    """Rename a conversation owned by the authenticated user."""

    conversation = await find_user_conversation(database, user_id, conversation_id)
    if not conversation:
        raise ValueError("Conversation not found")

    await database[CONVERSATIONS_COLLECTION].update_one(
        {"_id": conversation["_id"], "user_id": user_id},
        {"$set": {"title": title.strip(), "updated_at": utc_now()}},
    )
    refreshed = await database[CONVERSATIONS_COLLECTION].find_one({"_id": conversation["_id"], "user_id": user_id})
    return serialize_conversation(refreshed)


async def clear_user_conversation(database, user_id: ObjectId, conversation_id: str) -> dict[str, Any]:
    """Remove all messages from a conversation while keeping the shell."""

    conversation = await find_user_conversation(database, user_id, conversation_id)
    if not conversation:
        raise ValueError("Conversation not found")

    await database[CONVERSATIONS_COLLECTION].update_one(
        {"_id": conversation["_id"], "user_id": user_id},
        {
            "$set": {
                "messages": [],
                "message_count": 0,
                "confidence": None,
                "sources": [],
                "structured_sources": [],
                "updated_at": utc_now(),
            }
        },
    )
    refreshed = await database[CONVERSATIONS_COLLECTION].find_one({"_id": conversation["_id"], "user_id": user_id})
    return serialize_conversation(refreshed)


async def delete_user_conversation(database, user_id: ObjectId, conversation_id: str) -> None:
    """Delete a conversation owned by the authenticated user."""

    conversation = await find_user_conversation(database, user_id, conversation_id)
    if not conversation:
        raise ValueError("Conversation not found")
    await database[CONVERSATIONS_COLLECTION].delete_one({"_id": conversation["_id"], "user_id": user_id})
