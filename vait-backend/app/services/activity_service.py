"""Activity log helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from bson import ObjectId

from app.core.constants import ACTIVITY_LOGS_COLLECTION


async def log_activity(database, user_id: str | ObjectId | None, action: str, metadata: dict[str, Any] | None = None) -> None:
    """Persist a user activity record."""

    record = {
        "user_id": ObjectId(user_id) if user_id else None,
        "action": action,
        "metadata": metadata or {},
        "timestamp": datetime.now(timezone.utc),
    }
    await database[ACTIVITY_LOGS_COLLECTION].insert_one(record)


def serialize_activity(log: dict[str, Any], user_lookup: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Convert an activity document into a JSON-safe structure."""

    user_info = {}
    if user_lookup and log.get("user_id") in user_lookup:
        user_doc = user_lookup[log["user_id"]]
        user_info = {
            "id": str(user_doc["_id"]),
            "username": user_doc.get("username"),
            "email": user_doc.get("email"),
            "role": user_doc.get("role"),
        }

    return {
        "id": str(log["_id"]),
        "user_id": str(log["user_id"]) if log.get("user_id") else None,
        "action": log["action"],
        "metadata": log.get("metadata", {}),
        "timestamp": log["timestamp"].isoformat(),
        "user": user_info or None,
    }
