"""User lifecycle and serialization helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from bson import ObjectId

from app.core.constants import ADMIN_ROLE, ALL_ROLES, USERS_COLLECTION
from app.services.security_service import hash_password


def serialize_user(user: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe representation of a user."""

    return {
        "id": str(user["_id"]),
        "username": user["username"],
        "email": user["email"],
        "role": user["role"],
        "full_name": user.get("full_name", ""),
        "must_change_password": user.get("must_change_password", False),
        "created_at": user["created_at"].isoformat(),
        "updated_at": user["updated_at"].isoformat(),
        "last_login_at": user.get("last_login_at").isoformat() if user.get("last_login_at") else None,
        "session_version": user.get("session_version", 0),
    }


async def ensure_default_admin(database, username: str, email: str, password: str) -> None:
    """Create the default admin account if the system has none."""

    existing_admin = await database[USERS_COLLECTION].find_one({"role": ADMIN_ROLE})
    if existing_admin:
        return

    now = datetime.now(timezone.utc)
    admin_user = {
        "username": username,
        "email": email,
        "password_hash": hash_password(password),
        "role": ADMIN_ROLE,
        "full_name": "System Administrator",
        "session_version": 0,
        "must_change_password": True,
        "created_at": now,
        "updated_at": now,
        "last_login_at": None,
    }
    await database[USERS_COLLECTION].insert_one(admin_user)


async def get_user_by_identifier(database, identifier: str) -> dict[str, Any] | None:
    """Find a user by email or username."""

    normalized = identifier.strip().lower()
    return await database[USERS_COLLECTION].find_one(
        {
            "$or": [
                {"email": normalized},
                {"username": normalized},
            ]
        }
    )


async def get_user_by_id(database, user_id: str | ObjectId) -> dict[str, Any] | None:
    """Find a user by ObjectId."""

    return await database[USERS_COLLECTION].find_one({"_id": ObjectId(user_id)})


def validate_role(role: str) -> str:
    """Normalize and validate a role value."""

    normalized = role.strip().lower()
    if normalized not in ALL_ROLES:
        raise ValueError(f"Invalid role '{role}'")
    return normalized
