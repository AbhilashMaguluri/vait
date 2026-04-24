"""User lifecycle, defaults, and serialization helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from bson import ObjectId

from app.core.constants import ADMIN_ROLE, ALL_ROLES, USERS_COLLECTION
from app.services.security_service import hash_password


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


def normalize_email(email: str) -> str:
    """Normalize an email address for unique account lookup."""

    return email.strip().lower()


def normalize_username(username: str) -> str:
    """Normalize a username while keeping only user-safe characters."""

    normalized = username.strip().lower()
    return "".join(ch if ch.isalnum() or ch in {"_", "."} else "_" for ch in normalized).strip("_.")


def build_user_document(
    *,
    username: str,
    email: str,
    role: str,
    full_name: str,
    password_hash: str = "",
    auth_provider: str = "password",
    must_change_password: bool = False,
    google_profile: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Create a complete user document with stable places for account data."""

    timestamp = now or utc_now()
    google_profile = google_profile or {}
    picture = str(google_profile.get("picture") or "")
    google_sub = str(google_profile.get("sub") or "")
    providers = ["google"] if auth_provider == "google" else ["password"]

    return {
        "username": normalize_username(username),
        "email": normalize_email(email),
        "password_hash": password_hash,
        "role": role,
        "full_name": full_name.strip(),
        "auth_provider": auth_provider,
        "auth_providers": providers,
        "google_sub": google_sub,
        "picture": picture,
        "profile": {
            "full_name": full_name.strip(),
            "avatar_url": picture,
            "email_verified": bool(google_profile.get("email_verified", auth_provider == "password")),
            "locale": str(google_profile.get("locale") or ""),
        },
        "preferences": {
            "department": "",
            "academic_year": "",
        },
        "usage": {
            "chat_messages": 0,
            "chat_conversations": 0,
            "last_chat_at": None,
        },
        "must_change_password": must_change_password,
        "session_version": 0,
        "created_at": timestamp,
        "updated_at": timestamp,
        "last_login_at": None,
    }


def user_default_updates(user: dict[str, Any]) -> dict[str, Any]:
    """Return missing fields needed by the current account schema."""

    updates: dict[str, Any] = {}
    full_name = user.get("full_name", "")
    picture = user.get("picture", "")

    if "auth_provider" not in user:
        if user.get("google_sub") and user.get("password_hash"):
            updates["auth_provider"] = "password"
        elif user.get("google_sub"):
            updates["auth_provider"] = "google"
        else:
            updates["auth_provider"] = "password"

    if "auth_providers" not in user:
        providers = []
        if user.get("password_hash"):
            providers.append("password")
        if user.get("google_sub"):
            providers.append("google")
        updates["auth_providers"] = providers or [updates.get("auth_provider", user.get("auth_provider", "password"))]

    if "profile" not in user:
        updates["profile"] = {
            "full_name": full_name,
            "avatar_url": picture,
            "email_verified": True,
            "locale": "",
        }

    if "preferences" not in user:
        updates["preferences"] = {
            "department": "",
            "academic_year": "",
        }

    if "usage" not in user:
        updates["usage"] = {
            "chat_messages": 0,
            "chat_conversations": 0,
            "last_chat_at": None,
        }

    if "must_change_password" not in user:
        updates["must_change_password"] = False
    if "session_version" not in user:
        updates["session_version"] = 0
    if "created_at" not in user:
        updates["created_at"] = utc_now()
    if "updated_at" not in user:
        updates["updated_at"] = utc_now()
    if "last_login_at" not in user:
        updates["last_login_at"] = None

    return updates


async def ensure_user_record(database, user: dict[str, Any]) -> dict[str, Any]:
    """Backfill missing account fields and return the updated user document."""

    updates = user_default_updates(user)
    if updates:
        updates["updated_at"] = utc_now()
        await database[USERS_COLLECTION].update_one({"_id": user["_id"]}, {"$set": updates})
        user.update(updates)
    return user


def serialize_user(user: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe representation of a user."""

    created_at = user.get("created_at")
    updated_at = user.get("updated_at")
    last_login_at = user.get("last_login_at")
    profile = user.get("profile") or {}
    usage = user.get("usage") or {}

    return {
        "id": str(user["_id"]),
        "username": user.get("username", ""),
        "email": user.get("email", ""),
        "role": user.get("role", ""),
        "full_name": user.get("full_name", ""),
        "auth_provider": user.get("auth_provider", "password"),
        "auth_providers": user.get("auth_providers", [user.get("auth_provider", "password")]),
        "picture": user.get("picture") or profile.get("avatar_url", ""),
        "profile": {
            "full_name": profile.get("full_name", user.get("full_name", "")),
            "avatar_url": profile.get("avatar_url", user.get("picture", "")),
            "email_verified": profile.get("email_verified", True),
            "locale": profile.get("locale", ""),
        },
        "preferences": user.get("preferences", {"department": "", "academic_year": ""}),
        "usage": {
            "chat_messages": usage.get("chat_messages", 0),
            "chat_conversations": usage.get("chat_conversations", 0),
            "last_chat_at": usage.get("last_chat_at").isoformat()
            if hasattr(usage.get("last_chat_at"), "isoformat")
            else usage.get("last_chat_at"),
        },
        "must_change_password": user.get("must_change_password", False),
        "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else created_at,
        "updated_at": updated_at.isoformat() if hasattr(updated_at, "isoformat") else updated_at,
        "last_login_at": last_login_at.isoformat() if hasattr(last_login_at, "isoformat") else last_login_at,
        "session_version": user.get("session_version", 0),
    }


async def ensure_default_admin(database, username: str, email: str, password: str) -> None:
    """Create the default admin account if the system has none."""

    existing_admin = await database[USERS_COLLECTION].find_one({"role": ADMIN_ROLE})
    if existing_admin:
        await ensure_user_record(database, existing_admin)
        return

    now = utc_now()
    admin_user = build_user_document(
        username=username,
        email=email,
        password_hash=hash_password(password),
        role=ADMIN_ROLE,
        full_name="System Administrator",
        must_change_password=True,
        now=now,
    )
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
