"""Password hashing, JWT handling, and secret encryption helpers."""

from __future__ import annotations

import base64
import hashlib
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt
from cryptography.fernet import Fernet

from app.utils.config import Settings

JWT_ALGORITHM = "HS256"
_DURATION_RE = re.compile(r"^(?P<value>\d+)(?P<unit>[smhd])$")
_PLACEHOLDER_MARKERS = ("your_", "change-me", "placeholder", "replace")


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""

    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Validate a plain password against a bcrypt hash."""

    if not password_hash:
        return False
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def parse_expiry(expiry: str) -> timedelta:
    """Parse strings like 15m, 12h, or 7d."""

    match = _DURATION_RE.match(expiry.strip().lower())
    if not match:
        raise ValueError("Invalid JWT expiry format. Use values like 15m, 12h, or 7d.")

    value = int(match.group("value"))
    unit = match.group("unit")

    if unit == "s":
        return timedelta(seconds=value)
    if unit == "m":
        return timedelta(minutes=value)
    if unit == "h":
        return timedelta(hours=value)
    return timedelta(days=value)


def create_access_token(user: dict[str, Any], settings: Settings) -> str:
    """Create a signed JWT for the authenticated user."""

    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user["_id"]),
        "role": user["role"],
        "ver": user.get("session_version", 0),
        "iat": int(now.timestamp()),
        "exp": now + parse_expiry(settings.jwt_expiry),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str, settings: Settings) -> dict[str, Any]:
    """Decode and validate the access token."""

    return jwt.decode(token, settings.jwt_secret, algorithms=[JWT_ALGORITHM])


def _fernet_key(secret: str) -> bytes:
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def encrypt_secret(value: str, settings: Settings) -> str:
    """Encrypt a secret for storage."""

    if not value:
        return ""
    return Fernet(_fernet_key(settings.jwt_secret)).encrypt(value.encode("utf-8")).decode("utf-8")


def decrypt_secret(value: str, settings: Settings) -> str:
    """Decrypt a previously stored secret."""

    if not value:
        return ""
    return Fernet(_fernet_key(settings.jwt_secret)).decrypt(value.encode("utf-8")).decode("utf-8")


def mask_secret(value: str) -> str:
    """Mask a sensitive value for safe UI display."""

    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}{'*' * (len(value) - 8)}{value[-4:]}"


def looks_like_placeholder(value: str | None) -> bool:
    """Return True if the value looks like a placeholder string."""

    if not value:
        return True
    lowered = value.strip().lower()
    return any(marker in lowered for marker in _PLACEHOLDER_MARKERS)
