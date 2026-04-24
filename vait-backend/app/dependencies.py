"""Authentication and RBAC dependencies."""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import ExpiredSignatureError, InvalidTokenError

from app.db import get_database
from app.services.user_service import ensure_user_record, get_user_by_id, serialize_user
from app.services.security_service import decode_access_token
from app.utils.config import get_settings

bearer_scheme = HTTPBearer(auto_error=True)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> dict:
    """Resolve the current authenticated user from a bearer token."""

    settings = get_settings()
    try:
        database = get_database()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Authentication database is not available. {exc}",
        ) from exc

    try:
        payload = decode_access_token(credentials.credentials, settings)
    except ExpiredSignatureError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired") from exc
    except InvalidTokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid access token") from exc

    user = await get_user_by_id(database, payload["sub"])
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    user = await ensure_user_record(database, user)

    if user.get("session_version", 0) != payload.get("ver", 0):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session is no longer valid")

    return user


def require_roles(*roles: str):
    """Require the current user to have one of the allowed roles."""

    async def dependency(user: dict = Depends(get_current_user)) -> dict:
        if user["role"] not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return user

    return dependency


async def get_current_user_public(user: dict = Depends(get_current_user)) -> dict:
    """Return a serialized user object for route handlers that only need public fields."""

    return serialize_user(user)
