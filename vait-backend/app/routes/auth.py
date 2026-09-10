"""Authentication and profile routes."""

from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import urlencode

import httpx
import jwt as pyjwt
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr, Field
from pymongo.errors import DuplicateKeyError

from app.core.constants import USER_ROLE, USERS_COLLECTION
from app.db import get_database, get_database_async
from app.dependencies import get_current_user
from app.services.activity_service import log_activity
from app.services.security_service import create_access_token, hash_password, verify_password
from app.services.user_service import (
    build_user_document,
    ensure_user_record,
    get_user_by_identifier,
    normalize_email,
    normalize_username,
    serialize_user,
)
from app.utils.config import get_settings

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    identifier: str = Field(..., min_length=3, max_length=120)
    password: str = Field(..., min_length=1, max_length=256)


class SignupRequest(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=120)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=256)
    username: str | None = Field(default=None, min_length=3, max_length=50)


class ProfileUpdateRequest(BaseModel):
    username: str | None = Field(default=None, min_length=3, max_length=50)
    email: EmailStr | None = None
    full_name: str | None = Field(default=None, max_length=120)


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(..., min_length=1, max_length=256)
    new_password: str = Field(..., min_length=6, max_length=256)


async def _get_database_or_503():
    try:
        return await get_database_async()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Authentication database is not available. {exc}",
        ) from exc


async def _unique_username(database, email: str, preferred: str | None = None) -> str:
    base = normalize_username(preferred or email.split("@", 1)[0]) or "user"
    base = base[:40]

    candidate = base
    suffix = 1
    while await database[USERS_COLLECTION].find_one({"username": candidate}):
        suffix += 1
        candidate = f"{base[:36]}{suffix}"
    return candidate


def _issue_auth_response(user: dict, settings):
    token = create_access_token(user, settings)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": serialize_user(user),
    }


def _oauth_redirect(frontend_url: str, payload: dict[str, str]) -> RedirectResponse:
    fragment = urlencode(payload)
    return RedirectResponse(f"{frontend_url}#{fragment}")


@router.post("/login")
async def login(request: LoginRequest):
    """Authenticate a user and issue a session JWT."""

    settings = get_settings()
    database = await _get_database_or_503()

    user = await get_user_by_identifier(database, request.identifier)
    if user:
        user = await ensure_user_record(database, user)
    if not user or not verify_password(request.password, user.get("password_hash", "")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    now = datetime.now(timezone.utc)
    await database[USERS_COLLECTION].update_one(
        {"_id": user["_id"]},
        {"$set": {"last_login_at": now, "updated_at": now}},
    )
    user["last_login_at"] = now
    user["updated_at"] = now

    await log_activity(database, user["_id"], "auth.login", {"role": user["role"]})

    return _issue_auth_response(user, settings)


@router.post("/signup", status_code=status.HTTP_201_CREATED)
async def signup(request: SignupRequest):
    """Create a student/user account and issue a session JWT."""

    settings = get_settings()
    database = await _get_database_or_503()

    email = normalize_email(request.email)
    preferred_username = normalize_username(request.username) if request.username else None
    if request.username and not preferred_username:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Username must contain letters or numbers.")

    existing = await database[USERS_COLLECTION].find_one(
        {
            "$or": [
                {"email": email},
                *([{"username": preferred_username}] if preferred_username else []),
            ]
        }
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with that email or username already exists.",
        )

    now = datetime.now(timezone.utc)
    user = build_user_document(
        username=await _unique_username(database, email, preferred_username),
        email=email,
        password_hash=hash_password(request.password),
        role=USER_ROLE,
        full_name=request.full_name,
        auth_provider="password",
        now=now,
    )
    user["last_login_at"] = now

    try:
        result = await database[USERS_COLLECTION].insert_one(user)
    except DuplicateKeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with that email or username already exists.",
        ) from exc
    user["_id"] = result.inserted_id
    await log_activity(database, user["_id"], "auth.signup", {"role": USER_ROLE})

    return _issue_auth_response(user, settings)


@router.get("/google/login")
async def google_login(next_url: str | None = Query(default=None, alias="next")):
    """Redirect to Google's OAuth consent screen."""

    settings = get_settings()
    if not settings.google_client_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google OAuth is not configured.",
        )

    now_ts = int(datetime.now(timezone.utc).timestamp())
    state_payload = {
        "next": next_url or settings.frontend_auth_redirect_url,
        "iat": now_ts,
        "exp": now_ts + 600,
    }
    state = pyjwt.encode(state_payload, settings.jwt_secret, algorithm="HS256")
    query = urlencode(
        {
            "client_id": settings.google_client_id,
            "redirect_uri": settings.google_redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "access_type": "offline",
            "prompt": "select_account",
            "state": state,
        }
    )
    return RedirectResponse(f"https://accounts.google.com/o/oauth2/v2/auth?{query}")


@router.get("/google/callback")
async def google_callback(
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
):
    """Complete Google OAuth and redirect the browser back to the frontend."""

    settings = get_settings()
    database = await _get_database_or_503()
    default_frontend_redirect_url = settings.frontend_auth_redirect_url

    if error:
        return _oauth_redirect(default_frontend_redirect_url, {"error": f"Google sign-in cancelled: {error}"})
    if not code:
        return _oauth_redirect(default_frontend_redirect_url, {"error": "Missing OAuth code."})
    if not settings.google_client_id or not settings.google_client_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google OAuth is not configured.",
        )

    try:
        state_payload = pyjwt.decode(state or "", settings.jwt_secret, algorithms=["HS256"])
    except pyjwt.PyJWTError as exc:
        return _oauth_redirect(default_frontend_redirect_url, {"error": "Invalid OAuth state."})

    frontend_redirect_url = state_payload.get("next") or settings.frontend_auth_redirect_url

    async with httpx.AsyncClient(timeout=15) as client:
        token_response = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "redirect_uri": settings.google_redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        if token_response.status_code != 200:
            return _oauth_redirect(frontend_redirect_url, {"error": "Google OAuth token exchange failed."})

        access_token = token_response.json().get("access_token")
        if not access_token:
            return _oauth_redirect(frontend_redirect_url, {"error": "Google OAuth did not return an access token."})
        userinfo_response = await client.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if userinfo_response.status_code != 200:
            return _oauth_redirect(frontend_redirect_url, {"error": "Could not read Google profile."})

    profile = userinfo_response.json()
    email = normalize_email(str(profile.get("email", "")))
    if not email or profile.get("email_verified") is False:
        return _oauth_redirect(frontend_redirect_url, {"error": "Google account email is not verified."})

    now = datetime.now(timezone.utc)
    user = await database[USERS_COLLECTION].find_one({"email": email})
    google_updates = {
        "google_sub": profile.get("sub", ""),
        "picture": profile.get("picture", ""),
        "auth_provider": "google",
        "profile.full_name": profile.get("name", "") or email.split("@", 1)[0],
        "profile.avatar_url": profile.get("picture", ""),
        "profile.email_verified": bool(profile.get("email_verified", True)),
        "profile.locale": profile.get("locale", ""),
        "last_login_at": now,
        "updated_at": now,
    }

    if user:
        user = await ensure_user_record(database, user)
        await database[USERS_COLLECTION].update_one(
            {"_id": user["_id"]},
            {
                "$set": google_updates,
                "$addToSet": {"auth_providers": "google"},
            },
        )
        user.update(
            {
                "google_sub": google_updates["google_sub"],
                "picture": google_updates["picture"],
                "auth_provider": "google",
                "last_login_at": now,
                "updated_at": now,
            }
        )
        providers = set(user.get("auth_providers", []))
        providers.add("google")
        user["auth_providers"] = list(providers)
        user["profile"] = {
            **(user.get("profile") or {}),
            "full_name": google_updates["profile.full_name"],
            "avatar_url": google_updates["profile.avatar_url"],
            "email_verified": google_updates["profile.email_verified"],
            "locale": google_updates["profile.locale"],
        }
    else:
        user = build_user_document(
            username=await _unique_username(database, email),
            email=email,
            password_hash="",
            role=USER_ROLE,
            full_name=profile.get("name", "") or email.split("@", 1)[0],
            auth_provider="google",
            google_profile=profile,
            now=now,
        )
        user["last_login_at"] = now
        try:
            result = await database[USERS_COLLECTION].insert_one(user)
        except DuplicateKeyError:
            return _oauth_redirect(frontend_redirect_url, {"error": "A user with that Google email already exists."})
        user["_id"] = result.inserted_id

    await log_activity(database, user["_id"], "auth.google_login", {"role": user["role"]})
    token = create_access_token(user, settings)
    return _oauth_redirect(frontend_redirect_url, {"access_token": token, "token_type": "bearer"})


@router.post("/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """Invalidate the current user's active session token."""

    database = await _get_database_or_503()
    await database[USERS_COLLECTION].update_one(
        {"_id": current_user["_id"]},
        {
            "$set": {"updated_at": datetime.now(timezone.utc)},
            "$inc": {"session_version": 1},
        },
    )
    await log_activity(database, current_user["_id"], "auth.logout", {})
    return {"success": True}


@router.get("/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    """Return the current user profile."""

    return {"user": serialize_user(current_user)}


@router.put("/profile")
async def update_profile(request: ProfileUpdateRequest, current_user: dict = Depends(get_current_user)):
    """Update the authenticated user's account profile."""

    database = await _get_database_or_503()
    updates = {"updated_at": datetime.now(timezone.utc)}

    if request.username:
        normalized = normalize_username(request.username)
        if not normalized:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Username must contain letters or numbers.")
        existing = await database[USERS_COLLECTION].find_one({"username": normalized, "_id": {"$ne": current_user["_id"]}})
        if existing:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username is already in use")
        updates["username"] = normalized

    if request.email:
        normalized_email = normalize_email(request.email)
        existing = await database[USERS_COLLECTION].find_one({"email": normalized_email, "_id": {"$ne": current_user["_id"]}})
        if existing:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email is already in use")
        updates["email"] = normalized_email

    if request.full_name is not None:
        full_name = request.full_name.strip()
        updates["full_name"] = full_name
        updates["profile.full_name"] = full_name

    await database[USERS_COLLECTION].update_one({"_id": current_user["_id"]}, {"$set": updates})
    await log_activity(database, current_user["_id"], "profile.updated", {k: v for k, v in updates.items() if k != "updated_at"})
    refreshed = await database[USERS_COLLECTION].find_one({"_id": current_user["_id"]})
    return {"user": serialize_user(refreshed)}


@router.put("/password")
async def change_password(request: PasswordChangeRequest, current_user: dict = Depends(get_current_user)):
    """Change the current user's password and rotate the session version."""

    database = await _get_database_or_503()
    settings = get_settings()

    if not current_user.get("password_hash"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password sign-in is not enabled for this account.",
        )
    if not verify_password(request.current_password, current_user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Current password is incorrect")

    now = datetime.now(timezone.utc)
    await database[USERS_COLLECTION].update_one(
        {"_id": current_user["_id"]},
        {
            "$set": {
                "password_hash": hash_password(request.new_password),
                "must_change_password": False,
                "updated_at": now,
            },
            "$addToSet": {"auth_providers": "password"},
            "$inc": {"session_version": 1},
        },
    )

    refreshed = await database[USERS_COLLECTION].find_one({"_id": current_user["_id"]})
    token = create_access_token(refreshed, settings)
    await log_activity(database, current_user["_id"], "profile.password_changed", {})

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": serialize_user(refreshed),
    }
