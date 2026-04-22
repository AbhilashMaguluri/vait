"""Authentication and profile routes."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field

from app.core.constants import USERS_COLLECTION
from app.db import get_database
from app.dependencies import get_current_user
from app.services.activity_service import log_activity
from app.services.security_service import create_access_token, hash_password, verify_password
from app.services.user_service import get_user_by_identifier, serialize_user
from app.utils.config import get_settings

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    identifier: str = Field(..., min_length=3, max_length=120)
    password: str = Field(..., min_length=1, max_length=256)


class ProfileUpdateRequest(BaseModel):
    username: str | None = Field(default=None, min_length=3, max_length=50)
    email: EmailStr | None = None
    full_name: str | None = Field(default=None, max_length=120)


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(..., min_length=1, max_length=256)
    new_password: str = Field(..., min_length=6, max_length=256)


@router.post("/login")
async def login(request: LoginRequest):
    """Authenticate a user and issue a session JWT."""

    settings = get_settings()
    database = get_database()

    user = await get_user_by_identifier(database, request.identifier)
    if not user or not verify_password(request.password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    now = datetime.now(timezone.utc)
    await database[USERS_COLLECTION].update_one(
        {"_id": user["_id"]},
        {"$set": {"last_login_at": now, "updated_at": now}},
    )
    user["last_login_at"] = now
    user["updated_at"] = now

    token = create_access_token(user, settings)
    await log_activity(database, user["_id"], "auth.login", {"role": user["role"]})

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": serialize_user(user),
    }


@router.post("/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """Invalidate the current user's active session token."""

    database = get_database()
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

    database = get_database()
    updates = {"updated_at": datetime.now(timezone.utc)}

    if request.username:
        normalized = request.username.strip().lower()
        existing = await database[USERS_COLLECTION].find_one({"username": normalized, "_id": {"$ne": current_user["_id"]}})
        if existing:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username is already in use")
        updates["username"] = normalized

    if request.email:
        normalized_email = request.email.strip().lower()
        existing = await database[USERS_COLLECTION].find_one({"email": normalized_email, "_id": {"$ne": current_user["_id"]}})
        if existing:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email is already in use")
        updates["email"] = normalized_email

    if request.full_name is not None:
        updates["full_name"] = request.full_name.strip()

    await database[USERS_COLLECTION].update_one({"_id": current_user["_id"]}, {"$set": updates})
    await log_activity(database, current_user["_id"], "profile.updated", {k: v for k, v in updates.items() if k != "updated_at"})
    refreshed = await database[USERS_COLLECTION].find_one({"_id": current_user["_id"]})
    return {"user": serialize_user(refreshed)}


@router.put("/password")
async def change_password(request: PasswordChangeRequest, current_user: dict = Depends(get_current_user)):
    """Change the current user's password and rotate the session version."""

    database = get_database()
    settings = get_settings()

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
