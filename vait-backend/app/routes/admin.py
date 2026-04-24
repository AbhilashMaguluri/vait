"""Protected admin and trainer platform routes."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from bson import ObjectId
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, EmailStr, Field, HttpUrl

from app.core.constants import (
    ADMIN_ROLE,
    ACTIVITY_LOGS_COLLECTION,
    DEFAULT_AI_SETTINGS,
    DOCUMENT_SOURCE_TYPES,
    USERS_COLLECTION,
)
from app.db import get_database
from app.dependencies import get_current_user, require_roles
from app.services.activity_service import log_activity, serialize_activity
from app.services.document_service import (
    create_document_record,
    deactivate_document,
    get_knowledge_base_dashboard,
    list_documents,
    process_document,
    rebuild_index_from_documents,
    save_uploaded_file,
    schedule_background_job,
    update_document,
)
from app.services.security_service import hash_password
from app.services.settings_service import (
    get_ai_settings,
    get_api_key_settings,
    get_system_settings,
    save_api_key_settings,
    upsert_setting_value,
)
from app.services.user_service import (
    build_user_document,
    get_user_by_id,
    normalize_email,
    normalize_username,
    serialize_user,
    validate_role,
)
from app.utils.config import get_settings

router = APIRouter(prefix="/admin", tags=["platform"])


class TextDocumentRequest(BaseModel):
    title: str = Field(..., min_length=3, max_length=180)
    content: str = Field(..., min_length=10)


class UrlDocumentRequest(BaseModel):
    title: str = Field(..., min_length=3, max_length=180)
    url: HttpUrl


class DocumentUpdateRequest(BaseModel):
    title: str | None = Field(default=None, min_length=3, max_length=180)
    content: str | None = Field(default=None, min_length=10)
    source_url: HttpUrl | None = None


class UserCreateRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=256)
    role: str = Field(default="trainer")
    full_name: str | None = Field(default="", max_length=120)


class RoleUpdateRequest(BaseModel):
    role: str


class AISettingsRequest(BaseModel):
    provider: str = Field(..., min_length=3, max_length=40)
    model: str = Field(..., min_length=1, max_length=120)
    temperature: float = Field(..., ge=0, le=2)
    max_tokens: int = Field(..., ge=64, le=4096)


class ApiKeysRequest(BaseModel):
    openai_api_key: str | None = None
    groq_api_key: str | None = None
    openrouter_api_key: str | None = None
    nvidia_api_key: str | None = None


class SystemSettingsRequest(BaseModel):
    official_mode_label: str | None = Field(default=None, max_length=120)
    admin_indicator_label: str | None = Field(default=None, max_length=120)
    knowledge_empty_message: str | None = Field(default=None, max_length=240)


async def _get_recent_activity(limit: int = 30) -> list[dict]:
    database = get_database()
    logs = await database[ACTIVITY_LOGS_COLLECTION].find().sort("timestamp", -1).limit(limit).to_list(length=limit)
    user_ids = [log["user_id"] for log in logs if log.get("user_id")]
    user_lookup = {}

    if user_ids:
        users = await database[USERS_COLLECTION].find({"_id": {"$in": user_ids}}).to_list(length=None)
        user_lookup = {user["_id"]: user for user in users}

    return [serialize_activity(log, user_lookup) for log in logs]


@router.get("/knowledge-base/status")
async def knowledge_base_status(current_user: dict = Depends(require_roles("admin", "trainer"))):
    """Return knowledge-base readiness and document counts."""

    return await get_knowledge_base_dashboard()


@router.get("/documents")
async def get_documents(current_user: dict = Depends(require_roles("admin", "trainer"))):
    """Return the document catalog."""

    return {"documents": await list_documents()}


@router.post("/documents/text", status_code=status.HTTP_202_ACCEPTED)
async def add_text_document(request: TextDocumentRequest, current_user: dict = Depends(require_roles("admin", "trainer"))):
    """Create a text-based knowledge source."""

    document = await create_document_record(
        current_user,
        title=request.title,
        source_type="text",
        content=request.content,
    )
    schedule_background_job(process_document(str(document["_id"]), current_user["_id"]))
    return {"success": True, "document_id": str(document["_id"]), "status": "pending"}


@router.post("/documents/url", status_code=status.HTTP_202_ACCEPTED)
async def add_url_document(request: UrlDocumentRequest, current_user: dict = Depends(require_roles("admin", "trainer"))):
    """Create a URL-based knowledge source."""

    document = await create_document_record(
        current_user,
        title=request.title,
        source_type="url",
        source_url=str(request.url),
    )
    schedule_background_job(process_document(str(document["_id"]), current_user["_id"]))
    return {"success": True, "document_id": str(document["_id"]), "status": "pending"}


@router.post("/documents/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(require_roles("admin", "trainer")),
):
    """Upload a PDF, DOCX, or TXT knowledge source."""

    extension = Path(file.filename or "").suffix.lower().lstrip(".")
    if extension not in {"pdf", "docx", "txt"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF, DOCX, and TXT files are supported")

    document = await create_document_record(
        current_user,
        title=Path(file.filename).stem,
        source_type=extension,
        file_name=file.filename,
    )
    contents = await file.read()
    original_name, file_path = await save_uploaded_file(file.filename, contents, str(document["_id"]))
    database = get_database()
    await database["documents"].update_one(
        {"_id": document["_id"]},
        {"$set": {"file_name": original_name, "file_path": file_path, "updated_at": datetime.now(timezone.utc)}},
    )
    schedule_background_job(process_document(str(document["_id"]), current_user["_id"]))
    return {"success": True, "document_id": str(document["_id"]), "status": "pending"}


@router.put("/documents/{document_id}")
async def edit_document(
    document_id: str,
    request: DocumentUpdateRequest,
    current_user: dict = Depends(require_roles("admin")),
):
    """Update a document title or editable source content."""

    try:
        await update_document(document_id, request.model_dump(exclude_none=False), current_user)
        return {"success": True}
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str, current_user: dict = Depends(require_roles("admin"))):
    """Delete a knowledge source and rebuild the index."""

    try:
        await deactivate_document(document_id, current_user)
        return {"success": True}
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/documents/retrain", status_code=status.HTTP_202_ACCEPTED)
async def retrain_knowledge_base(current_user: dict = Depends(require_roles("admin", "trainer"))):
    """Trigger a full knowledge-base rebuild."""

    database = get_database()
    await log_activity(database, current_user["_id"], "knowledge_base.retrain_requested", {})
    schedule_background_job(rebuild_index_from_documents(current_user["_id"]))
    return {"success": True, "status": "processing"}


@router.get("/users")
async def list_users(current_user: dict = Depends(require_roles("admin"))):
    """Return all users with recent activity metadata."""

    database = get_database()
    activity = await database[ACTIVITY_LOGS_COLLECTION].aggregate(
        [
            {"$sort": {"timestamp": -1}},
            {
                "$group": {
                    "_id": "$user_id",
                    "last_action": {"$first": "$action"},
                    "last_activity_at": {"$first": "$timestamp"},
                }
            },
        ]
    ).to_list(length=None)
    activity_map = {item["_id"]: item for item in activity}

    users = await database[USERS_COLLECTION].find().sort("created_at", 1).to_list(length=None)
    payload = []
    for user in users:
        serialized = serialize_user(user)
        recent = activity_map.get(user["_id"])
        serialized["last_action"] = recent.get("last_action") if recent else None
        serialized["last_activity_at"] = recent.get("last_activity_at").isoformat() if recent and recent.get("last_activity_at") else None
        payload.append(serialized)

    return {"users": payload}


@router.post("/users", status_code=status.HTTP_201_CREATED)
async def create_user(request: UserCreateRequest, current_user: dict = Depends(require_roles("admin"))):
    """Create a new platform user."""

    database = get_database()
    role = validate_role(request.role)
    username = normalize_username(request.username)
    email = normalize_email(request.email)
    if not username:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Username must contain letters or numbers.")

    if await database[USERS_COLLECTION].find_one({"$or": [{"username": username}, {"email": email}]}):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="A user with that email or username already exists")

    user = build_user_document(
        username=username,
        email=email,
        password_hash=hash_password(request.password),
        role=role,
        full_name=request.full_name or "",
        must_change_password=True,
        now=datetime.now(timezone.utc),
    )
    result = await database[USERS_COLLECTION].insert_one(user)
    user["_id"] = result.inserted_id
    await log_activity(database, current_user["_id"], "user.created", {"user_id": str(user["_id"]), "role": role})
    return {"user": serialize_user(user)}


@router.put("/users/{user_id}/role")
async def change_user_role(
    user_id: str,
    request: RoleUpdateRequest,
    current_user: dict = Depends(require_roles("admin")),
):
    """Update a user's RBAC role."""

    database = get_database()
    target = await get_user_by_id(database, user_id)
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    new_role = validate_role(request.role)
    if target["role"] == ADMIN_ROLE and new_role != ADMIN_ROLE:
        admin_count = await database[USERS_COLLECTION].count_documents({"role": ADMIN_ROLE})
        if admin_count <= 1:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one admin must remain")

    await database[USERS_COLLECTION].update_one(
        {"_id": target["_id"]},
        {"$set": {"role": new_role, "updated_at": datetime.now(timezone.utc)}},
    )
    await log_activity(database, current_user["_id"], "user.role_updated", {"user_id": user_id, "role": new_role})
    updated = await get_user_by_id(database, user_id)
    return {"user": serialize_user(updated)}


@router.delete("/users/{user_id}")
async def remove_user(user_id: str, current_user: dict = Depends(require_roles("admin"))):
    """Delete a user while protecting the final admin account."""

    if str(current_user["_id"]) == user_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="You cannot delete your own account")

    database = get_database()
    target = await get_user_by_id(database, user_id)
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if target["role"] == ADMIN_ROLE:
        admin_count = await database[USERS_COLLECTION].count_documents({"role": ADMIN_ROLE})
        if admin_count <= 1:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The final admin account cannot be deleted")

    await database[USERS_COLLECTION].delete_one({"_id": target["_id"]})
    await log_activity(database, current_user["_id"], "user.deleted", {"user_id": user_id})
    return {"success": True}


@router.get("/activity")
async def get_activity(current_user: dict = Depends(require_roles("admin"))):
    """Return recent platform activity."""

    return {"activity": await _get_recent_activity()}


@router.get("/settings/ai")
async def read_ai_settings(current_user: dict = Depends(require_roles("admin"))):
    """Return current AI runtime settings."""

    settings = get_settings()
    database = get_database()
    ai_settings = await get_ai_settings(database, settings)
    return {
        "ai_settings": ai_settings,
        "available_providers": ["openai", "groq", "openrouter", "nvidia"],
    }


@router.put("/settings/ai")
async def update_ai_settings(request: AISettingsRequest, current_user: dict = Depends(require_roles("admin"))):
    """Update model provider/runtime parameters."""

    database = get_database()
    payload = DEFAULT_AI_SETTINGS.copy()
    payload.update(request.model_dump())
    await upsert_setting_value(database, "ai_settings", payload)
    await log_activity(database, current_user["_id"], "settings.ai_updated", payload)
    return {"ai_settings": payload}


@router.get("/settings/api-keys")
async def read_api_keys(current_user: dict = Depends(require_roles("admin"))):
    """Return masked provider key configuration."""

    settings = get_settings()
    database = get_database()
    return {"api_keys": await get_api_key_settings(database, settings)}


@router.put("/settings/api-keys")
async def update_api_keys(request: ApiKeysRequest, current_user: dict = Depends(require_roles("admin"))):
    """Update stored provider keys and endpoints."""

    settings = get_settings()
    database = get_database()
    await save_api_key_settings(database, settings, request.model_dump())
    await log_activity(database, current_user["_id"], "settings.api_keys_updated", {"fields": list(request.model_dump(exclude_none=True).keys())})
    return {"api_keys": await get_api_key_settings(database, settings)}


@router.get("/settings/system")
async def read_system_settings(current_user: dict = Depends(require_roles("admin"))):
    """Return admin-controlled UI/system settings."""

    database = get_database()
    return {"system_settings": await get_system_settings(database)}


@router.put("/settings/system")
async def update_system_settings(request: SystemSettingsRequest, current_user: dict = Depends(require_roles("admin"))):
    """Update admin-controlled system settings."""

    database = get_database()
    current = await get_system_settings(database)
    updates = {key: value for key, value in request.model_dump().items() if value is not None}
    current.update(updates)
    await upsert_setting_value(database, "system_settings", current)
    await log_activity(database, current_user["_id"], "settings.system_updated", updates)
    return {"system_settings": current}
