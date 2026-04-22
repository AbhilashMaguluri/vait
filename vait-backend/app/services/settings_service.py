"""Helpers for persisted platform settings."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.core.constants import (
    AI_SETTINGS_KEY,
    API_KEYS_SETTINGS_KEY,
    DEFAULT_AI_SETTINGS,
    DEFAULT_KNOWLEDGE_BASE_STATE,
    DEFAULT_PROVIDER_CONFIG,
    DEFAULT_SYSTEM_SETTINGS,
    KNOWLEDGE_BASE_SETTINGS_KEY,
    SETTINGS_COLLECTION,
    SYSTEM_SETTINGS_KEY,
)
from app.services.security_service import (
    decrypt_secret,
    encrypt_secret,
    looks_like_placeholder,
    mask_secret,
)
from app.utils.config import Settings


async def _ensure_setting(database, key: str, default_value: dict[str, Any]) -> dict[str, Any]:
    existing = await database[SETTINGS_COLLECTION].find_one({"key": key})
    if existing:
        return existing

    now = datetime.now(timezone.utc)
    doc = {
        "key": key,
        "value": default_value,
        "updated_at": now,
    }
    await database[SETTINGS_COLLECTION].insert_one(doc)
    return doc


async def ensure_default_settings(database) -> None:
    """Create the default settings documents if they do not exist."""

    await _ensure_setting(database, AI_SETTINGS_KEY, DEFAULT_AI_SETTINGS.copy())
    await _ensure_setting(database, API_KEYS_SETTINGS_KEY, DEFAULT_PROVIDER_CONFIG.copy())
    await _ensure_setting(database, SYSTEM_SETTINGS_KEY, DEFAULT_SYSTEM_SETTINGS.copy())
    await _ensure_setting(database, KNOWLEDGE_BASE_SETTINGS_KEY, DEFAULT_KNOWLEDGE_BASE_STATE.copy())


async def get_setting_value(database, key: str, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fetch a settings payload by key."""

    doc = await database[SETTINGS_COLLECTION].find_one({"key": key})
    if doc:
        return doc.get("value", {})
    return (fallback or {}).copy()


async def upsert_setting_value(database, key: str, value: dict[str, Any]) -> None:
    """Update a settings payload."""

    await database[SETTINGS_COLLECTION].update_one(
        {"key": key},
        {"$set": {"value": value, "updated_at": datetime.now(timezone.utc)}},
        upsert=True,
    )


async def get_ai_settings(database, settings: Settings) -> dict[str, Any]:
    """Return persisted AI configuration with defaults applied."""

    stored = await get_setting_value(database, AI_SETTINGS_KEY, DEFAULT_AI_SETTINGS)
    return {
        "provider": stored.get("provider", settings.default_llm_provider),
        "model": stored.get("model", settings.default_llm_model),
        "temperature": stored.get("temperature", settings.default_temperature),
        "max_tokens": stored.get("max_tokens", settings.default_max_tokens),
    }


async def get_system_settings(database) -> dict[str, Any]:
    """Return non-sensitive system settings."""

    stored = await get_setting_value(database, SYSTEM_SETTINGS_KEY, DEFAULT_SYSTEM_SETTINGS)
    merged = DEFAULT_SYSTEM_SETTINGS.copy()
    merged.update(stored)
    return merged


async def get_knowledge_base_state(database) -> dict[str, Any]:
    """Return knowledge base runtime state."""

    stored = await get_setting_value(database, KNOWLEDGE_BASE_SETTINGS_KEY, DEFAULT_KNOWLEDGE_BASE_STATE)
    merged = DEFAULT_KNOWLEDGE_BASE_STATE.copy()
    merged.update(stored)
    return merged


async def update_knowledge_base_state(database, *, status: str, last_error: str | None = None, trained: bool = False) -> None:
    """Persist the current knowledge base state."""

    current = await get_knowledge_base_state(database)
    current["status"] = status
    current["last_error"] = last_error
    if trained:
        current["last_trained_at"] = datetime.now(timezone.utc).isoformat()
    await upsert_setting_value(database, KNOWLEDGE_BASE_SETTINGS_KEY, current)


async def get_api_key_settings(database, settings: Settings) -> dict[str, Any]:
    """Return masked provider configuration for UI use."""

    stored = await get_setting_value(database, API_KEYS_SETTINGS_KEY, DEFAULT_PROVIDER_CONFIG)
    openai_key = stored.get("openai_api_key") or (
        "" if looks_like_placeholder(settings.openai_api_key) else settings.openai_api_key
    )
    groq_key = stored.get("groq_api_key") or (
        "" if looks_like_placeholder(settings.groq_api_key) else settings.groq_api_key
    )
    openrouter_key = stored.get("openrouter_api_key") or (
        "" if looks_like_placeholder(settings.openrouter_api_key) else settings.openrouter_api_key
    )
    nvidia_key = stored.get("nvidia_api_key") or (
        "" if looks_like_placeholder(settings.nvidia_api_key) else settings.nvidia_api_key
    )

    return {
        "openai_api_key": mask_secret(decrypt_secret(openai_key, settings) if openai_key else ""),
        "groq_api_key": mask_secret(decrypt_secret(groq_key, settings) if groq_key else ""),
        "openrouter_api_key": mask_secret(decrypt_secret(openrouter_key, settings) if openrouter_key else ""),
        "nvidia_api_key": mask_secret(decrypt_secret(nvidia_key, settings) if nvidia_key else ""),
    }


async def get_runtime_provider_config(database, settings: Settings) -> dict[str, Any]:
    """Return decrypted provider configuration for server-side use."""

    stored = await get_setting_value(database, API_KEYS_SETTINGS_KEY, DEFAULT_PROVIDER_CONFIG)

    def _resolve(raw_encrypted: str, env_value: str) -> str:
        if raw_encrypted:
            return decrypt_secret(raw_encrypted, settings)
        if looks_like_placeholder(env_value):
            return ""
        return env_value

    return {
        "openai_api_key": _resolve(stored.get("openai_api_key", ""), settings.openai_api_key),
        "groq_api_key": _resolve(stored.get("groq_api_key", ""), settings.groq_api_key),
        "openrouter_api_key": _resolve(stored.get("openrouter_api_key", ""), settings.openrouter_api_key),
        "nvidia_api_key": _resolve(stored.get("nvidia_api_key", ""), settings.nvidia_api_key),
    }


async def save_api_key_settings(database, settings: Settings, payload: dict[str, Any]) -> None:
    """Store provider configuration with encrypted secrets."""

    current = await get_setting_value(database, API_KEYS_SETTINGS_KEY, DEFAULT_PROVIDER_CONFIG)

    for field in ("openai_api_key", "groq_api_key", "openrouter_api_key", "nvidia_api_key"):
        if field in payload and payload[field] is not None:
            current[field] = encrypt_secret(payload[field], settings) if payload[field] else ""

    await upsert_setting_value(database, API_KEYS_SETTINGS_KEY, current)
