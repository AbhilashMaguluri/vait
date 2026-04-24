"""MongoDB connection and collection helpers."""

from __future__ import annotations

import logging

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import OperationFailure

from app.core.constants import (
    ACTIVITY_LOGS_COLLECTION,
    CONVERSATIONS_COLLECTION,
    DOCUMENTS_COLLECTION,
    SETTINGS_COLLECTION,
    USERS_COLLECTION,
)
from app.utils.config import Settings

logger = logging.getLogger("vait.db")

_mongo_client: AsyncIOMotorClient | None = None
_database: AsyncIOMotorDatabase | None = None
_last_connection_error: str | None = None


async def connect_to_mongo(settings: Settings) -> AsyncIOMotorDatabase:
    """Initialize the shared Mongo client and database."""

    global _mongo_client, _database, _last_connection_error

    if _database is not None:
        return _database

    _mongo_client = AsyncIOMotorClient(
        settings.mongodb_uri,
        serverSelectionTimeoutMS=settings.mongodb_server_selection_timeout_ms,
    )
    try:
        await _mongo_client.admin.command("ping")
    except OperationFailure as exc:
        _mongo_client.close()
        _mongo_client = None
        _last_connection_error = "MongoDB authentication failed. Check MONGODB_URI username/password."
        logger.error(_last_connection_error)
        raise RuntimeError(_last_connection_error) from exc
    except Exception as exc:
        if _mongo_client is not None:
            _mongo_client.close()
        _mongo_client = None
        _last_connection_error = f"MongoDB connection failed: {exc}"
        logger.error(_last_connection_error)
        raise RuntimeError(_last_connection_error) from exc

    _database = _mongo_client.get_default_database(default="vait")
    logger.info("Connected to MongoDB database '%s'", _database.name)
    await create_indexes(_database)
    _last_connection_error = None
    return _database


def get_database() -> AsyncIOMotorDatabase:
    """Return the initialized Mongo database."""

    if _database is None:
        raise RuntimeError(_last_connection_error or "MongoDB has not been initialized")
    return _database


async def close_mongo_connection() -> None:
    """Close the shared Mongo connection."""

    global _mongo_client, _database

    if _mongo_client is not None:
        _mongo_client.close()
    _mongo_client = None
    _database = None


async def create_indexes(database: AsyncIOMotorDatabase) -> None:
    """Create the indexes required for platform operations."""

    await database[USERS_COLLECTION].create_index("email", unique=True)
    await database[USERS_COLLECTION].create_index("username", unique=True)
    await database[USERS_COLLECTION].create_index("role")

    await database[CONVERSATIONS_COLLECTION].create_index([("user_id", 1), ("updated_at", -1)])
    await database[CONVERSATIONS_COLLECTION].create_index(
        [("user_id", 1), ("client_conversation_id", 1)],
        unique=True,
    )

    await database[DOCUMENTS_COLLECTION].create_index("status")
    await database[DOCUMENTS_COLLECTION].create_index("uploaded_by")
    await database[DOCUMENTS_COLLECTION].create_index("created_at")

    await database[SETTINGS_COLLECTION].create_index("key", unique=True)

    await database[ACTIVITY_LOGS_COLLECTION].create_index("user_id")
    await database[ACTIVITY_LOGS_COLLECTION].create_index("timestamp")
