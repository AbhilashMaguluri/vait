"""MongoDB connection and collection helpers."""

from __future__ import annotations

import logging

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.constants import (
    ACTIVITY_LOGS_COLLECTION,
    DOCUMENTS_COLLECTION,
    SETTINGS_COLLECTION,
    USERS_COLLECTION,
)
from app.utils.config import Settings

logger = logging.getLogger("vait.db")

_mongo_client: AsyncIOMotorClient | None = None
_database: AsyncIOMotorDatabase | None = None


async def connect_to_mongo(settings: Settings) -> AsyncIOMotorDatabase:
    """Initialize the shared Mongo client and database."""

    global _mongo_client, _database

    if _database is not None:
        return _database

    _mongo_client = AsyncIOMotorClient(settings.mongodb_uri)
    await _mongo_client.admin.command("ping")
    _database = _mongo_client.get_default_database(default="vait")
    logger.info("Connected to MongoDB database '%s'", _database.name)
    await create_indexes(_database)
    return _database


def get_database() -> AsyncIOMotorDatabase:
    """Return the initialized Mongo database."""

    if _database is None:
        raise RuntimeError("MongoDB has not been initialized")
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

    await database[DOCUMENTS_COLLECTION].create_index("status")
    await database[DOCUMENTS_COLLECTION].create_index("uploaded_by")
    await database[DOCUMENTS_COLLECTION].create_index("created_at")

    await database[SETTINGS_COLLECTION].create_index("key", unique=True)

    await database[ACTIVITY_LOGS_COLLECTION].create_index("user_id")
    await database[ACTIVITY_LOGS_COLLECTION].create_index("timestamp")
