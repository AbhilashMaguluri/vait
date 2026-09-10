"""MongoDB connection and collection helpers."""

from __future__ import annotations

import asyncio
import logging

import certifi

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import (
    ConfigurationError,
    ConnectionFailure,
    OperationFailure,
    ServerSelectionTimeoutError,
)

from app.core.constants import (
    ACTIVITY_LOGS_COLLECTION,
    CONVERSATIONS_COLLECTION,
    DOCUMENTS_COLLECTION,
    SETTINGS_COLLECTION,
    USERS_COLLECTION,
)
from app.utils.config import Settings, mask_connection_uri

logger = logging.getLogger("vait.db")

_mongo_client: AsyncIOMotorClient | None = None
_database: AsyncIOMotorDatabase | None = None
_last_connection_error: str | None = None
_connect_lock: asyncio.Lock | None = None


def _get_connect_lock() -> asyncio.Lock:
    global _connect_lock
    if _connect_lock is None:
        _connect_lock = asyncio.Lock()
    return _connect_lock


def _cleanup_failed_client(client: AsyncIOMotorClient | None = None) -> None:
    global _mongo_client, _database
    if client is not None:
        try:
            client.close()
        except Exception:
            pass
    if _mongo_client is not None:
        try:
            _mongo_client.close()
        except Exception:
            pass
    _mongo_client = None
    _database = None


async def connect_to_mongo(settings: Settings | None = None) -> AsyncIOMotorDatabase:
    """Initialize the shared Mongo client and database."""

    global _mongo_client, _database, _last_connection_error

    if _database is not None:
        return _database

    lock = _get_connect_lock()
    async with lock:
        if _database is not None:
            return _database

        if settings is None:
            from app.utils.config import get_settings
            settings = get_settings()

        masked_uri = mask_connection_uri(settings.mongodb_uri)
        logger.info("Initializing MongoDB connection to %s", masked_uri)

        client: AsyncIOMotorClient | None = None
        try:
            client = AsyncIOMotorClient(
                settings.mongodb_uri,
                serverSelectionTimeoutMS=settings.mongodb_server_selection_timeout_ms,
                tlsCAFile=certifi.where(),
            )
            # Verify server responsiveness by pinging admin database
            await client.admin.command("ping")

            database = client.get_default_database(default="vait")
            logger.info("Connected to MongoDB database '%s'", database.name)

            await create_indexes(database)

            _mongo_client = client
            _database = database
            _last_connection_error = None
            return _database
        except ConfigurationError as exc:
            _cleanup_failed_client(client)
            _last_connection_error = (
                f"MongoDB configuration or DNS resolution failed: {exc}. "
                f"Please check your MONGODB_URI (or MONGO_URI) connection string and verify the cluster hostname."
            )
            logger.error("MongoDB initialization failed: %s", _last_connection_error, exc_info=True)
            raise RuntimeError(_last_connection_error) from exc
        except OperationFailure as exc:
            _cleanup_failed_client(client)
            _last_connection_error = (
                f"MongoDB authentication failed: {exc}. "
                f"Check MONGODB_URI (or MONGO_URI) username, password, and authentication database."
            )
            logger.error("MongoDB initialization failed: %s", _last_connection_error, exc_info=True)
            raise RuntimeError(_last_connection_error) from exc
        except ServerSelectionTimeoutError as exc:
            _cleanup_failed_client(client)
            _last_connection_error = (
                f"MongoDB server selection timed out ({settings.mongodb_server_selection_timeout_ms}ms): {exc}. "
                f"Verify that your MongoDB cluster is active and Network Access allows Render (IP access list 0.0.0.0/0)."
            )
            logger.error("MongoDB initialization failed: %s", _last_connection_error, exc_info=True)
            raise RuntimeError(_last_connection_error) from exc
        except ConnectionFailure as exc:
            _cleanup_failed_client(client)
            _last_connection_error = (
                f"MongoDB connection failure: {exc}. "
                f"Verify cluster connectivity and firewall rules."
            )
            logger.error("MongoDB initialization failed: %s", _last_connection_error, exc_info=True)
            raise RuntimeError(_last_connection_error) from exc
        except Exception as exc:
            _cleanup_failed_client(client)
            _last_connection_error = f"MongoDB connection failed: {exc}"
            logger.error("MongoDB initialization failed: %s", _last_connection_error, exc_info=True)
            raise RuntimeError(_last_connection_error) from exc


async def get_database_async(settings: Settings | None = None) -> AsyncIOMotorDatabase:
    """Return the initialized Mongo database, or attempt connection if not yet connected."""

    global _database
    if _database is not None:
        return _database
    return await connect_to_mongo(settings)


def get_database() -> AsyncIOMotorDatabase:
    """Return the initialized Mongo database."""

    if _database is None:
        raise RuntimeError(
            _last_connection_error
            or "MongoDB has not been initialized. Check MONGODB_URI/MONGO_URI configuration."
        )
    return _database


def is_db_connected() -> bool:
    """Check if the Mongo database is currently initialized."""
    return _database is not None


def get_last_connection_error() -> str | None:
    """Return the last connection error message if any."""
    return _last_connection_error


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
