"""Document catalog management and knowledge-base processing."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import PyPDF2
import docx
import faiss
import requests
from bson import ObjectId
from bs4 import BeautifulSoup

from app.core.constants import DOCUMENTS_COLLECTION, EDITABLE_DOCUMENT_TYPES
from app.db import get_database
from app.services.activity_service import log_activity
from app.services.settings_service import get_knowledge_base_state, get_system_settings, update_knowledge_base_state
from app.utils.config import UPLOADS_DIR

logger = logging.getLogger("vait.documents")

_processing_lock = asyncio.Lock()
_background_tasks: set[asyncio.Task] = set()


def _get_rag_service():
    from app.main import get_rag_service

    return get_rag_service()


def _current_academic_year() -> str:
    now = datetime.now(timezone.utc)
    year = now.year
    if now.month >= 8:
        return f"{year}-{str(year + 1)[2:]}"
    return f"{year - 1}-{str(year)[2:]}"


def _clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\t", " ")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _infer_authority_level(document: dict[str, Any]) -> str:
    source_type = document.get("source_type", "")
    source_url = (document.get("source_url") or "").lower()

    if source_type == "pdf":
        return "high"
    if source_type == "url" and any(domain in source_url for domain in ("vvitguntur.com", "vvitu.ac.in")):
        return "high"
    return "medium"


def _build_rag_metadata(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_type": "knowledge_source",
        "academic_year": _current_academic_year(),
        "department": document.get("department", "General"),
        "authority_level": _infer_authority_level(document),
        "source_type": document.get("source_type", "text"),
        "source_tier": "primary_official" if _infer_authority_level(document) == "high" else "related_web",
        "url": document.get("source_url", ""),
        "document_id": str(document["_id"]),
    }


def _extract_text_from_pdf(file_path: Path) -> str:
    text_parts = []
    with file_path.open("rb") as file_handle:
        reader = PyPDF2.PdfReader(file_handle)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(page_text.strip())
    return _clean_text("\n\n".join(text_parts))


def _extract_text_from_docx(file_path: Path) -> str:
    document = docx.Document(file_path)
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    return _clean_text("\n\n".join(paragraphs))


def _extract_text_from_txt(file_path: Path) -> str:
    return _clean_text(file_path.read_text(encoding="utf-8", errors="replace"))


def _extract_text_from_url(url: str) -> str:
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text("\n")
    return _clean_text(text)


def _resolve_document_text(document: dict[str, Any]) -> str:
    source_type = document.get("source_type")
    if source_type == "text":
        return _clean_text(document.get("content", ""))
    if source_type == "url":
        return _extract_text_from_url(document["source_url"])

    file_path = Path(document["file_path"])
    if source_type == "pdf":
        return _extract_text_from_pdf(file_path)
    if source_type == "docx":
        return _extract_text_from_docx(file_path)
    if source_type == "txt":
        return _extract_text_from_txt(file_path)
    raise ValueError(f"Unsupported source_type '{source_type}'")


def _serialize_document(document: dict[str, Any], uploaded_by: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": str(document["_id"]),
        "title": document["title"],
        "source_type": document["source_type"],
        "status": document["status"],
        "uploaded_by": str(document["uploaded_by"]) if document.get("uploaded_by") else None,
        "uploaded_by_user": {
            "id": str(uploaded_by["_id"]),
            "username": uploaded_by.get("username"),
            "email": uploaded_by.get("email"),
            "role": uploaded_by.get("role"),
        } if uploaded_by else None,
        "source_url": document.get("source_url"),
        "file_name": document.get("file_name"),
        "chunk_count": document.get("chunk_count", 0),
        "error_message": document.get("error_message"),
        "created_at": document["created_at"].isoformat(),
        "updated_at": document["updated_at"].isoformat(),
        "processed_at": document.get("processed_at").isoformat() if document.get("processed_at") else None,
    }


def schedule_background_job(coro) -> None:
    """Schedule a background coroutine and keep a reference until completion."""

    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def create_document_record(
    user: dict[str, Any],
    *,
    title: str,
    source_type: str,
    content: str | None = None,
    source_url: str | None = None,
    file_name: str | None = None,
    file_path: str | None = None,
) -> dict[str, Any]:
    """Persist a pending document catalog entry."""

    database = get_database()
    now = datetime.now(timezone.utc)
    document = {
        "title": title.strip(),
        "source_type": source_type,
        "uploaded_by": user["_id"],
        "status": "pending",
        "content": content or "",
        "source_url": source_url or "",
        "file_name": file_name or "",
        "file_path": file_path or "",
        "chunk_count": 0,
        "error_message": None,
        "created_at": now,
        "updated_at": now,
        "processed_at": None,
        "is_active": True,
    }
    result = await database[DOCUMENTS_COLLECTION].insert_one(document)
    document["_id"] = result.inserted_id
    await update_knowledge_base_state(database, status="processing")
    await log_activity(database, user["_id"], "document.created", {"document_id": str(document["_id"]), "title": title})
    return document


async def process_document(document_id: str, actor_id: str | ObjectId | None = None) -> None:
    """Extract, chunk, embed, and index a single document."""

    async with _processing_lock:
        database = get_database()
        rag_service = _get_rag_service()
        if rag_service is None:
            raise RuntimeError("RAG service is not initialized")

        document = await database[DOCUMENTS_COLLECTION].find_one({"_id": ObjectId(document_id), "is_active": True})
        if not document:
            return

        await database[DOCUMENTS_COLLECTION].update_one(
            {"_id": document["_id"]},
            {"$set": {"status": "processing", "updated_at": datetime.now(timezone.utc), "error_message": None}},
        )

        try:
            extracted_text = _resolve_document_text(document)
            if not extracted_text:
                raise ValueError("No text could be extracted from the document")

            chunks_added = await rag_service.add_documents(
                [{"content": extracted_text, "name": document["title"]}],
                [_build_rag_metadata(document)],
            )

            await database[DOCUMENTS_COLLECTION].update_one(
                {"_id": document["_id"]},
                {
                    "$set": {
                        "status": "completed",
                        "chunk_count": chunks_added,
                        "extracted_text": extracted_text,
                        "processed_at": datetime.now(timezone.utc),
                        "updated_at": datetime.now(timezone.utc),
                    }
                },
            )
            await update_knowledge_base_state(database, status="ready", trained=True)
            await log_activity(
                database,
                actor_id or document["uploaded_by"],
                "document.processed",
                {"document_id": str(document["_id"]), "title": document["title"], "chunks_added": chunks_added},
            )
        except Exception as exc:
            logger.error("Document processing failed for %s: %s", document_id, exc, exc_info=True)
            await database[DOCUMENTS_COLLECTION].update_one(
                {"_id": document["_id"]},
                {
                    "$set": {
                        "status": "failed",
                        "error_message": str(exc),
                        "updated_at": datetime.now(timezone.utc),
                    }
                },
            )
            await update_knowledge_base_state(database, status="empty", last_error=str(exc))
            await log_activity(
                database,
                actor_id or document["uploaded_by"],
                "document.failed",
                {"document_id": str(document["_id"]), "title": document["title"], "error": str(exc)},
            )


async def rebuild_index_from_documents(actor_id: str | ObjectId | None = None) -> None:
    """Rebuild the FAISS index from all active documents."""

    async with _processing_lock:
        database = get_database()
        rag_service = _get_rag_service()
        if rag_service is None:
            raise RuntimeError("RAG service is not initialized")

        await update_knowledge_base_state(database, status="processing")

        documents_cursor = database[DOCUMENTS_COLLECTION].find({"is_active": True}).sort("created_at", 1)
        documents = await documents_cursor.to_list(length=None)

        rag_service.index = faiss.IndexFlatIP(rag_service.embedding_service.embedding_dimension)
        rag_service.chunks = []

        if not documents:
            await rag_service._save_index()
            await update_knowledge_base_state(database, status="empty", trained=True)
            return

        for document in documents:
            await database[DOCUMENTS_COLLECTION].update_one(
                {"_id": document["_id"]},
                {"$set": {"status": "processing", "updated_at": datetime.now(timezone.utc), "error_message": None}},
            )

        any_completed = False
        for document in documents:
            try:
                extracted_text = document.get("extracted_text") or _resolve_document_text(document)
                if not extracted_text:
                    raise ValueError("No text could be extracted from the document")

                chunks_added = await rag_service.add_documents(
                    [{"content": extracted_text, "name": document["title"]}],
                    [_build_rag_metadata(document)],
                )
                any_completed = True

                await database[DOCUMENTS_COLLECTION].update_one(
                    {"_id": document["_id"]},
                    {
                        "$set": {
                            "status": "completed",
                            "chunk_count": chunks_added,
                            "extracted_text": extracted_text,
                            "processed_at": datetime.now(timezone.utc),
                            "updated_at": datetime.now(timezone.utc),
                        }
                    },
                )
            except Exception as exc:
                logger.error("Rebuild failed for %s: %s", document["_id"], exc, exc_info=True)
                await database[DOCUMENTS_COLLECTION].update_one(
                    {"_id": document["_id"]},
                    {"$set": {"status": "failed", "error_message": str(exc), "updated_at": datetime.now(timezone.utc)}},
                )

        await update_knowledge_base_state(database, status="ready" if any_completed else "empty", trained=True)
        await log_activity(
            database,
            actor_id,
            "knowledge_base.rebuilt",
            {"documents_processed": len(documents), "ready": any_completed},
        )


async def list_documents() -> list[dict[str, Any]]:
    """Return the active document catalog with uploader information."""

    database = get_database()
    documents = await database[DOCUMENTS_COLLECTION].find({"is_active": True}).sort("created_at", -1).to_list(length=None)
    user_ids = {document["uploaded_by"] for document in documents if document.get("uploaded_by")}

    users = {}
    if user_ids:
        cursor = get_database()["users"].find({"_id": {"$in": list(user_ids)}})
        async for user in cursor:
            users[user["_id"]] = user

    return [_serialize_document(document, users.get(document.get("uploaded_by"))) for document in documents]


async def update_document(document_id: str, payload: dict[str, Any], actor: dict[str, Any]) -> None:
    """Update a document record and schedule a rebuild when content changes."""

    database = get_database()
    document = await database[DOCUMENTS_COLLECTION].find_one({"_id": ObjectId(document_id), "is_active": True})
    if not document:
        raise ValueError("Document not found")

    updates = {"updated_at": datetime.now(timezone.utc)}
    reprocess_required = False

    if payload.get("title"):
        updates["title"] = payload["title"].strip()

    if document["source_type"] in EDITABLE_DOCUMENT_TYPES:
        if payload.get("content") is not None:
            updates["content"] = payload["content"]
            reprocess_required = True
        if payload.get("source_url") is not None:
            updates["source_url"] = payload["source_url"]
            reprocess_required = True

    if reprocess_required:
        updates["status"] = "pending"
        updates["error_message"] = None

    await database[DOCUMENTS_COLLECTION].update_one({"_id": document["_id"]}, {"$set": updates})
    await log_activity(database, actor["_id"], "document.updated", {"document_id": document_id, "reprocess": reprocess_required})

    if reprocess_required:
        await update_knowledge_base_state(database, status="processing")
        schedule_background_job(rebuild_index_from_documents(actor["_id"]))


async def deactivate_document(document_id: str, actor: dict[str, Any]) -> None:
    """Soft-delete a document and trigger a rebuild."""

    database = get_database()
    document = await database[DOCUMENTS_COLLECTION].find_one({"_id": ObjectId(document_id), "is_active": True})
    if not document:
        raise ValueError("Document not found")

    await database[DOCUMENTS_COLLECTION].update_one(
        {"_id": document["_id"]},
        {"$set": {"is_active": False, "status": "deleted", "updated_at": datetime.now(timezone.utc)}},
    )
    await update_knowledge_base_state(database, status="processing")
    await log_activity(database, actor["_id"], "document.deleted", {"document_id": document_id, "title": document["title"]})
    schedule_background_job(rebuild_index_from_documents(actor["_id"]))


async def get_knowledge_base_dashboard() -> dict[str, Any]:
    """Return a dashboard-friendly knowledge-base summary."""

    database = get_database()
    state = await get_knowledge_base_state(database)
    system_settings = await get_system_settings(database)

    total_documents = await database[DOCUMENTS_COLLECTION].count_documents({"is_active": True})
    completed_documents = await database[DOCUMENTS_COLLECTION].count_documents({"is_active": True, "status": "completed"})
    processing_documents = await database[DOCUMENTS_COLLECTION].count_documents({"is_active": True, "status": {"$in": ["pending", "processing"]}})
    failed_documents = await database[DOCUMENTS_COLLECTION].count_documents({"is_active": True, "status": "failed"})

    status = state.get("status", "empty")
    if processing_documents > 0:
        status = "processing"
    elif completed_documents > 0:
        status = "ready"
    else:
        status = "empty"

    rag_service = _get_rag_service()
    rag_stats = rag_service.get_stats() if rag_service is not None else {"total_vectors": 0}

    return {
        "status": status,
        "documents": {
            "total": total_documents,
            "completed": completed_documents,
            "processing": processing_documents,
            "failed": failed_documents,
        },
        "vectors": rag_stats.get("total_vectors", 0),
        "last_trained_at": state.get("last_trained_at"),
        "last_error": state.get("last_error"),
        "empty_message": system_settings["knowledge_empty_message"],
    }


async def save_uploaded_file(file_name: str, content: bytes, document_id: str) -> tuple[str, str]:
    """Persist an uploaded file under the uploads directory."""

    extension = Path(file_name).suffix.lower()
    safe_name = f"{document_id}{extension}"
    target = UPLOADS_DIR / safe_name
    target.write_bytes(content)
    return file_name, str(target)
