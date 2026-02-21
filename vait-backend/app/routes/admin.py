"""
VAIT Admin Routes
API endpoints for administrative operations.

Includes:
  - Single/bulk document ingestion
  - File upload ingestion
  - Website crawl + reindex
  - Social media ingestion
  - Knowledge base stats & clear
"""

import json
import logging

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Optional


def _get_rag_service():
    """Lazy import to avoid circular dependency with app.main."""
    from app.main import get_rag_service
    return _get_rag_service()


router = APIRouter(prefix="/admin", tags=["admin"])

logger = logging.getLogger("vait.routes.admin")


class DocumentMetadata(BaseModel):
    """Metadata model for document ingestion."""
    document_type: str = Field(..., description="Type of document")
    academic_year: str = Field(..., description="Academic year")
    department: str = Field(..., description="Responsible department")
    authority_level: str = Field(
        ...,
        description="Authority level: official, department, or informational"
    )


class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    content: str = Field(..., description="Document content")
    name: str = Field(..., description="Document name")
    metadata: DocumentMetadata


class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    success: bool
    chunks_added: int
    document_name: str


class BulkIngestRequest(BaseModel):
    """Request model for bulk document ingestion."""
    documents: List[IngestRequest]


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest):
    """
    Ingest a single document into the VAIT knowledge base.
    
    The document will be chunked, embedded, and indexed for retrieval.
    """
    rag_service = _get_rag_service()
    
    if rag_service is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service is not initialized"
        )
    
    try:
        documents = [{
            "content": request.content,
            "name": request.name
        }]
        
        metadata = [request.metadata.model_dump()]
        
        chunks_added = await rag_service.add_documents(documents, metadata)
        
        return IngestResponse(
            success=True,
            chunks_added=chunks_added,
            document_name=request.name
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest document: {str(e)}"
        )


@router.post("/ingest/bulk")
async def ingest_documents_bulk(request: BulkIngestRequest):
    """
    Ingest multiple documents into the VAIT knowledge base.
    """
    rag_service = _get_rag_service()
    
    if rag_service is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service is not initialized"
        )
    
    results = []
    total_chunks = 0
    
    for doc_request in request.documents:
        try:
            documents = [{
                "content": doc_request.content,
                "name": doc_request.name
            }]
            metadata = [doc_request.metadata.model_dump()]
            
            chunks = await rag_service.add_documents(documents, metadata)
            total_chunks += chunks
            
            results.append({
                "name": doc_request.name,
                "success": True,
                "chunks": chunks
            })
        except Exception as e:
            results.append({
                "name": doc_request.name,
                "success": False,
                "error": str(e)
            })
    
    return {
        "total_documents": len(request.documents),
        "total_chunks_added": total_chunks,
        "results": results
    }


@router.post("/ingest/file")
async def ingest_file(
    file: UploadFile = File(...),
    document_type: str = Form(...),
    academic_year: str = Form(...),
    department: str = Form(...),
    authority_level: str = Form(...)
):
    """
    Ingest a document file into the VAIT knowledge base.
    
    Supports .txt files.
    """
    rag_service = _get_rag_service()
    
    if rag_service is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service is not initialized"
        )
    
    if not file.filename.endswith('.txt'):
        raise HTTPException(
            status_code=400,
            detail="Only .txt files are supported"
        )
    
    try:
        content = await file.read()
        content = content.decode('utf-8')
        
        documents = [{
            "content": content,
            "name": file.filename.rsplit('.', 1)[0]
        }]
        
        metadata = [{
            "document_type": document_type,
            "academic_year": academic_year,
            "department": department,
            "authority_level": authority_level
        }]
        
        chunks_added = await rag_service.add_documents(documents, metadata)
        
        return {
            "success": True,
            "filename": file.filename,
            "chunks_added": chunks_added
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest file: {str(e)}"
        )


@router.get("/knowledge-base/stats")
async def get_knowledge_base_stats():
    """
    Get detailed statistics about the knowledge base.
    """
    rag_service = _get_rag_service()
    
    if rag_service is None:
        return {"status": "initializing", "ready": False}
    
    stats = rag_service.get_stats()
    stats["ready"] = True
    
    return stats


@router.delete("/knowledge-base/clear")
async def clear_knowledge_base():
    """
    Clear the entire knowledge base.
    
    WARNING: This action cannot be undone.
    """
    rag_service = _get_rag_service()
    
    if rag_service is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service is not initialized"
        )
    
    try:
        # Reinitialize with empty index
        import faiss
        dimension = rag_service.embedding_service.embedding_dimension
        rag_service.index = faiss.IndexFlatIP(dimension)
        rag_service.chunks = []
        
        # Save empty state
        await rag_service._save_index()
        
        return {"success": True, "message": "Knowledge base cleared"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear knowledge base: {str(e)}"
        )


# =====================================================================
# WEBSITE REINDEX (via API)
# =====================================================================

class WebsiteReindexRequest(BaseModel):
    """Optional overrides for website reindexing."""
    depth: Optional[int] = Field(None, description="Override crawl depth")
    max_pages: Optional[int] = Field(None, description="Override max pages")
    seed_urls: Optional[List[str]] = Field(None, description="Override seed URLs")


@router.post("/reindex/websites")
async def reindex_websites(request: WebsiteReindexRequest = None):
    """
    Crawl allowed websites, chunk, embed, and append to FAISS.

    This endpoint performs incremental ingestion â€” it will NOT
    duplicate content that already exists in the index.
    """
    rag_service = _get_rag_service()
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    try:
        from app.utils.config import get_settings
        from app.services.website_crawler import WebsiteCrawler
        import hashlib

        settings = get_settings()

        depth = (request.depth if request and request.depth else settings.crawl_depth_limit)
        max_pages = (request.max_pages if request and request.max_pages else settings.crawl_max_pages)
        seed_urls = (request.seed_urls if request and request.seed_urls else settings.crawl_seed_urls)

        crawler = WebsiteCrawler(
            allowed_domains=settings.allowed_domains,
            depth_limit=depth,
            max_pages=max_pages,
            delay=settings.crawl_delay,
        )
        pages = crawler.crawl_as_dicts(seed_urls)

        if not pages:
            return {"success": True, "pages_crawled": 0, "chunks_added": 0, "message": "No pages crawled"}

        # Prepare documents for add_documents (which handles dedup)
        documents = []
        metadata_list = []
        for page in pages:
            documents.append({
                "content": page.get("text", ""),
                "name": page.get("title", page.get("url", "website")),
            })
            metadata_list.append({
                "document_type": page.get("document_type", "website_page"),
                "source_type": page.get("source_type", "website"),
                "authority_level": page.get("authority_level", "medium"),
                "academic_year": "2025-26",
                "department": "General",
                "url": page.get("url", ""),
            })

        chunks_added = await rag_service.add_documents(documents, metadata_list)

        logger.info(
            "Website reindex API: crawled %d pages, added %d chunks",
            len(pages), chunks_added,
        )

        return {
            "success": True,
            "pages_crawled": len(pages),
            "chunks_added": chunks_added,
            "total_index_size": rag_service.index.ntotal,
        }

    except Exception as e:
        logger.error("Website reindex failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Website reindex failed: {str(e)}")


# =====================================================================
# SOCIAL MEDIA INGESTION (via API)
# =====================================================================

class SocialIngestRequest(BaseModel):
    """Request with social media entries."""
    entries: List[dict] = Field(..., description="List of social media entry objects")


@router.post("/ingest/social")
async def ingest_social(request: SocialIngestRequest):
    """
    Ingest structured social media entries into the knowledge base.

    Each entry should have: title, content, date, platform, url
    """
    rag_service = _get_rag_service()
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    try:
        from app.services.social_ingestor import SocialIngestor

        ingestor = SocialIngestor()
        processed = ingestor.from_entries(request.entries)

        if not processed:
            return {"success": True, "entries_processed": 0, "chunks_added": 0}

        documents = []
        metadata_list = []
        for entry in processed:
            documents.append({
                "content": entry.get("text", ""),
                "name": f"Social: {entry.get('platform', 'post')}",
            })
            metadata_list.append({
                "document_type": entry.get("document_type", "announcement"),
                "source_type": entry.get("source_type", "social_media"),
                "authority_level": entry.get("authority_level", "low"),
                "academic_year": "2025-26",
                "department": "General",
                "platform": entry.get("platform", ""),
                "url": entry.get("url", ""),
            })

        chunks_added = await rag_service.add_documents(documents, metadata_list)

        logger.info(
            "Social ingest API: %d entries â†’ %d chunks added",
            len(processed), chunks_added,
        )

        return {
            "success": True,
            "entries_processed": len(processed),
            "chunks_added": chunks_added,
            "total_index_size": rag_service.index.ntotal,
        }

    except Exception as e:
        logger.error("Social ingestion failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Social ingestion failed: {str(e)}")

