"""VAIT — Main FastAPI Application Entry Point."""

import logging
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.utils.config import get_settings, ensure_directories, LOGS_DIR
from app.db import close_mongo_connection, connect_to_mongo, get_last_connection_error, is_db_connected
from app.routes.auth import router as auth_router
from app.routes.chat import router as chat_router
from app.routes.admin import router as admin_router
from app.services.rag_service import RAGService
from app.services.file_watcher import FileWatcher, IncrementalIngestor, WATCHDOG_AVAILABLE
from app.services.user_service import ensure_default_admin


# =============================================================================
# LOGGING SETUP
# =============================================================================

def _configure_logging() -> None:
    """Configure application-wide logging: console + file."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("vait")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File
    fh = logging.FileHandler(LOGS_DIR / "vait.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)


_configure_logging()
logger = logging.getLogger("vait.main")


# Global RAG service instance
rag_service: RAGService = None
file_watcher: FileWatcher = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Initializes and cleans up resources.
    """
    global rag_service, file_watcher

    # Ensure all required directories exist
    ensure_directories()

    settings = get_settings()

    try:
        database = await connect_to_mongo(settings)
        await ensure_default_admin(
            database,
            username=settings.admin_username,
            email=settings.admin_email,
            password=settings.admin_password,
        )
        logger.info("Authentication database initialized successfully on '%s'", database.name)
    except Exception as exc:
        logger.error(
            "CRITICAL: Failed to initialize MongoDB on startup: %s. "
            "Authentication and database-backed routes will return 503 until MongoDB is accessible.",
            exc,
            exc_info=True,
        )

    # Initialize RAG service on startup
    try:
        rag_service = RAGService(settings)
        await rag_service.initialize()
    except Exception as exc:
        logger.error("Failed to initialize RAG service: %s", exc, exc_info=True)
        # Allow the server to start (health check will show degraded status)
        rag_service = None

    # Start file watcher if enabled
    if settings.auto_watch_enabled and WATCHDOG_AVAILABLE and rag_service is not None:
        try:
            from pathlib import Path
            watch_dirs = [Path(d) for d in settings.watch_directories]
            ingestor = IncrementalIngestor(rag_service)
            file_watcher = FileWatcher(
                directories=watch_dirs,
                on_new_file=ingestor.ingest_file,
            )
            file_watcher.start()
            logger.info("File watcher started: %d directories", len(watch_dirs))
        except Exception as exc:
            logger.warning("File watcher failed to start: %s", exc)
            file_watcher = None
    else:
        if settings.auto_watch_enabled and not WATCHDOG_AVAILABLE:
            logger.warning("auto_watch_enabled=True but watchdog not installed")

    logger.info("VAIT Backend v%s started successfully", settings.app_version)
    logger.info(
        "RAG Configuration: chunk_size=%d (~%d chars), top_k=%d, threshold=%.2f",
        settings.chunk_size,
        settings.chunk_size * 4,
        settings.top_k_retrieval,
        settings.similarity_threshold,
    )
    logger.info("FAISS Index Path: %s", settings.faiss_index_path)
    logger.info("Metadata Path:    %s", settings.metadata_path)
    logger.info("Allowed Domains:  %s", settings.allowed_domains)
    logger.info("File Watcher:     %s", "running" if file_watcher else "disabled")

    yield

    # Cleanup on shutdown
    if file_watcher is not None:
        file_watcher.stop()
    await close_mongo_connection()
    logger.info("VAIT Backend shutting down...")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="VAIT — VVIT's Artificial Intelligence Technology with strict RAG-based responses",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register routers
    app.include_router(auth_router, prefix=settings.api_prefix)
    app.include_router(chat_router, prefix=settings.api_prefix)
    app.include_router(admin_router, prefix=settings.api_prefix)
    
    @app.get("/")
    async def root():
        """Root endpoint - health check."""
        db_ok = is_db_connected()
        return {
            "service": "VAIT",
            "version": settings.app_version,
            "status": "operational" if db_ok else "degraded",
            "database": "connected" if db_ok else "disconnected",
        }
    
    @app.get("/health")
    async def health_check():
        """
        VAIT Institutional AI Dashboard — Health & Metrics.

        Returns comprehensive system status for demo impact:
          - Service info & operational status
          - Database connection & collection status
          - Total vectors / website / social / PDF breakdown
          - Average adjusted score over last 10 queries
          - Last reindex timestamp
          - RAG configuration summary
        """
        db_ok = is_db_connected()
        overall_status = "operational" if (rag_service is not None and db_ok) else "degraded"
        base = {
            "service": "VAIT — VVIT's Artificial Intelligence Technology",
            "version": settings.app_version,
            "status": overall_status,
            "database": {
                "status": "connected" if db_ok else "disconnected",
                "detail": None if db_ok else get_last_connection_error(),
            },
            "engine": "FAISS + Groq/OpenRouter RAG Pipeline",
            "llm_primary_model": settings.groq_model,
            "llm_fallback_model": settings.openrouter_model,
            "embedding_model": settings.embedding_model,
        }

        if rag_service is not None:
            stats = rag_service.get_stats()
            base.update({
                "total_vectors": stats.get("total_vectors", 0),
                "website_vectors": stats.get("website_vectors", 0),
                "social_vectors": stats.get("social_vectors", 0),
                "pdf_vectors": stats.get("pdf_vectors", 0),
                "source_breakdown": stats.get("source_breakdown", {}),
                "similarity_threshold": stats.get("similarity_threshold"),
                "top_k_retrieval": stats.get("top_k"),
                "avg_adjusted_score_last_10_queries": stats.get(
                    "avg_adjusted_score_last_10_queries"
                ),
                "last_reindex_time": stats.get("last_reindex_time"),
                "allowed_domains": settings.allowed_domains,
                "cache_hits": stats.get("cache_hits", 0),
                "cache_misses": stats.get("cache_misses", 0),
                "cache_size": stats.get("cache_size", 0),
                "uptime_seconds": stats.get("uptime_seconds", 0),
            })
        else:
            base.update({
                "total_vectors": 0,
                "note": "RAG service is initializing — knowledge base not loaded yet.",
            })

        return base
    
    return app


# Create the application instance
app = create_app()


def get_rag_service() -> RAGService:
    """Get the RAG service instance."""
    global rag_service
    return rag_service


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
