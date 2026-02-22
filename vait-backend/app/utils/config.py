"""
VAIT Configuration Module
Centralized configuration for the VAIT backend system.

This module defines all configuration constants and settings for the VAIT
Retrieval-Augmented Generation system. All paths, thresholds, and operational
parameters are defined here for consistency across the application.

Startup behaviour:
  1. Load .env from project root using python-dotenv
  2. Log configuration status
"""

import os
import sys
import logging
from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List

from dotenv import load_dotenv

# =============================================================================
# LOGGING
# =============================================================================

logger = logging.getLogger("vait.config")

# =============================================================================
# BASE PATHS
# =============================================================================

# Project root is vait-backend/
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load .env before anything else reads environment variables
_dotenv_path = PROJECT_ROOT / ".env"
_dotenv_loaded = load_dotenv(_dotenv_path)

# Data directory structure
DATA_DIR = PROJECT_ROOT / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_docs"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
LOGS_DIR = DATA_DIR / "logs"

# Knowledge directory structure
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"

# Watch directories (auto-ingestion)
UPLOADS_DIR = PROJECT_ROOT / "uploads"
NOTICES_DIR = KNOWLEDGE_DIR / "notices"
ANNOUNCEMENTS_DIR = PROJECT_ROOT / "announcements"
SOCIAL_DIR = PROJECT_ROOT / "social"

# Prompts directory
PROMPTS_DIR = PROJECT_ROOT / "prompts"


# =============================================================================
# METADATA CONFIGURATION
# =============================================================================

# Document types (in order of authority)
DOCUMENT_TYPES = [
    "regulation",      # Official university regulations
    "syllabus",        # Course syllabi and academic content
    "notice",          # Official notices and announcements
    "website"          # Website content and general information
]

# Authority levels (determines weight in retrieval)
AUTHORITY_LEVELS = {
    "high": 1.0,       # Regulations, official policies
    "medium": 0.8,     # Academic calendar, department docs
    "low": 0.6         # Website content, general info
}

# Required metadata fields for document ingestion
# These fields must be provided when ingesting documents
REQUIRED_METADATA_FIELDS = [
    "document_type",
    "academic_year",
    "department",
    "authority_level"
]

# Additional metadata fields added during processing
# source_file - added during ingestion
# document_name - added during chunking
# text/content - stored in chunk content

# Question types for classification
QUESTION_TYPES = {
    "factual": "Questions seeking specific facts or data",
    "policy": "Questions about institutional policies and regulations",
    "procedure": "Questions about processes and how to do things",
    "definition": "Questions asking for definitions or explanations of terms"
}


# =============================================================================
# REFUSAL MESSAGE (MANDATORY)
# =============================================================================

REFUSAL_MESSAGE = (
    "This information is not available in the official VAIT records at this time."
)


# =============================================================================
# SETTINGS CLASS
# =============================================================================

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    These settings control the behavior of the VAIT RAG system.
    All paths are configured relative to the project root.
    """
    
    # Application
    app_name: str = "VAIT - Institutional University AI Assistant"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Ollama Configuration (fully offline LLM)
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "phi3:mini"
    embedding_model: str = "nomic-embed-text"
    
    # ==========================================================================
    # RAG v2 CONFIGURATION (PRODUCTION)
    # ==========================================================================
    
    # Chunk size: ~400 tokens (~1600 characters)
    chunk_size: int = 400
    chunk_overlap: int = 50
    
    # Top-K retrieval: 6
    top_k_retrieval: int = 6
    
    # Strict similarity threshold - below this triggers refusal
    similarity_threshold: float = 0.65
    
    # Maximum context characters sent to LLM (prevent overflow)
    max_context_chars: int = 8000
    
    # ==========================================================================
    # WEBSITE CRAWLER CONFIGURATION
    # ==========================================================================
    
    # Allowed domains (crawler will NEVER leave these)
    allowed_domains: List[str] = [
        "vvitguntur.com",
        "www.vvitguntur.com",
        "vvitu.ac.in",
        "www.vvitu.ac.in",
    ]
    
    # Seed URLs for the website crawler
    crawl_seed_urls: List[str] = [
        "https://www.vvitguntur.com/",
        "https://www.vvitu.ac.in/",
    ]
    
    # Maximum crawl depth (0 = seed page only, 2 = two hops)
    crawl_depth_limit: int = 2
    
    # Maximum pages to crawl per run
    crawl_max_pages: int = 200
    
    # Seconds between HTTP requests (polite crawl)
    crawl_delay: float = 0.5
    
    # ==========================================================================
    # AUTO-INGESTION CONFIGURATION
    # ==========================================================================
    
    # Whether to reindex website content on server startup
    auto_reindex_on_startup: bool = False
    
    # Whether to start the file watcher on server startup
    auto_watch_enabled: bool = False
    
    # Directories monitored by the file watcher
    watch_directories: List[str] = [
        str(UPLOADS_DIR),
        str(NOTICES_DIR),
        str(ANNOUNCEMENTS_DIR),
        str(SOCIAL_DIR),
    ]
    
    # ==========================================================================
    # VECTOR STORE PATHS
    # ==========================================================================
    
    # FAISS index file path
    faiss_index_path: str = str(VECTOR_STORE_DIR / "vait.index")
    
    # Metadata JSON file path (stores text + metadata per vector)
    metadata_path: str = str(VECTOR_STORE_DIR / "metadata.json")
    
    # ==========================================================================
    # DATA PATHS
    # ==========================================================================
    
    # Raw documents input directory
    raw_docs_path: str = str(RAW_DOCS_DIR)
    
    # Query logs output directory
    logs_path: str = str(LOGS_DIR)
    
    # System prompt file path
    system_prompt_path: str = str(PROMPTS_DIR / "vait_system_prompt.txt")
    
    # ==========================================================================
    # API CONFIGURATION
    # ==========================================================================
    
    api_prefix: str = "/api/vait"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to ensure settings are only loaded once and reused
    throughout the application lifecycle.
    
    Validates that critical configuration values are present and logs
    the configuration status.
    """
    settings = Settings()

    # ── Log configuration status ─────────────────────────────────────
    logger.info("Configuration loaded successfully")
    logger.info("  .env file          : %s (loaded=%s)", _dotenv_path, _dotenv_loaded)
    logger.info("  Ollama URL         : %s", settings.ollama_url)
    logger.info("  LLM model          : %s", settings.ollama_model)
    logger.info("  Embedding model    : %s", settings.embedding_model)
    logger.info("  FAISS index path   : %s", settings.faiss_index_path)
    logger.info("  Similarity thresh  : %.2f", settings.similarity_threshold)

    return settings


def ensure_directories() -> None:
    """
    Ensure all required directories exist.
    
    Called during application startup to create necessary directories
    if they don't already exist.
    """
    directories = [
        DATA_DIR,
        RAW_DOCS_DIR,
        VECTOR_STORE_DIR,
        LOGS_DIR,
        KNOWLEDGE_DIR,
        KNOWLEDGE_DIR / "regulations",
        KNOWLEDGE_DIR / "academic-calendar",
        KNOWLEDGE_DIR / "examinations",
        KNOWLEDGE_DIR / "syllabus",
        KNOWLEDGE_DIR / "departments",
        KNOWLEDGE_DIR / "notices",
        UPLOADS_DIR,
        ANNOUNCEMENTS_DIR,
        SOCIAL_DIR,
        PROMPTS_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
