"""Centralized configuration for the VAIT backend."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("vait.config")


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Support standard local .env as well as Render's Secret File location (/etc/secrets/.env)
RENDER_SECRETS_PATH = Path("/etc/secrets/.env")
_local_env_path = PROJECT_ROOT / ".env"

_local_loaded = load_dotenv(_local_env_path) if _local_env_path.exists() else False
_secrets_loaded = load_dotenv(RENDER_SECRETS_PATH, override=False) if RENDER_SECRETS_PATH.exists() else False

_dotenv_path = RENDER_SECRETS_PATH if _secrets_loaded else _local_env_path
_dotenv_loaded = _secrets_loaded or _local_loaded

DATA_DIR = PROJECT_ROOT / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_docs"
VECTOR_STORE_DIR = PROJECT_ROOT / "vectorstore"
LOGS_DIR = DATA_DIR / "logs"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
NOTICES_DIR = KNOWLEDGE_DIR / "notices"
ANNOUNCEMENTS_DIR = PROJECT_ROOT / "announcements"
SOCIAL_DIR = PROJECT_ROOT / "social"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

DOCUMENT_TYPES = [
    "regulation",
    "syllabus",
    "notice",
    "website",
    "knowledge_source",
]

AUTHORITY_LEVELS = {
    "high": 1.0,
    "medium": 0.8,
    "low": 0.6,
}

REQUIRED_METADATA_FIELDS = [
    "document_type",
    "academic_year",
    "department",
    "authority_level",
]

QUESTION_TYPES = {
    "factual": "Questions seeking specific facts or data",
    "policy": "Questions about institutional policies and regulations",
    "procedure": "Questions about processes and how to do things",
    "definition": "Questions asking for definitions or explanations of terms",
}

REFUSAL_MESSAGE = (
    "Based on available VVIT / VVITU sources, this information is not clearly specified."
)


def mask_connection_uri(uri: str) -> str:
    """Hide credentials in database URIs before writing them to logs."""

    if "://" not in uri or "@" not in uri:
        return uri
    scheme, rest = uri.split("://", 1)
    _, host_part = rest.split("@", 1)
    return f"{scheme}://***:***@{host_part}"


class Settings(BaseSettings):
    """Environment-backed application settings."""

    model_config = SettingsConfigDict(
        env_file=tuple(
            str(p) for p in [RENDER_SECRETS_PATH, _local_env_path, Path(".env")] if p.exists()
        ) or ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "VAIT — VVIT's Artificial Intelligence Technology"
    app_version: str = "2.0.0"
    debug: bool = False
    node_env: str = Field(default="development", alias="NODE_ENV")
    port: int = Field(default=5000, alias="PORT")
    api_prefix: str = "/api/vait"

    mongodb_uri: str = Field(
        default="mongodb://localhost:27017/vait",
        validation_alias=AliasChoices("MONGODB_URI", "MONGO_URI"),
    )
    mongodb_server_selection_timeout_ms: int = Field(
        default=5000,
        validation_alias=AliasChoices(
            "MONGODB_SERVER_SELECTION_TIMEOUT_MS",
            "MONGO_SERVER_SELECTION_TIMEOUT_MS",
        ),
    )
    jwt_secret: str = Field(default="change-me-in-production", alias="JWT_SECRET")
    jwt_expiry: str = Field(default="7d", alias="JWT_EXPIRY")

    admin_username: str = Field(default="admin", alias="ADMIN_USERNAME")
    admin_email: str = Field(default="admin@vvit.net", alias="ADMIN_EMAIL")
    admin_password: str = Field(default="admin", alias="ADMIN_PASSWORD")

    google_client_id: str = Field(default="", alias="GOOGLE_CLIENT_ID")
    google_client_secret: str = Field(default="", alias="GOOGLE_CLIENT_SECRET")
    google_redirect_uri: str = Field(
        default="http://127.0.0.1:8000/api/vait/auth/google/callback",
        alias="GOOGLE_REDIRECT_URI",
    )
    frontend_auth_redirect_url: str = Field(
        default="http://localhost:5173/auth/callback",
        alias="FRONTEND_AUTH_REDIRECT_URL",
    )

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    nvidia_api_key: str = Field(default="", alias="NVIDIA_API_KEY")

    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dimension: int = 1536

    openai_model: str = "gpt-4o-mini"
    groq_model: str = "openai/gpt-oss-120b"
    openrouter_model: str = "meta-llama/llama-3.1-8b-instruct"
    nvidia_model: str = "meta/llama-3.1-70b-instruct"
    groq_chat_completions_url: str = "https://api.groq.com/openai/v1/chat/completions"
    openrouter_chat_completions_url: str = "https://openrouter.ai/api/v1/chat/completions"
    openrouter_embeddings_url: str = "https://openrouter.ai/api/v1/embeddings"
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    groq_timeout_seconds: float = 12.0
    openrouter_timeout_seconds: float = 18.0
    embeddings_timeout_seconds: float = 20.0

    default_llm_provider: str = "groq"
    default_llm_model: str = "openai/gpt-oss-120b"
    default_temperature: float = 0.3
    default_max_tokens: int = 800

    chunk_size: int = 400
    chunk_overlap: int = 50
    top_k_retrieval: int = 6
    similarity_threshold: float = 0.65
    max_context_chars: int = 8000

    # Institutional domains & sources (Dual-Source Strategy)
    current_university_domain: str = "vvitu.ac.in"
    current_university_url: str = "https://vvitu.ac.in/"
    legacy_institute_domain: str = "vvitguntur.com"
    legacy_institute_url: str = "https://vvitguntur.com/"

    allowed_domains: List[str] = [
        "vvitu.ac.in",
        "www.vvitu.ac.in",
        "vvitguntur.com",
        "www.vvitguntur.com",
    ]
    # Crawl seeds: CURRENT university (vvitu.ac.in) is primary, followed by LEGACY institute (vvitguntur.com)
    crawl_seed_urls: List[str] = [
        "https://www.vvitu.ac.in/",
        "https://www.vvitguntur.com/",
    ]
    crawl_depth_limit: int = 2
    crawl_max_pages: int = 200
    crawl_delay: float = 0.5
    auto_ingest_official_sites_on_empty: bool = True
    official_bootstrap_depth_limit: int = 1
    official_bootstrap_max_pages: int = 24

    auto_reindex_on_startup: bool = False
    auto_watch_enabled: bool = False
    watch_directories: List[str] = [
        str(UPLOADS_DIR),
        str(NOTICES_DIR),
        str(ANNOUNCEMENTS_DIR),
        str(SOCIAL_DIR),
    ]

    vector_db_path: str = Field(default=str(VECTOR_STORE_DIR), alias="VECTOR_DB_PATH")
    faiss_index_path: str = str(VECTOR_STORE_DIR / "vait.index")
    metadata_path: str = str(VECTOR_STORE_DIR / "metadata.json")
    raw_docs_path: str = str(RAW_DOCS_DIR)
    logs_path: str = str(LOGS_DIR)
    system_prompt_path: str = str(PROMPTS_DIR / "vait_system_prompt.txt")

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug_flag(cls, value):
        """Accept common environment labels for the DEBUG setting."""
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "prod", "production", "false", "0", "no", "off"}:
                return False
            if normalized in {"debug", "dev", "development", "true", "1", "yes", "on"}:
                return True
        return value

    @field_validator("groq_model", "default_llm_model", mode="before")
    @classmethod
    def validate_groq_model(cls, value):
        """Ensure decommissioned or deprecated Llama models in env vars are upgraded."""
        if not value or "llama" in str(value).lower():
            logger.warning(
                "Deprecated Groq model '%s' detected in environment. Upgrading to openai/gpt-oss-120b.",
                value,
            )
            return "openai/gpt-oss-120b"
        return value

    @field_validator("openrouter_model", mode="before")
    @classmethod
    def validate_openrouter_model(cls, value):
        """Ensure deprecated OpenRouter fallback models in env vars are upgraded."""
        if not value or "mistral" in str(value).lower():
            logger.warning(
                "Invalid OpenRouter fallback model '%s' detected in environment. Upgrading to meta-llama/llama-3.1-8b-instruct.",
                value,
            )
            return "meta-llama/llama-3.1-8b-instruct"
        return value

    @field_validator("embedding_model", mode="before")
    @classmethod
    def validate_embedding_model(cls, value):
        """Ensure incompatible embedding models in env vars are upgraded to 1536-dim standard."""
        if not value or "nomic" in str(value).lower():
            logger.warning(
                "Incompatible embedding model '%s' detected in environment. Upgrading to openai/text-embedding-3-small.",
                value,
            )
            return "openai/text-embedding-3-small"
        return value


@lru_cache()
def get_settings() -> Settings:
    """Return the cached settings instance."""

    settings = Settings()
    logger.info("Configuration loaded successfully")
    logger.info("  Render secrets file : %s (exists=%s, loaded=%s)", RENDER_SECRETS_PATH, RENDER_SECRETS_PATH.exists(), _secrets_loaded)
    logger.info("  Local .env file     : %s (exists=%s, loaded=%s)", _local_env_path, _local_env_path.exists(), _local_loaded)
    logger.info("  Environment         : %s", settings.node_env)
    logger.info("  MongoDB URI         : %s", mask_connection_uri(settings.mongodb_uri))
    logger.info("  JWT expiry          : %s", settings.jwt_expiry)
    logger.info("  Default admin email : %s", settings.admin_email)
    logger.info("  Groq model          : %s", settings.groq_model)
    logger.info("  Groq endpoint       : %s", settings.groq_chat_completions_url)
    logger.info("  OpenRouter model    : %s", settings.openrouter_model)
    logger.info("  OpenRouter endpoint : %s", settings.openrouter_chat_completions_url)
    logger.info("  Groq key configured : %s", bool(settings.groq_api_key))
    logger.info("  OpenRouter key cfgd : %s", bool(settings.openrouter_api_key))
    logger.info("  Vector store path   : %s", settings.vector_db_path)
    logger.info("  API prefix          : %s", settings.api_prefix)
    if "localhost" in settings.mongodb_uri and settings.node_env.lower() in {"production", "prod", "release"}:
        logger.warning(
            "MONGODB_URI is pointing to localhost in production (%s). Set MONGODB_URI (or MONGO_URI) in your Render environment variables.",
            settings.node_env,
        )
    return settings


def ensure_directories() -> None:
    """Ensure required runtime directories exist."""

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
        NOTICES_DIR,
        UPLOADS_DIR,
        ANNOUNCEMENTS_DIR,
        SOCIAL_DIR,
        PROMPTS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
