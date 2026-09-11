"""Core constants shared across the platform."""

ADMIN_ROLE = "admin"
TRAINER_ROLE = "trainer"
USER_ROLE = "user"
ALL_ROLES = {ADMIN_ROLE, TRAINER_ROLE, USER_ROLE}

USERS_COLLECTION = "users"
CONVERSATIONS_COLLECTION = "conversations"
DOCUMENTS_COLLECTION = "documents"
SETTINGS_COLLECTION = "settings"
ACTIVITY_LOGS_COLLECTION = "activity_logs"

AI_SETTINGS_KEY = "ai_settings"
API_KEYS_SETTINGS_KEY = "api_keys"
SYSTEM_SETTINGS_KEY = "system_settings"
KNOWLEDGE_BASE_SETTINGS_KEY = "knowledge_base"

DEFAULT_AI_SETTINGS = {
    "provider": "groq",
    "model": "openai/gpt-oss-120b",
    "temperature": 0.3,
    "max_tokens": 800,
}

DEFAULT_SYSTEM_SETTINGS = {
    "official_mode_label": "Official Academic Mode",
    "admin_indicator_label": "Admin Control Enabled",
    "knowledge_empty_message": "Knowledge base not populated. Please upload documents.",
}

DEFAULT_KNOWLEDGE_BASE_STATE = {
    "status": "empty",
    "last_trained_at": None,
    "last_error": None,
}

DEFAULT_PROVIDER_CONFIG = {
    "openai_api_key": "",
    "groq_api_key": "",
    "openrouter_api_key": "",
    "nvidia_api_key": "",
}

DOCUMENT_SOURCE_TYPES = {"pdf", "docx", "txt", "url", "text"}
EDITABLE_DOCUMENT_TYPES = {"url", "text"}
