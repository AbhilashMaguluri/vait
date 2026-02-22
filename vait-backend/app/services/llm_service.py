"""VAIT LLM Service — Ollama response generation."""

import logging
from typing import List, Dict, Optional

from app.services.ollama_service import OllamaService
from app.utils.config import Settings

logger = logging.getLogger("vait.llm")


class LLMService:
    """Service for generating responses using a local Ollama LLM."""

    def __init__(self, settings: Settings):
        """Initialize the LLM service."""
        self.settings = settings
        self.llm = OllamaService(model=settings.ollama_model)

    async def generate_response(
        self,
        user_message: str,
        context: str,
        system_prompt: str,
        sources: List[str],
    ) -> str:
        """Generate a response from retrieved context via Ollama."""
        contextual_prompt = self._build_contextual_prompt(
            user_message=user_message,
            context=context,
            sources=sources,
        )

        final_prompt = f"{system_prompt}\n\n{contextual_prompt}"

        logger.info("User message: %s", user_message)
        logger.debug("Final prompt length: %d chars", len(final_prompt))
        logger.debug("Final prompt (first 500 chars): %s", final_prompt[:500])

        try:
            response = self.llm.generate(final_prompt)
            logger.info("Ollama response length: %d chars", len(response))
            logger.debug("Ollama response (first 300 chars): %s", response[:300])
            return response
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            raise RuntimeError(f"LLM generation failed: {exc}") from exc

    def _build_contextual_prompt(
        self,
        user_message: str,
        context: str,
        sources: List[str]
    ) -> str:
        """Build contextual prompt separating retrieved records from the user question."""
        source_list = "\n".join(f"  - {source}" for source in sources)

        return (
            "RETRIEVED INSTITUTIONAL RECORDS:\n"
            "---\n"
            f"{context}\n"
            "---\n\n"
            "SOURCE DOCUMENTS:\n"
            f"{source_list}\n\n"
            "USER QUESTION:\n"
            f"{user_message}\n\n"
            "INSTRUCTIONS:\n"
            "1. Answer ONLY using the RETRIEVED INSTITUTIONAL RECORDS above.\n"
            "2. If the records do not contain sufficient information, respond "
            "with the exact refusal message from your system prompt.\n"
            "3. Cite the source documents when providing information.\n"
            "4. Structure the answer with a title, explanation, bullet points "
            "(if applicable), and a source citation section.\n"
            "5. Do not speculate or provide information beyond the records.\n"
        )
