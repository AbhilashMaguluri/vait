"""
VAIT LLM Service
Handles interactions with the local Ollama LLM for response generation.

This service manages all LLM interactions for the VAIT system, including
response generation. Uses Ollama (Mistral) for fully offline inference.
"""

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
        """
        Generate a response based on retrieved context.

        Args:
            user_message: The user's question.
            context: Retrieved context from RAG.
            system_prompt: The system prompt defining VAIT's behaviour.
            sources: List of source document names.

        Returns:
            Generated response string.

        Raises:
            RuntimeError: If Ollama fails to generate a response.
        """
        contextual_prompt = self._build_contextual_prompt(
            user_message=user_message,
            context=context,
            sources=sources,
        )

        final_prompt = f"{system_prompt}\n\n{contextual_prompt}"

        logger.info("[DEBUG] User message: %s", user_message)
        logger.debug("[DEBUG] Final prompt length: %d chars", len(final_prompt))
        logger.debug("[DEBUG] Final prompt (first 500 chars): %s", final_prompt[:500])

        try:
            response = self.llm.generate(final_prompt)
            logger.info("[DEBUG] Ollama response length: %d chars", len(response))
            logger.debug("[DEBUG] Ollama response (first 300 chars): %s", response[:300])
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
        """
        Build the contextual prompt with retrieved information.

        The prompt clearly separates context from the user question so the
        LLM can ground its answer exclusively in the retrieved records.
        """
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
