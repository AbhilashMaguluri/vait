"""VAIT LLM Service with Groq primary and OpenRouter fallback."""

import logging
from typing import List

from app.services.groq_service import GroqService
from app.services.openrouter_service import OpenRouterService
from app.utils.config import Settings

logger = logging.getLogger("vait.llm")
SAFE_UNAVAILABLE_MESSAGE = "AI service is currently unavailable."


class LLMService:
    """Service for generating responses using Groq with OpenRouter fallback."""

    def __init__(self, settings: Settings):
        """Initialize the LLM service."""
        self.settings = settings
        self.groq = GroqService(settings)
        self.openrouter = OpenRouterService(settings)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate a response using Groq as primary and OpenRouter as fallback.

        Args:
            system_prompt: System-level policy and guardrails.
            user_prompt: Context + user question payload.

        Returns:
            Model response text, or a safe availability message.
        """
        logger.info("Starting LLM generation with Groq model: %s", self.settings.groq_model)
        full_prompt = f"{system_prompt}\n\n{user_prompt}".strip()

        try:
            return self.groq.generate(full_prompt)
        except Exception as exc:
            logger.error("Groq generation failed: %s", exc, exc_info=True)
            logger.warning("Fallback triggered: switching to OpenRouter after Groq failure")
            return self._generate_with_openrouter(full_prompt)

    async def generate_response(
        self,
        user_message: str,
        context: str,
        system_prompt: str,
        sources: List[str],
    ) -> str:
        """Backward-compatible async wrapper around generate()."""
        contextual_prompt = self._build_contextual_prompt(
            user_message=user_message,
            context=context,
            sources=sources,
        )

        logger.info("User message: %s", user_message)
        logger.debug("Final prompt length: %d chars", len(contextual_prompt))
        logger.debug("Final prompt (first 500 chars): %s", contextual_prompt[:500])

        return self.generate(
            system_prompt=system_prompt,
            user_prompt=contextual_prompt,
        )

    def _generate_with_openrouter(self, prompt: str) -> str:
        """Generate with OpenRouter fallback and return a safe message on failure."""

        try:
            return self.openrouter.generate(prompt)
        except Exception as exc:
            logger.error("OpenRouter fallback failed: %s", exc, exc_info=True)
            return SAFE_UNAVAILABLE_MESSAGE

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
            "3. Use this output structure exactly: Title, Explanation (2-4 sentences), "
            "Key Points (bullet points), Source(s).\n"
            "4. If sources conflict, prefer higher-authority sources in this order: "
            "Official VVIT/VVITU websites, LinkedIn sources, other social sources, then related web sources.\n"
            "5. Do not speculate, infer beyond the records, or use outside knowledge.\n"
        )
