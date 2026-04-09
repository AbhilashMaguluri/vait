"""VAIT LLM Service — Groq primary generation with Ollama fallback."""

import logging
from typing import List

from openai import OpenAI

from app.services.ollama_service import OllamaService
from app.utils.config import Settings

logger = logging.getLogger("vait.llm")
SAFE_UNAVAILABLE_MESSAGE = "AI service is currently unavailable. Please try again later."


class LLMService:
    """Service for generating responses using Groq with Ollama fallback."""

    def __init__(self, settings: Settings):
        """Initialize the LLM service."""
        self.settings = settings
        self.groq_client = OpenAI(
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
        )
        self.fallback_llm = OllamaService(model=settings.ollama_model)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate a response using Groq as primary and Ollama as fallback.

        Args:
            system_prompt: System-level policy and guardrails.
            user_prompt: Context + user question payload.

        Returns:
            Model response text, or a safe availability message.
        """
        logger.info("Starting LLM generation with Groq model: %s", self.settings.groq_model)

        try:
            response = self.groq_client.chat.completions.create(
                model=self.settings.groq_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=800,
            )
            content = (response.choices[0].message.content or "").strip()
            if not content:
                raise RuntimeError("Groq returned an empty response")

            logger.info("Groq response generation succeeded (%d chars)", len(content))
            return content
        except Exception as exc:
            logger.error("Groq generation failed: %s", exc, exc_info=True)
            logger.warning(
                "Using Ollama fallback model after Groq failure: %s",
                self.settings.ollama_model,
            )
            return self._generate_with_fallback(system_prompt, user_prompt)

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

    def _generate_with_fallback(self, system_prompt: str, user_prompt: str) -> str:
        """Generate with Ollama fallback and return a safe message on failure."""
        fallback_prompt = f"{system_prompt}\n\n{user_prompt}"

        try:
            response = self.fallback_llm.generate(fallback_prompt)
            response = (response or "").strip()
            if not response:
                raise RuntimeError("Ollama fallback returned an empty response")

            logger.info("Ollama fallback generation succeeded (%d chars)", len(response))
            return response
        except Exception as exc:
            logger.error("Ollama fallback failed: %s", exc, exc_info=True)
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
            "Regulations, Academic calendars, Syllabi/Notices, Website content, Social media.\n"
            "5. Do not speculate, infer beyond the records, or use outside knowledge.\n"
        )
