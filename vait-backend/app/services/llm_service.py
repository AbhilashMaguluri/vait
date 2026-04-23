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

    def generate(self, system_prompt: str, user_prompt: str, history: list = None) -> str:
        """
        Generate a response using Groq as primary and OpenRouter as fallback.

        Args:
            system_prompt: System-level policy and guardrails.
            user_prompt: Context + user question payload.
            history: Optional conversation history to include.

        Returns:
            Model response text, or a safe availability message.
        """
        logger.info("Starting LLM generation with Groq model: %s", self.settings.groq_model)
        
        messages = [{"role": "system", "content": system_prompt}]
        
        if history:
            limit = 10
            for msg in history[-limit:]:
                role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else msg.role)
                content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else msg.content)
                if role and content:
                    norm_role = "assistant" if role == "assistant" else "user"
                    messages.append({"role": norm_role, "content": content})
                    
        messages.append({"role": "user", "content": user_prompt})

        try:
            return self.groq.generate(messages)
        except Exception as exc:
            logger.error("Groq generation failed: %s", exc, exc_info=True)
            logger.warning("Fallback triggered: switching to OpenRouter after Groq failure")
            return self._generate_with_openrouter(messages)

    async def generate_response(
        self,
        user_message: str,
        context: str,
        system_prompt: str,
        sources: List[str],
        history: list = None,
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
            history=history,
        )

    def _generate_with_openrouter(self, messages: list) -> str:
        """Generate with OpenRouter fallback and return a safe message on failure."""

        try:
            return self.openrouter.generate(messages)
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
            "1. Answer the user's question using the RETRIEVED INSTITUTIONAL RECORDS above whenever applicable.\n"
            "2. Use this output structure exactly: Title, Explanation (2-4 sentences), "
            "Key Points (bullet points), Source(s) (if used).\n"
            "3. If sources conflict, prefer higher-authority sources in this order: "
            "Official VVIT/VVITU websites, LinkedIn sources, other social sources, then related web sources.\n"
        )

    async def generate_stream(self, system_prompt: str, user_prompt: str, history: list = None):
        logger.info("Starting streaming LLM generation with Groq model: %s", self.settings.groq_model)
        
        messages = [{"role": "system", "content": system_prompt}]
        
        if history:
            limit = 10
            for msg in history[-limit:]:
                role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else msg.role)
                content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else msg.content)
                if role and content:
                    norm_role = "assistant" if role == "assistant" else "user"
                    messages.append({"role": norm_role, "content": content})
                    
        messages.append({"role": "user", "content": user_prompt})

        fallback_needed = False
        try:
            async for chunk in self.groq.generate_stream(messages):
                yield chunk
            logger.info("Groq stream completed successfully")
        except Exception as exc:
            logger.error("Groq streaming failed mid-way: %s", exc)
            fallback_needed = True

        if fallback_needed:
            logger.warning("Fallback triggered: switching to OpenRouter after Groq streaming failure")
            yield "[FALLBACK_TRIGGERED]"
            try:
                async for chunk in self.openrouter.generate_stream(messages):
                    yield chunk
                logger.info("OpenRouter stream completed successfully")
            except Exception as exc:
                logger.error("OpenRouter fallback streaming failed: %s", exc)
                yield "\n\nAI service is currently unavailable."
