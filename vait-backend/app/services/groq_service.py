"""Groq chat completion provider service."""

from __future__ import annotations

import logging

import httpx

from app.utils.config import Settings

logger = logging.getLogger("vait.groq")


class GroqService:
    """Thin Groq API client for chat completions."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.endpoint = settings.groq_chat_completions_url
        self.timeout_seconds = settings.groq_timeout_seconds

    def generate(self, prompt: str) -> str:
        """Generate a completion using Groq chat completions."""
        if not self.settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not configured")

        headers = {
            "Authorization": f"Bearer {self.settings.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.groq_model,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            response = httpx.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise RuntimeError("Groq request timed out") from exc
        except httpx.HTTPError as exc:
            raise RuntimeError("Groq request failed") from exc

        if response.status_code != 200:
            body = response.text[:400]
            raise RuntimeError(
                f"Groq non-200 response: status={response.status_code}, body={body}"
            )

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("Groq response did not include choices")

        message = choices[0].get("message") or {}
        content = (message.get("content") or "").strip()
        if not content:
            raise RuntimeError("Groq returned an empty response")

        logger.info("Groq response generation succeeded (%d chars)", len(content))
        return content
