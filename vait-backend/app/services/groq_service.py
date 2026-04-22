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

    async def generate_stream(self, prompt: str):
        """Generate a streaming completion using Groq chat completions."""
        if not self.settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not configured")

        headers = {
            "Authorization": f"Bearer {self.settings.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.groq_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }

        import json
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                ) as response:
                    if response.status_code != 200:
                        raise RuntimeError(f"Groq non-200 streaming response: {response.status_code}")

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            if not data_str:
                                continue
                            try:
                                data = json.loads(data_str)
                                choices = data.get("choices") or []
                                if choices:
                                    chunk = choices[0].get("delta", {}).get("content", "")
                                    if chunk:
                                        yield chunk
                            except Exception:
                                continue
            except Exception as exc:
                logger.error("Groq stream failed: %s", exc)
                raise RuntimeError("Groq streaming request failed mid-way") from exc
