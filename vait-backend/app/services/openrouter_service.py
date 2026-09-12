"""OpenRouter chat completion provider service."""

from __future__ import annotations

import logging

import httpx

from app.utils.config import Settings

logger = logging.getLogger("vait.openrouter")


class OpenRouterService:
    """Thin OpenRouter API client for chat completions."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.endpoint = settings.openrouter_chat_completions_url
        self.timeout_seconds = settings.openrouter_timeout_seconds

    def generate(self, messages: list) -> str:
        """Generate a completion using OpenRouter chat completions."""
        if not self.settings.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")

        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.openrouter_model,
            "messages": messages,
        }

        try:
            response = httpx.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise RuntimeError("OpenRouter request timed out") from exc
        except httpx.HTTPError as exc:
            raise RuntimeError("OpenRouter request failed") from exc

        if response.status_code != 200:
            body = response.text[:400]
            raise RuntimeError(
                "OpenRouter non-200 response: "
                f"status={response.status_code}, body={body}"
            )

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("OpenRouter response did not include choices")

        message = choices[0].get("message") or {}
        content = (message.get("content") or "").strip()
        if not content:
            raise RuntimeError("OpenRouter returned an empty response")

        logger.info("OpenRouter fallback generation succeeded (%d chars)", len(content))
        return content

    async def generate_stream(self, messages: list):
        """Generate a streaming completion using OpenRouter chat completions."""
        if not self.settings.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")

        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://vait.web",
        }
        payload = {
            "model": self.settings.openrouter_model,
            "messages": messages,
            "stream": True,
        }

        import json
        async with httpx.AsyncClient() as client:
            try:
                stream_timeout = httpx.Timeout(connect=15.0, read=45.0, write=15.0, pool=15.0)
                async with client.stream(
                    "POST",
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=stream_timeout,
                ) as response:
                    if response.status_code != 200:
                        raise RuntimeError(f"OpenRouter non-200 streaming response: {response.status_code}")

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
                logger.error("OpenRouter stream failed: %s", exc)
                raise RuntimeError("OpenRouter streaming request failed mid-way") from exc
