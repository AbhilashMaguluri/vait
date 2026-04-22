"""VAIT Embeddings Service using OpenRouter embeddings API."""

from __future__ import annotations

import logging
from typing import List

import httpx
import numpy as np

from app.utils.config import Settings

logger = logging.getLogger("vait.embedding")

EMBEDDING_DIMENSION = 768


class EmbeddingService:
    """Service for generating text embeddings via OpenRouter."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedding_dimension = settings.embedding_dimension or EMBEDDING_DIMENSION

    async def get_embedding(self, text: str) -> np.ndarray:
        """Generate an embedding for one text."""
        vectors = self.get_embeddings_sync([text])
        return vectors[0]

    async def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        return self.get_embeddings_sync(texts)

    def get_embeddings_sync(self, texts: List[str]) -> np.ndarray:
        """Synchronous embedding generation used by ingestion and management scripts."""
        if not texts:
            return np.array([], dtype=np.float32)

        if not self.settings.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for embeddings")

        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.embedding_model,
            "input": texts,
        }

        try:
            response = httpx.post(
                self.settings.openrouter_embeddings_url,
                headers=headers,
                json=payload,
                timeout=self.settings.embeddings_timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise RuntimeError("OpenRouter embeddings request timed out") from exc
        except httpx.HTTPError as exc:
            raise RuntimeError("OpenRouter embeddings request failed") from exc

        if response.status_code != 200:
            body = response.text[:400]
            raise RuntimeError(
                "OpenRouter embeddings non-200 response: "
                f"status={response.status_code}, body={body}"
            )

        data = response.json().get("data") or []
        if not data:
            raise RuntimeError("OpenRouter embeddings response did not include vectors")

        sorted_data = sorted(data, key=lambda item: item.get("index", 0))
        vectors = [item.get("embedding") for item in sorted_data]
        if not all(vectors):
            raise RuntimeError("OpenRouter embeddings response had empty vector values")

        arr = np.array(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if arr.shape[1] != self.embedding_dimension:
            logger.warning(
                "Embedding dimension mismatch: got=%d expected=%d; resizing vectors",
                arr.shape[1],
                self.embedding_dimension,
            )
            if arr.shape[1] > self.embedding_dimension:
                arr = arr[:, : self.embedding_dimension]
            else:
                pad_width = self.embedding_dimension - arr.shape[1]
                arr = np.pad(arr, ((0, 0), (0, pad_width)), mode="constant")

        return arr
