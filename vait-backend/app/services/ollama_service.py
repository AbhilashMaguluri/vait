"""VAIT Ollama Service — local LLM and embedding client."""

import logging
import numpy as np
import requests
from typing import List

logger = logging.getLogger("vait.ollama")

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_LLM_MODEL = "mistral"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
EMBEDDING_DIMENSION = 768  # nomic-embed-text output dimension


class OllamaService:
    """Unified Ollama client for LLM generation and embeddings."""

    def __init__(self, model: str = DEFAULT_LLM_MODEL, embed_model: str = DEFAULT_EMBED_MODEL):
        self.model = model
        self.embed_model = embed_model
        self.base_url = OLLAMA_BASE_URL

    # ─── LLM Generation ─────────────────────────────────────────────

    def generate(self, prompt: str) -> str:
        """Generate a text response from Ollama."""
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )

        if response.status_code != 200:
            logger.error("Ollama generate error %d: %s", response.status_code, response.text)
            raise RuntimeError(f"Ollama service error (status {response.status_code})")

        return response.json().get("response", "")

    # ─── Single Embedding ────────────────────────────────────────────

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        response = requests.post(
            f"{self.base_url}/api/embed",
            json={
                "model": self.embed_model,
                "input": text,
            },
            timeout=60,
        )

        if response.status_code != 200:
            logger.error("Ollama embed error %d: %s", response.status_code, response.text)
            raise RuntimeError(f"Ollama embedding error (status {response.status_code})")

        data = response.json()
        embeddings = data.get("embeddings", [])
        if not embeddings:
            raise RuntimeError("Ollama returned no embeddings")

        return np.array(embeddings[0], dtype=np.float32)

    # ─── Batch Embeddings ────────────────────────────────────────────

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        if not texts:
            return np.array([])

        response = requests.post(
            f"{self.base_url}/api/embed",
            json={
                "model": self.embed_model,
                "input": texts,
            },
            timeout=300,
        )

        if response.status_code != 200:
            logger.error("Ollama batch embed error %d: %s", response.status_code, response.text)
            raise RuntimeError(f"Ollama batch embedding error (status {response.status_code})")

        data = response.json()
        embeddings = data.get("embeddings", [])
        if not embeddings:
            raise RuntimeError("Ollama returned no embeddings")

        return np.array(embeddings, dtype=np.float32)

    # ─── Health Check ────────────────────────────────────────────────

    @staticmethod
    def is_running() -> bool:
        """Check whether the Ollama server is reachable."""
        try:
            r = requests.get(OLLAMA_BASE_URL, timeout=3)
            return r.status_code == 200
        except Exception:
            return False
