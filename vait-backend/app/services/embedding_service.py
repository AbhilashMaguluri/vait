"""
VAIT Embeddings Service
Handles text embedding generation using Ollama's local embedding models.

This service provides the core embedding functionality for the VAIT RAG system.
Embeddings are generated via a local Ollama instance (nomic-embed-text, 768 dim)
for fully offline semantic similarity matching.
"""

import logging
import numpy as np
from typing import List

from app.services.ollama_service import OllamaService, EMBEDDING_DIMENSION
from app.utils.config import Settings

logger = logging.getLogger("vait.embedding")


class EmbeddingService:
    """Service for generating text embeddings using Ollama."""

    def __init__(self, settings: Settings):
        """Initialize the embedding service."""
        self.settings = settings
        self.ollama = OllamaService(
            model=settings.ollama_model,
            embed_model=settings.embedding_model,
        )
        self.embedding_dimension = EMBEDDING_DIMENSION

    async def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            Numpy array of the embedding vector.
        """
        return self.ollama.embed(text)

    async def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            Numpy array of embedding vectors (shape: [n_texts, embedding_dim]).
        """
        if not texts:
            return np.array([])

        return self.ollama.embed_batch(texts)
