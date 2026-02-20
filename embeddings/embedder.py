"""
Embedding providers for converting text to vector representations.

Provides an abstract base class with two implementations:
- OllamaEmbedder: Uses a locally running Ollama instance.
- SentenceTransformerEmbedder: Uses sentence-transformers as a fallback.
"""

from abc import ABC, abstractmethod

import ollama
from sentence_transformers import SentenceTransformer


class Embedder(ABC):
    """
    Abstract base class for text embedding providers.
    """

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts into vector representations.

        Args:
            texts (list[str]): List of text strings to embed.

        Returns:
            list[list[float]]: List of embedding vectors.
        """
        ...

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query text.

        Args:
            text (str): The query text to embed.

        Returns:
            list[float]: The embedding vector.
        """
        return self.embed_batch([text])[0]


class OllamaEmbedder(Embedder):
    """
    Embedding provider using a locally running Ollama instance.

    Args:
        model (str): The Ollama embedding model name.
    """

    def __init__(self, model: str = "nomic-embed-text") -> None:
        self.model = model

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed texts using Ollama's embed endpoint.

        Args:
            texts (list[str]): List of text strings to embed.

        Returns:
            list[list[float]]: List of embedding vectors.

        Raises:
            ollama.ResponseError: If the Ollama server returns an error.
        """
        response = ollama.embed(model=self.model, input=texts)
        return response["embeddings"]


class SentenceTransformerEmbedder(Embedder):
    """
    Embedding provider using the sentence-transformers library.

    Useful as a fallback when Ollama is not available.

    Args:
        model (str): The sentence-transformers model name.
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed texts using sentence-transformers.

        Args:
            texts (list[str]): List of text strings to embed.

        Returns:
            list[list[float]]: List of embedding vectors.
        """
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


def get_embedder(
    provider: str = "ollama",
    model: str | None = None,
) -> Embedder:
    """
    Factory function to create the appropriate embedder.

    Args:
        provider (str): Either "ollama" or "sentence_transformers".
        model (str | None): Model name override. Uses provider defaults if None.

    Returns:
        Embedder: An initialized embedder instance.

    Raises:
        ValueError: If the provider is not recognized.
    """
    if provider == "ollama":
        return OllamaEmbedder(model=model or "nomic-embed-text")
    elif provider == "sentence_transformers":
        return SentenceTransformerEmbedder(model=model or "all-MiniLM-L6-v2")
    else:
        raise ValueError(
            f"Unknown embedding provider: {provider}. "
            "Use 'ollama' or 'sentence_transformers'."
        )
