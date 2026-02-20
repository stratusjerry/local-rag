"""
Abstract base class for LLM client providers.

All LLM providers (Ollama, LM Studio, Bedrock) implement this interface
so the RAG chain and UI can work with any backend interchangeably.
"""

from abc import ABC, abstractmethod
from collections.abc import Generator


class LLMClient(ABC):
    """
    Abstract base class for LLM chat providers.

    Subclasses must implement connectivity checking, synchronous chat,
    and streaming chat.
    """

    @abstractmethod
    def check_connectivity(self) -> tuple[bool, str]:
        """
        Verify that the LLM provider is reachable and configured.

        Returns:
            tuple[bool, str]: (is_connected, status_message).
        """
        ...

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.7,
    ) -> str:
        """
        Send a chat request and return the full response.

        Args:
            messages (list[dict]): Conversation messages.
            model (str): The model identifier.
            temperature (float): Sampling temperature.

        Returns:
            str: The assistant's response text.
        """
        ...

    @abstractmethod
    def chat_stream(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """
        Send a streaming chat request, yielding response chunks.

        Args:
            messages (list[dict]): Conversation messages.
            model (str): The model identifier.
            temperature (float): Sampling temperature.

        Yields:
            str: Individual text chunks of the response.
        """
        ...
