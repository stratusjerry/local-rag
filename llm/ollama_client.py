"""
Ollama LLM client implementation.

Wraps the Ollama Python client for local LLM chat with connectivity
checking and streaming support.
"""

from collections.abc import Generator

import ollama

from llm.base import LLMClient


class OllamaLLM(LLMClient):
    """
    LLM client using a locally running Ollama instance.
    """

    def check_connectivity(self) -> tuple[bool, str]:
        """
        Verify that Ollama is running and reachable.

        Returns:
            tuple[bool, str]: (is_connected, status_message).
        """
        try:
            ollama.list()
            return True, "Ollama is running."
        except Exception as e:
            return False, f"Cannot connect to Ollama: {e}"

    def chat(
        self,
        messages: list[dict],
        model: str = "llama3.2",
        temperature: float = 0.7,
    ) -> str:
        """
        Send a chat request to Ollama and return the full response.

        Args:
            messages (list[dict]): Conversation messages.
            model (str): The Ollama model to use.
            temperature (float): Sampling temperature.

        Returns:
            str: The assistant's response text.
        """
        response = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature},
        )
        return response["message"]["content"]

    def chat_stream(
        self,
        messages: list[dict],
        model: str = "llama3.2",
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """
        Send a streaming chat request to Ollama, yielding response chunks.

        Args:
            messages (list[dict]): Conversation messages.
            model (str): The Ollama model to use.
            temperature (float): Sampling temperature.

        Yields:
            str: Individual text chunks of the response.
        """
        stream = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature},
            stream=True,
        )
        for chunk in stream:
            content = chunk["message"]["content"]
            if content:
                yield content
