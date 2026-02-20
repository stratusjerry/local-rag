"""
LM Studio LLM client implementation.

LM Studio exposes an OpenAI-compatible API at a local endpoint.
This client uses the openai Python package to communicate with it.
"""

from collections.abc import Generator

from openai import OpenAI

from llm.base import LLMClient

DEFAULT_LMSTUDIO_URL = "http://localhost:1234/v1"


class LMStudioLLM(LLMClient):
    """
    LLM client for a locally running LM Studio instance.

    LM Studio serves models via an OpenAI-compatible REST API.

    Args:
        base_url (str): The LM Studio API base URL.
    """

    def __init__(self, base_url: str = DEFAULT_LMSTUDIO_URL) -> None:
        self._client = OpenAI(
            base_url=base_url,
            api_key="lm-studio",  # Reason: LM Studio ignores the key but the SDK requires one.
        )

    def check_connectivity(self) -> tuple[bool, str]:
        """
        Verify that LM Studio is running and reachable.

        Returns:
            tuple[bool, str]: (is_connected, status_message).
        """
        try:
            self._client.models.list()
            return True, "LM Studio is running."
        except Exception as e:
            return False, f"Cannot connect to LM Studio: {e}"

    def chat(
        self,
        messages: list[dict],
        model: str = "default",
        temperature: float = 0.7,
    ) -> str:
        """
        Send a chat request to LM Studio and return the full response.

        Args:
            messages (list[dict]): Conversation messages.
            model (str): The model identifier loaded in LM Studio.
            temperature (float): Sampling temperature.

        Returns:
            str: The assistant's response text.
        """
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content

    def chat_stream(
        self,
        messages: list[dict],
        model: str = "default",
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """
        Send a streaming chat request to LM Studio, yielding response chunks.

        Args:
            messages (list[dict]): Conversation messages.
            model (str): The model identifier loaded in LM Studio.
            temperature (float): Sampling temperature.

        Yields:
            str: Individual text chunks of the response.
        """
        stream = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
