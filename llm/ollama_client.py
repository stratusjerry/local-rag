"""
Thin wrapper around the Ollama Python client for LLM chat operations.

Provides connectivity checking and both streaming and non-streaming
chat interfaces.
"""

from collections.abc import Generator

import ollama


def check_connectivity() -> tuple[bool, str]:
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


def check_model_available(model: str) -> tuple[bool, str]:
    """
    Check if a specific model is available in Ollama.

    Args:
        model (str): The model name to check.

    Returns:
        tuple[bool, str]: (is_available, status_message).
    """
    try:
        models = ollama.list()
        model_names = [m.model for m in models.models]
        # Reason: model names in Ollama may include a ":latest" tag,
        # so we check both exact match and with the tag appended.
        if model in model_names or f"{model}:latest" in model_names:
            return True, f"Model '{model}' is available."
        return False, (
            f"Model '{model}' not found. "
            f"Available models: {', '.join(model_names)}. "
            f"Run 'ollama pull {model}' to download it."
        )
    except Exception as e:
        return False, f"Cannot check models: {e}"


def chat(
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
