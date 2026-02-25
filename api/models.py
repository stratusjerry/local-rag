"""
Pydantic schemas for the OpenAI-compatible Chat Completions API.

Defines request and response models that match the OpenAI API format,
allowing Open WebUI and other compatible clients to communicate with
the local RAG pipeline.
"""

import time
import uuid

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """
    A single message in the chat conversation.

    Attributes:
        role: The role of the message sender (system, user, assistant).
        content: The text content of the message.
    """

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """
    Request body for the /v1/chat/completions endpoint.

    Follows the OpenAI Chat Completions API format. Extra fields sent by
    Open WebUI (e.g., max_tokens, top_p) are accepted but ignored since
    the RAG pipeline controls its own generation parameters.

    Attributes:
        model: Model identifier (ignored — the RAG pipeline uses its configured model).
        messages: List of conversation messages.
        temperature: Sampling temperature (overrides settings if provided).
        stream: Whether to stream the response via SSE.
    """

    model: str = "local-rag"
    messages: list[ChatMessage]
    temperature: float | None = None
    stream: bool = False

    model_config = {"extra": "allow"}


class Choice(BaseModel):
    """
    A single choice in a chat completion response.

    Attributes:
        index: The index of this choice.
        message: The assistant's response message.
        finish_reason: Why the model stopped generating (e.g., "stop").
    """

    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    """
    Token usage statistics (placeholder values for compatibility).

    Attributes:
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the completion.
        total_tokens: Total tokens used.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """
    Response body for non-streaming /v1/chat/completions requests.

    Follows the OpenAI Chat Completions API response format.

    Attributes:
        id: Unique identifier for this completion.
        object: Object type (always "chat.completion").
        created: Unix timestamp of when the completion was created.
        model: The model used for generation.
        choices: List of completion choices.
        usage: Token usage statistics.
    """

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "local-rag"
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)


class StreamDelta(BaseModel):
    """
    Delta content for a streaming chunk.

    Attributes:
        role: The role (only set in the first chunk).
        content: The text content of this chunk.
    """

    role: str | None = None
    content: str | None = None


class StreamChoice(BaseModel):
    """
    A single choice in a streaming chat completion chunk.

    Attributes:
        index: The index of this choice.
        delta: The delta content for this chunk.
        finish_reason: Why the model stopped (None while streaming, "stop" at end).
    """

    index: int = 0
    delta: StreamDelta
    finish_reason: str | None = None


class ChatCompletionStreamChunk(BaseModel):
    """
    A single chunk in a streaming chat completion response.

    Attributes:
        id: Unique identifier (same across all chunks of one completion).
        object: Object type (always "chat.completion.chunk").
        created: Unix timestamp.
        model: The model used.
        choices: List of streaming choices.
    """

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "local-rag"
    choices: list[StreamChoice]


class ModelEntry(BaseModel):
    """
    A single model entry for the /v1/models response.

    Attributes:
        id: The model identifier.
        object: Object type (always "model").
        created: Unix timestamp.
        owned_by: The owner of the model.
    """

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "local-rag"


class ModelListResponse(BaseModel):
    """
    Response body for the /v1/models endpoint.

    Attributes:
        object: Object type (always "list").
        data: List of available models.
    """

    object: str = "list"
    data: list[ModelEntry]
