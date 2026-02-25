"""
Route handlers for the OpenAI-compatible Chat Completions API.

Implements /v1/models and /v1/chat/completions endpoints that delegate
to the existing RAG pipeline (rag/chain.py) for retrieval and generation.
"""

import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChunk,
    ChatMessage,
    Choice,
    ModelEntry,
    ModelListResponse,
    StreamChoice,
    StreamDelta,
)
from config.settings import Settings
from rag.chain import query_rag, query_rag_stream

router = APIRouter()


def _extract_query_and_history(
    messages: list[ChatMessage],
) -> tuple[str, list[dict]]:
    """
    Extract the RAG query and chat history from incoming messages.

    The last user message becomes the RAG query. All prior user/assistant
    messages become chat_history. System messages from the client are
    stripped because the RAG pipeline injects its own context-aware
    system prompt.

    Args:
        messages (list[ChatMessage]): The incoming conversation messages.

    Returns:
        tuple[str, list[dict]]: (query, chat_history) where query is the
            last user message and chat_history is prior user/assistant turns.

    Raises:
        ValueError: If no user message is found in the messages list.
    """
    # Filter out system messages — the RAG pipeline builds its own
    non_system = [m for m in messages if m.role != "system"]

    # Find the last user message as the query
    query = ""
    query_index = -1
    for i in range(len(non_system) - 1, -1, -1):
        if non_system[i].role == "user":
            query = non_system[i].content
            query_index = i
            break

    if not query:
        raise ValueError("No user message found in the request.")

    # Everything before the query becomes chat history
    chat_history = []
    for msg in non_system[:query_index]:
        chat_history.append({"role": msg.role, "content": msg.content})

    return query, chat_history


@router.get("/v1/models")
async def list_models(request: Request) -> ModelListResponse:
    """
    List available models.

    Returns a single model entry representing the local RAG pipeline.
    The model name includes the configured LLM provider for clarity.

    Args:
        request (Request): The incoming HTTP request (provides app state).

    Returns:
        ModelListResponse: A list containing the single RAG model entry.
    """
    settings: Settings = request.app.state.settings
    provider = settings.llm_provider
    model_id = f"local-rag ({provider})"

    return ModelListResponse(
        data=[ModelEntry(id=model_id)]
    )


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
) -> ChatCompletionResponse | StreamingResponse:
    """
    Handle chat completion requests (streaming and non-streaming).

    Extracts the user query and chat history from the OpenAI-format
    messages, runs the RAG pipeline, and returns the response in
    OpenAI-compatible format.

    Args:
        body (ChatCompletionRequest): The chat completion request body.
        request (Request): The incoming HTTP request (provides app state).

    Returns:
        ChatCompletionResponse | StreamingResponse: The completion response,
            either as a single JSON object or an SSE stream.

    Raises:
        HTTPException: 400 if no user message is found in the request.
    """
    try:
        query, chat_history = _extract_query_and_history(body.messages)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    settings: Settings = request.app.state.settings
    embedder = request.app.state.embedder
    store = request.app.state.store
    llm_client = request.app.state.llm_client

    # Apply request temperature if provided, otherwise use settings default
    effective_settings = settings
    if body.temperature is not None:
        effective_settings = settings.model_copy(
            update={"temperature": body.temperature}
        )

    if body.stream:
        return _stream_response(
            query, chat_history, embedder, store, llm_client, effective_settings
        )

    # Non-streaming response
    result = query_rag(
        query=query,
        embedder=embedder,
        store=store,
        llm_client=llm_client,
        settings=effective_settings,
        chat_history=chat_history or None,
    )

    provider = settings.llm_provider
    model_id = f"local-rag ({provider})"

    return ChatCompletionResponse(
        model=model_id,
        choices=[
            Choice(
                message=ChatMessage(role="assistant", content=result["answer"]),
            )
        ],
    )


def _stream_response(
    query: str,
    chat_history: list[dict],
    embedder: object,
    store: object,
    llm_client: object,
    settings: Settings,
) -> StreamingResponse:
    """
    Create a streaming SSE response for chat completions.

    Wraps the synchronous RAG stream generator in an SSE-formatted
    async generator that yields `data: {json}` lines.

    Args:
        query (str): The user's question.
        chat_history (list[dict]): Prior conversation turns.
        embedder: The embedding provider.
        store: The vector store.
        llm_client: The LLM client.
        settings (Settings): Application configuration.

    Returns:
        StreamingResponse: An SSE streaming response.
    """
    provider = settings.llm_provider
    model_id = f"local-rag ({provider})"
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    def generate():
        """
        Generator that yields SSE-formatted chunks.

        Yields:
            str: SSE data lines in the format 'data: {json}\\n\\n'.
        """
        stream, _sources = query_rag_stream(
            query=query,
            embedder=embedder,
            store=store,
            llm_client=llm_client,
            settings=settings,
            chat_history=chat_history or None,
        )

        # First chunk with role
        first_chunk = ChatCompletionStreamChunk(
            id=completion_id,
            created=created,
            model=model_id,
            choices=[
                StreamChoice(
                    delta=StreamDelta(role="assistant", content=""),
                )
            ],
        )
        yield f"data: {first_chunk.model_dump_json()}\n\n"

        # Content chunks
        for token in stream:
            chunk = ChatCompletionStreamChunk(
                id=completion_id,
                created=created,
                model=model_id,
                choices=[
                    StreamChoice(
                        delta=StreamDelta(content=token),
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        # Final chunk with finish_reason
        final_chunk = ChatCompletionStreamChunk(
            id=completion_id,
            created=created,
            model=model_id,
            choices=[
                StreamChoice(
                    delta=StreamDelta(),
                    finish_reason="stop",
                )
            ],
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
