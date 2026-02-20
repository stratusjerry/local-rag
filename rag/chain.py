"""
RAG query chain: embed query → retrieve → build prompt → call LLM.

Orchestrates the full retrieval-augmented generation pipeline from
user query to sourced answer.
"""

from collections.abc import Generator

from config.settings import Settings
from embeddings.embedder import Embedder
from llm.base import LLMClient
from rag.prompts import build_messages, format_context
from vectorstore.store import QueryResult, VectorStore


def retrieve(
    query: str,
    embedder: Embedder,
    store: VectorStore,
    top_k: int = 5,
) -> list[QueryResult]:
    """
    Embed a query and retrieve the most similar document chunks.

    Args:
        query (str): The user's question.
        embedder (Embedder): Embedding provider for the query.
        store (VectorStore): Vector store to search.
        top_k (int): Number of results to retrieve.

    Returns:
        list[QueryResult]: Retrieved chunks ranked by similarity.
    """
    query_embedding = embedder.embed_query(query)
    return store.query(query_embedding=query_embedding, n_results=top_k)


def query_rag(
    query: str,
    embedder: Embedder,
    store: VectorStore,
    llm_client: LLMClient,
    settings: Settings,
    chat_history: list[dict] | None = None,
) -> dict:
    """
    Run the full RAG pipeline: retrieve context, build prompt, call LLM.

    Args:
        query (str): The user's question.
        embedder (Embedder): Embedding provider.
        store (VectorStore): Vector store for retrieval.
        llm_client (LLMClient): LLM provider for generating answers.
        settings (Settings): Application configuration.
        chat_history (list[dict] | None): Previous conversation messages.

    Returns:
        dict: {"answer": str, "sources": list[QueryResult]}.
    """
    results = retrieve(query, embedder, store, top_k=settings.top_k)
    context = format_context(results)
    messages = build_messages(query, context, chat_history)

    model = _resolve_model(settings)
    answer = llm_client.chat(
        messages=messages,
        model=model,
        temperature=settings.temperature,
    )

    return {"answer": answer, "sources": results}


def query_rag_stream(
    query: str,
    embedder: Embedder,
    store: VectorStore,
    llm_client: LLMClient,
    settings: Settings,
    chat_history: list[dict] | None = None,
) -> tuple[Generator[str, None, None], list[QueryResult]]:
    """
    Run the RAG pipeline with streaming LLM output.

    Retrieval happens upfront; the LLM response is streamed. Returns both
    the stream generator and the sources so the UI can display attribution
    after streaming completes.

    Args:
        query (str): The user's question.
        embedder (Embedder): Embedding provider.
        store (VectorStore): Vector store for retrieval.
        llm_client (LLMClient): LLM provider for generating answers.
        settings (Settings): Application configuration.
        chat_history (list[dict] | None): Previous conversation messages.

    Returns:
        tuple: (response_stream, sources) where response_stream is a
            Generator[str] and sources is list[QueryResult].
    """
    results = retrieve(query, embedder, store, top_k=settings.top_k)
    context = format_context(results)
    messages = build_messages(query, context, chat_history)

    model = _resolve_model(settings)
    stream = llm_client.chat_stream(
        messages=messages,
        model=model,
        temperature=settings.temperature,
    )

    return stream, results


def _resolve_model(settings: Settings) -> str:
    """
    Return the model identifier based on the active LLM provider.

    Bedrock uses its own model ID field; other providers use llm_model.

    Args:
        settings (Settings): Application configuration.

    Returns:
        str: The model identifier to pass to the LLM client.
    """
    if settings.llm_provider == "bedrock":
        return settings.bedrock_model_id
    return settings.llm_model
