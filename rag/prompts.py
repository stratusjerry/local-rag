"""
Prompt templates for the RAG query chain.

Contains the system prompt and context formatting logic for
grounding LLM responses in retrieved document chunks.
"""

from vectorstore.store import QueryResult

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context from local documents. Follow these rules:

1. Answer ONLY based on the provided context. If the context doesn't contain enough information to answer the question, say so clearly.
2. Cite the source document when referencing specific information.
3. Be concise and direct in your responses.
4. If the question is ambiguous, ask for clarification.

Context from documents:
{context}"""


def format_context(results: list[QueryResult]) -> str:
    """
    Format retrieved query results into a context string for the LLM prompt.

    Each chunk is labeled with its source filename and chunk index for
    attribution.

    Args:
        results (list[QueryResult]): Retrieved chunks from the vector store.

    Returns:
        str: Formatted context string with source attribution.
    """
    if not results:
        return "No relevant documents found."

    context_parts = []
    for i, result in enumerate(results, start=1):
        source = result.metadata.get("filename", "Unknown")
        context_parts.append(
            f"[Source {i}: {source}]\n{result.text}"
        )

    return "\n\n---\n\n".join(context_parts)


def build_messages(
    query: str,
    context: str,
    chat_history: list[dict] | None = None,
) -> list[dict]:
    """
    Build the message list for the LLM, including system prompt,
    chat history, and the current user query.

    Args:
        query (str): The user's current question.
        context (str): Formatted context from retrieved documents.
        chat_history (list[dict] | None): Previous conversation messages
            in the format [{"role": "user"|"assistant", "content": "..."}].

    Returns:
        list[dict]: Complete message list for the LLM API call.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
    ]

    if chat_history:
        messages.extend(chat_history)

    messages.append({"role": "user", "content": query})

    return messages
