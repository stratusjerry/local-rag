"""Tests for the RAG chain module."""

from unittest.mock import MagicMock, patch

import pytest

from config.settings import Settings
from rag.chain import query_rag, retrieve
from rag.prompts import build_messages, format_context
from vectorstore.store import QueryResult


class TestFormatContext:
    """Tests for the context formatting function."""

    def test_format_with_results(self) -> None:
        """Context should include source attribution for each result."""
        results = [
            QueryResult(
                text="First chunk content.",
                metadata={"filename": "doc1.txt", "chunk_index": 0},
                distance=0.1,
            ),
            QueryResult(
                text="Second chunk content.",
                metadata={"filename": "doc2.docx", "chunk_index": 2},
                distance=0.3,
            ),
        ]

        context = format_context(results)

        assert "[Source 1: doc1.txt]" in context
        assert "First chunk content." in context
        assert "[Source 2: doc2.docx]" in context
        assert "Second chunk content." in context

    def test_format_empty_results(self) -> None:
        """Empty results should return a 'no documents' message."""
        context = format_context([])
        assert "No relevant documents found" in context

    def test_format_missing_filename(self) -> None:
        """Results with missing filename should show 'Unknown'."""
        results = [
            QueryResult(text="Some text.", metadata={}, distance=0.5),
        ]
        context = format_context(results)
        assert "Unknown" in context


class TestBuildMessages:
    """Tests for the message builder function."""

    def test_basic_messages(self) -> None:
        """Should produce system + user messages."""
        messages = build_messages("What is X?", "Context about X")

        assert messages[0]["role"] == "system"
        assert "Context about X" in messages[0]["content"]
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "What is X?"

    def test_with_chat_history(self) -> None:
        """Chat history should be inserted between system and user messages."""
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]

        messages = build_messages("New question", "Context", chat_history=history)

        assert len(messages) == 4  # system + 2 history + user
        assert messages[1]["content"] == "Previous question"
        assert messages[2]["content"] == "Previous answer"

    def test_without_chat_history(self) -> None:
        """Without history, should only have system + user messages."""
        messages = build_messages("Question", "Context")
        assert len(messages) == 2


class TestRetrieve:
    """Tests for the retrieve function."""

    def test_retrieve_calls_embedder_and_store(self) -> None:
        """Retrieve should embed the query and search the store."""
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]

        expected_result = QueryResult(
            text="Found text", metadata={"filename": "test.txt"}, distance=0.1
        )
        mock_store = MagicMock()
        mock_store.query.return_value = [expected_result]

        results = retrieve("test query", mock_embedder, mock_store, top_k=3)

        mock_embedder.embed_query.assert_called_once_with("test query")
        mock_store.query.assert_called_once_with(
            query_embedding=[0.1, 0.2, 0.3], n_results=3
        )
        assert len(results) == 1
        assert results[0].text == "Found text"


class TestQueryRag:
    """Tests for the full RAG query chain."""

    def test_query_rag_returns_answer_and_sources(self) -> None:
        """Full RAG query should return answer text and source list."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "This is the answer."

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]

        mock_store = MagicMock()
        mock_store.query.return_value = [
            QueryResult(
                text="Relevant chunk",
                metadata={"filename": "doc.txt"},
                distance=0.2,
            )
        ]

        settings = Settings(top_k=3, llm_model="llama3.2", temperature=0.5)

        result = query_rag(
            query="What is this about?",
            embedder=mock_embedder,
            store=mock_store,
            llm_client=mock_llm,
            settings=settings,
        )

        assert result["answer"] == "This is the answer."
        assert len(result["sources"]) == 1
        mock_llm.chat.assert_called_once()

    def test_query_rag_with_no_results(self) -> None:
        """RAG query with no retrieved results should still call the LLM."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "I couldn't find relevant information."

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.0, 0.0, 0.0]

        mock_store = MagicMock()
        mock_store.query.return_value = []

        settings = Settings()

        result = query_rag(
            query="Unknown question",
            embedder=mock_embedder,
            store=mock_store,
            llm_client=mock_llm,
            settings=settings,
        )

        assert "sources" in result
        assert len(result["sources"]) == 0

    def test_query_rag_bedrock_uses_bedrock_model_id(self) -> None:
        """Bedrock provider should pass bedrock_model_id as the model."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "Bedrock answer."

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]

        mock_store = MagicMock()
        mock_store.query.return_value = []

        settings = Settings(
            llm_provider="bedrock",
            bedrock_model_id="us.anthropic.claude-sonnet-4-6",
        )

        query_rag(
            query="Test",
            embedder=mock_embedder,
            store=mock_store,
            llm_client=mock_llm,
            settings=settings,
        )

        # Should have been called with the bedrock model ID, not llm_model
        call_kwargs = mock_llm.chat.call_args
        assert call_kwargs.kwargs["model"] == "us.anthropic.claude-sonnet-4-6"
