"""Tests for the OpenAI-compatible RAG API server."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.models import ChatMessage
from api.routes import _extract_query_and_history, router
from api.server import AuthMiddleware, create_app
from config.settings import Settings
from vectorstore.store import QueryResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_app(
    settings: Settings | None = None,
    embedder: object | None = None,
    store: object | None = None,
    llm_client: object | None = None,
) -> FastAPI:
    """
    Build a FastAPI app with mocked dependencies for testing.

    Bypasses the lifespan handler by directly setting app.state attributes.

    Args:
        settings (Settings | None): Application settings override.
        embedder: Mock embedder instance.
        store: Mock vector store instance.
        llm_client: Mock LLM client instance.

    Returns:
        FastAPI: A configured test app with mocked dependencies.
    """
    app = FastAPI()
    app.include_router(router)

    test_settings = settings or Settings()
    app.state.settings = test_settings
    app.state.embedder = embedder or MagicMock()
    app.state.store = store or MagicMock()
    app.state.llm_client = llm_client or MagicMock()

    return app


# ---------------------------------------------------------------------------
# _extract_query_and_history tests
# ---------------------------------------------------------------------------


class TestExtractQueryAndHistory:
    """Tests for the message extraction helper."""

    def test_single_user_message(self) -> None:
        """A single user message should be the query with no history."""
        messages = [ChatMessage(role="user", content="What is RAG?")]
        query, history = _extract_query_and_history(messages)

        assert query == "What is RAG?"
        assert history == []

    def test_strips_system_messages(self) -> None:
        """System messages from the client should be filtered out."""
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Hello"),
        ]
        query, history = _extract_query_and_history(messages)

        assert query == "Hello"
        assert history == []

    def test_multi_turn_conversation(self) -> None:
        """Prior user/assistant turns should become chat_history."""
        messages = [
            ChatMessage(role="system", content="System prompt"),
            ChatMessage(role="user", content="First question"),
            ChatMessage(role="assistant", content="First answer"),
            ChatMessage(role="user", content="Follow-up question"),
        ]
        query, history = _extract_query_and_history(messages)

        assert query == "Follow-up question"
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "First question"}
        assert history[1] == {"role": "assistant", "content": "First answer"}

    def test_no_user_message_raises(self) -> None:
        """Should raise ValueError when no user message is found."""
        messages = [ChatMessage(role="system", content="System only")]

        with pytest.raises(ValueError, match="No user message found"):
            _extract_query_and_history(messages)

    def test_empty_messages_raises(self) -> None:
        """Should raise ValueError when messages list is empty."""
        with pytest.raises(ValueError, match="No user message found"):
            _extract_query_and_history([])


# ---------------------------------------------------------------------------
# /v1/models tests
# ---------------------------------------------------------------------------


class TestModelsEndpoint:
    """Tests for the /v1/models endpoint."""

    def test_returns_model_list(self) -> None:
        """Should return a list with one model entry."""
        app = _make_test_app(settings=Settings(llm_provider="ollama"))
        client = TestClient(app)

        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "local-rag (ollama)"
        assert data["data"][0]["object"] == "model"

    def test_model_name_includes_provider(self) -> None:
        """Model ID should reflect the configured LLM provider."""
        app = _make_test_app(settings=Settings(llm_provider="bedrock"))
        client = TestClient(app)

        response = client.get("/v1/models")

        data = response.json()
        assert "bedrock" in data["data"][0]["id"]


# ---------------------------------------------------------------------------
# /v1/chat/completions (non-streaming) tests
# ---------------------------------------------------------------------------


class TestChatCompletions:
    """Tests for the /v1/chat/completions endpoint (non-streaming)."""

    def test_valid_request_returns_completion(self) -> None:
        """A valid request should return a properly formatted completion."""
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

        app = _make_test_app(
            embedder=mock_embedder,
            store=mock_store,
            llm_client=mock_llm,
        )
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Summarize the docs"}],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "This is the answer."
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_missing_user_message_returns_400(self) -> None:
        """Request with only system messages should return 400."""
        app = _make_test_app()
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "system", "content": "You are helpful."}],
            },
        )

        assert response.status_code == 400
        assert "No user message found" in response.json()["detail"]

    def test_temperature_override(self) -> None:
        """Request temperature should override settings default."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "Answer."

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1, 0.2]

        mock_store = MagicMock()
        mock_store.query.return_value = []

        app = _make_test_app(
            settings=Settings(temperature=0.7),
            embedder=mock_embedder,
            store=mock_store,
            llm_client=mock_llm,
        )
        client = TestClient(app)

        client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Test"}],
                "temperature": 0.2,
            },
        )

        # The LLM should have been called with temperature 0.2
        call_kwargs = mock_llm.chat.call_args
        assert call_kwargs.kwargs["temperature"] == 0.2

    def test_extra_fields_accepted(self) -> None:
        """Extra fields like max_tokens should be accepted and ignored."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "Answer."

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1]

        mock_store = MagicMock()
        mock_store.query.return_value = []

        app = _make_test_app(
            embedder=mock_embedder,
            store=mock_store,
            llm_client=mock_llm,
        )
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": 1024,
                "top_p": 0.9,
            },
        )

        assert response.status_code == 200


# ---------------------------------------------------------------------------
# /v1/chat/completions (streaming) tests
# ---------------------------------------------------------------------------


class TestChatCompletionsStreaming:
    """Tests for the /v1/chat/completions endpoint (streaming)."""

    def test_streaming_response_format(self) -> None:
        """Streaming response should use SSE format ending with [DONE]."""
        mock_llm = MagicMock()
        mock_llm.chat_stream.return_value = iter(["Hello", " world"])

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1, 0.2]

        mock_store = MagicMock()
        mock_store.query.return_value = []

        app = _make_test_app(
            embedder=mock_embedder,
            store=mock_store,
            llm_client=mock_llm,
        )
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Test"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        # Parse SSE lines
        lines = response.text.strip().split("\n\n")
        assert len(lines) >= 3  # role chunk + content chunks + final + [DONE]

        # Last line should be [DONE]
        assert lines[-1].strip() == "data: [DONE]"

    def test_streaming_chunks_have_correct_object(self) -> None:
        """Each SSE chunk should have object type 'chat.completion.chunk'."""
        import json

        mock_llm = MagicMock()
        mock_llm.chat_stream.return_value = iter(["token"])

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1]

        mock_store = MagicMock()
        mock_store.query.return_value = []

        app = _make_test_app(
            embedder=mock_embedder,
            store=mock_store,
            llm_client=mock_llm,
        )
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Test"}],
                "stream": True,
            },
        )

        lines = response.text.strip().split("\n\n")
        # Check first data line (role chunk)
        first_data = lines[0].removeprefix("data: ")
        parsed = json.loads(first_data)
        assert parsed["object"] == "chat.completion.chunk"
        assert parsed["choices"][0]["delta"]["role"] == "assistant"


# ---------------------------------------------------------------------------
# Auth middleware tests
# ---------------------------------------------------------------------------


class TestAuthMiddleware:
    """Tests for the optional Bearer token authentication middleware."""

    def test_no_auth_when_key_empty(self) -> None:
        """Requests should pass through when no API key is configured."""
        app = _make_test_app(settings=Settings(api_key=""))
        client = TestClient(app)

        response = client.get("/v1/models")
        assert response.status_code == 200

    def test_valid_key_passes(self) -> None:
        """Valid Bearer token should allow the request through."""
        app = _make_test_app(settings=Settings(api_key="test-secret-key"))
        app.add_middleware(AuthMiddleware, api_key="test-secret-key")
        client = TestClient(app)

        response = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer test-secret-key"},
        )
        assert response.status_code == 200

    def test_invalid_key_returns_401(self) -> None:
        """Invalid Bearer token should return 401."""
        app = _make_test_app(settings=Settings(api_key="correct-key"))
        app.add_middleware(AuthMiddleware, api_key="correct-key")
        client = TestClient(app)

        response = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert response.status_code == 401

    def test_missing_auth_header_returns_401(self) -> None:
        """Missing Authorization header should return 401 when key is set."""
        app = _make_test_app(settings=Settings(api_key="my-key"))
        app.add_middleware(AuthMiddleware, api_key="my-key")
        client = TestClient(app)

        response = client.get("/v1/models")
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# /health endpoint tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    @patch("api.server.get_settings")
    @patch("api.server.get_embedder")
    @patch("api.server.VectorStore")
    @patch("api.server.get_llm_client")
    def test_health_returns_ok(
        self,
        mock_get_llm: MagicMock,
        mock_store_cls: MagicMock,
        mock_get_embedder: MagicMock,
        mock_get_settings: MagicMock,
    ) -> None:
        """Health endpoint should return status ok."""
        mock_get_settings.return_value = Settings()
        mock_get_embedder.return_value = MagicMock()
        mock_store_cls.return_value = MagicMock()
        mock_get_llm.return_value = MagicMock()

        app = create_app(settings=Settings())
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
