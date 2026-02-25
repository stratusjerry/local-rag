"""
FastAPI application factory for the OpenAI-compatible RAG API server.

Creates and configures the FastAPI app with lifespan startup (initializes
embedder, vector store, and LLM client), CORS middleware, optional Bearer
token authentication, and a health endpoint.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from api.routes import router
from config.settings import Settings, get_settings
from embeddings.embedder import get_embedder
from llm import get_llm_client
from vectorstore.store import VectorStore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler for startup/shutdown.

    Initializes the embedder, vector store, and LLM client on startup
    and stores them on app.state so route handlers can access them.
    Mirrors the Streamlit @st.cache_resource initialization pattern.

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        None: Control is yielded to the application during its lifetime.
    """
    settings = get_settings()

    # Initialize shared resources
    embedder = get_embedder(
        provider=settings.embedding_provider,
        model=settings.embedding_model,
    )
    store = VectorStore(
        persist_dir=settings.chroma_path,
        collection_name=settings.collection_name,
    )
    llm_client = get_llm_client(
        provider=settings.llm_provider,
        lmstudio_url=settings.lmstudio_url,
        bedrock_region=settings.bedrock_region,
        bedrock_api_key=settings.bedrock_api_key,
        bedrock_endpoint_url=settings.bedrock_endpoint_url,
        bedrock_ca_bundle=settings.bedrock_ca_bundle,
    )

    app.state.settings = settings
    app.state.embedder = embedder
    app.state.store = store
    app.state.llm_client = llm_client

    yield


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Optional Bearer token authentication middleware.

    When an API key is configured (RAG_API_KEY), all requests to /v1/
    endpoints must include a valid Authorization header. The /health
    endpoint is always accessible without authentication.

    Args:
        app: The ASGI application.
        api_key (str): The expected Bearer token. Empty string disables auth.
    """

    def __init__(self, app: object, api_key: str = "") -> None:
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next: object) -> Response:
        """
        Check the Authorization header for /v1/ routes when auth is enabled.

        Args:
            request (Request): The incoming HTTP request.
            call_next: The next middleware or route handler.

        Returns:
            Response: The HTTP response (401 if auth fails).
        """
        if self.api_key and request.url.path.startswith("/v1/"):
            auth_header = request.headers.get("Authorization", "")
            if auth_header != f"Bearer {self.api_key}":
                return Response(
                    content='{"detail":"Invalid or missing API key"}',
                    status_code=401,
                    media_type="application/json",
                )
        return await call_next(request)


def create_app(settings: Settings | None = None) -> FastAPI:
    """
    Build and configure the FastAPI application.

    Sets up CORS, optional auth middleware, the /health endpoint,
    and registers the API routes.

    Args:
        settings (Settings | None): Application settings. If None,
            settings are loaded from environment/dotenv during lifespan.

    Returns:
        FastAPI: The configured FastAPI application instance.
    """
    app = FastAPI(
        title="Local RAG API",
        description="OpenAI-compatible API for the local RAG pipeline",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS — allow all origins for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Optional Bearer token auth
    resolved_settings = settings or get_settings()
    if resolved_settings.api_key:
        app.add_middleware(AuthMiddleware, api_key=resolved_settings.api_key)

    @app.get("/health")
    async def health() -> dict:
        """
        Health check endpoint.

        Returns:
            dict: Status message indicating the API is running.
        """
        return {"status": "ok"}

    app.include_router(router)

    return app
