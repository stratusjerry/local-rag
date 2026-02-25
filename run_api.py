"""
Entry point for the OpenAI-compatible RAG API server.

Starts the FastAPI server using uvicorn. Configure host, port, and
optional API key via environment variables (RAG_API_HOST, RAG_API_PORT,
RAG_API_KEY) or the .env file.

Usage:
    python run_api.py
"""

import uvicorn

from config.settings import get_settings


def main() -> None:
    """
    Load settings and start the uvicorn server.

    Reads api_host and api_port from the application settings and
    launches the FastAPI app factory.
    """
    settings = get_settings()

    uvicorn.run(
        "api.server:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
