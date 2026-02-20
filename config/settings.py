"""
Application settings using Pydantic BaseSettings.

Single source of truth for all configuration. Values are loaded from
environment variables (prefixed with RAG_) and .env file.
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    RAG application configuration.

    All settings can be overridden via environment variables
    with the RAG_ prefix (e.g., RAG_DOCUMENTS_DIR).
    """

    documents_dir: str = "files"
    chroma_db_path: str = "output/chromadb"
    collection_name: str = "local_rag"
    embedding_provider: str = "ollama"
    embedding_model: str = "nomic-embed-text"
    llm_model: str = "llama3.2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    temperature: float = 0.7

    model_config = {
        "env_prefix": "RAG_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }

    @property
    def documents_path(self) -> Path:
        """
        Return the documents directory as an absolute Path.

        Returns:
            Path: Absolute path to the documents directory.
        """
        return Path(self.documents_dir).resolve()

    @property
    def chroma_path(self) -> Path:
        """
        Return the ChromaDB storage directory as an absolute Path.

        Returns:
            Path: Absolute path to the ChromaDB directory.
        """
        return Path(self.chroma_db_path).resolve()


def get_settings() -> Settings:
    """
    Create and return a Settings instance.

    Returns:
        Settings: Application configuration loaded from env/.env.
    """
    return Settings()
