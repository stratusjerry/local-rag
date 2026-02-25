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
    embedding_provider: str = "sentence_transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_provider: str = "lmstudio"
    llm_model: str = "llama3.2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    temperature: float = 0.7

    # LM Studio
    lmstudio_url: str = "http://localhost:1234/v1"

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = ""

    # AWS Bedrock
    bedrock_region: str = "us-east-1"
    bedrock_model_id: str = "us.anthropic.claude-sonnet-4-6"
    bedrock_api_key: str = ""
    bedrock_endpoint_url: str = ""
    bedrock_ca_bundle: str = ""

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
