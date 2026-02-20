"""
LLM client package.

Provides a factory function to create the appropriate LLM client
based on the configured provider.
"""

from llm.base import LLMClient


def get_llm_client(
    provider: str = "ollama",
    lmstudio_url: str = "http://localhost:1234/v1",
    bedrock_region: str = "us-east-1",
    bedrock_access_key: str = "",
    bedrock_secret_key: str = "",
    bedrock_session_token: str = "",
) -> LLMClient:
    """
    Factory function to create the appropriate LLM client.

    Args:
        provider (str): One of "ollama", "lmstudio", or "bedrock".
        lmstudio_url (str): LM Studio API base URL (only used when provider is "lmstudio").
        bedrock_region (str): AWS region (only used when provider is "bedrock").
        bedrock_access_key (str): AWS access key ID (only used when provider is "bedrock").
        bedrock_secret_key (str): AWS secret access key (only used when provider is "bedrock").
        bedrock_session_token (str): AWS session token (only used when provider is "bedrock").

    Returns:
        LLMClient: An initialized LLM client instance.

    Raises:
        ValueError: If the provider is not recognized.
    """
    if provider == "ollama":
        from llm.ollama_client import OllamaLLM

        return OllamaLLM()

    elif provider == "lmstudio":
        from llm.lmstudio_client import LMStudioLLM

        return LMStudioLLM(base_url=lmstudio_url)

    elif provider == "bedrock":
        from llm.bedrock_client import BedrockLLM

        return BedrockLLM(
            region=bedrock_region,
            access_key=bedrock_access_key,
            secret_key=bedrock_secret_key,
            session_token=bedrock_session_token,
        )

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            "Use 'ollama', 'lmstudio', or 'bedrock'."
        )
