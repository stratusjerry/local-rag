"""
AWS Bedrock LLM client implementation.

Uses the Bedrock Runtime Converse API to call Claude models hosted
on AWS. Supports both static credentials (access key + secret + optional
session token) and default credential chain (IAM roles, env vars, etc.).
"""

import json
from collections.abc import Generator

import boto3

from llm.base import LLMClient

DEFAULT_BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-5-v1"


def _messages_to_bedrock_format(
    messages: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Convert standard chat messages to Bedrock Converse API format.

    Separates the system prompt from conversation turns and converts
    content to the Bedrock content block format.

    Args:
        messages (list[dict]): Standard chat messages with role/content keys.

    Returns:
        tuple: (system_prompts, conversation_messages) in Bedrock format.
    """
    system_prompts = []
    conversation = []

    for msg in messages:
        if msg["role"] == "system":
            system_prompts.append({"text": msg["content"]})
        else:
            conversation.append({
                "role": msg["role"],
                "content": [{"text": msg["content"]}],
            })

    return system_prompts, conversation


class BedrockLLM(LLMClient):
    """
    LLM client for AWS Bedrock using the Converse API.

    Authenticates via explicit credentials or the default AWS credential
    chain (environment variables, IAM role, ~/.aws/credentials, etc.).

    Args:
        region (str): AWS region for the Bedrock endpoint.
        access_key (str): AWS access key ID. Empty string uses default chain.
        secret_key (str): AWS secret access key. Empty string uses default chain.
        session_token (str): Optional AWS session token for temporary credentials.
    """

    def __init__(
        self,
        region: str = "us-east-1",
        access_key: str = "",
        secret_key: str = "",
        session_token: str = "",
    ) -> None:
        kwargs: dict = {"region_name": region}

        if access_key and secret_key:
            kwargs["aws_access_key_id"] = access_key
            kwargs["aws_secret_access_key"] = secret_key
            if session_token:
                kwargs["aws_session_token"] = session_token

        self._client = boto3.client("bedrock-runtime", **kwargs)

    def check_connectivity(self) -> tuple[bool, str]:
        """
        Verify that Bedrock is reachable with the configured credentials.

        Makes a lightweight list-foundation-models call to validate access.

        Returns:
            tuple[bool, str]: (is_connected, status_message).
        """
        try:
            # Reason: bedrock-runtime doesn't have a simple list/ping endpoint,
            # so we create a separate bedrock client just for the connectivity check.
            bedrock = boto3.client(
                "bedrock",
                region_name=self._client.meta.region_name,
                aws_access_key_id=self._client._request_signer._credentials.access_key
                if self._client._request_signer._credentials
                else None,
                aws_secret_access_key=self._client._request_signer._credentials.secret_key
                if self._client._request_signer._credentials
                else None,
            )
            bedrock.list_foundation_models(maxResults=1)
            return True, "AWS Bedrock is connected."
        except Exception as e:
            return False, f"Cannot connect to AWS Bedrock: {e}"

    def chat(
        self,
        messages: list[dict],
        model: str = DEFAULT_BEDROCK_MODEL,
        temperature: float = 0.7,
    ) -> str:
        """
        Send a chat request to Bedrock and return the full response.

        Args:
            messages (list[dict]): Conversation messages.
            model (str): The Bedrock model ID.
            temperature (float): Sampling temperature.

        Returns:
            str: The assistant's response text.
        """
        system_prompts, conversation = _messages_to_bedrock_format(messages)

        response = self._client.converse(
            modelId=model,
            system=system_prompts,
            messages=conversation,
            inferenceConfig={"temperature": temperature},
        )

        return response["output"]["message"]["content"][0]["text"]

    def chat_stream(
        self,
        messages: list[dict],
        model: str = DEFAULT_BEDROCK_MODEL,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """
        Send a streaming chat request to Bedrock, yielding response chunks.

        Args:
            messages (list[dict]): Conversation messages.
            model (str): The Bedrock model ID.
            temperature (float): Sampling temperature.

        Yields:
            str: Individual text chunks of the response.
        """
        system_prompts, conversation = _messages_to_bedrock_format(messages)

        response = self._client.converse_stream(
            modelId=model,
            system=system_prompts,
            messages=conversation,
            inferenceConfig={"temperature": temperature},
        )

        for event in response["stream"]:
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {})
                text = delta.get("text", "")
                if text:
                    yield text
