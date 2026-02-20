"""
AWS Bedrock LLM client implementation.

Uses the Bedrock Runtime Converse API to call Claude models hosted
on AWS. Supports static credentials, default credential chain,
custom endpoint URLs, and custom CA bundles for isolated regions.
"""

from collections.abc import Generator

import boto3
from botocore.config import Config

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


def _build_boto3_kwargs(
    region: str,
    access_key: str,
    secret_key: str,
    session_token: str,
    endpoint_url: str,
    ca_bundle: str,
) -> dict:
    """
    Build the keyword arguments dict for boto3.client().

    Args:
        region (str): AWS region name.
        access_key (str): AWS access key ID (empty = default chain).
        secret_key (str): AWS secret access key (empty = default chain).
        session_token (str): AWS session token (empty = omitted).
        endpoint_url (str): Custom endpoint URL (empty = default).
        ca_bundle (str): Path to a custom CA bundle .pem file (empty = default).

    Returns:
        dict: kwargs ready to pass to boto3.client().
    """
    kwargs: dict = {"region_name": region}

    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key
        if session_token:
            kwargs["aws_session_token"] = session_token

    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url

    if ca_bundle:
        kwargs["verify"] = ca_bundle

    return kwargs


class BedrockLLM(LLMClient):
    """
    LLM client for AWS Bedrock using the Converse API.

    Authenticates via explicit credentials or the default AWS credential
    chain. Supports custom endpoint URLs and CA bundles for isolated
    or air-gapped regions.

    Args:
        region (str): AWS region for the Bedrock endpoint.
        access_key (str): AWS access key ID. Empty string uses default chain.
        secret_key (str): AWS secret access key. Empty string uses default chain.
        session_token (str): Optional AWS session token for temporary credentials.
        endpoint_url (str): Custom Bedrock Runtime endpoint URL. Empty uses default.
        ca_bundle (str): Path to a custom CA bundle .pem file. Empty uses system default.
    """

    def __init__(
        self,
        region: str = "us-east-1",
        access_key: str = "",
        secret_key: str = "",
        session_token: str = "",
        endpoint_url: str = "",
        ca_bundle: str = "",
    ) -> None:
        self._boto3_kwargs = _build_boto3_kwargs(
            region=region,
            access_key=access_key,
            secret_key=secret_key,
            session_token=session_token,
            endpoint_url=endpoint_url,
            ca_bundle=ca_bundle,
        )
        self._client = boto3.client("bedrock-runtime", **self._boto3_kwargs)

    def check_connectivity(self) -> tuple[bool, str]:
        """
        Verify that Bedrock is reachable with the configured credentials.

        Makes a lightweight list-foundation-models call to validate access.
        Uses the same endpoint/CA settings as the runtime client.

        Returns:
            tuple[bool, str]: (is_connected, status_message).
        """
        try:
            # Reason: bedrock-runtime doesn't have a list/ping endpoint,
            # so we create a bedrock (control plane) client with the same
            # connection settings for the connectivity check.
            control_kwargs = dict(self._boto3_kwargs)
            # Reason: the control plane endpoint differs from the runtime
            # endpoint, so we drop endpoint_url and let boto3 resolve it
            # unless the user is in an isolated region where both share
            # a custom base URL.
            control_kwargs.pop("endpoint_url", None)
            bedrock = boto3.client("bedrock", **control_kwargs)
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
