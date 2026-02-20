"""
AWS Bedrock LLM client implementation.

Uses the Bedrock Runtime Converse API to call Claude models hosted
on AWS. Authenticates via a Bedrock API key passed in the x-api-key
header. Supports custom endpoint URLs and CA bundles for isolated regions.
"""

from collections.abc import Generator

import boto3
from botocore import UNSIGNED
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


def _build_bedrock_client(
    service: str,
    region: str,
    api_key: str,
    endpoint_url: str,
    ca_bundle: str,
) -> boto3.client:
    """
    Build a boto3 client for a Bedrock service.

    When an API key is provided, SigV4 signing is disabled and the key
    is injected as an x-api-key header on every request. When no API key
    is provided, the default AWS credential chain is used.

    Args:
        service (str): The boto3 service name ("bedrock-runtime" or "bedrock").
        region (str): AWS region name.
        api_key (str): Bedrock API key (empty = use default AWS credential chain).
        endpoint_url (str): Custom endpoint URL (empty = default).
        ca_bundle (str): Path to a custom CA bundle .pem file (empty = default).

    Returns:
        boto3.client: Configured boto3 client.
    """
    kwargs: dict = {"region_name": region}

    if api_key:
        # Reason: Bedrock API key auth replaces SigV4 signing. We disable
        # signing entirely and inject the key via an event handler.
        kwargs["config"] = Config(signature_version=UNSIGNED)

    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url

    if ca_bundle:
        kwargs["verify"] = ca_bundle

    client = boto3.client(service, **kwargs)

    if api_key:
        def _inject_api_key(request, **kwargs):
            request.headers["x-api-key"] = api_key

        client.meta.events.register("before-send", _inject_api_key)

    return client


class BedrockLLM(LLMClient):
    """
    LLM client for AWS Bedrock using the Converse API.

    Authenticates via a Bedrock API key (passed as x-api-key header).
    Falls back to the default AWS credential chain if no key is provided.
    Supports custom endpoint URLs and CA bundles for isolated regions.

    Args:
        region (str): AWS region for the Bedrock endpoint.
        api_key (str): Bedrock API key. Empty string uses default AWS credential chain.
        endpoint_url (str): Custom Bedrock Runtime endpoint URL. Empty uses default.
        ca_bundle (str): Path to a custom CA bundle .pem file. Empty uses system default.
    """

    def __init__(
        self,
        region: str = "us-east-1",
        api_key: str = "",
        endpoint_url: str = "",
        ca_bundle: str = "",
    ) -> None:
        self._region = region
        self._api_key = api_key
        self._endpoint_url = endpoint_url
        self._ca_bundle = ca_bundle
        self._client = _build_bedrock_client(
            service="bedrock-runtime",
            region=region,
            api_key=api_key,
            endpoint_url=endpoint_url,
            ca_bundle=ca_bundle,
        )

    def check_connectivity(self) -> tuple[bool, str]:
        """
        Verify that Bedrock is reachable with the configured credentials.

        Makes a lightweight list-foundation-models call to validate access.
        Uses the same API key and CA settings as the runtime client.

        Returns:
            tuple[bool, str]: (is_connected, status_message).
        """
        try:
            control = _build_bedrock_client(
                service="bedrock",
                region=self._region,
                api_key=self._api_key,
                endpoint_url="",  # Control plane has its own endpoint
                ca_bundle=self._ca_bundle,
            )
            control.list_foundation_models(maxResults=1)
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
