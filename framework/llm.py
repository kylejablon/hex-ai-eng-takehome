"""OpenRouter API client for LLM inference.

This module provides a client for the OpenRouter API, supporting
streaming chat completions with tool calling.
"""

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

# Set up logging for rate limit retries (logs to stderr when retrying)
_logger = logging.getLogger(__name__)
if not _logger.handlers:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    _logger.setLevel(logging.WARNING)


def _is_rate_limit_error(exception: BaseException) -> bool:
    """Check if the exception is a rate limit (429) error."""
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code == 429
    return False


def _is_read_timeout_error(exception: BaseException) -> bool:
    """Check if the exception is a read timeout error."""
    return isinstance(exception, httpx.ReadTimeout)


def _is_retryable_error(exception: BaseException) -> bool:
    """Check if the exception is retryable (rate limit or read timeout)."""
    return _is_rate_limit_error(exception) or _is_read_timeout_error(exception)


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class OpenRouterConfig:
    """Configuration for the OpenRouter API client.

    Defaults to gpt-oss-120b on Cerebras provider.
    """

    model: str = "openai/gpt-oss-120b:nitro"
    api_key: str = ""
    max_tokens: int = 100000
    # Temperature 0 => most reliable tool-call JSON (gpt-oss-120b otherwise emits
    # malformed argument JSON for SQL-bearing args) and less over-exploration.
    temperature: float = 0.0
    max_iterations: int = 30
    # Provider routing - no restriction by default
    provider: dict[str, Any] = field(default_factory=dict)
    # Reasoning configuration
    reasoning: dict[str, Any] | None = None
    # Timeout for receiving first token (also applies between subsequent chunks)
    # If no data is received within this time, the request will timeout
    first_token_timeout: float = 10.0
    # Context compression settings (reduces token usage on long conversations)
    compress_context: bool = False  # Enable context compression
    compress_keep_recent: int = 3  # Number of recent tool results to keep in full
    compress_max_chars: int = 150  # Max chars for truncated older results


@dataclass
class TokenUsage:
    """Token usage statistics from an API call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens used (prompt + completion)."""
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Add two TokenUsage instances together."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


@dataclass
class StreamChunk:
    """A single chunk from a streaming response."""

    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    reasoning_details: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    usage: TokenUsage | None = None


class OpenRouterClient:
    """Client for the OpenRouter API with streaming support."""

    def __init__(self, config: OpenRouterConfig) -> None:
        """Initialize the client with configuration."""
        self.config = config
        if not config.api_key:
            raise ValueError("OpenRouter API key is required")

        self._client = httpx.Client(
            timeout=httpx.Timeout(
                300.0,  # Overall timeout
                connect=30.0,  # Connection establishment timeout
                read=config.first_token_timeout,  # Time to wait for data between chunks
            ),
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/hex-inc/takehome",
                "X-Title": "Hex Takehome Agent",
            },
        )

    def _build_request_body(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = True,
    ) -> dict[str, Any]:
        """Build the request body for the API call."""
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": stream,
        }

        # Request usage data in streaming mode (OpenAI-compatible extension)
        if stream:
            body["stream_options"] = {"include_usage": True}

        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        if self.config.provider:
            body["provider"] = self.config.provider

        if self.config.reasoning:
            body["reasoning"] = self.config.reasoning

        return body

    def chat_completion_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[StreamChunk]:
        """Make a streaming chat completion request.

        Yields StreamChunk objects as they arrive via SSE.
        """
        body = self._build_request_body(messages, tools, stream=True)

        # Use Retrying for retryable errors (rate limits, read timeouts) with streaming
        retryer = Retrying(
            retry=retry_if_exception(_is_retryable_error),
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            reraise=True,
        )

        try:
            for attempt in retryer:
                with attempt:
                    with self._client.stream(
                        "POST", OPENROUTER_API_URL, json=body
                    ) as response:
                        response.raise_for_status()

                        # Accumulated tool call data (streamed incrementally)
                        tool_calls_buffer: dict[int, dict[str, Any]] = {}

                        for line in response.iter_lines():
                            # Skip empty lines and comments
                            if not line or line.startswith(":"):
                                continue

                            # SSE format: "data: {json}"
                            if not line.startswith("data: "):
                                continue

                            data_str = line[6:]  # Remove "data: " prefix

                            # Handle stream end
                            if data_str == "[DONE]":
                                break

                            try:
                                data = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            # Extract from choices[0].delta
                            choices = data.get("choices", [])
                            if not choices:
                                continue

                            choice = choices[0]
                            delta = choice.get("delta", {})
                            finish_reason = choice.get("finish_reason")

                            chunk = StreamChunk(finish_reason=finish_reason)

                            # Content
                            if "content" in delta and delta["content"]:
                                chunk.content = delta["content"]

                            # Reasoning details
                            if "reasoning_details" in delta and delta["reasoning_details"]:
                                chunk.reasoning_details = delta["reasoning_details"]

                            # Tool calls (accumulated incrementally)
                            if "tool_calls" in delta and delta["tool_calls"]:
                                for tc in delta["tool_calls"]:
                                    idx = tc.get("index", 0)
                                    if idx not in tool_calls_buffer:
                                        tool_calls_buffer[idx] = {
                                            "id": tc.get("id", ""),
                                            "type": "function",
                                            "function": {"name": "", "arguments": ""},
                                        }

                                    if "id" in tc and tc["id"]:
                                        tool_calls_buffer[idx]["id"] = tc["id"]

                                    if "function" in tc:
                                        func = tc["function"]
                                        if "name" in func and func["name"]:
                                            tool_calls_buffer[idx]["function"][
                                                "name"
                                            ] = func["name"]
                                        if "arguments" in func:
                                            tool_calls_buffer[idx]["function"][
                                                "arguments"
                                            ] += func["arguments"]

                            # When finish_reason is set, include accumulated tool calls
                            if finish_reason == "tool_calls" and tool_calls_buffer:
                                chunk.tool_calls = [
                                    tool_calls_buffer[i]
                                    for i in sorted(tool_calls_buffer.keys())
                                ]

                            # Parse usage data if present (comes in final chunk)
                            if "usage" in data and data["usage"]:
                                usage_data = data["usage"]
                                chunk.usage = TokenUsage(
                                    prompt_tokens=usage_data.get("prompt_tokens", 0),
                                    completion_tokens=usage_data.get(
                                        "completion_tokens", 0
                                    ),
                                )

                            yield chunk
        except KeyboardInterrupt:
            # Re-raise to allow proper cleanup at higher levels
            raise

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()
