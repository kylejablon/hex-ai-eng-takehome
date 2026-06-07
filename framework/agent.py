"""Agent framework for autonomous task execution with tool calling.

This module implements an agent that uses the OpenRouter API for LLM inference,
supporting streaming responses, tool calling, and reasoning token display.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import cache
from typing import Any

from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic_core import from_json

from framework.llm import OpenRouterClient, OpenRouterConfig, TokenUsage

# Prefix that indicates the agent should stop (answer was submitted)
# This avoids global state - the tool result signals completion
ANSWER_SUBMITTED_PREFIX = "ANSWER_SUBMITTED:"

type ToolFunction = Callable[..., str]


class EventType(Enum):
    """Types of events emitted during agent execution."""

    # Generation events
    GENERATION_START = auto()
    THINKING_START = auto()
    THINKING_CHUNK = auto()
    THINKING_END = auto()
    RESPONSE_CHUNK = auto()
    GENERATION_END = auto()

    # Tool events
    TOOL_CALL_START = auto()
    TOOL_CALL_PARSED = auto()
    TOOL_EXECUTION_START = auto()
    TOOL_EXECUTION_END = auto()

    # Agent loop events
    ITERATION_START = auto()
    ITERATION_END = auto()
    AGENT_COMPLETE = auto()
    AGENT_ERROR = auto()


@dataclass
class AgentEvent:
    """An event emitted during agent execution."""

    type: EventType
    data: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Format event for display."""
        return f"[{self.type.name}] {self.data}"


@dataclass
class Tool:
    """Represents a tool that can be called by the agent.

    Tool functions must return a string that will be shown to the LLM.

    `args_model` is a pydantic model describing the tool's arguments. It is used to
    validate (and partially recover) the JSON the model emits for a tool call -- see
    `_parse_arguments`. `parameters` remains the hand-tuned JSON Schema advertised to
    the API; the two describe the same arguments from the prompt and runtime sides.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    function: ToolFunction
    args_model: type[BaseModel] | None = None


@dataclass
class ToolCall:
    """Represents a (parsed) tool call request from the agent."""

    id: str  # Required for OpenAI-compatible API
    name: str
    arguments: dict[str, Any]
    error: str | None = None


@dataclass
class Message:
    """Represents a message in the conversation."""

    role: str  # "system", "user", "assistant", or "tool"
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None  # For assistant messages with tool calls
    tool_call_id: str | None = None  # For tool result messages


@dataclass
class ContextCompressionSettings:
    """Settings for context compression to reduce token usage."""

    enabled: bool = False
    keep_recent: int = 3  # Number of recent tool results to keep in full
    max_chars: int = 150  # Max chars for truncated older results


@dataclass
class Conversation:
    """Represents a conversation between the agent and the user."""

    messages: list[Message] = field(default_factory=list)

    def to_api_format(
        self,
        compression: ContextCompressionSettings | None = None,
    ) -> list[dict[str, Any]]:
        """Convert the conversation to OpenAI-compatible API format.

        Args:
            compression: Optional compression settings. If enabled, older tool
                results are truncated and duplicate consecutive tool calls are
                deduplicated.
        """
        messages_to_convert = self.messages

        if compression and compression.enabled:
            messages_to_convert = _compress_messages(
                self.messages,
                keep_recent=compression.keep_recent,
                max_chars=compression.max_chars,
            )

        result: list[dict[str, Any]] = []
        for message in messages_to_convert:
            msg: dict[str, Any] = {"role": message.role}

            if message.content is not None:
                msg["content"] = message.content

            if message.tool_calls is not None:
                msg["tool_calls"] = message.tool_calls

            if message.tool_call_id is not None:
                msg["tool_call_id"] = message.tool_call_id

            result.append(msg)
        return result


def _truncate_tool_result(content: str, max_chars: int) -> str:
    """Truncate a tool result to max_chars with a summary prefix."""
    if len(content) <= max_chars:
        return content

    # Extract first line as summary (often contains row/column counts)
    first_line = content.split("\n")[0]
    if len(first_line) <= max_chars - 20:
        return f"[Truncated] {first_line}"

    return f"[Truncated] {content[:max_chars - 15]}..."


def _compress_messages(
    messages: list[Message],
    keep_recent: int,
    max_chars: int,
) -> list[Message]:
    """Compress messages by truncating old tool results and deduplicating.

    Applies two optimizations:
    1. Truncates tool results older than keep_recent to max_chars
    2. Removes duplicate consecutive tool calls with identical results
    """
    # Find all tool message indices (for determining which are "recent")
    tool_indices: list[int] = [
        i for i, m in enumerate(messages) if m.role == "tool"
    ]

    # Indices of tool messages to keep in full (the most recent ones)
    recent_tool_indices = set(tool_indices[-keep_recent:]) if tool_indices else set()

    # Build compressed message list
    result: list[Message] = []
    seen_tool_calls: dict[str, str] = {}  # (name, args_json) -> full result

    for i, msg in enumerate(messages):
        if msg.role == "tool":
            # Check for deduplication: same tool call with same result
            # Find the corresponding assistant message's tool call
            tool_key: str | None = None
            for j in range(i - 1, -1, -1):
                assistant_tool_calls = messages[j].tool_calls
                if messages[j].role == "assistant" and assistant_tool_calls:
                    for tc in assistant_tool_calls:
                        if tc.get("id") == msg.tool_call_id:
                            name = tc.get("function", {}).get("name", "")
                            args = tc.get("function", {}).get("arguments", "")
                            tool_key = f"{name}:{args}"
                            break
                    break

            # Deduplicate: if we've seen this exact call before with same result
            if tool_key and msg.content:
                if tool_key in seen_tool_calls:
                    prev_content = seen_tool_calls[tool_key]
                    if prev_content == msg.content:
                        # Skip this duplicate - but we need to keep the message
                        # structure for the API, so mark it as deduplicated
                        result.append(
                            Message(
                                role=msg.role,
                                content="[Duplicate call - see earlier result]",
                                tool_call_id=msg.tool_call_id,
                            )
                        )
                        continue
                seen_tool_calls[tool_key] = msg.content

            # Truncate if not in recent set
            if i not in recent_tool_indices and msg.content:
                result.append(
                    Message(
                        role=msg.role,
                        content=_truncate_tool_result(msg.content, max_chars),
                        tool_call_id=msg.tool_call_id,
                    )
                )
            else:
                result.append(msg)
        else:
            result.append(msg)

    return result


class Agent:
    """Implements a tiny, generic agent framework.

    Built on top of the OpenRouter API client.

    Only supports a single model, streaming, and an extensible tool set.
    """

    def __init__(self, config: OpenRouterConfig, tools: dict[str, Tool]):
        self.config = config
        self.tools: dict[str, Tool] = tools  # mapping from tool name to tool object
        self.client: OpenRouterClient = OpenRouterClient(config)
        self.conversation: Conversation = Conversation()
        self._compression = ContextCompressionSettings(
            enabled=config.compress_context,
            keep_recent=config.compress_keep_recent,
            max_chars=config.compress_max_chars,
        )
        self.reset_conversation()

    def _get_tool_definitions(self) -> list[dict[str, Any]]:
        """Get tool definitions in OpenAI-compatible format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self.tools.values()
        ]

    def _execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a tool call and return the result as a string.

        Guaranteed to return a string, swallowing exceptions.
        """
        # gpt-oss-120b often emits a phantom, nameless second tool call (with junk
        # arguments) alongside a real one. It maps to no tool, so treat it as a no-op
        # instead of echoing a confusing parse error back to the model.
        if not tool_call.name.strip():
            return "(empty tool call ignored)"

        if tool_call.error:
            return f"Error parsing arguments for tool '{tool_call.name}': {tool_call.error}"

        if tool_call.name not in self.tools:
            return f"Error: Unknown tool '{tool_call.name}'"
        tool = self.tools[tool_call.name]
        try:
            return tool.function(**tool_call.arguments)
        except Exception as e:
            return f"Error executing {tool_call.name}: {e}"

    def _generate_response(self, conversation: Conversation) -> Iterator[AgentEvent]:
        """Generate a response from the model, streaming the events out."""
        yield AgentEvent(type=EventType.GENERATION_START)

        messages = conversation.to_api_format(compression=self._compression)
        tools = self._get_tool_definitions() if self.tools else None

        full_content = ""
        tool_calls: list[dict[str, Any]] = []
        in_thinking = False
        finish_reason: str | None = None
        usage: TokenUsage | None = None

        for chunk in self.client.chat_completion_stream(messages, tools):
            # Handle reasoning/thinking tokens
            if chunk.reasoning_details:
                for detail in chunk.reasoning_details:
                    if detail.get("type") == "reasoning.text":
                        text = detail.get("text", "")
                        if text:
                            if not in_thinking:
                                in_thinking = True
                                yield AgentEvent(type=EventType.THINKING_START)
                            yield AgentEvent(
                                type=EventType.THINKING_CHUNK,
                                data={"chunk": text},
                            )

            # Handle regular content
            if chunk.content:
                # Close thinking block if we were in it
                if in_thinking:
                    in_thinking = False
                    yield AgentEvent(type=EventType.THINKING_END)

                full_content += chunk.content
                yield AgentEvent(
                    type=EventType.RESPONSE_CHUNK,
                    data={"chunk": chunk.content},
                )

            # Handle tool calls (accumulated at the end)
            if chunk.tool_calls:
                tool_calls = chunk.tool_calls

            if chunk.finish_reason:
                finish_reason = chunk.finish_reason

            # Capture usage data (comes in final chunk)
            if chunk.usage:
                usage = chunk.usage

        # Close thinking if still open
        if in_thinking:
            yield AgentEvent(type=EventType.THINKING_END)

        event_data: dict[str, Any] = {
            "full_response": full_content,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
        }
        if usage:
            event_data["usage"] = usage

        yield AgentEvent(type=EventType.GENERATION_END, data=event_data)

    def _get_system_message(self) -> str:
        """Get the system message for the agent."""
        return (
            "You are an autonomous SQL agent. You must complete tasks independently "
            "without asking the user for clarification or additional information. "
            "Use the available tools to gather any information you need. "
            "If you're uncertain, make your best assumptions and proceed.\n\n"
            "CRITICAL: You MUST call the 'submit_answer' tool to complete EVERY task. "
            "NEVER stop without calling submit_answer. Even if you've computed the answer, "
            "you MUST submit it via submit_answer with a valid SQL query.\n\n"
            "Do not provide answers as plain text - always use the submit_answer tool "
            "with a valid SQL query that generates a dataframe with the intended answer.\n\n"
            "KNOWLEDGE BASE: You have a knowledge base of markdown files about our SQL "
            "data. Call list_guides to find relevant guides, then open_guide to read "
            "them.\n\n"
            "TOOL CALL FORMATTING: A tool's arguments must be a single valid JSON object. "
            "When a query argument contains SQL, write the SQL as a SINGLE-LINE JSON string "
            "with NO literal newline characters, and use single quotes for SQL string "
            "literals so you never need to escape double quotes. Malformed JSON will be "
            "rejected and waste a turn.\n\n"
            "WORKFLOW: Be decisive and efficient - you have a limited number of steps. "
            "Use explore_schema and explore_table only as much as needed to find the right "
            "tables and columns; do NOT repeatedly re-explore the same things. Use run_query "
            "to test your query, and as soon as its results look correct, immediately call "
            "submit_answer with that same query. Always submit a working answer well before "
            "you run out of steps rather than over-exploring."
        )

    def run(self, prompt: str) -> Iterator[AgentEvent]:
        """Run the agent with streaming output, from the user's natural language prompt."""
        # Add the new user message to the ongoing conversation
        self.conversation.messages.append(Message(role="user", content=prompt))

        # Track cumulative token usage across all iterations
        total_usage = TokenUsage()

        budget_msg: Message | None = None
        for iteration in range(self.config.max_iterations):
            yield AgentEvent(type=EventType.ITERATION_START, data={"iteration": iteration + 1})

            # Show the step budget to the agent every turn so it can pace itself.
            # Keep a single, always-current note at the end of the conversation.
            if budget_msg is not None and budget_msg in self.conversation.messages:
                self.conversation.messages.remove(budget_msg)
            remaining = self.config.max_iterations - iteration
            budget_msg = Message(
                role="user",
                content=(
                    f"[Step {iteration + 1} of {self.config.max_iterations}, "
                    f"{remaining} step(s) remaining]"
                ),
            )
            self.conversation.messages.append(budget_msg)

            full_response = ""
            tool_calls_data: list[dict[str, Any]] = []

            for event in self._generate_response(self.conversation):
                yield event
                if event.type == EventType.GENERATION_END:
                    full_response = event.data.get("full_response", "")
                    tool_calls_data = event.data.get("tool_calls", [])
                    # Accumulate token usage
                    if "usage" in event.data and event.data["usage"]:
                        total_usage = total_usage + event.data["usage"]

            # Parse tool calls from the structured response
            tool_calls = _parse_tool_calls_from_api(tool_calls_data, self.tools)

            if not tool_calls:
                # Check if this is an empty response (model just stopped)
                is_empty_response = not full_response or not full_response.strip()

                # Check if response looks like a malformed tool call (JSON with "query" key)
                # This happens when the model outputs tool call arguments as plain text
                looks_like_failed_tool_call = (full_response and "{" in full_response)

                if is_empty_response or looks_like_failed_tool_call:
                    # Model returned empty response or malformed tool call
                    # Inject a continuation prompt to remind it to properly call submit_answer
                    if looks_like_failed_tool_call:
                        print("\n[DEBUG] Response looks like failed tool call "
                              "- injecting continuation prompt")
                    else:
                        print("\n[DEBUG] Empty response detected "
                              "- injecting continuation prompt")

                    self.conversation.messages.append(
                        Message(role="assistant", content=full_response if full_response else "")
                    )
                    self.conversation.messages.append(
                        Message(
                            role="user",
                            content=(
                                "You must use the submit_answer TOOL to submit your "
                                "answer - do not output JSON directly. "
                                "Call the submit_answer tool now with your SQL query."
                            ),
                        )
                    )
                    # Continue to next iteration instead of returning
                    continue

                # Non-empty response without tool calls - agent is done
                self.conversation.messages.append(
                    Message(role="assistant", content=full_response)
                )
                yield AgentEvent(
                    type=EventType.AGENT_COMPLETE,
                    data={"response": full_response, "usage": total_usage},
                )
                return

            yield AgentEvent(type=EventType.TOOL_CALL_START, data={"count": len(tool_calls)})

            # Save assistant message with tool calls
            self.conversation.messages.append(
                Message(
                    role="assistant",
                    content=full_response if full_response else None,
                    tool_calls=tool_calls_data,
                )
            )

            for tool_call in tool_calls:
                yield AgentEvent(
                    type=EventType.TOOL_CALL_PARSED,
                    data={"name": tool_call.name, "arguments": tool_call.arguments},
                )
                yield AgentEvent(
                    type=EventType.TOOL_EXECUTION_START,
                    data={"name": tool_call.name},
                )
                tool_result = self._execute_tool(tool_call)
                yield AgentEvent(
                    type=EventType.TOOL_EXECUTION_END,
                    data={"name": tool_call.name, "result": tool_result},
                )

                # Add tool result message with tool_call_id
                self.conversation.messages.append(
                    Message(
                        role="tool",
                        content=tool_result,
                        tool_call_id=tool_call.id,
                    )
                )

                # Check if this tool signals agent completion (e.g., answer submitted)
                if tool_result.startswith(ANSWER_SUBMITTED_PREFIX):
                    yield AgentEvent(
                        type=EventType.AGENT_COMPLETE,
                        data={
                            "reason": "answer_submitted",
                            "tool": tool_call.name,
                            "usage": total_usage,
                        },
                    )
                    return

            yield AgentEvent(type=EventType.ITERATION_END, data={"iteration": iteration + 1})

        yield AgentEvent(
            type=EventType.AGENT_ERROR,
            data={"error": "Max iterations reached", "usage": total_usage},
        )

    def reset_conversation(self) -> None:
        """Reset the conversation to the initial state (with system message)."""
        self.conversation = Conversation()
        self.conversation.messages.append(
            Message(role="system", content=self._get_system_message())
        )


@cache
def _adapter_for(model: type[BaseModel]) -> TypeAdapter[BaseModel]:
    """Cache one TypeAdapter per args model (building one is not free)."""
    return TypeAdapter(model)


def _format_validation_error(error: ValidationError) -> str:
    """Condense a pydantic ValidationError into a short, model-readable message."""
    problems = "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or '(root)'}: {err['msg']}"
        for err in error.errors()
    )
    return f"Invalid arguments: {problems}"


def _parse_arguments(
    arguments_str: str, model: type[BaseModel] | None
) -> tuple[dict[str, Any], str | None]:
    """Validate a tool call's raw argument JSON against its pydantic model.

    Recovers the "phantom split" gpt-oss exhibits, where it peels a call's closing
    brace off into a separate, nameless tool-call slot and leaves the real call's
    argument JSON truncated. Pydantic partial JSON parsing fills in the missing tail.

    A strict (lossless) parse is tried first; partial parsing is only the fallback,
    for input that is genuinely cut short, since it can silently truncate a string
    rather than reject malformed input.

    Returns (kwargs, None) on success, or ({}, error_message) on failure.
    """
    error: str | None = None
    for allow_partial in (False, True):
        try:
            if model is None:
                # A tool without a declared model: parse leniently, no schema.
                parsed = from_json(arguments_str, allow_partial=allow_partial)
                if not isinstance(parsed, dict):
                    error = "Invalid arguments: expected a JSON object"
                    continue
                return parsed, None
            obj = _adapter_for(model).validate_json(
                arguments_str, experimental_allow_partial=allow_partial
            )
            # by_alias so an aliased field (e.g. ExploreSchemaArgs.schema_name)
            # dumps back to the function's real keyword ("schema").
            return obj.model_dump(by_alias=True), None
        except ValidationError as e:
            error = _format_validation_error(e)
        except ValueError as e:
            error = f"Invalid arguments: {e}"

    return {}, error


def _parse_tool_calls_from_api(
    tool_calls_data: list[dict[str, Any]], tools: dict[str, Tool]
) -> list[ToolCall]:
    """Parse tool calls from OpenAI-compatible API response format.

    Each call's arguments are validated against the target tool's pydantic args
    model (see `_parse_arguments`). A call naming an unknown or empty tool -- such
    as the nameless phantom call gpt-oss emits alongside a real one -- carries no
    arguments to act on and is passed through for `_execute_tool` to ignore.
    """
    tool_calls: list[ToolCall] = []

    for tc in tool_calls_data:
        tc_id = tc.get("id", "")
        function = tc.get("function", {})
        name = function.get("name", "")
        arguments_str = function.get("arguments", "{}")

        tool = tools.get(name)
        if tool is None:
            # Unknown / nameless call: nothing to validate. _execute_tool reports
            # the unknown or empty tool name.
            tool_calls.append(ToolCall(id=tc_id, name=name, arguments={}))
            continue

        arguments, error = _parse_arguments(arguments_str, tool.args_model)
        tool_calls.append(
            ToolCall(id=tc_id, name=name, arguments=arguments, error=error)
        )

    return tool_calls
