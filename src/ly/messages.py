"""Define conversation messages and their session JSON representation.

Structure:
    AgentMessage
    ├── UserMessage: content (text)
    ├── AssistantMessage: content (list of blocks), model, api, usage, …
    │   ├── TextContent: text
    │   └── ToolCall: id, name, arguments
    └── ToolResultMessage: tool_call_id, tool_name, output, is_error

Every message has a fixed role and a timestamp in Unix milliseconds. Tool
results are separate messages linked to an assistant's tool call by its id.
An assistant message can contain both text and tool calls.

Session JSON uses camelCase for Python's snake_case fields (for example,
stop_reason becomes stopReason). The role identifies the message variant;
each content block's type identifies text or toolCall. Optional error and
reasoning fields are omitted when empty. These are ly's internal messages,
not provider request bodies; each backend converts them to its API format.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


def now_ms() -> int:
    return int(time.time() * 1000)


def new_id() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Content blocks


@dataclass
class TextContent:
    """Hold one text block in an assistant message."""

    text: str

    type: str = field(default="text", init=False)


@dataclass
class ToolCall:
    """Request a named tool with parsed arguments and an id for its result."""

    id: str
    name: str
    arguments: dict[str, Any]

    type: str = field(default="toolCall", init=False)


# ---------------------------------------------------------------------------
# Usage


@dataclass
class Usage:
    """Count input, output, and total tokens reported by the provider."""

    input: int = 0
    output: int = 0
    total_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.input += other.input
        self.output += other.output
        self.total_tokens += other.total_tokens


# ---------------------------------------------------------------------------
# Messages


@dataclass
class UserMessage:
    """Hold a user's text with the fixed role 'user'."""

    content: str

    role: str = field(default="user", init=False)
    timestamp: int = field(default_factory=now_ms, init=False)


@dataclass
class AssistantMessage:
    """Hold an assistant response, including tool calls and provider metadata.

    Args:
        content: Ordered text and tool-call blocks; may be empty on failure.
        model: Model identifier used to generate the response.
        api: API format, not service identity: 'openai-completions' or
            'openai-responses'. Empty when unspecified; does not route requests.
        stop_reason: 'stop' for completion, 'toolUse' for tool calls, 'length'
            for a limit, 'error' for failure, or 'aborted' for cancellation.
        usage: Provider-reported token counts; defaults to zero counts.
        error_message: Failure details, when present.
        reasoning_content: Readable reasoning or summary text. Chat backends
            may replay it when required by the service.
        reasoning_item: Responses-only replay data: type='reasoning', summary
            (a list), and encrypted_content (an opaque string), without an id.
            None when no encrypted reasoning is available.
    """

    content: list[TextContent | ToolCall]
    model: str

    api: str = ""
    stop_reason: str = "stop"  # stop | toolUse | length | error | aborted
    usage: Usage = field(default_factory=Usage)
    error_message: str | None = None
    # Thinking-mode chain of thought. DeepSeek requires this to be sent back
    # on every subsequent request when tools are present.
    reasoning_content: str | None = None
    # Responses-API only: the reasoning item to replay on the next request.
    # `summary` is a required field on input and `encrypted_content` is the
    # stateless (store=false) stand-in for server-side storage.
    reasoning_item: dict[str, Any] | None = None
    role: str = field(default="assistant", init=False)
    timestamp: int = field(default_factory=now_ms, init=False)

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextContent))

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [b for b in self.content if isinstance(b, ToolCall)]


@dataclass
class ToolResultMessage:
    """Hold tool output linked to a ToolCall.id by tool_call_id.

    tool_name identifies the executed tool. output is text, including error
    details when is_error is true. The fixed role is 'toolResult'.
    """

    tool_call_id: str
    tool_name: str
    output: str

    is_error: bool = False
    role: str = field(default="toolResult", init=False)
    timestamp: int = field(default_factory=now_ms, init=False)


AgentMessage = UserMessage | AssistantMessage | ToolResultMessage

# ---------------------------------------------------------------------------
# Serialization (session JSONL)


def serialize_message(message: AgentMessage) -> dict[str, Any]:
    data: dict[str, Any] = {"role": message.role, "timestamp": message.timestamp}
    if isinstance(message, UserMessage):
        data["content"] = message.content

    elif isinstance(message, AssistantMessage):
        data["model"] = message.model
        data["api"] = message.api
        data["stopReason"] = message.stop_reason
        data["usage"] = {
            "input": message.usage.input,
            "output": message.usage.output,
            "totalTokens": message.usage.total_tokens,
        }
        if message.error_message:
            data["errorMessage"] = message.error_message
        if message.reasoning_content:
            data["reasoningContent"] = message.reasoning_content
        if message.reasoning_item:
            data["reasoningItem"] = message.reasoning_item
        blocks: list[dict[str, Any]] = []
        for block in message.content:
            if isinstance(block, TextContent):
                blocks.append({"type": "text", "text": block.text})
            else:
                blocks.append(
                    {
                        "type": "toolCall",
                        "id": block.id,
                        "name": block.name,
                        "arguments": block.arguments,
                    }
                )
        data["content"] = blocks
    elif isinstance(message, ToolResultMessage):
        data["toolCallId"] = message.tool_call_id
        data["toolName"] = message.tool_name
        data["output"] = message.output
        data["isError"] = message.is_error
    return data


def deserialize_message(data: dict[str, Any]) -> AgentMessage:
    role = data["role"]
    message: AgentMessage
    if role == "user":
        message = UserMessage(content=data["content"])
    elif role == "assistant":
        content: list[TextContent | ToolCall] = []
        for block in data["content"]:
            if block["type"] == "text":
                content.append(TextContent(text=block["text"]))
            else:
                content.append(
                    ToolCall(
                        id=block["id"], name=block["name"], arguments=block["arguments"]
                    )
                )
        message = AssistantMessage(
            content=content,
            model=data["model"],
            # Legacy logs used adapter names as providers, not service identities.
            api=data.get(
                "api",
                {
                    "openai": "openai-completions",
                    "openai-responses": "openai-responses",
                }.get(data.get("provider", ""), ""),
            ),
            stop_reason=data["stopReason"],
            usage=Usage(
                input=data["usage"]["input"],
                output=data["usage"]["output"],
                total_tokens=data["usage"]["totalTokens"],
            ),
            error_message=data.get("errorMessage"),
            reasoning_content=data.get("reasoningContent"),
            reasoning_item=data.get("reasoningItem"),
        )
    elif role == "toolResult":
        message = ToolResultMessage(
            tool_call_id=data["toolCallId"],
            tool_name=data["toolName"],
            output=data["output"],
            is_error=data.get("isError", False),
        )
    else:
        raise ValueError(f"unknown message role: {role!r}")
    message.timestamp = data.get("timestamp", message.timestamp)
    return message
