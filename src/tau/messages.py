"""Agent message types, modeled after pi's AgentMessage union.

Timestamps are Unix milliseconds. Assistant messages contain content blocks
(text and tool calls); tool results are standalone messages referencing a
tool call id, as in pi's transcript model.
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
    text: str

    type: str = field(default="text", init=False)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    type: str = field(default="toolCall", init=False)


# ---------------------------------------------------------------------------
# Usage


@dataclass
class Usage:
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
    content: str

    role: str = field(default="user", init=False)
    timestamp: int = field(default_factory=now_ms, init=False)


@dataclass
class AssistantMessage:
    content: list[TextContent | ToolCall]
    model: str
    provider: str

    stop_reason: str = "stop"  # stop | toolUse | length | error | aborted
    usage: Usage = field(default_factory=Usage)
    error_message: str | None = None
    # Thinking-mode chain of thought. DeepSeek requires this to be sent back
    # on every subsequent request when tools are present.
    reasoning_content: str | None = None
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
        data["provider"] = message.provider
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
        data["content"] = [
            (
                {"type": "text", "text": b.text}
                if isinstance(b, TextContent)
                else {
                    "type": "toolCall",
                    "id": b.id,
                    "name": b.name,
                    "arguments": b.arguments,
                }
            )
            for b in message.content
        ]
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
            provider=data["provider"],
            stop_reason=data["stopReason"],
            usage=Usage(
                input=data["usage"]["input"],
                output=data["usage"]["output"],
                total_tokens=data["usage"]["totalTokens"],
            ),
            error_message=data.get("errorMessage"),
            reasoning_content=data.get("reasoningContent"),
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
