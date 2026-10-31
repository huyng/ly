"""The chat-completions backend: one `messages` array in, `choices[0].delta` out.

This is the default and the only endpoint DeepSeek, z.ai and the rest of the
OpenAI-compatible world speak.
"""

import json
import threading
import uuid
from collections.abc import Generator, Mapping
from typing import Any

from ..events import ProviderEvent, ReasoningDeltaEvent, TextDeltaEvent, ToolNameEvent
from ..messages import (
    AgentMessage,
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from ..tools import Tool
from .base import (
    HTTPProvider,
    _as_int,
    _close_stream,
)

# From GPT-5.4 on, chat-completions refuses function tools unless the effort is
# exactly "none", and an omitted effort is filled in with "medium". These models
# both need it and accept it; gpt-6-astra and gpt-6.1-sol reject "none" outright
# and are left out on purpose — they need /v1/responses.
_ACCEPTS_NONE = (
    "gpt-5.4", "gpt-5.5", "gpt-5.6",
    "gpt-6-sol", "gpt-6-luna", "gpt-6.1-terra",
)


class OpenAIProvider(HTTPProvider):
    """Streaming provider for any OpenAI-compatible chat-completions endpoint."""

    # -- request construction ------------------------------------------------

    def _request_messages(
        self, system_prompt: str, messages: list[AgentMessage]
    ) -> list[dict[str, Any]]:
        request: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]
        # Only the trailing assistant message is still in flight. Echoing reasoning
        # for finished turns is dead weight — no model reads it again, and it was
        # ~40% of all input tokens. Dropping the earlier blocks also leaves the
        # request prefix stable enough for provider-side caching.
        in_flight = max(
            (index for index, m in enumerate(messages) if isinstance(m, AssistantMessage)),
            default=-1,
        )
        for index, message in enumerate(messages):
            if isinstance(message, UserMessage):
                request.append({"role": "user", "content": message.content})
            elif isinstance(message, AssistantMessage):
                calls = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in message.tool_calls
                ]
                entry: dict[str, Any] = {"role": "assistant"}
                if message.text:
                    entry["content"] = message.text
                if (
                    index == in_flight
                    and message.reasoning_content
                    and self.pass_reasoning_content
                ):
                    # DeepSeek-only field: outside the OpenAI request schema.
                    entry["reasoning_content"] = message.reasoning_content
                if calls:
                    entry["tool_calls"] = calls
                if not entry.get("content") and "tool_calls" not in entry:
                    # An aborted or failed turn leaves an assistant message with
                    # nothing in it. Providers reject that shape outright, and it
                    # carries nothing a later request needs.
                    continue
                request.append(entry)
            elif isinstance(message, ToolResultMessage):
                request.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id,
                        "content": message.output,
                    }
                )
        return request

    @staticmethod
    def _tool_schemas(tools: list[Tool]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def _request_body(
        self,
        system_prompt: str,
        messages: list[AgentMessage],
        tools: list[Tool],
    ) -> bytes:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": self._request_messages(system_prompt, messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        # Omit `tools` entirely when there are none: some endpoints reject an
        # empty array where a missing key is fine.
        if tools:
            body["tools"] = self._tool_schemas(tools)
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        elif tools and self.model.lower().startswith(_ACCEPTS_NONE):
            body["reasoning_effort"] = "none"
        return json.dumps(body).encode("utf-8")

    # -- streaming ------------------------------------------------------------

    def stream(
        self,
        system_prompt: str,
        messages: list[AgentMessage],
        tools: list[Tool],
        abort_event: threading.Event | None = None,
    ) -> Generator[ProviderEvent, None, AssistantMessage]:
        """Stream one turn; yields events, returns the AssistantMessage."""
        url = self.base_url.rstrip("/") + "/chat/completions"
        chunks = self.transport(
            url,
            self._headers(),
            self._request_body(system_prompt, messages, tools),
            timeout=self.timeout,
        )

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: dict[int, dict] = {}
        usage = Usage()
        stop_reason = "stop"
        failure_reason: str | None = None
        aborted = False

        try:
            for chunk in chunks:
                if abort_event is not None and abort_event.is_set():
                    aborted = True
                    break
                parsed_usage = _usage_from(chunk.get("usage"))
                if parsed_usage is not None:
                    usage = parsed_usage
                choices = chunk.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0]
                if not isinstance(choice, Mapping):
                    continue
                delta = choice.get("delta")
                if not isinstance(delta, Mapping):
                    continue
                # The stream is untrusted JSON, so narrow before use: a
                # malformed event is skipped rather than crashing the turn.
                piece = delta.get("content")
                if isinstance(piece, str) and piece:
                    text_parts.append(piece)
                    yield TextDeltaEvent(delta=piece)
                reasoning = delta.get("reasoning_content")
                if isinstance(reasoning, str) and reasoning:
                    reasoning_parts.append(reasoning)
                    yield ReasoningDeltaEvent(delta=reasoning)
                for call in _as_list(delta.get("tool_calls")):
                    if not isinstance(call, Mapping):
                        continue
                    index = call.get("index", 0)
                    acc = tool_calls.setdefault(
                        index if isinstance(index, int) else 0,
                        {"id": None, "name": "", "arguments": ""},
                    )
                    if isinstance(call.get("id"), str):
                        acc["id"] = call["id"]
                    function = call.get("function")
                    if isinstance(function, Mapping):
                        name = function.get("name")
                        if isinstance(name, str) and name:
                            acc["name"] += name
                            yield ToolNameEvent(name=acc["name"])
                        fragment = function.get("arguments")
                        if isinstance(fragment, str) and fragment:
                            acc["arguments"] += fragment
                finish_reason = choice.get("finish_reason")
                if isinstance(finish_reason, str) and finish_reason:
                    stop_reason = _STOP_REASONS.get(finish_reason, "stop")
                    if stop_reason == "error":
                        failure_reason = finish_reason
        except KeyboardInterrupt:
            # Ctrl-C surfaced inside the stream: finalize as aborted.
            aborted = True
        finally:
            # Deterministic: drop the connection instead of draining it.
            _close_stream(chunks)

        content: list[TextContent | ToolCall] = []
        # No empty text block when nothing was streamed: it would be the whole
        # content of an interrupted turn, which providers reject.
        if text := "".join(text_parts):
            content.append(TextContent(text=text))
        for index in sorted(tool_calls):
            acc = tool_calls[index]
            try:
                arguments = json.loads(acc["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {"_raw": acc["arguments"]}
            content.append(
                ToolCall(id=acc["id"] or new_call_id(), name=acc["name"], arguments=arguments)
            )

        return AssistantMessage(
            content=content,
            model=self.model,
            provider=self.name,
            stop_reason="aborted" if aborted else stop_reason,
            usage=usage,
            error_message=failure_reason,
            reasoning_content="".join(reasoning_parts) or None,
        )


# OpenAI's documented reasons, plus the DeepSeek-specific failures tau has seen.
_STOP_REASONS = {
    "tool_calls": "toolUse",
    "length": "length",
    "aborted": "aborted",
    "content_filter": "error",
    "insufficient_system_resource": "error",
}


def _usage_from(payload: Any) -> Usage | None:
    if not isinstance(payload, Mapping):
        return None

    return Usage(
        input=_as_int(payload.get("prompt_tokens")),
        output=_as_int(payload.get("completion_tokens")),
        total_tokens=_as_int(payload.get("total_tokens")),
    )


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"
