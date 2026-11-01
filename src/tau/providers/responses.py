"""Responses-API backend: the only route to tools on the newest models.

From GPT-5.4 on, chat-completions accepts function tools only alongside
``reasoning_effort: "none"``, and gpt-6-astra / gpt-6.1-sol reject ``none``
outright, so those models are reachable only through ``/v1/responses``. Both
the request (input items, flat tool schemas, ``instructions``) and the stream
(named events instead of ``choices[0].delta``) differ from chat-completions,
which is why this is a backend of its own rather than a branch in
``chat.ChatCompletionsBackend``.
"""

import json
import threading
from collections import OrderedDict
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
from .base import HTTPBackend, ProviderError


class ResponsesBackend(HTTPBackend):
    """Streaming provider for OpenAI's /v1/responses endpoint."""

    api = "openai-responses"

    # -- request construction ------------------------------------------------

    def _input_items(self, messages: list[AgentMessage]) -> list[dict[str, Any]]:
        """Convert the transcript into Responses input items.

        The system prompt is not an item here — it travels as `instructions`.
        Text and each tool call become separate items, where chat-completions
        packs both into one assistant message.
        """
        items: list[dict[str, Any]] = []
        for message in messages:
            if isinstance(message, UserMessage):
                items.append({"role": "user", "content": message.content})
            elif isinstance(message, AssistantMessage):
                if message.text:
                    items.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": message.text}],
                        }
                    )
                for call in message.tool_calls:
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": call.id,
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        }
                    )
            elif isinstance(message, ToolResultMessage):
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_call_id,
                        "output": message.output,
                    }
                )
        return items

    def _request_body(
        self,
        system_prompt: str,
        messages: list[AgentMessage],
        tools: list[Tool],
    ) -> bytes:
        body: dict[str, Any] = {
            "model": self.model,
            "instructions": system_prompt,
            "input": self._input_items(messages),
            "stream": True,
            # tau owns the transcript, so nothing needs keeping server-side.
            "store": False,
            # Request summaries so reasoning text appears on the stream.
            "reasoning": {"summary": "auto"},
        }
        if tools:
            # Flat on Responses: no `function` wrapper around the schema.
            body["tools"] = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
                for tool in tools
            ]
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
        url = self.base_url.rstrip("/") + "/responses"
        chunks = self.transport(
            url,
            self._headers(),
            self._request_body(system_prompt, messages, tools),
            timeout=self.timeout,
        )

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        calls: OrderedDict[str, dict[str, str]] = OrderedDict()
        usage = Usage()
        stop_reason = "stop"
        failure_reason: str | None = None
        retryable = False
        aborted = False

        try:
            for chunk in chunks:
                if abort_event is not None and abort_event.is_set():
                    aborted = True
                    break
                kind = chunk.get("type")
                if kind == "response.output_text.delta":
                    delta = _text(chunk.get("delta"))
                    if delta:
                        text_parts.append(delta)
                        yield TextDeltaEvent(delta=delta)
                elif kind == "response.reasoning_summary_text.delta":
                    delta = _text(chunk.get("delta"))
                    if delta:
                        reasoning_parts.append(delta)
                        yield ReasoningDeltaEvent(delta=delta)
                elif kind == "response.output_item.added":
                    item = _mapping(chunk.get("item"))
                    if item.get("type") == "function_call":
                        call = _call_entry(item)
                        calls[_item_key(chunk, item)] = call
                        if call["name"]:
                            yield ToolNameEvent(name=call["name"])
                elif kind == "response.function_call_arguments.delta":
                    call = calls.get(_item_key(chunk))
                    if call is not None:
                        call["arguments"] += _text(chunk.get("delta"))
                elif kind == "response.output_item.done":
                    # The done event is authoritative: deltas may be incomplete.
                    item = _mapping(chunk.get("item"))
                    if item.get("type") == "function_call":
                        call = calls.setdefault(
                            _item_key(chunk, item), _call_entry(item)
                        )
                        call["id"] = call["id"] or _text(item.get("call_id"))
                        call["name"] = call["name"] or _text(item.get("name"))
                        call["arguments"] = (
                            _text(item.get("arguments")) or call["arguments"]
                        )
                elif kind == "response.completed":
                    usage = (
                        _usage_from(_mapping(chunk.get("response")).get("usage"))
                        or usage
                    )
                elif kind == "response.incomplete":
                    stop_reason = "length"
                    response = _mapping(chunk.get("response"))
                    details = _mapping(response.get("incomplete_details"))
                    if details.get("reason") == "content_filter":
                        stop_reason = "error"
                        failure_reason = "content_filter"
                elif kind in ("response.failed", "error"):
                    stop_reason = "error"
                    response = _mapping(chunk.get("response"))
                    usage = _usage_from(response.get("usage")) or usage
                    error = _mapping(response.get("error")) or chunk
                    retryable = error.get("code") in (
                        "server_error",
                        "rate_limit_exceeded",
                        "insufficient_system_resource",
                    )
                    failure_reason = (
                        _text(error.get("message"))
                        or _text(chunk.get("message"))
                        or "response failed"
                    )
        except ProviderError as exc:
            if exc.usage is None:
                exc.usage = usage
            raise
        except KeyboardInterrupt:
            aborted = True
        finally:
            _close_stream(chunks)

        if not aborted and stop_reason == "error":
            raise ProviderError(
                failure_reason or "response failed",
                retryable=retryable,
                usage=usage,
            )

        content: list[TextContent | ToolCall] = []
        if text := "".join(text_parts):
            content.append(TextContent(text=text))
        for key, call in calls.items():
            try:
                arguments = json.loads(call["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {"_raw": call["arguments"]}
            content.append(
                ToolCall(id=call["id"] or key, name=call["name"], arguments=arguments)
            )
        if stop_reason == "stop" and calls:
            stop_reason = "toolUse"

        return AssistantMessage(
            content=content,
            model=self.model,
            api=self.api,
            stop_reason="aborted" if aborted else stop_reason,
            usage=usage,
            error_message=failure_reason,
            reasoning_content="".join(reasoning_parts) or None,
        )


def _item_key(chunk: Mapping[str, Any], item: Mapping[str, Any] | None = None) -> str:
    """Name for one function call as it appears across several events."""
    if item is not None and item.get("id"):
        return str(item["id"])
    if item_id := _text(chunk.get("item_id")):
        return item_id
    return f"#{chunk.get('output_index', 0)}"


def _call_entry(item: Mapping[str, Any]) -> dict[str, str]:
    return {
        "id": _text(item.get("call_id")),
        "name": _text(item.get("name")),
        "arguments": "",
    }


def _usage_from(payload: Any) -> Usage | None:
    """Responses reports `input_tokens`/`output_tokens`, not prompt/completion."""
    if not isinstance(payload, Mapping):
        return None
    return Usage(
        input=_as_int(payload.get("input_tokens")),
        output=_as_int(payload.get("output_tokens")),
        total_tokens=_as_int(payload.get("total_tokens")),
    )


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _close_stream(stream: Any) -> None:
    """Close a transport generator if it is closable, ignoring the rest."""
    close = getattr(stream, "close", None)
    if callable(close):
        close()


def _as_int(value: Any) -> int:
    return value if isinstance(value, int) else 0
