"""Every wire format tau can speak, from the transport up.

All backends share one contract: `stream` takes the transcript, yields events,
and returns the AssistantMessage. What to talk to — model, endpoint, key,
timeout — arrives as a `Provider` given to the constructor, and each request
reads it fresh, so `provider.model = ...` takes effect on the next request
rather than being shadowed by a copy taken at construction.

- `ChatCompletionsBackend` — `/chat/completions`, the default, and the only
  flavour DeepSeek, z.ai and the rest of the OpenAI-compatible world speak.
- `ResponsesBackend` — `/v1/responses`, the only route to tools on the newest
  models. From GPT-5.4 on, chat-completions accepts function tools only
  alongside `reasoning_effort: "none"`, and gpt-6-astra / gpt-6.1-sol reject
  `none` outright. Both the request (input items, flat tool schemas,
  `instructions`) and the stream (named events instead of `choices[0].delta`)
  differ, so it is a backend of its own rather than a branch in the chat one.
"""

import json
import threading
import uuid
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
from .errors import ProviderError
from .provider import Provider
from .sse import Chunk, StreamReader


class ModelBackend:
    """Wire format for one API flavour; subclasses speak it in `stream`.

    Holds the provider it reads configuration from, and the reader that turns a
    POST into chunks. What varies between subclasses is the request body and
    how the stream is interpreted.
    """

    api = ""

    def __init__(self, provider: Provider, reader: type[StreamReader] | None = None):
        self.provider = provider
        self.reader = reader or StreamReader

    def _make_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.provider.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Accept-Encoding": "identity",
        }

    def _post(self, path: str, body: bytes) -> Generator[Chunk, None, None]:
        """POST `body` under the provider's base URL and stream the chunks.

        Closing the returned stream closes the connection underneath it, so
        abandoning a turn drops the response instead of draining it.
        """
        url = self.provider.base_url.rstrip("/") + path
        chunks = self.reader(url, headers=self._make_headers(), body=body).read_chunks()
        try:
            yield from chunks
        finally:
            close = getattr(chunks, "close", None)
            if callable(close):
                close()

    def stream(
        self,
        system_prompt: str,
        messages: list[AgentMessage],
        tools: list[Tool],
        abort_event: threading.Event | None = None,
    ) -> Generator[ProviderEvent, None, AssistantMessage]:
        """Stream one turn; yields events, returns the AssistantMessage."""
        raise NotImplementedError


# -- chat-completions ---------------------------------------------------------

# OpenAI's documented reasons, plus the DeepSeek-specific failures tau has seen.
_STOP_REASONS = {
    "tool_calls": "toolUse",
    "length": "length",
    "aborted": "aborted",
    "content_filter": "error",
    "insufficient_system_resource": "error",
}


class ChatCompletionsBackend(ModelBackend):
    """One `messages` array in, `choices[0].delta` out."""

    api = "openai-completions"

    def _request_messages(
        self,
        system_prompt: str,
        messages: list[AgentMessage],
    ) -> list[dict[str, Any]]:
        request: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        # Only the trailing assistant message is still in flight. Echoing reasoning
        # for finished turns is dead weight — no model reads it again, and it was
        # ~40% of all input tokens. Dropping the earlier blocks also leaves the
        # request prefix stable enough for provider-side caching.
        in_flight = -1
        for index in range(len(messages) - 1, -1, -1):
            if isinstance(messages[index], AssistantMessage):
                in_flight = index
                break
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
                    and self.provider.pass_reasoning_content
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

    def _make_request_body(
        self,
        system_prompt: str,
        messages: list[AgentMessage],
        tools: list[Tool],
    ) -> bytes:
        body: dict[str, Any] = {
            "model": self.provider.model,
            "messages": self._request_messages(system_prompt, messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        # Omit `tools` entirely when there are none: some endpoints reject an
        # empty array where a missing key is fine.
        if tools:
            body["tools"] = self._tool_schemas(tools)
        return json.dumps(body).encode("utf-8")

    def stream(
        self,
        system_prompt: str,
        messages: list[AgentMessage],
        tools: list[Tool],
        abort_event: threading.Event | None = None,
    ) -> Generator[ProviderEvent, None, AssistantMessage]:
        """Stream one turn; yields events, returns the AssistantMessage."""
        body = self._make_request_body(system_prompt, messages, tools)
        chunks = self._post("/chat/completions", body)

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
                parsed_usage = _chat_usage_from(chunk.get("usage"))
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
        except ProviderError as exc:
            if exc.usage is None:
                exc.usage = usage
            raise
        except KeyboardInterrupt:
            # Ctrl-C surfaced inside the stream: finalize as aborted.
            aborted = True
        finally:
            chunks.close()  # drop the connection instead of draining it

        if not aborted and stop_reason == "error":
            raise ProviderError(
                failure_reason or "provider error",
                retryable=failure_reason == "insufficient_system_resource",
                usage=usage,
            )

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
                ToolCall(
                    id=acc["id"] or new_call_id(), name=acc["name"], arguments=arguments
                )
            )

        return AssistantMessage(
            content=content,
            model=self.provider.model,
            api=self.api,
            stop_reason="aborted" if aborted else stop_reason,
            usage=usage,
            error_message=failure_reason,
            reasoning_content="".join(reasoning_parts) or None,
        )


def _chat_usage_from(payload: Any) -> Usage | None:
    """chat-completions reports `prompt_tokens`/`completion_tokens`."""
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


# -- responses ----------------------------------------------------------------


class ResponsesBackend(ModelBackend):
    """Streaming backend for OpenAI's /v1/responses endpoint."""

    api = "openai-responses"

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

    def _make_request_body(
        self,
        system_prompt: str,
        messages: list[AgentMessage],
        tools: list[Tool],
    ) -> bytes:
        body: dict[str, Any] = {
            "model": self.provider.model,
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

    def stream(
        self,
        system_prompt: str,
        messages: list[AgentMessage],
        tools: list[Tool],
        abort_event: threading.Event | None = None,
    ) -> Generator[ProviderEvent, None, AssistantMessage]:
        """Stream one turn; yields events, returns the AssistantMessage."""
        body = self._make_request_body(system_prompt, messages, tools)
        chunks = self._post("/responses", body)

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
                        _responses_usage_from(
                            _mapping(chunk.get("response")).get("usage")
                        )
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
                    usage = _responses_usage_from(response.get("usage")) or usage
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
            chunks.close()  # drop the connection instead of draining it

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
            model=self.provider.model,
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


def _responses_usage_from(payload: Any) -> Usage | None:
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


def _as_int(value: Any) -> int:
    """Shared by both usage parsers."""
    return value if isinstance(value, int) else 0


def make_backend(
    provider: Provider, api: str | None = None, reader: type[StreamReader] | None = None
) -> ModelBackend:
    """Build the backend that speaks `api`, defaulting to chat-completions.

    Args:
        provider: The endpoint, credentials and model to read on each request.
        api: Wire format to speak; unknown names are treated as chat-completions
            so a blank flag picks the common flavour.
        reader: Stand-in for the HTTP layer, for tests.

    Returns:
        A backend bound to `provider`.
    """
    if (api or "chat").strip().lower() == "responses":
        return ResponsesBackend(provider, reader)
    return ChatCompletionsBackend(provider, reader)
