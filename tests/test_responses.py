"""Responses-API provider tests: request shapes and the named event stream."""

import json
from typing import Any, ClassVar

import pytest

from tau.events import ReasoningDeltaEvent, TextDeltaEvent, ToolNameEvent
from tau.messages import (
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from tau.providers import ResponsesProvider, make_provider
from tau.tools import Tool


class FakeTransport:
    """Stands in for `stream_chunk_transport`, recording the request bytes."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.url: str | None = None
        self.body: dict[str, Any] = {}
        self.closed = False

    def __call__(self, url, headers, body, timeout=None):
        self.url = url
        self.body = json.loads(body)
        try:
            yield from self.chunks
        finally:
            self.closed = True


class NoopTool(Tool):
    name = "noop"
    description = "Does nothing."
    parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}

    def execute(self, **kwargs: Any):
        raise NotImplementedError


def make_provider_for(chunks, *, model="gpt-6.1-sol") -> tuple[ResponsesProvider, FakeTransport]:
    transport = FakeTransport(chunks)
    provider = ResponsesProvider(
        model=model,
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        transport=transport,
    )
    return provider, transport


def event(kind, **fields):
    return {"type": kind, **fields}


def collect(provider, messages, tools=()):
    """Drive provider.stream to completion, returning (events, message)."""
    stream = provider.stream("system", list(messages), list(tools))
    events = []
    while True:
        try:
            events.append(next(stream))
        except StopIteration as stop:
            return events, stop.value


def test_text_deltas_accumulate_and_usage_is_mapped():
    chunks = [
        event("response.output_text.delta", delta="Hel"),
        event("response.output_text.delta", delta="lo"),
        event(
            "response.completed",
            response={
                "status": "completed",
                "usage": {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15},
            },
        ),
    ]
    provider, transport = make_provider_for(chunks)

    events, message = collect(provider, [UserMessage(content="hi")])

    assert "".join(e.delta for e in events if isinstance(e, TextDeltaEvent)) == "Hello"
    assert message.text == "Hello"
    assert message.stop_reason == "stop"
    assert message.usage.input == 11
    assert message.usage.output == 4
    assert message.usage.total_tokens == 15
    assert transport.url == "https://api.openai.com/v1/responses"


def test_reasoning_summary_deltas_are_streamed_and_stored():
    chunks = [
        event("response.reasoning_summary_text.delta", delta="Thinking"),
        event("response.reasoning_summary_text.delta", delta=" hard."),
        event("response.output_text.delta", delta="Done."),
    ]
    provider, _ = make_provider_for(chunks)

    events, message = collect(provider, [UserMessage(content="hi")])

    assert "".join(e.delta for e in events if isinstance(e, ReasoningDeltaEvent)) == "Thinking hard."
    assert message.reasoning_content == "Thinking hard."
    assert message.text == "Done."


def test_function_call_deltas_become_a_tool_call():
    chunks = [
        event(
            "response.output_item.added",
            output_index=0,
            item={"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "read"},
        ),
        event("response.function_call_arguments.delta", item_id="fc_1", delta='{"path": '),
        event("response.function_call_arguments.delta", item_id="fc_1", delta='"a.py"}'),
        event("response.output_item.done", item={
            "id": "fc_1",
            "type": "function_call",
            "call_id": "call_1",
            "name": "read",
            "arguments": '{"path": "a.py"}',
        }),
        event("response.completed", response={"status": "completed"}),
    ]
    provider, _ = make_provider_for(chunks)

    events, message = collect(provider, [UserMessage(content="read a.py")])

    names = [e.name for e in events if isinstance(e, ToolNameEvent)]
    assert names[-1] == "read"
    assert message.stop_reason == "toolUse"
    call = message.tool_calls[0]
    assert call.id == "call_1"
    assert call.name == "read"
    assert call.arguments == {"path": "a.py"}


def test_function_calls_preserve_stream_order_not_item_id_order():
    chunks = []
    for index, item_id in enumerate(["fc_z", "fc_a"]):
        chunks.append(event(
            "response.output_item.added",
            output_index=index,
            item={
                "id": item_id, "type": "function_call",
                "call_id": f"call_{index}", "name": "read",
            },
        ))
        chunks.append(event(
            "response.function_call_arguments.delta",
            item_id=item_id, delta=json.dumps({"path": f"{index}.py"}),
        ))
    chunks.append(event("response.completed", response={"status": "completed"}))
    provider, _ = make_provider_for(chunks)

    _, message = collect(provider, [UserMessage(content="read files")])

    assert [call.id for call in message.tool_calls] == ["call_0", "call_1"]
    assert [call.arguments for call in message.tool_calls] == [
        {"path": "0.py"}, {"path": "1.py"},
    ]


@pytest.mark.parametrize("final_arguments", ['{"path": "a.py"}', None])
def test_done_item_prefers_complete_arguments_and_falls_back_to_deltas(final_arguments):
    chunks = [
        event("response.output_item.added", item={
            "id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "read",
        }),
        event(
            "response.function_call_arguments.delta", item_id="fc_1",
            delta='{"path":' if final_arguments else '{"path": "a.py"}',
        ),
        event("response.output_item.done", item={
            "id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "read",
            **({"arguments": final_arguments} if final_arguments is not None else {}),
        }),
        event("response.completed", response={"status": "completed"}),
    ]
    provider, _ = make_provider_for(chunks)

    _, message = collect(provider, [UserMessage(content="read a.py")])

    assert message.tool_calls[0].arguments == {"path": "a.py"}


def test_done_item_supplies_arguments_the_deltas_missed():
    chunks = [
        event(
            "response.output_item.done",
            item={
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "glob",
                "arguments": '{"pattern": "*.py"}',
            },
        ),
    ]
    provider, _ = make_provider_for(chunks)

    _, message = collect(provider, [UserMessage(content="find py")])

    assert message.tool_calls[0].arguments == {"pattern": "*.py"}


def test_failed_response_reports_an_error():
    chunks = [
        event("response.failed", response={"status": "failed", "error": {"message": "boom"}})
    ]
    provider, _ = make_provider_for(chunks)

    _, message = collect(provider, [UserMessage(content="hi")])

    assert message.stop_reason == "error"
    assert message.error_message == "boom"


def test_incomplete_response_maps_to_length():
    chunks = [
        event(
            "response.incomplete",
            response={"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
        )
    ]
    provider, _ = make_provider_for(chunks)

    _, message = collect(provider, [UserMessage(content="hi")])

    assert message.stop_reason == "length"


# -- request construction ----------------------------------------------------


def test_request_uses_instructions_and_input_items():
    provider, transport = make_provider_for([])
    history = [
        UserMessage(content="read a.py"),
        AssistantMessage(
            content=[
                TextContent(text="Reading."),
                ToolCall(id="call_1", name="read", arguments={"path": "a.py"}),
            ],
            model="gpt-6.1-sol",
            provider="openai",
            stop_reason="toolUse",
        ),
        ToolResultMessage(tool_call_id="call_1", tool_name="read", output="print(1)"),
    ]

    collect(provider, history)

    assert transport.body["instructions"] == "system"
    assert transport.body["store"] is False
    assert transport.body["input"] == [
        {"role": "user", "content": "read a.py"},
        {"role": "assistant", "content": [{"type": "output_text", "text": "Reading."}]},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "read",
            "arguments": json.dumps({"path": "a.py"}),
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "print(1)"},
    ]


def test_tool_schemas_are_flat_on_requests():
    provider, transport = make_provider_for([])

    collect(provider, [UserMessage(content="hi")], tools=[NoopTool()])

    assert transport.body["tools"] == [
        {
            "type": "function",
            "name": "noop",
            "description": "Does nothing.",
            "parameters": {"type": "object", "properties": {}},
        }
    ]


def test_reasoning_effort_travels_as_a_reasoning_object(monkeypatch):
    monkeypatch.setenv("TAU_REASONING_EFFORT", "high")
    provider, transport = make_provider_for([])

    collect(provider, [UserMessage(content="hi")])

    assert transport.body["reasoning"] == {"effort": "high", "summary": "auto"}


def test_no_reasoning_object_by_default(monkeypatch):
    monkeypatch.delenv("TAU_REASONING_EFFORT", raising=False)
    provider, transport = make_provider_for([])

    collect(provider, [UserMessage(content="hi")])

    assert "reasoning" not in transport.body


# -- backend selection -------------------------------------------------------


@pytest.mark.parametrize(
    ("api", "expected"),
    [(None, "chat"), ("chat", "chat"), ("responses", "responses"), (" RESPONSES ", "responses")],
)
def test_make_provider_picks_the_backend(monkeypatch, api, expected):
    monkeypatch.delenv("TAU_API", raising=False)
    provider = make_provider(api=api, model="m", api_key="test-key")

    assert isinstance(provider, ResponsesProvider) is (expected == "responses")


def test_make_provider_honours_tau_api(monkeypatch):
    monkeypatch.setenv("TAU_API", "responses")

    assert isinstance(make_provider(model="m", api_key="test-key"), ResponsesProvider)
