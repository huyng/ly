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
from tau.providers import Provider, ProviderError, make_backend
from tau.providers.backend import ChatCompletionsBackend, ResponsesBackend
from tau.providers.sse import StreamReader
from tau.tools import Tool


class FakeReader(StreamReader):
    """Stands in for the real reader: replays `chunks`, records the request."""

    chunks: ClassVar[list[dict[str, Any]]] = []
    built: ClassVar[list["FakeReader"]] = []

    def __init__(self, url, headers, body, timeout=None):
        self.url = url
        self.body = json.loads(body)
        self.closed = False
        FakeReader.built.append(self)

    def read_chunks(self):
        try:
            yield from self.chunks
        finally:
            self.closed = True


def last_request() -> FakeReader:
    """The reader built for the most recent POST."""
    assert FakeReader.built, "no request was posted"
    return FakeReader.built[-1]


class NoopTool(Tool):
    name = "noop"
    description = "Does nothing."
    parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}

    def execute(self, **kwargs: Any):
        raise NotImplementedError


def make_backend_for(chunks, *, model="gpt-6.1-sol") -> ResponsesBackend:
    """A responses backend whose reader replays `chunks` as the whole stream."""
    FakeReader.chunks = chunks
    FakeReader.built.clear()
    provider = Provider(
        model=model,
        base_url="https://api.openai.com/v1",
        api_key="test-key",
    )
    return ResponsesBackend(provider, FakeReader)


def event(kind, **fields):
    return {"type": kind, **fields}


def collect(backend, messages, tools=()):
    """Drive backend.stream to completion, returning (events, message)."""
    stream = backend.stream("system", list(messages), list(tools))
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
    backend = make_backend_for(chunks)

    events, message = collect(backend, [UserMessage(content="hi")])

    assert "".join(e.delta for e in events if isinstance(e, TextDeltaEvent)) == "Hello"
    assert message.text == "Hello"
    assert message.api == "openai-responses"
    assert message.stop_reason == "stop"
    assert message.usage.input == 11
    assert message.usage.output == 4
    assert message.usage.total_tokens == 15
    assert last_request().url == "https://api.openai.com/v1/responses"


def test_reasoning_summary_deltas_are_streamed_and_stored():
    chunks = [
        event("response.reasoning_summary_text.delta", delta="Thinking"),
        event("response.reasoning_summary_text.delta", delta=" hard."),
        event("response.output_text.delta", delta="Done."),
    ]
    backend = make_backend_for(chunks)

    events, message = collect(backend, [UserMessage(content="hi")])

    assert (
        "".join(e.delta for e in events if isinstance(e, ReasoningDeltaEvent))
        == "Thinking hard."
    )
    assert message.reasoning_content == "Thinking hard."
    assert message.text == "Done."


def test_function_call_deltas_become_a_tool_call():
    chunks = [
        event(
            "response.output_item.added",
            output_index=0,
            item={
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "read",
            },
        ),
        event(
            "response.function_call_arguments.delta", item_id="fc_1", delta='{"path": '
        ),
        event(
            "response.function_call_arguments.delta", item_id="fc_1", delta='"a.py"}'
        ),
        event(
            "response.output_item.done",
            item={
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "read",
                "arguments": '{"path": "a.py"}',
            },
        ),
        event("response.completed", response={"status": "completed"}),
    ]
    backend = make_backend_for(chunks)

    events, message = collect(backend, [UserMessage(content="read a.py")])

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
        chunks.append(
            event(
                "response.output_item.added",
                output_index=index,
                item={
                    "id": item_id,
                    "type": "function_call",
                    "call_id": f"call_{index}",
                    "name": "read",
                },
            )
        )
        chunks.append(
            event(
                "response.function_call_arguments.delta",
                item_id=item_id,
                delta=json.dumps({"path": f"{index}.py"}),
            )
        )
    chunks.append(event("response.completed", response={"status": "completed"}))
    backend = make_backend_for(chunks)

    _, message = collect(backend, [UserMessage(content="read files")])

    assert [call.id for call in message.tool_calls] == ["call_0", "call_1"]
    assert [call.arguments for call in message.tool_calls] == [
        {"path": "0.py"},
        {"path": "1.py"},
    ]


@pytest.mark.parametrize("final_arguments", ['{"path": "a.py"}', None])
def test_done_item_prefers_complete_arguments_and_falls_back_to_deltas(final_arguments):
    chunks = [
        event(
            "response.output_item.added",
            item={
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "read",
            },
        ),
        event(
            "response.function_call_arguments.delta",
            item_id="fc_1",
            delta='{"path":' if final_arguments else '{"path": "a.py"}',
        ),
        event(
            "response.output_item.done",
            item={
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "read",
                **(
                    {"arguments": final_arguments}
                    if final_arguments is not None
                    else {}
                ),
            },
        ),
        event("response.completed", response={"status": "completed"}),
    ]
    backend = make_backend_for(chunks)

    _, message = collect(backend, [UserMessage(content="read a.py")])

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
    backend = make_backend_for(chunks)

    _, message = collect(backend, [UserMessage(content="find py")])

    assert message.tool_calls[0].arguments == {"pattern": "*.py"}


def test_failed_response_reports_an_error():
    chunks = [
        event(
            "response.failed",
            response={"status": "failed", "error": {"message": "boom"}},
        )
    ]
    backend = make_backend_for(chunks)

    with pytest.raises(ProviderError, match="boom") as caught:
        collect(backend, [UserMessage(content="hi")])

    assert caught.value.retryable is False


@pytest.mark.parametrize("kind", ["response.failed", "error"])
@pytest.mark.parametrize(
    "code", ["server_error", "rate_limit_exceeded", "invalid_prompt"]
)
def test_stream_errors_are_classified_by_code(kind, code):
    error = {"code": code, "message": "failure details"}
    payload = (
        {
            "response": {
                "error": error,
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
            }
        }
        if kind == "response.failed"
        else error
    )
    backend = make_backend_for([event(kind, **payload)])

    with pytest.raises(ProviderError, match="failure details") as caught:
        collect(backend, [UserMessage(content="hi")])

    assert caught.value.retryable is (code != "invalid_prompt")
    if kind == "response.failed":
        assert caught.value.usage is not None
        assert caught.value.usage.total_tokens == 15
    assert last_request().closed


def test_content_filtered_response_is_not_retryable():
    backend = make_backend_for(
        [
            event(
                "response.incomplete",
                response={"incomplete_details": {"reason": "content_filter"}},
            )
        ]
    )

    with pytest.raises(ProviderError, match="content_filter") as caught:
        collect(backend, [UserMessage(content="hi")])

    assert caught.value.retryable is False


def test_incomplete_response_maps_to_length():
    chunks = [
        event(
            "response.incomplete",
            response={
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
            },
        )
    ]
    backend = make_backend_for(chunks)

    _, message = collect(backend, [UserMessage(content="hi")])

    assert message.stop_reason == "length"


# -- request construction ----------------------------------------------------


def test_request_uses_instructions_and_input_items():
    backend = make_backend_for([])
    history = [
        UserMessage(content="read a.py"),
        AssistantMessage(
            content=[
                TextContent(text="Reading."),
                ToolCall(id="call_1", name="read", arguments={"path": "a.py"}),
            ],
            model="gpt-6.1-sol",
            stop_reason="toolUse",
        ),
        ToolResultMessage(tool_call_id="call_1", tool_name="read", output="print(1)"),
    ]

    collect(backend, history)

    assert last_request().body["instructions"] == "system"
    assert last_request().body["store"] is False
    assert last_request().body["input"] == [
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
    backend = make_backend_for([])

    collect(backend, [UserMessage(content="hi")], tools=[NoopTool()])

    assert last_request().body["tools"] == [
        {
            "type": "function",
            "name": "noop",
            "description": "Does nothing.",
            "parameters": {"type": "object", "properties": {}},
        }
    ]


def test_reasoning_summary_uses_default_effort(monkeypatch):
    monkeypatch.setenv("TAU_REASONING_EFFORT", "high")
    backend = make_backend_for([])

    collect(backend, [UserMessage(content="hi")])

    assert last_request().body["reasoning"] == {"summary": "auto"}


# -- backend selection -------------------------------------------------------


@pytest.mark.parametrize(
    ("api", "expected"),
    [
        (None, "chat"),
        ("chat", "chat"),
        ("responses", "responses"),
        (" RESPONSES ", "responses"),
    ],
)
def test_make_backend_picks_the_backend(api, expected):
    provider = Provider(
        model="m",
        base_url="https://explicit.example/v1",
        api_key="test-key",
    )

    backend = make_backend(provider, api)

    assert isinstance(backend, ResponsesBackend) is (expected == "responses")


def test_make_backend_defaults_to_chat():
    provider = Provider(
        model="m", base_url="https://explicit.example/v1", api_key="test-key"
    )

    assert isinstance(make_backend(provider), ChatCompletionsBackend)
