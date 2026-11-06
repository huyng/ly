"""Provider tests, focused on DeepSeek thinking-mode compatibility."""

import email.message
import http.client
import json
import urllib.error
from typing import Any, ClassVar, cast

import pytest

from tau.events import ReasoningDeltaEvent, TextDeltaEvent, ToolNameEvent
from tau.messages import (
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    deserialize_message,
    serialize_message,
)
from tau.providers import Provider, ProviderError
from tau.providers.backend import ChatCompletionsBackend
from tau.providers.errors import MissingBaseURLError, MissingCredentialsError
from tau.providers.sse import StreamReader
from tau.tools import Tool


class FakeReader(StreamReader):
    """Stands in for the real reader: replays `chunks`, records the request."""

    chunks: ClassVar[list[dict[str, Any]]] = []
    built: ClassVar[list["FakeReader"]] = []

    def __init__(self, url, headers, body, timeout=None):
        self.url = url
        self.headers = headers
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


def chunk(
    *, content=None, reasoning=None, tool_calls=None, finish_reason=None, usage=None
):
    delta: dict[str, Any] = {"content": content, "reasoning_content": reasoning}
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    return {
        "choices": [{"delta": delta, "finish_reason": finish_reason}],
        "usage": usage,
    }


def as_dict(value: object) -> dict[str, Any]:
    """View a request message as a plain dict for tests."""
    return cast(dict[str, Any], value)


def collect(backend, messages):
    """Drive backend.stream_response to completion, returning (events, message)."""
    stream = backend.stream_response("system", messages, [])
    events = []
    while True:
        try:
            events.append(next(stream))
        except StopIteration as stop:
            return events, stop.value


def make_backend(
    chunks, model="deepseek-flash", base_url="https://api.deepseek.com", **config
) -> tuple[ChatCompletionsBackend, type[FakeReader]]:
    """A chat backend whose reader replays `chunks` as the whole response."""
    FakeReader.chunks = chunks
    FakeReader.built.clear()
    provider = Provider(model=model, base_url=base_url, api_key="test-key", **config)
    return ChatCompletionsBackend(provider, FakeReader), FakeReader


class NoopTool(Tool):
    """Minimal tool, so a request body has a function schema to carry."""

    name = "noop"
    description = "Does nothing."
    parameters: ClassVar[dict[str, Any]] = {}

    def execute(self, **kwargs: Any):
        raise NotImplementedError


def body_with_tools(backend: ChatCompletionsBackend) -> dict[str, Any]:
    """The request body tau would send for one turn that has tools."""
    return json.loads(backend._make_request_body("system", [], [NoopTool()]))


def test_reasoning_is_streamed_stored_and_sent_back():
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    chunks = [
        chunk(reasoning="Let me "),
        chunk(reasoning="think."),
        chunk(content="Hello"),
        chunk(content=" there"),
        chunk(finish_reason="stop", usage=usage),
    ]
    backend, _ = make_backend(chunks)

    events, message = collect(backend, [UserMessage(content="hi")])

    assert (
        "".join(e.delta for e in events if isinstance(e, ReasoningDeltaEvent))
        == "Let me think."
    )
    assert (
        "".join(e.delta for e in events if isinstance(e, TextDeltaEvent))
        == "Hello there"
    )
    assert message.reasoning_content == "Let me think."
    assert message.text == "Hello there"
    assert message.stop_reason == "stop"
    assert message.usage.total_tokens == 15

    # DeepSeek needs reasoning_content echoed for the in-flight turn, so it has to
    # survive serialization and request construction; finished turns drop it.
    data = serialize_message(message)
    assert data["reasoningContent"] == "Let me think."
    restored = deserialize_message(data)
    assert isinstance(restored, AssistantMessage)
    assert restored.reasoning_content == "Let me think."

    request = backend._request_messages(
        "system",
        [
            UserMessage(content="hi"),
            message,
            ToolResultMessage(tool_call_id="c1", tool_name="read", output="ok"),
        ],
    )
    assert as_dict(request[2])["reasoning_content"] == "Let me think."


def test_reasoning_is_not_sent_to_non_deepseek_providers():
    backend, _ = make_backend(
        [], model="glm-4.6", base_url="https://api.z.ai/api/paas/v4/"
    )
    message = AssistantMessage(
        content=[TextContent(text="hi")],
        model="glm-4.6",
        reasoning_content="private reasoning",
    )

    request = backend._request_messages("system", [message])

    assert "reasoning_content" not in request[1]


def test_reasoning_passthrough_can_be_forced_explicitly():
    backend, _ = make_backend(
        [],
        model="glm-4.6",
        base_url="https://api.z.ai/api/paas/v4/",
        pass_reasoning_content=True,
    )
    message = AssistantMessage(
        content=[TextContent(text="hi")],
        model="glm-4.6",
        reasoning_content="private reasoning",
    )

    request = backend._request_messages("system", [message])

    assert as_dict(request[1])["reasoning_content"] == "private reasoning"


def test_reasoning_is_sent_back_alongside_tool_calls():
    backend, _ = make_backend([])
    message = AssistantMessage(
        content=[
            ToolCall(id="call_1", name="read", arguments={"path": "a.py"}),
        ],
        model="deepseek-flash",
        stop_reason="toolUse",
        reasoning_content="I should read the file.",
    )

    request = backend._request_messages("system", [message])
    assistant = as_dict(request[1])

    assert assistant["reasoning_content"] == "I should read the file."
    assert assistant["tool_calls"][0]["function"]["name"] == "read"


@pytest.mark.parametrize(
    ("reason", "retryable"),
    [("insufficient_system_resource", True), ("content_filter", False)],
)
def test_failure_finish_reasons_raise_classified_errors(reason, retryable):
    backend, _ = make_backend(
        [
            chunk(finish_reason=reason),
            chunk(
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            ),
        ]
    )

    with pytest.raises(ProviderError, match=reason) as caught:
        collect(backend, [UserMessage(content="hi")])

    assert caught.value.retryable is retryable
    assert caught.value.usage is not None
    assert caught.value.usage.total_tokens == 15
    assert last_request().closed


# -- reader: SSE framing and HTTP errors ----------------------------------


class FakeResponse:
    """Minimal stand-in for http.client.HTTPResponse: iterable of byte lines."""

    def __init__(self, lines):
        self.lines = [
            line.encode() if isinstance(line, str) else line for line in lines
        ]
        self.closed = False

    def __iter__(self):
        return iter(self.lines)

    def close(self):
        self.closed = True


def fake_urlopen(monkeypatch, response=None, error=None):
    calls = []

    def urlopen(request, timeout=None):
        calls.append({"request": request, "timeout": timeout})
        if error is not None:
            raise error
        return response

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    return calls


def test_reader_parses_sse_and_stops_at_done(monkeypatch):
    body = [
        ": ping\n",
        "event: message\n",
        "\n",
        'data: {"choices": [{"delta": {"content": "Hello"}}]}\n',
        'data: {"choices": [{"delta": {"content": " there"}}]}\r\n',
        "data: [DONE]\n",
        'data: {"choices": [{"delta": {"content": "ignored"}}]}\n',
    ]
    response = FakeResponse(body)
    calls = fake_urlopen(monkeypatch, response)

    chunks = list(
        StreamReader("https://x/v1/chat/completions", {}, b"{}").read_chunks()
    )

    assert [c["choices"][0]["delta"]["content"] for c in chunks] == ["Hello", " there"]
    assert response.closed
    assert calls[0]["request"].get_method() == "POST"
    assert calls[0]["request"].full_url == "https://x/v1/chat/completions"


def test_reader_decodes_split_utf8_lines(monkeypatch):
    payload = 'data: {"choices": [{"delta": {"content": "héllo ✓"}}]}\n'
    raw = payload.encode()  # multi-byte chars, still one complete line
    fake_urlopen(monkeypatch, FakeResponse([raw]))

    chunks = list(
        StreamReader("https://x/v1/chat/completions", {}, b"{}").read_chunks()
    )

    assert chunks[0]["choices"][0]["delta"]["content"] == "héllo ✓"


def test_reader_surfaces_api_error_message(monkeypatch):
    error = urllib.error.HTTPError(
        "https://x/v1/chat/completions",
        401,
        "Unauthorized",
        hdrs=email.message.Message(),
        fp=__import__("io").BytesIO(b'{"error": {"message": "bad api key"}}'),
    )
    fake_urlopen(monkeypatch, error=error)

    with pytest.raises(ProviderError) as caught:
        list(StreamReader("https://x/v1/chat/completions", {}, b"{}").read_chunks())

    assert "401" in str(caught.value)
    assert "bad api key" in str(caught.value)


def test_reader_reports_unreachable_host(monkeypatch):
    fake_urlopen(monkeypatch, error=urllib.error.URLError("name resolution failed"))

    with pytest.raises(ProviderError) as caught:
        list(
            StreamReader(
                "https://nope.invalid/chat/completions", {}, b"{}"
            ).read_chunks()
        )

    assert "cannot reach" in str(caught.value)


def test_reader_marks_unreachable_host_retryable(monkeypatch):
    fake_urlopen(monkeypatch, error=urllib.error.URLError("name resolution failed"))

    with pytest.raises(ProviderError) as caught:
        list(
            StreamReader(
                "https://nope.invalid/chat/completions", {}, b"{}"
            ).read_chunks()
        )

    assert caught.value.retryable is True


def http_error(code, reason, body=b""):
    return urllib.error.HTTPError(
        "https://x/v1/chat/completions",
        code,
        reason,
        hdrs=email.message.Message(),
        fp=__import__("io").BytesIO(body),
    )


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
def test_transient_statuses_are_retryable(monkeypatch, code):
    fake_urlopen(monkeypatch, error=http_error(code, "Transient"))

    with pytest.raises(ProviderError) as caught:
        list(StreamReader("https://x/v1/chat/completions", {}, b"{}").read_chunks())

    assert caught.value.retryable is True


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_permanent_statuses_are_not_retryable(monkeypatch, code):
    fake_urlopen(monkeypatch, error=http_error(code, "Permanent"))

    with pytest.raises(ProviderError) as caught:
        list(StreamReader("https://x/v1/chat/completions", {}, b"{}").read_chunks())

    assert caught.value.retryable is False


class TruncatedResponse(FakeResponse):
    """Yields one good line, then fails the way a dropped socket does."""

    def __iter__(self):
        yield b'data: {"choices": [{"delta": {"content": "partial"}}]}\n'
        raise http.client.IncompleteRead(b"", 512)


def test_reader_wraps_a_stream_that_drops_mid_turn(monkeypatch):
    fake_urlopen(monkeypatch, TruncatedResponse([]))

    with pytest.raises(ProviderError) as caught:
        list(StreamReader("https://x/v1/chat/completions", {}, b"{}").read_chunks())

    assert "stream interrupted" in str(caught.value)
    assert "IncompleteRead" in str(caught.value)
    assert caught.value.retryable is True


def test_reader_wraps_a_socket_timeout(monkeypatch):
    class StalledResponse(FakeResponse):
        def __iter__(self):
            raise TimeoutError("the read operation timed out")

    fake_urlopen(monkeypatch, StalledResponse([]))

    with pytest.raises(ProviderError) as caught:
        list(StreamReader("https://x/v1/chat/completions", {}, b"{}").read_chunks())

    assert "stream interrupted" in str(caught.value)
    assert caught.value.retryable is True


# urllib wraps failures while sending the request, but not while reading the
# status line: http.client's own exceptions escape `urlopen` unwrapped.
@pytest.mark.parametrize(
    "error",
    [
        http.client.RemoteDisconnected("Remote end closed connection without response"),
        TimeoutError("timed out"),
        ConnectionResetError(104, "Connection reset by peer"),
    ],
)
def test_reader_wraps_a_connection_lost_before_the_response(monkeypatch, error):
    fake_urlopen(monkeypatch, error=error)

    with pytest.raises(ProviderError) as caught:
        list(StreamReader("https://x/v1/chat/completions", {}, b"{}").read_chunks())

    assert "no response from" in str(caught.value)
    assert type(error).__name__ in str(caught.value)
    assert caught.value.retryable is True


def test_reader_keeps_a_hangup_distinct_from_an_unreachable_host(monkeypatch):
    """A hangup means the host was reached; it must not say 'cannot reach'."""
    fake_urlopen(monkeypatch, error=http.client.RemoteDisconnected("closed"))

    with pytest.raises(ProviderError) as caught:
        list(StreamReader("https://x/v1/chat/completions", {}, b"{}").read_chunks())

    assert "cannot reach" not in str(caught.value)


def test_reader_does_not_rewrap_its_own_errors(monkeypatch):
    """A malformed-event ProviderError must not be mistaken for a socket fault."""
    fake_urlopen(monkeypatch, FakeResponse(["data: not-json\n"]))

    with pytest.raises(ProviderError) as caught:
        list(StreamReader("https://x/v1/chat/completions", {}, b"{}").read_chunks())

    assert "malformed stream event" in str(caught.value)
    assert "no response from" not in str(caught.value)


# -- request construction ----------------------------------------------------


def test_request_omits_tools_when_none_are_present():
    backend, _ = make_backend([chunk(content="ok")])

    collect(backend, [UserMessage(content="hi")])

    assert "tools" not in last_request().body
    assert last_request().body["stream"] is True
    assert last_request().body["stream_options"] == {"include_usage": True}
    assert last_request().url == "https://api.deepseek.com/chat/completions"
    assert last_request().headers["Authorization"] == "Bearer test-key"


def test_request_includes_tool_schemas_when_present():
    backend, _ = make_backend([])
    stream = backend.stream_response("system", [], [NoopTool()])
    with pytest.raises(StopIteration):
        next(stream)

    assert last_request().body["tools"][0]["function"]["name"] == "noop"


def test_model_change_takes_effect_on_the_next_request():
    backend, _ = make_backend([chunk(finish_reason="stop")])
    backend.provider.model = "switched-model"

    collect(backend, [UserMessage(content="hi")])

    assert last_request().body["model"] == "switched-model"


def test_response_records_api_and_model():
    backend, _ = make_backend([chunk(content="ok")])
    _, message = collect(backend, [])

    assert message.api == "openai-completions"
    assert message.model == backend.provider.model


def test_request_omits_reasoning_effort(monkeypatch):
    monkeypatch.setenv("TAU_REASONING_EFFORT", "high")
    backend, _ = make_backend([])

    assert "reasoning_effort" not in body_with_tools(backend)


def test_tool_call_deltas_accumulate_across_chunks():
    chunks = [
        chunk(
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_1",
                    "function": {"name": "read", "arguments": '{"pa'},
                }
            ]
        ),
        chunk(tool_calls=[{"index": 0, "function": {"arguments": 'th": "a.py"}'}}]),
        chunk(finish_reason="tool_calls"),
    ]
    backend, _ = make_backend(chunks)

    events, message = collect(backend, [UserMessage(content="hi")])

    assert [e.name for e in events if isinstance(e, ToolNameEvent)] == ["read"]
    assert message.stop_reason == "toolUse"
    assert message.tool_calls[0].arguments == {"path": "a.py"}


def test_reasoning_from_finished_turns_is_not_sent_back():
    backend, _ = make_backend([])
    first = AssistantMessage(
        content=[ToolCall(id="c1", name="read", arguments={"path": "a.py"})],
        model="deepseek-flash",
        stop_reason="toolUse",
        reasoning_content="first thoughts",
    )
    second = AssistantMessage(
        content=[ToolCall(id="c2", name="read", arguments={"path": "b.py"})],
        model="deepseek-flash",
        stop_reason="toolUse",
        reasoning_content="recent thoughts",
    )

    request = backend._request_messages(
        "system",
        [
            UserMessage(content="go"),
            first,
            ToolResultMessage(tool_call_id="c1", tool_name="read", output="ok"),
            second,
        ],
    )

    assert "reasoning_content" not in as_dict(request[2])
    assert as_dict(request[4])["reasoning_content"] == "recent thoughts"


def test_request_omits_an_assistant_message_with_no_content():
    backend, _ = make_backend([])
    aborted = AssistantMessage(content=[], model="m", stop_reason="aborted")

    request = backend._request_messages(
        "system",
        [UserMessage(content="hi"), aborted, UserMessage(content="again")],
    )

    assert [as_dict(entry)["role"] for entry in request] == ["system", "user", "user"]


def test_stream_that_yields_no_text_has_no_empty_text_block():
    backend, _ = make_backend([chunk(finish_reason="stop")])

    _, message = collect(backend, [UserMessage(content="hi")])

    assert message.content == []
    assert message.stop_reason == "stop"


def test_missing_base_url_raises_at_construction():
    with pytest.raises(MissingBaseURLError) as exc:
        Provider(api_key="test-key")
    assert "TAU_BASE_URL" in str(exc.value)
    assert exc.value.retryable is False


@pytest.mark.parametrize(
    "base_url", ["https://api.openai.com/v1", "https://api.deepseek.com"]
)
def test_missing_credentials_raise_at_construction(base_url):
    with pytest.raises(MissingCredentialsError) as exc:
        Provider(base_url=base_url)
    assert str(exc.value) == "Missing credentials. Set TAU_API_KEY or pass --api-key."


def test_abort_mid_stream_closes_reader_and_reports_aborted():
    import threading

    abort = threading.Event()

    class AbortingReader(FakeReader):
        def read_chunks(self):
            try:
                yield chunk(content="first")
                abort.set()  # simulate Ctrl-C between chunks
                yield chunk(content="never")
            finally:
                self.closed = True

    FakeReader.chunks = []
    FakeReader.built.clear()
    backend = ChatCompletionsBackend(
        Provider(model="m", base_url="https://explicit.example/v1", api_key="k"),
        AbortingReader,
    )
    stream = backend.stream_response(
        "system", [UserMessage(content="hi")], [], abort_event=abort
    )

    events = []
    while True:
        try:
            events.append(next(stream))
        except StopIteration as stop:
            message = stop.value
            break

    assert [e.delta for e in events if isinstance(e, TextDeltaEvent)] == ["first"]
    assert message.text == "first"
    assert message.stop_reason == "aborted"
    assert last_request().closed
