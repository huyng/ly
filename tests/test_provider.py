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
from tau.providers import (
    ChatCompletionsBackend,
    MissingBaseURLError,
    MissingCredentialsError,
    Provider,
    ProviderError,
    stream_chunk_transport,
)
from tau.tools import Tool


class FakeTransport:
    """Stands in for `stream_chunk_transport`, recording the request bytes."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.url: str | None = None
        self.headers: dict[str, str] = {}
        self.body: dict[str, Any] = {}
        self.closed = False

    def __call__(self, url, headers, body, timeout=None):
        self.url, self.headers = url, headers
        self.body = json.loads(body)
        try:
            yield from self.chunks
        finally:
            self.closed = True


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


def collect(provider, messages):
    """Drive provider.stream to completion, returning (events, message)."""
    stream = provider.stream("system", messages, [])
    events = []
    while True:
        try:
            events.append(next(stream))
        except StopIteration as stop:
            return events, stop.value


def make_provider(chunks, model="deepseek-flash") -> tuple[Provider, FakeTransport]:
    transport = FakeTransport(chunks)
    provider = Provider(
        model=model,
        base_url="https://api.deepseek.com",
        api_key="test-key",
        reader=transport,
    )
    return provider, transport


class NoopTool(Tool):
    """Minimal tool, so a request body has a function schema to carry."""

    name = "noop"
    description = "Does nothing."
    parameters: ClassVar[dict[str, Any]] = {}

    def execute(self, **kwargs: Any):
        raise NotImplementedError


def body_with_tools(provider: Provider) -> dict[str, Any]:
    """The request body tau would send for one turn that has tools."""
    backend = cast(ChatCompletionsBackend, provider.backend)
    return json.loads(
        backend._request_body("system", [], [NoopTool()], provider.model, False)
    )


def test_reasoning_is_streamed_stored_and_sent_back():
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    chunks = [
        chunk(reasoning="Let me "),
        chunk(reasoning="think."),
        chunk(content="Hello"),
        chunk(content=" there"),
        chunk(finish_reason="stop", usage=usage),
    ]
    provider, _ = make_provider(chunks)

    events, message = collect(provider, [UserMessage(content="hi")])

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

    request = cast(ChatCompletionsBackend, provider.backend)._request_messages(
        "system",
        [
            UserMessage(content="hi"),
            message,
            ToolResultMessage(tool_call_id="c1", tool_name="read", output="ok"),
        ],
        provider.pass_reasoning_content,
    )
    assert as_dict(request[2])["reasoning_content"] == "Let me think."


def test_reasoning_is_not_sent_to_non_deepseek_providers():
    provider = Provider(
        model="glm-4.6",
        base_url="https://api.z.ai/api/paas/v4/",
        api_key="test-key",
        reader=FakeTransport([]),
    )
    message = AssistantMessage(
        content=[TextContent(text="hi")],
        model="glm-4.6",
        reasoning_content="private reasoning",
    )

    request = cast(ChatCompletionsBackend, provider.backend)._request_messages(
        "system", [message], provider.pass_reasoning_content
    )

    assert "reasoning_content" not in request[1]


def test_reasoning_passthrough_can_be_forced_explicitly():
    provider = Provider(
        model="glm-4.6",
        base_url="https://api.z.ai/api/paas/v4/",
        api_key="test-key",
        reader=FakeTransport([]),
        pass_reasoning_content=True,
    )
    message = AssistantMessage(
        content=[TextContent(text="hi")],
        model="glm-4.6",
        reasoning_content="private reasoning",
    )

    request = cast(ChatCompletionsBackend, provider.backend)._request_messages(
        "system", [message], provider.pass_reasoning_content
    )

    assert as_dict(request[1])["reasoning_content"] == "private reasoning"


def test_reasoning_is_sent_back_alongside_tool_calls():
    provider, _ = make_provider([])
    message = AssistantMessage(
        content=[
            ToolCall(id="call_1", name="read", arguments={"path": "a.py"}),
        ],
        model="deepseek-flash",
        stop_reason="toolUse",
        reasoning_content="I should read the file.",
    )

    request = cast(ChatCompletionsBackend, provider.backend)._request_messages(
        "system", [message], provider.pass_reasoning_content
    )
    assistant = as_dict(request[1])

    assert assistant["reasoning_content"] == "I should read the file."
    assert assistant["tool_calls"][0]["function"]["name"] == "read"


@pytest.mark.parametrize(
    ("reason", "retryable"),
    [("insufficient_system_resource", True), ("content_filter", False)],
)
def test_failure_finish_reasons_raise_classified_errors(reason, retryable):
    provider, transport = make_provider(
        [
            chunk(finish_reason=reason),
            chunk(
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            ),
        ]
    )

    with pytest.raises(ProviderError, match=reason) as caught:
        collect(provider, [UserMessage(content="hi")])

    assert caught.value.retryable is retryable
    assert caught.value.usage is not None
    assert caught.value.usage.total_tokens == 15
    assert transport.closed


# -- transport: SSE framing and HTTP errors ----------------------------------


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


def test_transport_parses_sse_and_stops_at_done(monkeypatch):
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

    chunks = list(stream_chunk_transport("https://x/v1/chat/completions", {}, b"{}"))

    assert [c["choices"][0]["delta"]["content"] for c in chunks] == ["Hello", " there"]
    assert response.closed
    assert calls[0]["request"].get_method() == "POST"
    assert calls[0]["request"].full_url == "https://x/v1/chat/completions"


def test_transport_decodes_split_utf8_lines(monkeypatch):
    payload = 'data: {"choices": [{"delta": {"content": "héllo ✓"}}]}\n'
    raw = payload.encode()  # multi-byte chars, still one complete line
    fake_urlopen(monkeypatch, FakeResponse([raw]))

    chunks = list(stream_chunk_transport("https://x/v1/chat/completions", {}, b"{}"))

    assert chunks[0]["choices"][0]["delta"]["content"] == "héllo ✓"


def test_transport_surfaces_api_error_message(monkeypatch):
    error = urllib.error.HTTPError(
        "https://x/v1/chat/completions",
        401,
        "Unauthorized",
        hdrs=email.message.Message(),
        fp=__import__("io").BytesIO(b'{"error": {"message": "bad api key"}}'),
    )
    fake_urlopen(monkeypatch, error=error)

    with pytest.raises(ProviderError) as caught:
        list(stream_chunk_transport("https://x/v1/chat/completions", {}, b"{}"))

    assert "401" in str(caught.value)
    assert "bad api key" in str(caught.value)


def test_transport_reports_unreachable_host(monkeypatch):
    fake_urlopen(monkeypatch, error=urllib.error.URLError("name resolution failed"))

    with pytest.raises(ProviderError) as caught:
        list(stream_chunk_transport("https://nope.invalid/chat/completions", {}, b"{}"))

    assert "cannot reach" in str(caught.value)


def test_transport_marks_unreachable_host_retryable(monkeypatch):
    fake_urlopen(monkeypatch, error=urllib.error.URLError("name resolution failed"))

    with pytest.raises(ProviderError) as caught:
        list(stream_chunk_transport("https://nope.invalid/chat/completions", {}, b"{}"))

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
        list(stream_chunk_transport("https://x/v1/chat/completions", {}, b"{}"))

    assert caught.value.retryable is True


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_permanent_statuses_are_not_retryable(monkeypatch, code):
    fake_urlopen(monkeypatch, error=http_error(code, "Permanent"))

    with pytest.raises(ProviderError) as caught:
        list(stream_chunk_transport("https://x/v1/chat/completions", {}, b"{}"))

    assert caught.value.retryable is False


class TruncatedResponse(FakeResponse):
    """Yields one good line, then fails the way a dropped socket does."""

    def __iter__(self):
        yield b'data: {"choices": [{"delta": {"content": "partial"}}]}\n'
        raise http.client.IncompleteRead(b"", 512)


def test_transport_wraps_a_stream_that_drops_mid_turn(monkeypatch):
    fake_urlopen(monkeypatch, TruncatedResponse([]))

    with pytest.raises(ProviderError) as caught:
        list(stream_chunk_transport("https://x/v1/chat/completions", {}, b"{}"))

    assert "stream interrupted" in str(caught.value)
    assert "IncompleteRead" in str(caught.value)
    assert caught.value.retryable is True


def test_transport_wraps_a_socket_timeout(monkeypatch):
    class StalledResponse(FakeResponse):
        def __iter__(self):
            raise TimeoutError("the read operation timed out")

    fake_urlopen(monkeypatch, StalledResponse([]))

    with pytest.raises(ProviderError) as caught:
        list(stream_chunk_transport("https://x/v1/chat/completions", {}, b"{}"))

    assert "stream interrupted" in str(caught.value)
    assert caught.value.retryable is True


def test_transport_does_not_rewrap_its_own_errors(monkeypatch):
    """A malformed-event ProviderError must not be mistaken for a socket fault."""
    fake_urlopen(monkeypatch, FakeResponse(["data: not-json\n"]))

    with pytest.raises(ProviderError) as caught:
        list(stream_chunk_transport("https://x/v1/chat/completions", {}, b"{}"))

    assert "malformed stream event" in str(caught.value)


# -- request construction ----------------------------------------------------


def test_request_omits_tools_when_none_are_present():
    provider, transport = make_provider([chunk(content="ok")])

    collect(provider, [UserMessage(content="hi")])

    assert "tools" not in transport.body
    assert transport.body["stream"] is True
    assert transport.body["stream_options"] == {"include_usage": True}
    assert transport.url == "https://api.deepseek.com/chat/completions"
    assert transport.headers["Authorization"] == "Bearer test-key"


def test_request_includes_tool_schemas_when_present():
    provider, transport = make_provider([])
    stream = provider.stream("system", [], [NoopTool()])
    with pytest.raises(StopIteration):
        next(stream)

    assert transport.body["tools"][0]["function"]["name"] == "noop"


def test_model_change_takes_effect_on_the_next_request():
    provider, transport = make_provider([chunk(finish_reason="stop")])
    provider.model = "switched-model"

    collect(provider, [UserMessage(content="hi")])

    assert transport.body["model"] == "switched-model"


def test_response_records_api_and_model():
    provider = Provider(
        base_url="https://api.deepseek.com",
        api_key="test",
        reader=FakeTransport([chunk(content="ok")]),
    )
    _, message = collect(provider, [])

    assert message.api == "openai-completions"
    assert message.model == provider.model


def test_request_omits_reasoning_effort(monkeypatch):
    monkeypatch.setenv("TAU_REASONING_EFFORT", "high")
    provider, _ = make_provider([])

    assert "reasoning_effort" not in body_with_tools(provider)


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
    provider, _ = make_provider(chunks)

    events, message = collect(provider, [UserMessage(content="hi")])

    assert [e.name for e in events if isinstance(e, ToolNameEvent)] == ["read"]
    assert message.stop_reason == "toolUse"
    assert message.tool_calls[0].arguments == {"path": "a.py"}


def test_reasoning_from_finished_turns_is_not_sent_back():
    provider, _ = make_provider([])
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

    request = cast(ChatCompletionsBackend, provider.backend)._request_messages(
        "system",
        [
            UserMessage(content="go"),
            first,
            ToolResultMessage(tool_call_id="c1", tool_name="read", output="ok"),
            second,
        ],
        provider.pass_reasoning_content,
    )

    assert "reasoning_content" not in as_dict(request[2])
    assert as_dict(request[4])["reasoning_content"] == "recent thoughts"


def test_request_omits_an_assistant_message_with_no_content():
    provider = Provider(
        model="deepseek-flash",
        base_url="https://api.deepseek.com",
        api_key="test-key",
        reader=FakeTransport([]),
    )
    aborted = AssistantMessage(content=[], model="m", stop_reason="aborted")

    request = cast(ChatCompletionsBackend, provider.backend)._request_messages(
        "system",
        [UserMessage(content="hi"), aborted, UserMessage(content="again")],
        provider.pass_reasoning_content,
    )

    assert [as_dict(entry)["role"] for entry in request] == ["system", "user", "user"]


def test_stream_that_yields_no_text_has_no_empty_text_block():
    provider, _ = make_provider([chunk(finish_reason="stop")])

    _, message = collect(provider, [UserMessage(content="hi")])

    assert message.content == []
    assert message.stop_reason == "stop"


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_missing_base_url_raises_at_construction(api):
    with pytest.raises(MissingBaseURLError) as exc:
        Provider(api=api, api_key="test-key")
    assert "TAU_BASE_URL" in str(exc.value)
    assert exc.value.retryable is False


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize(
    "base_url", ["https://api.openai.com/v1", "https://api.deepseek.com"]
)
def test_missing_credentials_raise_at_construction(api, base_url):
    with pytest.raises(MissingCredentialsError) as exc:
        Provider(api=api, base_url=base_url)
    assert str(exc.value) == "Missing credentials. Set TAU_API_KEY or pass --api-key."


def test_abort_mid_stream_closes_transport_and_reports_aborted():
    import threading

    abort = threading.Event()

    class AbortingTransport(FakeTransport):
        def __call__(self, url, headers, body, timeout=None):
            try:
                yield chunk(content="first")
                abort.set()  # simulate Ctrl-C between chunks
                yield chunk(content="never")
            finally:
                self.closed = True

    transport = AbortingTransport([])
    provider = Provider(
        model="m",
        base_url="https://explicit.example/v1",
        api_key="k",
        reader=transport,
    )
    stream = provider.stream(
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
    assert transport.closed
