"""Tests for the ly agent harness using a scripted fake backend."""

import json
from typing import Any, ClassVar, cast

import pytest

from ly.agent import AgentSession
from ly.compaction import CompactionConfig
from ly.events import (
    AgentEndEvent,
    AgentStartEvent,
    ErrorEvent,
    MessageEndEvent,
    MessageStartEvent,
    TextDeltaEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from ly.messages import (
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from ly.providers import Provider, ProviderError, make_backend
from ly.presets import Model, Preset
from ly.providers.backend import ChatCompletionsBackend, ModelBackend, ResponsesBackend
from ly.providers.errors import MissingCredentialsError, UnknownProviderError
from ly.providers.sse import StreamReader
from ly.session import SessionManager
from ly.settings import Settings
from ly.tools import Tool, ToolResult
from ly.tools.task import SubAgentTool


class FakeBackend(ModelBackend):
    """Scripts a sequence of turns: each turn is an AssistantMessage."""

    api_format = "fake"

    def __init__(self, turns: list[AssistantMessage]):
        super().__init__(Provider(model="fake-model", base_url="fake", api_key="fake"))
        self.turns = list(turns)
        self.requests: list[tuple[str, list, list]] = []

    def stream_response(self, system_prompt, messages, tools, abort_event=None):
        self.requests.append((system_prompt, list(messages), list(tools)))
        turn = self.turns.pop(0)

        def generate():
            for block in turn.content:
                if isinstance(block, TextContent):
                    for word in block.text.split(" "):
                        yield TextDeltaEvent(delta=word + " ")
            return turn

        return generate()


class EchoTool(Tool):
    name = "echo"
    description = "Echo back the input."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def __init__(self):
        self.calls: list[dict] = []

    def execute(self, text: str, **_: Any) -> ToolResult:
        self.calls.append({"text": text})
        return ToolResult(output=f"echo: {text}")


class SpyWriteTool(Tool):
    """Records writes instead of performing them.

    A blocked test asserts this never runs; an allowed test asserts it does.
    """

    name = "write"
    description = "Record a write without touching the filesystem."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }

    def __init__(self):
        self.calls: list[str] = []

    def execute(self, path: str, content: str, **_: Any) -> ToolResult:
        self.calls.append(path)
        return ToolResult(output=f"wrote {path}")


def make_session(tmp_path, turns) -> tuple[AgentSession, EchoTool, SessionManager]:
    manager = SessionManager(directory=tmp_path)
    manager.new_session()
    tool = EchoTool()
    session = AgentSession(
        backend=FakeBackend(turns),
        tools=[tool],
        session_manager=manager,
        system_prompt="test system prompt",
        cwd=str(tmp_path),
    )
    return session, tool, manager


def test_agent_loop_tool_call_and_final_text(tmp_path):
    turn1 = AssistantMessage(
        content=[ToolCall(id="c1", name="echo", arguments={"text": "hello"})],
        model="fake-model",
        stop_reason="toolUse",
    )
    turn2 = AssistantMessage(
        content=[TextContent(text="all done")],
        model="fake-model",
        stop_reason="stop",
    )
    session, tool, manager = make_session(tmp_path, [turn1, turn2])

    events = []
    session.subscribe(events.append)
    final = session.prompt("run the echo tool")

    assert final.text == "all done"
    assert tool.calls == [{"text": "hello"}]

    kinds = [type(e) for e in events]
    assert kinds[0] is AgentStartEvent
    assert kinds[-1] is AgentEndEvent
    assert kinds.count(MessageEndEvent) == 2
    assert kinds.count(ToolExecutionStartEvent) == 1
    assert kinds.count(ToolExecutionEndEvent) == 1
    assert any(isinstance(e, TextDeltaEvent) and "done" in e.delta for e in events)

    # active branch: user, assistant(toolCall), toolResult, assistant(text)
    messages = session.messages
    assert [type(m) for m in messages] == [
        UserMessage,
        AssistantMessage,
        ToolResultMessage,
        AssistantMessage,
    ]
    assert isinstance(messages[2], ToolResultMessage)
    assert messages[2].output == "echo: hello"
    assert session.last_assistant_text() == "all done"

    # the second request included the tool result
    backend = cast(FakeBackend, session.backend)
    _, second_request_messages, _ = backend.requests[1]
    assert isinstance(second_request_messages[-1], ToolResultMessage)
    assert second_request_messages[-1].output == "echo: hello"


def test_session_jsonl_persistence(tmp_path):
    turn = AssistantMessage(
        content=[TextContent(text="hi there")],
        model="fake-model",
        stop_reason="stop",
    )
    session, _, manager = make_session(tmp_path, [turn])
    session.prompt("say hi")

    session_file = manager.file
    assert session_file is not None
    lines = [json.loads(l) for l in session_file.read_text().splitlines() if l.strip()]
    entries = [l for l in lines if "id" in l]
    pointers = [l for l in lines if "current" in l]
    assert len(entries) == 2  # user + assistant
    assert entries[0]["parent"] is None
    assert entries[1]["parent"] == entries[0]["id"]
    assert pointers[-1]["current"] == entries[-1]["id"]
    assert entries[0]["message"]["role"] == "user"
    assert entries[1]["message"]["role"] == "assistant"

    # reload and verify the active branch is restored
    restored = SessionManager(directory=tmp_path)
    restored.load(session_file)
    assert [type(m) for m in restored.active_branch()] == [
        UserMessage,
        AssistantMessage,
    ]


def test_unknown_tool_records_error_result(tmp_path):
    turn1 = AssistantMessage(
        content=[ToolCall(id="c1", name="nonexistent", arguments={})],
        model="fake-model",
        stop_reason="toolUse",
    )
    turn2 = AssistantMessage(
        content=[TextContent(text="recovered")],
        model="fake-model",
        stop_reason="stop",
    )
    session, _, _ = make_session(tmp_path, [turn1, turn2])
    final = session.prompt("call a missing tool")

    assert final.text == "recovered"
    tool_result = next(m for m in session.messages if isinstance(m, ToolResultMessage))
    assert tool_result.is_error
    assert "unknown tool" in tool_result.output


def test_invalid_argument_type_is_rejected_before_execution(tmp_path):
    turn1 = AssistantMessage(
        content=[ToolCall(id="c1", name="echo", arguments={"text": 42})],
        model="fake-model",
        stop_reason="toolUse",
    )
    turn2 = AssistantMessage(
        content=[TextContent(text="retried")],
        model="fake-model",
        stop_reason="stop",
    )
    session, tool, _ = make_session(tmp_path, [turn1, turn2])
    session.prompt("call with a bad argument type")

    assert tool.calls == []  # never reached the tool
    result = next(m for m in session.messages if isinstance(m, ToolResultMessage))
    assert result.is_error
    assert "invalid arguments for echo" in result.output
    assert "arguments.text must be a string (got integer)" in result.output


def test_missing_required_argument_is_rejected(tmp_path):
    turn1 = AssistantMessage(
        content=[ToolCall(id="c1", name="echo", arguments={})],
        model="fake-model",
        stop_reason="toolUse",
    )
    turn2 = AssistantMessage(
        content=[TextContent(text="retried")],
        model="fake-model",
        stop_reason="stop",
    )
    session, tool, _ = make_session(tmp_path, [turn1, turn2])
    session.prompt("call without the required argument")

    assert tool.calls == []
    result = next(m for m in session.messages if isinstance(m, ToolResultMessage))
    assert result.is_error
    assert "arguments.text is required" in result.output


def test_abort_prevents_tool_execution(tmp_path):
    turn = AssistantMessage(
        content=[ToolCall(id="c1", name="echo", arguments={"text": "nope"})],
        model="fake-model",
        stop_reason="toolUse",
    )
    session, tool, _ = make_session(tmp_path, [turn])

    def aborter(event):
        if isinstance(event, MessageEndEvent):
            session.abort()

    session.subscribe(aborter)
    final = session.prompt("abort before tools run")

    assert final.stop_reason == "toolUse"  # message completed
    assert tool.calls == []  # but tools never ran
    # The call still needs a result, or the next request is rejected outright.
    last = session.messages[-1]
    assert isinstance(last, ToolResultMessage)
    assert (last.tool_call_id, last.is_error) == ("c1", True)


def test_aborted_tool_stream_answers_every_call_before_next_prompt(tmp_path):
    turn = AssistantMessage(
        content=[
            ToolCall(id="c1", name="echo", arguments={"text": "never run"}),
            ToolCall(id="c2", name="echo", arguments={"_raw": '{"text":'}),
        ],
        model="fake-model",
        stop_reason="aborted",
    )
    reply = AssistantMessage(content=[TextContent(text="resumed")], model="fake-model")
    backend = FakeBackend([turn, reply])
    tool = EchoTool()
    session = AgentSession(
        backend=backend,
        tools=[tool],
        session_manager=SessionManager(),
        system_prompt="test system prompt",
        cwd=str(tmp_path),
    )
    events = []
    session.subscribe(events.append)

    assert session.prompt("interrupt the stream") is turn
    assert tool.calls == []
    results = [m for m in session.messages if isinstance(m, ToolResultMessage)]
    assert [r.tool_call_id for r in results] == ["c1", "c2"]
    assert all(r.is_error and "aborted" in r.output for r in results)
    assert isinstance(events[-1], AgentEndEvent)
    assert events[-1].aborted

    history = session.messages
    assert session.prompt("continue") is reply
    request_messages = backend.requests[1][1]
    assert request_messages[:-1] == history
    assert isinstance(request_messages[-1], UserMessage)
    assert request_messages[-1].content == "continue"
    assert tool.calls == []


def test_interrupt_inside_a_tool_answers_every_call(tmp_path):
    """Ctrl-C mid-turn must leave a result for each call the turn announced."""

    class InterruptingTool(Tool):
        name = "echo"
        description = "Interrupts instead of running."
        parameters: ClassVar[dict[str, Any]] = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

        def execute(self, text: str, **_: Any) -> ToolResult:
            raise KeyboardInterrupt

    turn = AssistantMessage(
        content=[
            ToolCall(id="c1", name="echo", arguments={"text": "one"}),
            ToolCall(id="c2", name="echo", arguments={"text": "two"}),
        ],
        model="fake-model",
        stop_reason="toolUse",
    )
    manager = SessionManager(directory=tmp_path)
    manager.new_session()
    session = AgentSession(
        backend=FakeBackend([turn]),
        tools=[InterruptingTool()],
        session_manager=manager,
        system_prompt="test system prompt",
        cwd=str(tmp_path),
    )

    final = session.prompt("interrupt me")

    assert final.stop_reason == "toolUse"
    results = [m for m in session.messages if isinstance(m, ToolResultMessage)]
    assert [r.tool_call_id for r in results] == ["c1", "c2"]
    assert all(r.is_error for r in results)


def test_abort_during_retry_stops_without_another_request(tmp_path):
    class FailingBackend(FakeBackend):
        api_format = "fake"

        def __init__(self):
            super().__init__([])
            self.calls = 0

        def stream_response(self, *_args, **_kwargs):
            self.calls += 1
            raise ProviderError("boom")

    backend = FailingBackend()
    session = AgentSession(
        backend=backend,
        tools=[],
        session_manager=SessionManager(directory=tmp_path),
        system_prompt="test system prompt",
    )

    def abort_on_error(event):
        if isinstance(event, ErrorEvent):
            session.abort()

    session.subscribe(abort_on_error)
    final = session.prompt("hi")

    assert final.stop_reason == "aborted"
    assert final.api_format == "fake"
    assert backend.calls == 1


def test_permanent_error_is_not_retried(tmp_path):
    """A 401 must fail once, immediately — not three times over 3 seconds."""

    class UnauthorizedBackend(FakeBackend):
        api_format = "fake"

        def __init__(self):
            super().__init__([])
            self.calls = 0

        def stream_response(self, *_args, **_kwargs):
            self.calls += 1
            raise ProviderError("HTTP 401 Unauthorized: bad api key", retryable=False)

    backend = UnauthorizedBackend()
    session = AgentSession(
        backend=backend,
        tools=[],
        session_manager=SessionManager(directory=tmp_path),
        system_prompt="test system prompt",
    )
    errors: list[str] = []
    session.subscribe(
        lambda event: (
            errors.append(event.message) if isinstance(event, ErrorEvent) else None
        )
    )

    final = session.prompt("hi")

    assert backend.calls == 1
    assert final.stop_reason == "error"
    assert "401" in (final.error_message or "")
    # One error event only: no "retrying in Ns" chatter for a permanent fault.
    assert errors == ["HTTP 401 Unauthorized: bad api key"]


def test_transient_error_is_retried(tmp_path, monkeypatch):
    monkeypatch.setattr("ly.agent.RETRY_DELAYS", (0, 0, 0))

    class FlakyBackend(FakeBackend):
        api_format = "fake"

        def __init__(self):
            super().__init__([])
            self.calls = 0

        def stream_response(self, *_args, **_kwargs):
            self.calls += 1
            raise ProviderError("HTTP 503 Service Unavailable")

    backend = FlakyBackend()
    session = AgentSession(
        backend=backend,
        tools=[],
        session_manager=SessionManager(directory=tmp_path),
        system_prompt="test system prompt",
    )

    final = session.prompt("hi")

    assert backend.calls == 3
    assert final.stop_reason == "error"
    assert final.api_format == "fake"
    assert "503" in (final.error_message or "")


def test_retry_starts_a_fresh_message(tmp_path, monkeypatch):
    """A retried turn re-streams from its first token, so hosts must see one
    MessageStartEvent per attempt to reset a half-rendered message."""
    monkeypatch.setattr("ly.agent.RETRY_DELAYS", (0, 0, 0))

    class FlakyBackend(FakeBackend):
        api_format = "fake"

        def __init__(self):
            super().__init__([])
            self.calls = 0

        def stream_response(self, *_args, **_kwargs):
            self.calls += 1
            attempt = self.calls

            def gen():
                yield TextDeltaEvent(delta="partial ")
                if attempt == 1:
                    raise ProviderError("stream interrupted: boom")
                return AssistantMessage(
                    content=[TextContent(text="partial done")],
                    model="fake-model",
                )

            return gen()

    backend = FlakyBackend()
    session = AgentSession(
        backend=backend,
        tools=[],
        session_manager=SessionManager(directory=tmp_path),
        system_prompt="test system prompt",
    )
    events = []
    session.subscribe(events.append)

    final = session.prompt("hi")

    assert backend.calls == 2
    assert final.text == "partial done"
    assert [type(e) for e in events].count(MessageStartEvent) == 2


SDK_PRESETS = {
    "openai": Preset(
        base_url="https://openai.test/v1",
        api_format="openai-responses",
        models={"large": Model(context_window=200_000)},
    ),
    "deepseek": Preset(
        base_url="https://deepseek.test",
        pass_reasoning_content=True,
        models={"small": Model(context_window=65_536, keep_turns=3)},
    ),
}


def sdk_session(**overrides) -> AgentSession:
    from ly import create_agent_session

    return create_agent_session(
        model="openai/large",
        api_key="sk-openai",
        persistent=False,
        **overrides,
    )


def test_sdk_factory_sizes_compaction_and_honors_custom_tools(monkeypatch):
    monkeypatch.setattr("ly.settings.BUILT_IN", SDK_PRESETS)
    tool = EchoTool()

    session = sdk_session(tools=[tool])
    defaults = sdk_session()

    assert session.compaction == CompactionConfig(
        threshold_tokens=100_000, max_summary_input_tokens=100_000
    )
    assert session.tools == {"echo": tool}
    assert "read" in defaults.tools
    assert "echo" not in defaults.tools


def test_sdk_switch_replaces_the_whole_model_runtime(monkeypatch):
    monkeypatch.setattr("ly.settings.BUILT_IN", SDK_PRESETS)
    session = sdk_session()
    old_backend = session.backend
    reader = old_backend.reader
    session.session_manager.append(UserMessage("unchanged history"))
    history = session.messages

    settings = Settings.load(env={}, model="deepseek/small", api_key="sk-deepseek")
    backend = make_backend(settings.make_provider(), old_backend.reader)
    session.set_model(backend, settings.compaction)

    provider = session.backend.provider
    assert isinstance(old_backend, ResponsesBackend)
    assert isinstance(session.backend, ChatCompletionsBackend)
    assert session.backend.reader is reader
    assert (
        provider.model,
        provider.base_url,
        provider.api_key,
        provider.api_format,
        provider.pass_reasoning_content,
    ) == (
        "small",
        "https://deepseek.test",
        "sk-deepseek",
        "openai-completions",
        True,
    )
    assert session.compaction == CompactionConfig(
        threshold_tokens=32_768, keep_turns=3, max_summary_input_tokens=32_768
    )
    assert session.messages == history


def test_sdk_switch_honors_endpoint_and_wire_format_overrides(monkeypatch):
    monkeypatch.setattr("ly.settings.BUILT_IN", SDK_PRESETS)
    session = sdk_session()

    settings = Settings.load(
        env={},
        model="deepseek/small",
        api_key="sk-gateway",
        base_url="https://gateway.test/v1",
        api_format="openai-responses",
    )
    backend = make_backend(settings.make_provider(), session.backend.reader)
    session.set_model(backend, settings.compaction)

    provider = session.backend.provider
    assert isinstance(session.backend, ResponsesBackend)
    assert provider.base_url == "https://gateway.test/v1"
    assert provider.api_key == "sk-gateway"


@pytest.mark.parametrize("model", ["deepseek/small", "unknown/model"])
def test_failed_sdk_switch_changes_nothing(model, monkeypatch):
    monkeypatch.setattr("ly.settings.BUILT_IN", SDK_PRESETS)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "must-not-be-read")
    session = sdk_session()
    backend = session.backend
    compaction = session.compaction
    error = (
        MissingCredentialsError if model == "deepseek/small" else UnknownProviderError
    )

    with pytest.raises(error):
        settings = Settings.load(env={}, model=model)
        session.set_model(make_backend(settings.make_provider()), settings.compaction)

    assert session.backend is backend
    assert session.compaction is compaction
    assert backend.provider.api_key == "sk-openai"


def test_model_switching_requires_an_idle_session():
    session = AgentSession(
        backend=FakeBackend([]),
        tools=[],
        session_manager=SessionManager(),
        system_prompt="test",
    )

    backend = session.backend
    compaction = session.compaction
    session._busy = True
    with pytest.raises(RuntimeError, match="run is active"):
        session.set_model(backend, CompactionConfig())
    assert session.backend is backend
    assert session.compaction is compaction


def test_create_agent_session_binds_tools_to_the_requested_cwd(tmp_path):
    """Regression: `cwd` reached the system prompt and the policy guard but
    not the tools, so relative writes landed in the process cwd."""
    from ly import create_agent_session

    session = create_agent_session(
        model="openai/gpt-4.1",
        cwd=str(tmp_path),
        persistent=False,
        base_url="https://explicit.example/v1",
        api_key="test-key",
    )

    assert {getattr(tool, "cwd", None) for tool in session.tools.values()} == {
        str(tmp_path)
    }


def test_create_agent_session_wants_a_provider_with_its_model():
    """The SDK boundary follows the same rule as the CLI: the endpoint comes
    from the model's provider half, so a bare name is refused there too."""
    from ly import create_agent_session
    from ly.providers.errors import UnknownProviderError

    session = create_agent_session(
        model="deepseek/deepseek-chat",
        api_key="test-key",
        persistent=False,
    )
    provider = session.backend.provider
    assert (provider.base_url, provider.model) == (
        "https://api.deepseek.com",
        "deepseek-chat",
    )

    with pytest.raises(UnknownProviderError):
        create_agent_session(model="gpt-4.1", api_key="test-key", persistent=False)


def test_create_agent_session_without_a_model_says_so():
    """There is no built-in model, so the SDK refuses as the CLI does."""
    from ly import create_agent_session
    from ly.providers.errors import MissingModelError

    with pytest.raises(MissingModelError):
        create_agent_session(api_key="test-key", persistent=False)


def failing_backend(reason, *, retryable=False):
    class FailingBackend(FakeBackend):
        api_format = "fake"

        def __init__(self):
            super().__init__([])
            self.calls = 0

        def stream_response(self, *_args, **_kwargs):
            self.calls += 1
            raise ProviderError(
                reason,
                retryable=retryable,
                usage=Usage(input=10, output=5, total_tokens=15),
            )

    return FailingBackend()


def test_content_filter_is_not_retried(tmp_path, monkeypatch):
    monkeypatch.setattr("ly.agent.RETRY_DELAYS", (0, 0, 0))
    backend = failing_backend("content_filter")
    session = AgentSession(
        backend=backend,
        tools=[],
        session_manager=SessionManager(directory=tmp_path),
        system_prompt="test system prompt",
    )

    assert session.prompt("hi").stop_reason == "error"
    assert backend.calls == 1


def test_server_overload_is_retried(tmp_path, monkeypatch):
    monkeypatch.setattr("ly.agent.RETRY_DELAYS", (0, 0, 0))
    backend = failing_backend("insufficient_system_resource", retryable=True)
    session = AgentSession(
        backend=backend,
        tools=[],
        session_manager=SessionManager(directory=tmp_path),
        system_prompt="test system prompt",
    )

    assert session.prompt("hi").stop_reason == "error"
    assert backend.calls == 3
    assert session.usage.total_tokens == 45


@pytest.mark.parametrize("failure", ["subscriber", "persistence"])
def test_prompt_startup_failure_ends_the_run(failure, monkeypatch):
    session = AgentSession(
        backend=FakeBackend([]),
        tools=[],
        session_manager=SessionManager(),
        system_prompt="test",
    )
    if failure == "persistence":

        def fail_append(_message):
            raise OSError("write failed")

        monkeypatch.setattr(session.session_manager, "append", fail_append)
    events = []
    session.subscribe(events.append)
    if failure == "subscriber":

        def fail(event):
            if isinstance(event, AgentStartEvent):
                raise RuntimeError("callback failed")

        session.subscribe(fail)

    with pytest.raises(RuntimeError if failure == "subscriber" else OSError):
        session.prompt("hi")

    assert session.busy is False
    assert isinstance(events[-1], AgentEndEvent)
    assert events[-1].aborted is False


@pytest.mark.parametrize("event_type", [TextDeltaEvent, MessageEndEvent])
@pytest.mark.parametrize("error_type", [RuntimeError, ProviderError])
def test_subscriber_failure_is_not_retried(event_type, error_type):
    message = AssistantMessage(content=[TextContent(text="done")], model="fake")
    backend = FakeBackend([message])
    session = AgentSession(
        backend=backend,
        tools=[],
        session_manager=SessionManager(),
        system_prompt="test",
    )

    def fail(event):
        if isinstance(event, event_type):
            raise error_type("callback failed")

    session.subscribe(fail)
    with pytest.raises(error_type, match="callback failed"):
        session.prompt("hi")

    assert len(backend.requests) == 1
    assert session.busy is False
    saved = [m for m in session.messages if isinstance(m, AssistantMessage)]
    assert len(saved) == (1 if event_type is MessageEndEvent else 0)


@pytest.mark.parametrize("error_type", [OSError, ProviderError])
def test_persistence_failure_is_not_retried(monkeypatch, error_type):
    message = AssistantMessage(content=[TextContent(text="done")], model="fake")
    backend = FakeBackend([message])
    manager = SessionManager()
    session = AgentSession(
        backend=backend,
        tools=[],
        session_manager=manager,
        system_prompt="test",
    )
    append = manager.append

    def fail(message):
        if isinstance(message, AssistantMessage):
            raise error_type("write failed")
        return append(message)

    monkeypatch.setattr(manager, "append", fail)
    with pytest.raises(error_type, match="write failed"):
        session.prompt("hi")

    assert len(backend.requests) == 1
    assert session.busy is False


@pytest.mark.parametrize("during_stream", [False, True])
def test_unexpected_provider_failure_is_not_retried(during_stream):
    class BrokenBackend(FakeBackend):
        calls = 0

        def stream_response(self, *_args, **_kwargs):
            self.calls += 1
            if not during_stream:
                raise RuntimeError("provider bug")

            def generate():
                yield TextDeltaEvent(delta="partial")
                raise RuntimeError("provider bug")

            return generate()

    backend = BrokenBackend([])
    session = AgentSession(
        backend=backend,
        tools=[],
        session_manager=SessionManager(),
        system_prompt="test",
    )
    with pytest.raises(RuntimeError, match="provider bug"):
        session.prompt("hi")

    assert backend.calls == 1
    assert session.busy is False


@pytest.mark.parametrize("api_format", ["openai-completions", "openai-responses"])
def test_streamed_provider_failure_retries_and_counts_usage(api_format, monkeypatch):
    monkeypatch.setattr("ly.agent.RETRY_DELAYS", (0, 0, 0))
    requests = []

    class FakeReader(StreamReader):
        def __init__(self, url, headers, body, timeout=None):
            requests.append(url)

        def read_chunks(self):
            if api_format == "openai-completions":
                yield {
                    "choices": [
                        {"delta": {}, "finish_reason": "insufficient_system_resource"}
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                }
            else:
                yield {
                    "type": "response.failed",
                    "response": {
                        "error": {"code": "server_error", "message": "try later"},
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "total_tokens": 15,
                        },
                    },
                }

    provider = Provider(
        model="m",
        base_url="https://explicit.example/v1",
        api_key="test",
        api_format=api_format,
    )
    backend = make_backend(provider, FakeReader)
    session = AgentSession(
        backend=backend,
        tools=[],
        session_manager=SessionManager(),
        system_prompt="test",
    )

    assert session.prompt("hi").stop_reason == "error"
    assert len(requests) == 3
    assert session.usage.total_tokens == 45
    assert len(session.messages) == 2


def test_repl_renders_text_and_reasoning_deltas(capsys):
    """Regression: the interactive REPL must print streamed text (it used to
    only render tool events, so replies were invisible)."""
    from ly.cli import Repl
    from ly.events import MessageEndEvent, MessageStartEvent, ReasoningDeltaEvent

    class StubSession:
        backend = type("B", (), {"provider": type("P", (), {"model": "m"})()})()
        session_manager = type("S", (), {"file": "x"})
        cwd = None

        def subscribe(self, cb):
            return lambda: None

    repl = Repl(cast(AgentSession, StubSession()))
    repl._on_event(MessageStartEvent())
    repl._on_event(ReasoningDeltaEvent(delta="thinking... "))
    repl._on_event(TextDeltaEvent(delta="Hello"))
    repl._on_event(TextDeltaEvent(delta=" there"))
    message = AssistantMessage(content=[TextContent(text="Hello there")], model="m")
    repl._on_event(MessageEndEvent(message=message))

    out = capsys.readouterr().out
    assert "thinking... " in out
    assert "╭─ thinking\n" in out
    assert "╰─\n" in out
    assert "Hello there" in out
    assert out.endswith("\n")


def test_tool_guard_blocks_call_and_reports_to_model(tmp_path):
    turn1 = AssistantMessage(
        content=[ToolCall(id="c1", name="echo", arguments={"text": "hello"})],
        model="fake-model",
        stop_reason="toolUse",
    )
    turn2 = AssistantMessage(
        content=[TextContent(text="blocked")],
        model="fake-model",
        stop_reason="stop",
    )
    session, tool, _ = make_session(tmp_path, [turn1, turn2])
    session.tool_guard = lambda call: "not allowed"

    session.prompt("run the echo tool")

    assert tool.calls == []  # never executed
    result = next(m for m in session.messages if isinstance(m, ToolResultMessage))
    assert result.is_error
    assert "blocked by policy" in result.output


def test_default_guard_blocks_before_execute(tmp_path):
    turn1 = AssistantMessage(
        content=[
            ToolCall(
                id="c1",
                name="write",
                arguments={"path": "/etc/escape.txt", "content": "x"},
            )
        ],
        model="fake-model",
        stop_reason="toolUse",
    )
    turn2 = AssistantMessage(content=[TextContent(text="done")], model="fake-model")
    tool = SpyWriteTool()
    session = AgentSession(
        backend=FakeBackend([turn1, turn2]),
        tools=[tool],
        session_manager=SessionManager(directory=tmp_path),
        system_prompt="test system prompt",
        cwd=str(tmp_path),
    )

    session.prompt("write outside")

    assert tool.calls == []  # execute() was never reached
    result = next(m for m in session.messages if isinstance(m, ToolResultMessage))
    assert result.is_error
    assert "blocked by policy" in result.output


def test_default_guard_allows_write_inside_cwd(tmp_path):
    turn1 = AssistantMessage(
        content=[
            ToolCall(
                id="c1",
                name="write",
                arguments={"path": "ok.txt", "content": "x"},
            )
        ],
        model="fake-model",
        stop_reason="toolUse",
    )
    turn2 = AssistantMessage(content=[TextContent(text="done")], model="fake-model")
    tool = SpyWriteTool()
    session = AgentSession(
        backend=FakeBackend([turn1, turn2]),
        tools=[tool],
        session_manager=SessionManager(directory=tmp_path),
        system_prompt="test system prompt",
        cwd=str(tmp_path),
    )

    session.prompt("write inside")

    assert tool.calls == ["ok.txt"]


def test_task_subagent_inherits_tool_guard(tmp_path):
    seen: list[str] = []

    def guard(call: ToolCall) -> str | None:
        seen.append(call.name)
        return None

    turns = [
        AssistantMessage(
            content=[
                ToolCall(id="c1", name="subagent", arguments={"prompt": "investigate"})
            ],
            model="fake-model",
            stop_reason="toolUse",
        ),
        # The sub-agent's turn runs through the parent's guard too.
        AssistantMessage(
            content=[ToolCall(id="c2", name="read", arguments={"path": "x"})],
            model="fake-model",
            stop_reason="toolUse",
        ),
        AssistantMessage(
            content=[TextContent(text="sub report")],
            model="fake-model",
        ),
        AssistantMessage(content=[TextContent(text="final")], model="fake-model"),
    ]
    session = AgentSession(
        backend=FakeBackend(turns),
        tools=[SubAgentTool(cwd=str(tmp_path))],
        session_manager=SessionManager(directory=tmp_path),
        system_prompt="parent prompt",
        cwd=str(tmp_path),
        tool_guard=guard,
    )

    session.prompt("delegate")

    assert "read" in seen
