"""Tests for the REPL's status line: it must show the size of the next request,
not just the cumulative token totals."""

from tau.agent import AgentSession
from tau.cli import Repl
from tau.compaction import CompactionConfig
from tau.events import TaskActivityEvent, ToolExecutionEndEvent, ToolExecutionStartEvent
from tau.messages import (
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
)
from tau.session import SessionManager
from tau.theme import Theme


class FakeProvider:
    name = "fake"
    model = "fake-model"

    def stream(self, *_args, **_kwargs):
        raise AssertionError("the status line must not call the provider")


def make_repl(threshold_tokens: int = 100_000, enabled: bool = True) -> Repl:
    session = AgentSession(
        provider=FakeProvider(),  # type: ignore[arg-type]
        tools=[],
        session_manager=SessionManager(in_memory=True),
        system_prompt="system",
        compaction=CompactionConfig(
            enabled=enabled, threshold_tokens=threshold_tokens
        ),
    )
    # A run that billed 82k for its prompt and left nothing behind: the next
    # request is exactly what the provider measured last time.
    session.session_manager.append(
        AssistantMessage(
            content=[TextContent(text="done")], model="fake", provider="fake",
            usage=Usage(input=82_000, output=12, total_tokens=82_012),
        )
    )
    session.usage = Usage(input=1_234, output=567, total_tokens=1_801)
    return Repl(session, Theme(enabled=False))


def test_task_activity_shows_calls_and_status_without_child_output(capsys):
    repl = make_repl()
    parent = ToolCall(id="parent", name="task", arguments={"description": "explore"})
    repl._on_event(ToolExecutionStartEvent(parent))
    for name, arguments, is_error in [
        ("read", {"path": "a.py"}, False),
        ("bash", {"command": "pwd\ngit status --short"}, True),
    ]:
        # Child ids may coincide with the parent's; timings must stay separate.
        call = ToolCall(id="parent", name=name, arguments=arguments)
        repl._on_event(TaskActivityEvent("explore", ToolExecutionStartEvent(call)))
        assert f"→ {name}" in capsys.readouterr().out
        result = ToolResultMessage(call.id, name, "private child output", is_error)
        repl._on_event(TaskActivityEvent("explore", ToolExecutionEndEvent(call, result)))
        out = capsys.readouterr().out
        assert f"{'✗' if is_error else '✓'} {name}" in out
        assert "private child output" not in out
    assert "parent" in repl._tool_started
    result = ToolResultMessage(parent.id, "task", "report")
    repl._on_event(ToolExecutionEndEvent(parent, result))
    assert not repl._tool_started
    assert "report" in capsys.readouterr().out


def test_status_line_shows_the_next_request_against_the_compaction_threshold(
    capsys,
):
    make_repl()._print_usage(elapsed=2.0, output_tokens=100)

    out = capsys.readouterr().out
    assert "ctx 82,000 / 100,000" in out
    assert "1,234 in" in out  # cumulative totals are still reported


def test_status_line_omits_the_threshold_when_compaction_is_disabled(capsys):
    make_repl(enabled=False)._print_usage(elapsed=2.0, output_tokens=100)

    out = capsys.readouterr().out
    assert "ctx 82,000" in out
    assert "100,000" not in out
