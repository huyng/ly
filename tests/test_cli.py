"""Tests for the REPL's status line: it must show the size of the next request,
not just the cumulative token totals."""

from tau.agent import AgentSession
from tau.cli import Repl
from tau.compaction import CompactionConfig
from tau.messages import AssistantMessage, TextContent, Usage
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
