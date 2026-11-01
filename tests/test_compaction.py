"""Tests for context compaction: splitting, session re-rooting, and the
agent loop's threshold behavior."""

import threading
from typing import Any

import pytest

from tau.agent import AgentSession
from tau.compaction import (
    SUMMARY_PREFIX,
    SUMMARY_SYSTEM_PROMPT,
    CompactionConfig,
    estimate_tokens,
    split_index,
    summarize,
)
from tau.events import (
    CompactionEndEvent,
    CompactionStartEvent,
    ErrorEvent,
    TextDeltaEvent,
)
from tau.messages import (
    AgentMessage,
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from tau.providers import ProviderError
from tau.session import SessionManager


class FakeProvider:
    api = "fake"
    provider = "fake"

    def __init__(self, turns: list[AssistantMessage]):
        self.turns = list(turns)
        self.model = "fake-model"
        self.requests: list[dict[str, Any]] = []

    def stream(self, system_prompt, messages, tools, abort_event=None):
        self.requests.append(
            {
                "system": system_prompt,
                "messages": list(messages),
                "tools": [tool.name for tool in tools],
            }
        )
        turn = self.turns.pop(0)

        def generate():
            for block in turn.content:
                if isinstance(block, TextContent):
                    yield TextDeltaEvent(delta=block.text)
            return turn

        return generate()


def reply(text: str) -> AssistantMessage:
    return AssistantMessage(content=[TextContent(text=text)], model="fake", provider="fake")


def summary_requests(provider) -> list[dict[str, Any]]:
    return [r for r in provider.requests if r["system"] == SUMMARY_SYSTEM_PROMPT]


def unknown_tool_turn() -> AssistantMessage:
    return AssistantMessage(
        content=[ToolCall(id="call-x", name="nope", arguments={})],
        model="fake", provider="fake", stop_reason="toolUse",
    )


def failing(message: str = "boom") -> AssistantMessage:
    return AssistantMessage(
        content=[], model="fake", provider="fake",
        stop_reason="error", error_message=message,
    )


def user_texts(messages) -> list[str]:
    return [message.content for message in messages if isinstance(message, UserMessage)]


def final_text(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, AssistantMessage):
            return message.text
    return ""


def build_history(manager, turns: int, usage_input: int = 0) -> None:
    for index in range(turns):
        manager.append(UserMessage(content=f"question {index} " + "x" * 400))
        manager.append(
            AssistantMessage(
                content=[TextContent(text=f"answer {index} " + "y" * 400)],
                model="fake",
                provider="fake",
                usage=Usage(input=usage_input),
            )
        )


def make_session(turns: list[AssistantMessage], **overrides) -> tuple[AgentSession, FakeProvider]:
    provider = FakeProvider(turns)
    config = CompactionConfig(
        enabled=True,
        threshold_tokens=overrides.pop("threshold_tokens", 1),
        keep_turns=overrides.pop("keep_turns", 2),
        max_summary_input_tokens=overrides.pop("max_summary_input_tokens", 30_000),
    )
    session = AgentSession(
        provider=provider,
        tools=[],
        session_manager=SessionManager(in_memory=True),
        system_prompt="sys",
        compaction=config,
    )
    return session, provider


# -- split_index ---------------------------------------------------------------


def test_split_index_none_without_enough_turns():
    messages = [UserMessage(content="a"), reply("b")]
    assert split_index(messages, keep_turns=4) is None


def test_split_index_lands_on_a_user_message():
    messages = []
    for index in range(5):
        messages.append(UserMessage(content=f"u{index}"))
        messages.append(reply(f"a{index}"))
    index = split_index(messages, keep_turns=2)
    assert index is not None
    assert isinstance(messages[index], UserMessage)
    assert messages[index].content == "u3"


def test_split_index_keeps_tool_pairs_together():
    """A cut must never orphan a tool result from its assistant tool call."""
    messages: list[Any] = [
        UserMessage(content="u0"),
        AssistantMessage(
            content=[ToolCall(id="c1", name="read", arguments={"path": "x"})],
            model="fake", provider="fake",
        ),
        ToolResultMessage(tool_call_id="c1", tool_name="read", output="data"),
        UserMessage(content="u1"),
        reply("done"),
        UserMessage(content="u2"),
    ]
    index = split_index(messages, keep_turns=1)
    assert index is not None
    assert isinstance(messages[index], UserMessage)


def test_split_never_cuts_inside_a_turn():
    """Whatever the shape, the cut lands on a user message (or nowhere)."""
    shapes = [
        [reply("warmup"), UserMessage(content="u0"), reply("a0")],
        [UserMessage(content="u0"), reply("a0"), UserMessage(content="u1")],
        [
            UserMessage(content="u0"),
            AssistantMessage(
                content=[ToolCall(id="c1", name="read", arguments={})],
                model="fake", provider="fake",
            ),
            ToolResultMessage(tool_call_id="c1", tool_name="read", output="x"),
            UserMessage(content="u1"),
            reply("a1"),
        ],
    ]
    for messages in shapes:
        index = split_index(messages, keep_turns=1)
        if index is not None:
            assert isinstance(messages[index], UserMessage)


# -- estimate_tokens -----------------------------------------------------------


def test_estimate_tokens_counts_tool_output():
    small: list[AgentMessage] = [UserMessage(content="hi")]
    large: list[AgentMessage] = [
        ToolResultMessage(tool_call_id="c", tool_name="read", output="z" * 4000)
    ]
    assert estimate_tokens(large) > estimate_tokens(small) + 900


# -- session re-rooting --------------------------------------------------------


def test_compact_reroots_the_branch_and_persists(tmp_path):
    manager = SessionManager(directory=tmp_path)
    manager.new_session()
    build_history(manager, 3)
    before = len(manager.active_branch())
    keep = manager.active_branch()[-2:]

    manager.compact(UserMessage(content=f"{SUMMARY_PREFIX}\nnotes\n</conversation_summary>"), keep)

    branch = manager.active_branch()
    assert len(branch) == 3
    assert isinstance(branch[0], UserMessage)
    assert branch[0].content.startswith(SUMMARY_PREFIX)

    reloaded = SessionManager(directory=tmp_path)
    assert manager.file is not None
    reloaded.load(manager.file)
    assert [m.role for m in reloaded.active_branch()] == [m.role for m in branch]
    assert user_texts(reloaded.active_branch()) == user_texts(branch)
    assert len(reloaded.entries) > before  # old entries are still on disk


# -- agent loop ----------------------------------------------------------------


def test_compaction_triggers_above_the_threshold():
    session, provider = make_session([reply("SUM"), reply("final")])
    build_history(session.session_manager, 3)

    session.prompt("next question")

    summary_request, real_request = provider.requests[0], provider.requests[1]
    assert summary_request["tools"] == []
    assert len(summary_request["messages"]) == 1
    assert "Previous summary" in summary_request["messages"][0].content

    first = real_request["messages"][0]
    assert isinstance(first, UserMessage)
    assert first.content.startswith(SUMMARY_PREFIX)
    assert "SUM" in first.content

    assert user_texts(session.messages)[0].startswith(SUMMARY_PREFIX)
    assert final_text(session.messages) == "final"


def test_compaction_keeps_recent_user_turns_verbatim():
    session, _ = make_session([reply("SUM"), reply("final")], keep_turns=1)
    build_history(session.session_manager, 4)

    session.prompt("latest question")

    contents = [m.content for m in session.messages if isinstance(m, UserMessage)]
    assert contents[0].startswith(SUMMARY_PREFIX)
    assert contents[-1] == "latest question"


def test_no_compaction_below_the_threshold():
    session, provider = make_session([reply("final")], threshold_tokens=10_000_000)
    build_history(session.session_manager, 3)

    session.prompt("next")

    assert len(provider.requests) == 1
    assert not user_texts(session.messages)[0].startswith(SUMMARY_PREFIX)


def test_compaction_triggers_on_measured_prompt_tokens():
    # ~400 tokens by the char estimate, 120k as billed by the provider. Trusting
    # the estimate here is how requests ran past the budget the threshold exists
    # to enforce.
    session, provider = make_session(
        [reply("SUM"), reply("final")], threshold_tokens=100_000
    )
    build_history(session.session_manager, 3, usage_input=120_000)

    session.prompt("next question")

    assert len(summary_requests(provider)) == 1
    assert user_texts(session.messages)[0].startswith(SUMMARY_PREFIX)


def test_compaction_trusts_measured_usage_over_a_char_heavy_estimate():
    # The estimate reads this branch as ~10k tokens; the provider billed 1k for
    # it. Compacting on the estimate discards history that already fits.
    session, provider = make_session([reply("final")], threshold_tokens=5_000)
    manager = session.session_manager
    for index in range(3):
        manager.append(UserMessage(content=f"question {index}"))
        manager.append(
            AssistantMessage(
                content=[TextContent(text="z" * 40_000 if index == 2 else "answer")],
                model="fake",
                provider="fake",
                usage=Usage(input=1_000),
            )
        )

    session.prompt("next")

    assert summary_requests(provider) == []
    assert len(provider.requests) == 1


def test_disabled_compaction_never_calls_the_provider():
    session, provider = make_session([reply("final")])
    session.compaction = CompactionConfig(enabled=False, threshold_tokens=1)
    build_history(session.session_manager, 3)

    session.prompt("next")

    assert len(provider.requests) == 1


def test_failed_summary_leaves_history_untouched():
    session, provider = make_session([failing(), reply("final")])
    build_history(session.session_manager, 3)
    errors: list[str] = []
    session.subscribe(
        lambda event: errors.append(event.message) if isinstance(event, ErrorEvent) else None
    )

    session.prompt("next")

    assert any("compaction failed" in message for message in errors)
    assert user_texts(session.messages)[0].startswith("question 0")
    assert final_text(session.messages) == "final"
    assert len(provider.requests) == 2  # summary attempt, then the real turn


@pytest.mark.parametrize("stop_reason", ["aborted", "length", "toolUse"])
def test_unfinished_summary_leaves_history_untouched(stop_reason):
    summary = reply("partial notes")
    summary.stop_reason = stop_reason
    session, provider = make_session([summary])
    build_history(session.session_manager, 3)
    manager = session.session_manager
    entries = dict(manager.entries)
    current_id = manager.current_id

    assert session.compact() is False

    assert manager.entries == entries
    assert manager.current_id == current_id
    assert len(provider.requests) == 1


def test_cancellation_before_branch_update_preserves_history(monkeypatch):
    session, _ = make_session([])
    build_history(session.session_manager, 3)
    manager = session.session_manager
    entries = dict(manager.entries)
    current_id = manager.current_id

    def cancelled_summary(*_):
        session.abort()
        return "SUM"

    monkeypatch.setattr("tau.agent.summarize", cancelled_summary)

    assert session.compact() is False
    assert manager.entries == entries
    assert manager.current_id == current_id


@pytest.mark.parametrize("stop_reason", ["stop", "aborted"])
def test_cancelled_compaction_stops_the_run(stop_reason):
    summary = reply("partial notes")
    summary.stop_reason = stop_reason
    session, provider = make_session([summary], max_summary_input_tokens=200)
    build_history(session.session_manager, 3)
    current_id = session.session_manager.current_id

    class CancellingProvider(FakeProvider):
        def stream(self, system_prompt, messages, tools, abort_event=None):
            result = yield from super().stream(system_prompt, messages, tools, abort_event)
            if stop_reason == "stop":
                session.abort()
            return result

    provider = CancellingProvider(provider.turns)
    session.provider = provider

    result = session.prompt("next")

    assert result.stop_reason == "aborted"
    assert session.busy is False
    assert len(provider.requests) == 1
    assert len(summary_requests(provider)) == 1
    assert session.session_manager.current_id != current_id
    assert user_texts(session.messages)[0].startswith("question 0")
    assert user_texts(session.messages)[-1] == "next"
    assert len(session.messages) == 8  # original history, new prompt, aborted stub


def test_failed_summary_is_not_retried_within_the_run():
    # An unknown-tool call forces a second loop iteration, which would try to
    # compact again if the failure flag were not sticky for the run.
    session, provider = make_session(
        [failing(), unknown_tool_turn(), reply("final")]
    )
    build_history(session.session_manager, 3)
    session.prompt("next")
    assert len(summary_requests(provider)) == 1


def test_compaction_events_are_emitted():
    session, _ = make_session([reply("SUM"), reply("final")])
    build_history(session.session_manager, 3)
    events: list[Any] = []
    session.subscribe(events.append)

    session.prompt("next")

    starts = [e for e in events if isinstance(e, CompactionStartEvent)]
    ends = [e for e in events if isinstance(e, CompactionEndEvent)]
    assert starts and ends
    assert starts[0].messages == 4
    assert ends[0].tokens_after < ends[0].tokens_before


def test_summary_usage_counts_toward_session_usage():
    summary = AssistantMessage(
        content=[TextContent(text="SUM")],
        model="fake",
        provider="fake",
        usage=Usage(input=100, output=10, total_tokens=110),
    )
    session, _ = make_session([summary, reply("final")])
    build_history(session.session_manager, 3)

    session.prompt("next")

    assert session.usage.total_tokens == 110


def test_failed_summary_usage_counts_toward_session_usage():
    class FailingProvider(FakeProvider):
        def stream(self, *_args, **_kwargs):
            raise ProviderError(
                "summary failed", retryable=False,
                usage=Usage(input=100, output=10, total_tokens=110),
            )

    usage = Usage()
    with pytest.raises(ProviderError, match="summary failed"):
        summarize(
            FailingProvider([]), [UserMessage(content="hi")], CompactionConfig(), usage=usage,
        )

    assert usage.total_tokens == 110


def test_large_histories_are_summarized_in_batches():
    session, provider = make_session(
        [*(reply(f"SUM-{index}") for index in range(6)), reply("final")],
        max_summary_input_tokens=200,
    )
    build_history(session.session_manager, 4)

    session.prompt("next")

    requests = summary_requests(provider)
    assert len(requests) == 6
    assert "Previous summary" in requests[0]["messages"][0].content
    assert final_text(session.messages) == "final"


def test_explicit_compact_works_without_a_prompt():
    session, provider = make_session([reply("SUM")])
    build_history(session.session_manager, 3)

    assert session.compact() is True
    assert len(provider.requests) == 1
    assert user_texts(session.messages)[0].startswith(SUMMARY_PREFIX)


def test_compact_returns_false_with_too_few_turns():
    session, provider = make_session([reply("SUM")])
    session.session_manager.append(UserMessage(content="only one"))
    assert session.compact() is False
    assert provider.requests == []


# -- summarize() ---------------------------------------------------------------


def test_summarize_does_not_request_a_batch_after_cancellation():
    provider = FakeProvider([reply("SUM")])
    abort_event = threading.Event()
    abort_event.set()

    with pytest.raises(RuntimeError, match="compaction aborted"):
        summarize(provider, [UserMessage(content="hi")], CompactionConfig(), abort_event)

    assert provider.requests == []


def test_summarize_rejects_empty_summary():
    provider = FakeProvider([reply("   ")])
    with pytest.raises(RuntimeError, match="empty summary"):
        summarize(provider, [UserMessage(content="hi")], CompactionConfig())


# -- usefulness guard ----------------------------------------------------------


def test_summary_larger_than_the_history_is_not_applied():
    """A branch that is short, or dominated by one huge tool result, can
    summarize to something bigger than itself; verbatim wins."""
    session, _ = make_session([reply("z" * 5000), reply("final")])
    build_history(session.session_manager, 3)
    events: list[Any] = []
    session.subscribe(events.append)

    session.prompt("next")

    assert user_texts(session.messages)[0].startswith("question 0")
    assert final_text(session.messages) == "final"
    ends = [event for event in events if isinstance(event, CompactionEndEvent)]
    assert ends and ends[0].skipped


def test_unhelpful_summary_is_not_retried_within_the_run():
    session, provider = make_session([reply("z" * 5000), unknown_tool_turn(), reply("final")])
    build_history(session.session_manager, 3)

    session.prompt("next")

    assert len(summary_requests(provider)) == 1
