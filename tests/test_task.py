"""Tests for the `task` sub-agent tool."""

from typing import Any

import pytest

from tau.agent import AgentSession
from tau.events import (
    TaskActivityEvent,
    TextDeltaEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from tau.messages import (
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
)
from tau.session import SessionManager
from tau.tools import default_tools
from tau.tools.task import TaskTool


class FakeProvider:
    """Scripts a sequence of turns and records every request it receives.

    The parent and its sub-agent share one provider instance, so `requests`
    interleaves both conversations in call order: the sub-agent's turn lands
    between the parent's tool call and the parent's next turn.
    """

    name = "fake"

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
                "model": self.model,
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


def failing(message: str) -> AssistantMessage:
    return AssistantMessage(
        content=[], model="fake", provider="fake",
        stop_reason="error", error_message=message,
    )


def task_call(**arguments: Any) -> AssistantMessage:
    return AssistantMessage(
        content=[ToolCall(id="call-1", name="task", arguments=arguments)],
        model="fake",
        provider="fake",
        stop_reason="toolUse",
    )


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "a.py").write_text("def handler():\n    return 1\n")
    return tmp_path


def make_session(workspace, turns):
    provider = FakeProvider(turns)
    session = AgentSession(
        provider=provider,
        tools=default_tools(cwd=str(workspace)),
        session_manager=SessionManager(in_memory=True),
        system_prompt="parent prompt",
        cwd=str(workspace),
    )
    return session, provider


def test_task_runs_a_subagent_and_returns_only_its_report(workspace):
    session, provider = make_session(
        workspace,
        [
            task_call(description="explore", prompt="Where is handler defined?"),
            reply("handler is defined in a.py:1"),
            reply("done"),
        ],
    )
    session.prompt("go")

    tool_result = [
        message for message in session.messages if isinstance(message, ToolResultMessage)
    ][-1]
    assert "handler is defined in a.py:1" in tool_result.output
    assert "sub-agent: 0 tool calls" in tool_result.output
    assert not tool_result.is_error


@pytest.mark.parametrize("path", ["a.py", "missing.py"])
def test_subagent_activity_is_live_and_kept_out_of_parent_context(workspace, path):
    call = ToolCall(id="child-1", name="read", arguments={"path": path})
    session, provider = make_session(
        workspace,
        [
            task_call(description="explore", prompt="find handler"),
            AssistantMessage(
                content=[call], model="fake", provider="fake", stop_reason="toolUse"
            ),
            reply("report"),
            reply("done"),
        ],
    )
    activities = []

    def observe(event):
        if isinstance(event, TaskActivityEvent):
            # Forwarding happens during execution, before the report request.
            assert len(provider.requests) == 2
            activities.append(event)

    session.subscribe(observe)
    session.prompt("go")

    assert [type(item.event) for item in activities] == [
        ToolExecutionStartEvent, ToolExecutionEndEvent
    ]
    assert all(item.description == "explore" for item in activities)
    assert all(item.event.tool_call == call for item in activities)
    end = activities[-1].event
    assert isinstance(end, ToolExecutionEndEvent)
    assert end.result.is_error == (path == "missing.py")
    assert not any(
        isinstance(message, ToolResultMessage) and message.tool_call_id == call.id
        for message in session.messages
    )
    assert "def handler" not in str(provider.requests[-1]["messages"])


def test_subagent_works_in_a_fresh_context(workspace):
    session, provider = make_session(
        workspace,
        [
            task_call(prompt="Where is handler defined?"),
            reply("a.py:1"),
            reply("done"),
        ],
    )
    session.prompt("a long parent question " * 50)

    sub_request = provider.requests[1]
    assert len(sub_request["messages"]) == 1
    assert sub_request["messages"][0].content == "Where is handler defined?"
    assert "sub-agent" in sub_request["system"]


def test_subagent_tool_set_is_read_only_and_non_recursive(workspace):
    session, provider = make_session(
        workspace,
        [task_call(prompt="look around"), reply("ok"), reply("done")],
    )
    session.prompt("go")

    parent_tools = provider.requests[0]["tools"]
    sub_tools = provider.requests[1]["tools"]
    assert {"read", "grep", "glob", "bash"} == set(sub_tools)
    assert {"edit", "write", "task"}.isdisjoint(sub_tools)
    assert {"edit", "write", "task"} <= set(parent_tools)


def test_subagent_inherits_the_current_model(workspace):
    session, provider = make_session(
        workspace,
        [task_call(prompt="look around"), reply("ok"), reply("done")],
    )
    session.set_model("other-model")
    session.prompt("go")

    assert provider.requests[1]["model"] == "other-model"


def test_subagent_sees_the_delegated_prompt_verbatim(workspace):
    session, provider = make_session(
        workspace,
        [task_call(prompt="find the tests\nand list them"), reply("ok"), reply("done")],
    )
    session.prompt("go")
    assert provider.requests[1]["messages"][0].content == "find the tests\nand list them"


def test_empty_subagent_report_is_replaced_with_a_note(workspace):
    session, _ = make_session(
        workspace,
        [task_call(prompt="look"), reply("   "), reply("done")],
    )
    session.prompt("go")
    tool_result = [
        message for message in session.messages if isinstance(message, ToolResultMessage)
    ][-1]
    assert "without producing a final report" in tool_result.output


def test_unbound_task_tool_is_an_error(tmp_path):
    result = TaskTool(cwd=str(tmp_path)).execute(prompt="anything")
    assert result.is_error
    assert "not bound" in result.output


def test_task_requires_a_prompt(workspace):
    session, _ = make_session(
        workspace,
        [task_call(prompt="  "), reply("unused")],
    )
    session.prompt("go")
    tool_result = [
        message for message in session.messages if isinstance(message, ToolResultMessage)
    ][-1]
    assert tool_result.is_error
    assert "non-empty" in tool_result.output


def test_tool_schema_only_requires_prompt(workspace):
    from tau.tools.validation import validate_arguments

    tool = TaskTool(cwd=str(workspace))
    assert validate_arguments(tool.parameters, {"prompt": "x"}) == []
    assert validate_arguments(tool.parameters, {}) == ["arguments.prompt is required"]


def test_subagent_provider_error_is_a_tool_error(workspace, monkeypatch):
    monkeypatch.setattr("tau.agent.RETRY_DELAYS", (0, 0, 0))
    # The sub-agent's turn retries MAX_RETRIES times before giving up.
    session, _ = make_session(
        workspace,
        [
            task_call(prompt="look"),
            failing("rate limited"),
            failing("rate limited"),
            failing("rate limited"),
            reply("done"),
        ],
    )
    session.prompt("go")

    tool_result = [
        message for message in session.messages if isinstance(message, ToolResultMessage)
    ][-1]
    assert tool_result.is_error
    assert "sub-agent failed" in tool_result.output
    assert "rate limited" in tool_result.output


def test_subagent_spend_counts_toward_parent_usage(workspace):
    report = AssistantMessage(
        content=[TextContent(text="a.py:1")],
        model="fake",
        provider="fake",
        usage=Usage(input=100, output=5, total_tokens=105),
    )
    session, _ = make_session(
        workspace,
        [task_call(prompt="look"), report, reply("done")],
    )
    session.prompt("go")

    assert session.usage.total_tokens == 105
