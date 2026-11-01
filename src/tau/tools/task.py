"""Sub-agent delegation: the `task` tool.

A `task` call runs a **separate** `AgentSession` — same provider and model, a
fresh empty history, a read-only tool set — and returns only its final report.
The parent's context stays clean, which is the point: a five-call exploration
becomes one tool call plus a short summary instead of thousands of tokens of
transcript that the parent would otherwise carry for the rest of the run.

The sub-agent is deliberately unable to `edit`, `write`, or spawn further
sub-agents. It is an explorer, not a second pair of hands: two agents editing
one working tree concurrently is how changes get clobbered.
"""

import os
from typing import Any

from ..events import TaskActivityEvent, ToolExecutionEndEvent, ToolExecutionStartEvent
from ..messages import AssistantMessage
from .base import Tool, ToolResult
from .bash import BashTool
from .read import ReadTool
from .search import GlobTool, GrepTool

SUBAGENT_SYSTEM_PROMPT = """\
You are a sub-agent inside tau, a coding agent. You were spawned to answer one
self-contained question; the agent that spawned you cannot be asked anything,
so work with what you were given.

You have read-only search tools: `read`, `grep`, `glob`, and `bash`. You
cannot edit or create files, and you should not try. Use `bash` for commands
like `git log`, `git diff`, or a test run that merely reports information.

How to work:
- Start broad, then narrow. Prefer `grep`/`glob` over shell pipelines.
- Read enough to be sure; do not guess at code you have not opened.
- Stop as soon as the question is answered. You are not graded on thoroughness
  for its own sake.

Your final message is the *only* thing the parent sees. Make it a compact
report: what you found, with `path:line` references, plus anything surprising
or still unknown. No preamble, no narration of the steps you took.
"""


class TaskTool(Tool):
    name = "task"
    description = (
        "Delegate a self-contained investigation to a sub-agent that works in "
        "its own fresh context and reports back a short summary. Use this when "
        "answering would take many search/read calls whose raw output you do "
        "not need to keep — 'how does X work', 'where is Y handled', 'trace Z "
        "end to end', 'what would break if we changed W'. The sub-agent can "
        "read, search and run shell commands, but cannot edit files, and it "
        "cannot ask you questions, so the prompt must stand on its own."
    )
    parameters = {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "Short label for the task, e.g. 'locate session persistence'",
            },
            "prompt": {
                "type": "string",
                "description": "Complete, self-contained instructions for the sub-agent",
            },
        },
        "required": ["prompt"],
    }

    def __init__(self, cwd: str | None = None):
        self.cwd = cwd
        self._parent = None
        self.runs = 0

    def bind_session(self, session) -> None:
        """Called by AgentSession so the tool can share the parent's runtime.

        Sharing the provider means `/model` switches apply to sub-agents too;
        sharing the abort flag means one Ctrl-C stops both.
        """
        self._parent = session

    def execute(
        self, prompt: str, description: str = "", **_: Any
    ) -> ToolResult:
        if self._parent is None:
            return ToolResult(
                "Error: the task tool is not bound to a session. Build the "
                "agent with AgentSession(...) so it can bind its tools.",
                is_error=True,
            )
        if not isinstance(prompt, str) or not prompt.strip():
            return ToolResult(
                "Error: task.prompt must be a non-empty string of instructions.",
                is_error=True,
            )

        # Imported lazily: agent -> tools -> task would otherwise be circular.
        from ..agent import AgentSession
        from ..session import SessionManager

        cwd = self.cwd or os.getcwd()
        sub = AgentSession(
            provider=self._parent.provider,
            tools=_explorer_tools(cwd),
            session_manager=SessionManager(in_memory=True),
            system_prompt=(
                f"{SUBAGENT_SYSTEM_PROMPT}\nWorking directory: {os.path.abspath(cwd)}\n"
            ),
            cwd=cwd,
            abort_event=self._parent.abort_event,
            compaction=self._parent.compaction,
            tool_guard=self._parent.tool_guard,
        )
        label = (description or "").strip()
        emit = self._parent._emit

        def on_activity(event) -> None:
            if isinstance(event, (ToolExecutionStartEvent, ToolExecutionEndEvent)):
                emit(TaskActivityEvent(label, event))

        sub.subscribe(on_activity)
        self.runs += 1
        outcome = sub.prompt(prompt)
        # Sub-agent spend is real spend: fold it into the parent's totals so
        # the session's cumulative usage reflects the whole run, delegation
        # included.
        self._parent.usage.add(sub.usage)

        if outcome.stop_reason == "error":
            detail = outcome.error_message or "provider error"
            return ToolResult(f"Error: sub-agent failed: {detail}", is_error=True)
        if outcome.stop_reason == "aborted":
            return ToolResult("Error: sub-agent was aborted", is_error=True)

        report = sub.last_assistant_text().strip()
        if not report:
            report = "(the sub-agent finished without producing a final report)"
        prefix = f"[{label}] " if label else ""
        stats = (
            f"{prefix}sub-agent: {_tool_call_count(sub)} tool calls · "
            f"{sub.usage.total_tokens:,} tokens"
        )
        return ToolResult(f"{report}\n\n{stats}")


def _explorer_tools(cwd: str) -> list[Tool]:
    return [
        ReadTool(cwd=cwd),
        GrepTool(cwd=cwd),
        GlobTool(cwd=cwd),
        BashTool(cwd=cwd),
    ]


def _tool_call_count(session) -> int:
    return sum(
        len(message.tool_calls)
        for message in session.messages
        if isinstance(message, AssistantMessage)
    )
