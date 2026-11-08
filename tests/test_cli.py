"""Tests for the REPL's status line: it must show the size of the next request,
not just the cumulative token totals."""

import pytest

from ly.agent import AgentSession
from ly.cli import Repl, build_parser
from ly.compaction import CompactionConfig
from ly.events import TaskActivityEvent, ToolExecutionEndEvent, ToolExecutionStartEvent
from ly.messages import (
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
)
from ly.providers import Provider
from ly.providers.backend import ModelBackend
from ly.session import SessionManager
from ly.theme import Theme


class ScriptedBackend(ModelBackend):
    """A backend that never runs: the status line must not reach the model."""

    def __init__(self):
        super().__init__(Provider(model="fake-model", base_url="fake", api_key="fake"))

    def stream_response(self, *_args, **_kwargs):
        raise AssertionError("the status line must not call the provider")


def make_repl(threshold_tokens: int = 100_000, enabled: bool = True) -> Repl:
    session = AgentSession(
        backend=ScriptedBackend(),
        tools=[],
        session_manager=SessionManager(in_memory=True),
        system_prompt="system",
        compaction=CompactionConfig(enabled=enabled, threshold_tokens=threshold_tokens),
    )
    # A run that billed 82k for its prompt and left nothing behind: the next
    # request is exactly what the provider measured last time.
    session.session_manager.append(
        AssistantMessage(
            content=[TextContent(text="done")],
            model="fake",
            usage=Usage(input=82_000, output=12, total_tokens=82_012),
        )
    )
    session.usage = Usage(input=1_234, output=567, total_tokens=1_801)
    return Repl(session, Theme(enabled=False))


def test_task_activity_shows_calls_and_status_without_child_output(capsys):
    repl = make_repl()
    parent = ToolCall(
        id="parent", name="subagent", arguments={"description": "explore"}
    )
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
        repl._on_event(
            TaskActivityEvent("explore", ToolExecutionEndEvent(call, result))
        )
        out = capsys.readouterr().out
        assert f"{'✗' if is_error else '✓'} {name}" in out
        assert "private child output" not in out
    assert "parent" in repl._tool_started
    result = ToolResultMessage(parent.id, "subagent", "report")
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


# -- environment variables: the CLI is the only place that reads them ---------


def test_banner_names_the_container_image(capsys, monkeypatch):
    repl = make_repl()
    monkeypatch.setenv("LY_IN_CONTAINER", "1")
    monkeypatch.setenv("LY_IMAGE", "example/image:1")
    monkeypatch.setattr(repl, "_read", lambda prompt: None)

    repl.run()

    assert "docker example/image:1" in capsys.readouterr().out


def test_banner_names_the_host_directory_that_was_mounted(capsys, monkeypatch):
    repl = make_repl()
    repl.session.cwd = "/workspace"
    monkeypatch.setenv("LY_IN_CONTAINER", "1")
    monkeypatch.setenv("LY_IMAGE", "example/image:1")
    monkeypatch.setenv("LY_HOST_DIR", "/Users/someone/ly")
    monkeypatch.setattr(repl, "_read", lambda prompt: None)

    repl.run()

    out = capsys.readouterr().out
    assert "/Users/someone/ly → /workspace" in out


def test_banner_copes_when_the_host_directory_is_unknown(capsys, monkeypatch):
    repl = make_repl()
    repl.session.cwd = "/workspace"
    monkeypatch.setenv("LY_IN_CONTAINER", "1")
    monkeypatch.setenv("LY_IMAGE", "example/image:1")
    monkeypatch.delenv("LY_HOST_DIR", raising=False)
    monkeypatch.setattr(repl, "_read", lambda prompt: None)

    repl.run()

    out = capsys.readouterr().out
    assert "/workspace" in out
    assert "→" not in out


def test_banner_stays_quiet_on_the_host(capsys, monkeypatch):
    repl = make_repl()
    monkeypatch.delenv("LY_IN_CONTAINER", raising=False)
    monkeypatch.setattr(repl, "_read", lambda prompt: None)

    repl.run()

    assert "docker" not in capsys.readouterr().out


def test_print_mode_says_it_is_in_docker_on_stderr(monkeypatch, capsys):
    from ly.cli import print_mode

    monkeypatch.setenv("LY_IN_CONTAINER", "1")
    monkeypatch.setenv("LY_IMAGE", "example/image:1")

    def no_session(_args):
        raise SystemExit(0)

    monkeypatch.setattr("ly.cli.make_session", no_session)

    with pytest.raises(SystemExit):
        print_mode(build_parser().parse_args(["-p", "hi"]))

    captured = capsys.readouterr()
    assert captured.err == "ly: docker example/image:1\n"
    assert captured.out == ""  # stdout stays clean for pipes


def test_env_variables_become_argument_defaults(monkeypatch):
    from ly.cli import build_parser

    monkeypatch.setenv("LY_MODEL", "deepseek-flash")
    monkeypatch.setenv("LY_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("LY_API_KEY", "env-key")
    monkeypatch.setenv("LY_API", "responses")

    args = build_parser().parse_args([])

    assert args.model == "deepseek-flash"
    assert args.base_url == "https://api.deepseek.com"
    assert args.api_key == "env-key"
    assert args.api == "responses"


def test_flags_override_env_variables(monkeypatch):
    from ly.cli import build_parser

    monkeypatch.setenv("LY_BASE_URL", "https://env.example/v1")
    monkeypatch.setenv("LY_API_KEY", "env-key")

    args = build_parser().parse_args(
        ["--base-url", "https://flag.example/v1", "--api-key", "flag-key"]
    )

    assert args.base_url == "https://flag.example/v1"
    assert args.api_key == "flag-key"


def test_pass_reasoning_content_env_flag_is_tri_state(monkeypatch):
    from ly.cli import _flag

    monkeypatch.delenv("LY_PASS_REASONING_CONTENT", raising=False)
    assert _flag("LY_PASS_REASONING_CONTENT") is None

    monkeypatch.setenv("LY_PASS_REASONING_CONTENT", "yes")
    assert _flag("LY_PASS_REASONING_CONTENT") is True

    monkeypatch.setenv("LY_PASS_REASONING_CONTENT", "off")
    assert _flag("LY_PASS_REASONING_CONTENT") is False


def test_a_sessions_directory_that_cannot_be_written_is_explained(monkeypatch):
    """The traceback would be about os.mkdir; name the directory instead."""
    from ly.cli import main
    from ly.session import SessionManager

    monkeypatch.setenv("LY_IN_CONTAINER", "1")
    monkeypatch.setenv("LY_BASE_URL", "https://x/v1")
    monkeypatch.setenv("LY_API_KEY", "test")

    def refuse(self):
        raise PermissionError(13, "Permission denied", "/sessions/2026-11-06")

    monkeypatch.setattr(SessionManager, "new_session", refuse)

    with pytest.raises(SystemExit) as exited:
        main([])

    message = str(exited.value)
    assert "cannot save sessions" in message
    assert "Permission denied" in message
    # Nothing is guessed about why: this path is ours to create.
    assert "chown" not in message
