"""Tests for the REPL's status line: it must show the size of the next request,
not just the cumulative token totals."""

import json

import pytest

from ly.agent import AgentSession
from ly.cli import Repl, build_parser
from ly.compaction import CompactionConfig
from ly.events import (
    MessageEndEvent,
    MessageStartEvent,
    ReasoningDeltaEvent,
    TaskActivityEvent,
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
from ly.providers import Provider
from ly.providers.backend import ModelBackend, ResponsesBackend
from ly.session import SessionManager
from ly.cli import COMMANDS
from ly.commands import Command
from ly.presets import Model, Preset
from ly.settings import Settings
from ly.theme import Theme

#: A fixed catalogue for completion and sizing tests. The models ly ships are
#: the user's own data and change, so tests state the names and windows they
#: mean rather than reading the built-in table.
TEST_PRESETS = {
    "openai": Preset(
        base_url="https://api.openai.com/v1",
        key_env=("OPENAI_API_KEY",),
        models={"gpt-4.1": Model(context_window=1_047_576)},
    ),
    "deepseek": Preset(
        base_url="https://api.deepseek.com",
        key_env=("DEEPSEEK_API_KEY",),
        pass_reasoning_content=True,
        models={
            "deepseek-chat": Model(context_window=65_536, keep_turns=3),
            "deepseek-reasoner": Model(context_window=65_536, keep_turns=3),
        },
    ),
}


def _as_file_entry(preset: Preset) -> dict:
    """One preset written the way a settings file would hold it."""
    return {
        "base_url": preset.base_url,
        "key_env": list(preset.key_env),
        "models": {
            name: {"context_window": spec.context_window, "keep_turns": spec.keep_turns}
            for name, spec in preset.models.items()
        },
    }


#: The same catalogue as a settings file, for tests that go through `load()`.
TEST_SETTINGS = {name: _as_file_entry(preset) for name, preset in TEST_PRESETS.items()}


class ScriptedBackend(ModelBackend):
    """A backend that never runs: the status line must not reach the model."""

    def __init__(self):
        super().__init__(Provider(model="fake-model", base_url="fake", api_key="fake"))

    def stream_response(self, *_args, **_kwargs):
        raise AssertionError("the status line must not call the provider")


def catalogue_settings(model: str = "openai/gpt-4.1") -> Settings:
    """Settings over the test catalogue, so completion is not the user's data."""
    return Settings(presets=TEST_PRESETS, model=model)


def make_repl(
    threshold_tokens: int = 100_000,
    enabled: bool = True,
    settings: Settings | None = None,
) -> Repl:
    session = AgentSession(
        backend=ScriptedBackend(),
        tools=[],
        session_manager=SessionManager(),
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
    return Repl(session, Theme(enabled=False), settings or catalogue_settings())


def test_tab_completes_the_provider_half_of_a_model():
    repl = make_repl()

    # The reference in use leads the candidates, so Enter on it keeps it.
    assert repl._completions("/model ") == [
        "/model deepseek/",
        "/model openai/",
        "/model openai/gpt-4.1",
    ]
    assert repl._completions("/model de") == ["/model deepseek/"]
    assert repl._completions("/model x") == []


def test_tab_completes_the_models_a_provider_lists():
    repl = make_repl()

    assert repl._completions("/model deepseek/") == [
        "/model deepseek/deepseek-chat",
        "/model deepseek/deepseek-reasoner",
    ]
    assert repl._completions("/model deepseek/deepseek-r") == [
        "/model deepseek/deepseek-reasoner"
    ]


def test_tab_offers_a_reference_this_session_used():
    """The preset list is a guide, not the whole truth, so a reference you
    already typed is offered even when no preset names it."""
    repl = make_repl()
    repl._models.append("deepseek/deepseek-coder")

    assert repl._completions("/model deepseek/deepseek-c") == [
        "/model deepseek/deepseek-chat",
        "/model deepseek/deepseek-coder",
    ]


def test_tab_is_quiet_outside_a_model_reference():
    repl = make_repl()
    repl._models.append("deepseek/deepseek-chat")

    # Nothing after a complete reference: the line is done, not half-typed.
    assert repl._completions("/model deepseek/deepseek-chat ") == []
    assert repl._completions("/config ") == []
    assert repl._completions("explain this") == []


def test_switching_model_remembers_the_reference_for_tab(monkeypatch, capsys):
    monkeypatch.setenv("LY_API_KEY", "test-key")
    repl = make_repl()

    repl._command("/model deepseek/deepseek-chat")

    assert repl._models == ["openai/gpt-4.1", "deepseek/deepseek-chat"]
    assert repl._completions("/model deepseek/deepseek-chat") == [
        "/model deepseek/deepseek-chat"
    ]


def test_tab_lists_the_slash_commands():
    repl = make_repl()

    assert repl._completions("/co") == ["/compact", "/config"]
    # Markdown commands from this machine join the list, so only check that
    # every built-in is offered, not that they are the whole answer.
    assert set(COMMANDS) <= set(repl._completions("/"))


def test_tab_lists_markdown_commands_too():
    repl = make_repl()
    repl.commands = {"review": Command(name="review", path="", text="# Review")}

    assert repl._completions("/rev") == ["/review"]


def test_tab_completes_a_command_name_from_a_bare_slash(monkeypatch):
    """Libedit breaks the word at `/`, so the whole command name is the
    replacement only after the slash is accounted for."""
    repl = make_repl()

    assert complete_at(repl, "/mo", "mo", monkeypatch) == ["/model"]
    assert complete_at(repl, "/mo", "/mo", monkeypatch) == ["/model"]


def complete_at(repl, line, text, monkeypatch):
    """Answers for one Tab, plus the line each answer would produce."""
    import readline

    monkeypatch.setattr(readline, "get_line_buffer", lambda: line)
    start = len(line) - len(text)
    answers = []
    for state in range(5):
        answer = repl._complete(text, state)
        if answer is None:
            break
        answers.append(line[:start] + answer)
    return answers


def test_tab_after_the_slash_offers_the_model_alone(monkeypatch):
    """Regression: libedit breaks the word at `/`, so a full reference came
    back as `/model deepseek/deepseek/deepseek-...`."""
    repl = make_repl()

    assert complete_at(repl, "/model deepseek/", "", monkeypatch) == [
        "/model deepseek/deepseek-chat",
        "/model deepseek/deepseek-reasoner",
    ]


def test_tab_keeps_the_provider_when_readline_breaks_on_slashes(monkeypatch):
    repl = make_repl()

    assert complete_at(
        repl, "/model deepseek/deepseek-r", "deepseek-r", monkeypatch
    ) == ["/model deepseek/deepseek-reasoner"]


def test_tab_offers_the_whole_reference_when_the_word_is_the_reference(
    monkeypatch,
):
    """GNU readline treats the reference as one word, so the candidate has to
    carry the provider with it."""
    repl = make_repl()

    assert complete_at(repl, "/model deepseek/d", "deepseek/d", monkeypatch) == [
        "/model deepseek/deepseek-chat",
        "/model deepseek/deepseek-reasoner",
    ]


def config_values(out: str) -> dict[str, str]:
    """Parse `/config` output, which is one `name value` pair per line."""
    values: dict[str, str] = {}
    for line in out.splitlines():
        if line.strip():
            name, value = line.split(maxsplit=1)
            values[name] = value.strip()
    return values


def test_config_shows_the_settings_in_effect(capsys):
    repl = make_repl(threshold_tokens=8000)
    repl.session.set_model("switched-model")
    repl.show_thinking = False

    repl._command("/config")

    out = capsys.readouterr().out
    settings = config_values(out)
    assert settings["model"].endswith("switched-model")  # live, not the start
    assert settings["compaction.threshold_tokens"] == "8000"
    assert settings["thinking"] == "false"
    assert settings["base_url"] == "fake"


def test_config_lines_the_values_up_in_one_column(capsys):
    make_repl()._command("/config")

    lines = capsys.readouterr().out.splitlines()
    # Where each value starts: the line length less the value itself.
    columns = {len(line) - len(line.split(maxsplit=1)[1]) for line in lines}
    assert len(columns) == 1


def test_config_never_prints_the_api_key(capsys):
    repl = make_repl()
    repl.session.backend.provider.api_key = "sk-secret-1234"

    repl._command("/config")

    out = capsys.readouterr().out
    assert "sk-secret-1234" not in out
    assert config_values(out)["api_key"] == "set"


def test_config_says_when_no_api_key_is_set(capsys):
    repl = make_repl()
    repl.session.backend.provider.api_key = ""

    repl._command("/config")

    assert config_values(capsys.readouterr().out)["api_key"] == "unset"


def sessions_repl(monkeypatch, tmp_path, *prompts) -> Repl:
    """A REPL whose `/sessions` listing is exactly the files in `tmp_path`."""
    manager = SessionManager(directory=tmp_path)
    manager.new_session()
    for prompt in prompts:
        manager.append(UserMessage(prompt))
    session = AgentSession(
        backend=ScriptedBackend(),
        tools=[],
        session_manager=manager,
        system_prompt="system",
    )
    return Repl(session, Theme(enabled=False))


def test_sessions_lists_each_by_its_hash_and_last_prompt(monkeypatch, capsys, tmp_path):
    repl = sessions_repl(
        monkeypatch, tmp_path, "an old prompt", "fix the failing tests"
    )

    repl._command("/sessions")

    out = capsys.readouterr().out
    assert repl.session.session_manager.file is not None
    stem = repl.session.session_manager.file.stem
    assert stem in out  # the id, as /resume would need it
    assert "fix the failing tests" in out
    assert "an old prompt" not in out
    # Hash, then when it was last used, then what it is about.
    assert out.index(stem) < out.index("20") < out.index("fix the failing tests")


def test_sessions_synopsis_fits_a_narrow_terminal(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("ly.cli.terminal_width", lambda: 60)
    repl = sessions_repl(monkeypatch, tmp_path, "fix " * 40)

    repl._command("/sessions")

    lines = capsys.readouterr().out.splitlines()
    assert lines and lines[0].endswith("…")
    # A wrapped row would be two lines, so one short line is the whole check.
    assert all(len(line) <= 60 for line in lines)


def test_sessions_synopsis_stays_short_on_a_wide_terminal(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setattr("ly.cli.terminal_width", lambda: 300)
    repl = sessions_repl(monkeypatch, tmp_path, "fix " * 40)

    repl._command("/sessions")

    (line,) = capsys.readouterr().out.splitlines()
    assert line.endswith("…")
    assert len(line) < 100  # SYNOPSIS_CHARS, not the whole 300-column row


def test_sessions_survives_a_terminal_narrower_than_the_row(
    monkeypatch, capsys, tmp_path
):
    # The id and timestamp alone are wider than this, so the synopsis gets nothing.
    monkeypatch.setattr("ly.cli.terminal_width", lambda: 20)
    repl = sessions_repl(monkeypatch, tmp_path, "fix the failing tests")

    repl._command("/sessions")

    assert "fix" not in capsys.readouterr().out


def test_a_retry_finishes_the_reasoning_frame_before_starting_another(capsys):
    """A retry re-enters at MessageStart; its frame must not stack on the first."""
    repl = make_repl()
    repl._on_event(MessageStartEvent())
    repl._on_event(ReasoningDeltaEvent(delta="first attempt\n"))
    # The provider failed during reasoning and the loop started over.
    repl._on_event(MessageStartEvent())
    repl._on_event(ReasoningDeltaEvent(delta="second attempt\n"))
    repl._on_event(
        MessageEndEvent(
            AssistantMessage(content=[TextContent(text="done")], model="fake")
        )
    )

    out = capsys.readouterr().out
    assert out.count("╭─ thinking") == 2
    assert out.count("╰─") == 2  # every frame that opened also closed


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


def test_status_line_names_the_directory_the_run_happened_in(capsys, monkeypatch):
    # Tests run inside ly's own container often, where the host directory is
    # already in the environment and would win over the session's cwd.
    monkeypatch.delenv("LY_SB_IN_CONTAINER", raising=False)
    monkeypatch.delenv("LY_SB_HOST_DIR", raising=False)
    repl = make_repl()
    repl.session.cwd = "/tmp/project"

    repl._print_usage(elapsed=2.0, output_tokens=100)

    assert "/tmp/project" in capsys.readouterr().out


def test_status_line_names_the_host_directory_inside_a_container(capsys, monkeypatch):
    """The mount point `/workspace` says nothing about where the user is."""
    monkeypatch.setenv("LY_SB_IN_CONTAINER", "1")
    monkeypatch.setenv("LY_SB_HOST_DIR", "/Users/someone/project")
    repl = make_repl()
    repl.session.cwd = "/workspace"

    repl._print_usage(elapsed=2.0, output_tokens=100)

    out = capsys.readouterr().out
    assert "/Users/someone/project" in out
    assert "/workspace" not in out


def test_status_line_leaves_out_the_elapsed_time(capsys):
    make_repl()._print_usage(elapsed=2.0, output_tokens=100)

    out = capsys.readouterr().out
    assert "2.0s" not in out
    assert "50 tok/s" in out  # the rate still needs the elapsed time


# -- ! shell escape ----------------------------------------------------------


def drive(repl, monkeypatch, *lines) -> None:
    """Run the REPL over `lines`, then end input."""
    pending = iter([*lines, None])
    monkeypatch.setattr(repl, "_read", lambda prompt: next(pending))
    repl.run()


def test_bang_runs_the_command_without_asking_the_model(capsys, monkeypatch, tmp_path):
    repl = make_repl()
    repl.session.cwd = str(tmp_path)
    before = len(repl.session.messages)

    # ScriptedBackend raises if the provider is reached, so getting to the end
    # at all proves `!` never became a prompt.
    drive(repl, monkeypatch, "!printf 'shell output\\n'")

    assert "shell output" in capsys.readouterr().out
    assert len(repl.session.messages) == before


def test_bang_runs_in_the_session_directory(capsys, monkeypatch, tmp_path):
    repl = make_repl()
    repl.session.cwd = str(tmp_path)

    drive(repl, monkeypatch, "!pwd")

    assert str(tmp_path) in capsys.readouterr().out


def test_bang_keeps_a_failing_command_from_ending_the_session(
    capsys, monkeypatch, tmp_path
):
    repl = make_repl()
    repl.session.cwd = str(tmp_path)

    # A non-zero exit is the user's business, not the REPL's; a bare `!` is a
    # no-op rather than an error.
    drive(repl, monkeypatch, "!exit 3", "!", "!echo still here")

    assert "still here" in capsys.readouterr().out


# -- environment variables: the CLI is the only place that reads them ---------


def test_banner_names_the_container_image(capsys, monkeypatch):
    repl = make_repl()
    monkeypatch.setenv("LY_SB_IN_CONTAINER", "1")
    monkeypatch.setenv("LY_SB_IMAGE", "example/image:1")
    monkeypatch.setattr(repl, "_read", lambda prompt: None)

    repl.run()

    assert "docker example/image:1" in capsys.readouterr().out


def test_banner_names_the_host_directory_that_was_mounted(capsys, monkeypatch):
    repl = make_repl()
    repl.session.cwd = "/workspace"
    monkeypatch.setenv("LY_SB_IN_CONTAINER", "1")
    monkeypatch.setenv("LY_SB_IMAGE", "example/image:1")
    monkeypatch.setenv("LY_SB_HOST_DIR", "/Users/someone/ly")
    monkeypatch.setattr(repl, "_read", lambda prompt: None)

    repl.run()

    out = capsys.readouterr().out
    assert "/Users/someone/ly → /workspace" in out


def test_banner_copes_when_the_host_directory_is_unknown(capsys, monkeypatch):
    repl = make_repl()
    repl.session.cwd = "/workspace"
    monkeypatch.setenv("LY_SB_IN_CONTAINER", "1")
    monkeypatch.setenv("LY_SB_IMAGE", "example/image:1")
    monkeypatch.delenv("LY_SB_HOST_DIR", raising=False)
    monkeypatch.setattr(repl, "_read", lambda prompt: None)

    repl.run()

    out = capsys.readouterr().out
    assert "/workspace" in out
    assert "→" not in out


def test_banner_stays_quiet_on_the_host(capsys, monkeypatch):
    repl = make_repl()
    monkeypatch.delenv("LY_SB_IN_CONTAINER", raising=False)
    monkeypatch.setattr(repl, "_read", lambda prompt: None)

    repl.run()

    assert "docker" not in capsys.readouterr().out


def test_print_mode_says_it_is_in_docker_on_stderr(monkeypatch, capsys, tmp_path):
    from ly.cli import print_mode

    monkeypatch.setenv("LY_SB_IN_CONTAINER", "1")
    monkeypatch.setenv("LY_SB_IMAGE", "example/image:1")

    def no_session(_args, _provider, _sessions_dir):
        raise SystemExit(0)

    monkeypatch.setattr("ly.cli.make_session", no_session)
    provider = Provider(model="m", base_url="u", api_key="k")

    with pytest.raises(SystemExit):
        print_mode(build_parser().parse_args(["-p", "hi"]), provider, tmp_path)

    captured = capsys.readouterr()
    assert captured.err == "ly: docker example/image:1\n"
    assert captured.out == ""  # stdout stays clean for pipes


def test_no_provider_option_is_a_flag(monkeypatch):
    """They come from LY_* and the settings file, so host and container cannot
    disagree about which endpoint they are talking to."""
    from ly.cli import build_parser
    from ly.settings import Settings

    monkeypatch.setenv("LY_MODEL", "deepseek/deepseek-chat")
    monkeypatch.setenv("LY_API_KEY", "env-key")

    assert Settings.load().api_key == "env-key"

    for flag in ("--model", "--base-url", "--api-format", "--api-key", "--provider"):
        with pytest.raises(SystemExit):
            build_parser().parse_args([flag, "x"])


def test_config_picks_the_settings_file_to_read(monkeypatch, tmp_path):
    """`--config` names the file, so a model written there is the one used."""
    from ly.cli import main

    path = tmp_path / "custom.json"
    path.write_text(json.dumps({"model": "deepseek/deepseek-chat"}), encoding="utf-8")
    for name in ("LY_MODEL", "LY_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(SystemExit) as exited:
        main(["--config", str(path), "-p", "hi"])

    # The file supplied the model, so the missing piece is the key: the default
    # settings path, which has no model, would have said so instead.
    message = str(exited.value)
    assert "Missing credentials" in message
    assert "DEEPSEEK_API_KEY" in message


def test_a_sessions_directory_that_cannot_be_written_is_explained(monkeypatch):
    """The traceback would be about os.mkdir; name the directory instead."""
    from ly.cli import main
    from ly.session import SessionManager

    monkeypatch.setenv("LY_SB_IN_CONTAINER", "1")
    monkeypatch.setenv("LY_MODEL", "openai/gpt-4.1")
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


# -- switching models and providers -------------------------------------------


def provider_repl(monkeypatch, tmp_path) -> Repl:
    """A REPL whose switches resolve from the test catalogue, not this machine's."""
    path = tmp_path / "catalogue.json"
    path.write_text(json.dumps({"providers": TEST_SETTINGS}))
    monkeypatch.setattr("ly.settings.DEFAULT_PATH", str(path))
    for name in ("LY_MODEL", "LY_BASE_URL", "LY_API_FORMAT", "LY_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    return make_repl()


def test_the_model_command_shows_the_provider_with_the_model(
    monkeypatch, capsys, tmp_path
):
    repl = provider_repl(monkeypatch, tmp_path)

    repl._command("/model")

    assert "openai/gpt-4.1" in capsys.readouterr().out


def test_a_switch_changes_the_endpoint_model_and_key(monkeypatch, capsys, tmp_path):
    repl = provider_repl(monkeypatch, tmp_path)

    repl._command("/model deepseek/deepseek-chat")

    provider = repl.session.backend.provider
    assert provider.base_url == "https://api.deepseek.com"
    assert provider.model == "deepseek-chat"
    assert provider.api_key == "sk-deepseek"
    assert repl.settings.provider == "deepseek"
    assert "deepseek/deepseek-chat" in capsys.readouterr().out


def test_a_switch_resizes_compaction_to_the_new_model(monkeypatch, capsys, tmp_path):
    """Regression: the threshold stayed at the old model's, so a 64k model
    compacted only after the window had already overflowed."""
    repl = provider_repl(monkeypatch, tmp_path)
    before = repl.session.compaction.threshold_tokens

    repl._command("/model deepseek/deepseek-chat")

    assert repl.session.compaction.threshold_tokens == 32_768
    assert before == 100_000


def test_a_model_without_a_provider_changes_nothing(monkeypatch, capsys, tmp_path):
    repl = provider_repl(monkeypatch, tmp_path)
    before = repl.session.backend

    repl._command("/model deepseek-chat")

    assert repl.session.backend is before
    assert "provider/model" in capsys.readouterr().out


def test_a_slash_inside_the_model_name_stays_there(monkeypatch, capsys, tmp_path):
    repl = provider_repl(monkeypatch, tmp_path)

    repl._command("/model openai/moonshotai/kimi-k2")

    provider = repl.session.backend.provider
    assert provider.model == "moonshotai/kimi-k2"
    assert provider.base_url == "https://api.openai.com/v1"


def test_a_switch_rebuilds_the_backend_for_the_new_wire_format(
    monkeypatch, capsys, tmp_path
):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "providers": {
                    "gw": {
                        "base_url": "https://gw.test/v1",
                        "api_format": "openai-responses",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    repl = provider_repl(monkeypatch, tmp_path)
    monkeypatch.setattr("ly.settings.DEFAULT_PATH", str(path))
    monkeypatch.setenv("LY_API_KEY", "sk-gw")

    repl._command("/model gw/house-model")

    assert isinstance(repl.session.backend, ResponsesBackend)
    assert repl.session.backend.provider.api_format == "openai-responses"


def test_a_provider_without_a_key_says_so(monkeypatch, capsys, tmp_path):
    repl = provider_repl(monkeypatch, tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    before = repl.settings

    repl._command("/model deepseek/deepseek-chat")

    # Named, so a key kept under the provider's own variable is not a dead end.
    assert "Set DEEPSEEK_API_KEY or LY_API_KEY" in capsys.readouterr().out
    assert repl.settings is before
    assert repl.session.backend.provider.model == "fake-model"


def test_there_is_no_provider_command(monkeypatch, capsys, tmp_path):
    """One way to switch: the provider is part of the model, not a mode."""
    repl = provider_repl(monkeypatch, tmp_path)

    repl._command("/provider deepseek")

    assert "unknown command" in capsys.readouterr().out


def test_config_names_the_provider_in_effect(monkeypatch, capsys, tmp_path):
    repl = provider_repl(monkeypatch, tmp_path)
    repl._command("/model deepseek/deepseek-reasoner")

    repl._command("/config")

    out = capsys.readouterr().out
    assert "deepseek/deepseek-reasoner" in out
