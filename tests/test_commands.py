"""commands.py: slash commands defined by Markdown files in `.agents/commands`."""

from typing import cast

import pytest

from ly.agent import AgentSession
from ly.commands import discover
from ly.cli import Repl
from ly.messages import Usage
from ly.theme import Theme


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


def write_command(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_user_command_is_discovered(tmp_path, home):
    path = write_command(home / ".agents" / "commands" / "review.md", "# Review\nlook")
    commands = discover(str(tmp_path))
    assert list(commands) == ["review"]
    assert commands["review"].path == str(path)
    assert commands["review"].description == "Review"
    assert commands["review"].text == "# Review\nlook"


def test_name_is_the_lowercase_file_stem(tmp_path, home):
    write_command(home / ".agents" / "commands" / "Review.md", "look")
    assert list(discover(str(tmp_path))) == ["review"]


def test_project_command_overrides_user_command(tmp_path, home):
    write_command(home / ".agents" / "commands" / "review.md", "personal")
    write_command(tmp_path / ".agents" / "commands" / "review.md", "project")
    commands = discover(str(tmp_path))
    assert commands["review"].text == "project"


def test_both_scopes_are_merged(tmp_path, home):
    write_command(home / ".agents" / "commands" / "review.md", "personal")
    write_command(tmp_path / ".agents" / "commands" / "deploy.md", "project")
    assert sorted(discover(str(tmp_path))) == ["deploy", "review"]


def test_ancestor_commands_are_not_loaded(tmp_path, home):
    write_command(tmp_path / ".agents" / "commands" / "review.md", "outer")
    inner = tmp_path / "service"
    inner.mkdir()
    assert discover(str(inner)) == {}


def test_only_the_working_directory_is_read(tmp_path, home):
    write_command(tmp_path / ".agents" / "commands" / "review.md", "outer")
    inner = tmp_path / "service"
    write_command(inner / ".agents" / "commands" / "ship.md", "inner")
    assert list(discover(str(inner))) == ["ship"]


def test_other_files_and_empty_files_are_ignored(tmp_path, home):
    commands_dir = home / ".agents" / "commands"
    write_command(commands_dir / "notes.txt", "not a command")
    write_command(commands_dir / "blank.md", "   \n")
    assert discover(str(tmp_path)) == {}


def test_missing_directories_yield_no_commands(tmp_path, home):
    assert discover(str(tmp_path)) == {}


def test_expand_substitutes_arguments(tmp_path, home):
    write_command(
        home / ".agents" / "commands" / "fix.md", "fix $ARGUMENTS, then $ARGUMENTS"
    )
    command = discover(str(tmp_path))["fix"]
    assert command.expand("the tests") == "fix the tests, then the tests"


def test_expand_appends_arguments_without_placeholder(tmp_path, home):
    write_command(home / ".agents" / "commands" / "fix.md", "fix the tests")
    command = discover(str(tmp_path))["fix"]
    assert command.expand("quickly") == "fix the tests\n\nquickly"


def test_expand_leaves_text_alone_without_arguments(tmp_path, home):
    write_command(home / ".agents" / "commands" / "fix.md", "fix the tests")
    command = discover(str(tmp_path))["fix"]
    assert command.expand("") == "fix the tests"


class StubSession:
    """Just enough session for the REPL to take a prompt."""

    usage = Usage()
    tools: dict = {}

    def __init__(self, cwd):
        self.cwd = str(cwd)
        self.prompts: list[str] = []

    def subscribe(self, callback):
        return lambda: None

    def prompt(self, text):
        self.prompts.append(text)


def make_repl(tmp_path):
    session = StubSession(tmp_path)
    return session, Repl(cast(AgentSession, session), Theme(enabled=False))


def test_repl_runs_a_custom_command(tmp_path, home, capsys):
    write_command(home / ".agents" / "commands" / "fix.md", "fix: $ARGUMENTS")
    session, repl = make_repl(tmp_path)
    repl._command("/fix the tests")
    assert session.prompts == ["fix: the tests"]
    assert "unknown command" not in capsys.readouterr().out


def test_repl_prefers_builtins_over_command_files(tmp_path, home):
    write_command(home / ".agents" / "commands" / "help.md", "hijack")
    session, repl = make_repl(tmp_path)
    repl._command("/help")
    assert session.prompts == []


def test_unknown_command_still_reports_itself(tmp_path, home, capsys):
    session, repl = make_repl(tmp_path)
    repl._command("/nope")
    assert session.prompts == []
    assert "unknown command: /nope" in capsys.readouterr().out


def test_help_lists_custom_commands(tmp_path, home, capsys):
    write_command(home / ".agents" / "commands" / "review.md", "# Review\nthe diff")
    _, repl = make_repl(tmp_path)
    repl._command("/help")
    assert "/review: Review" in capsys.readouterr().out
