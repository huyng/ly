"""Prompt input preserves paste boundaries, not input delivery timing."""

import io
import json
import os
import pty
import select
import signal
import sys

import pytest

from ly.cli import Repl


def test_piped_input_stays_line_oriented(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("first\nsecond\n"))
    repl = Repl.__new__(Repl)
    assert repl._read("") == "first"
    assert repl._read("") == "second"
    assert repl._read("") is None


def test_gnu_readline_enables_bracketed_paste(monkeypatch):
    import readline

    bindings = []
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(readline, "__doc__", "GNU readline")
    monkeypatch.setattr(readline, "parse_and_bind", bindings.append)
    monkeypatch.setattr("builtins.input", lambda prompt: "first\nsecond")
    assert Repl.__new__(Repl)._read("") == "first\nsecond"
    # Bracketed paste leaves Tab inserting a literal tab, so it is rebound.
    assert bindings == ["set enable-bracketed-paste on", "tab: complete"]


@pytest.mark.parametrize(
    "backend, expected",
    [
        ("GNU readline", "\001\x1b[36m\002» \001\x1b[0m\002"),
        ("libedit", "» "),
    ],
)
def test_prompt_color_does_not_count_as_visible_columns(monkeypatch, backend, expected):
    import readline

    from ly.theme import Theme

    prompts = []

    def capture(prompt):
        prompts.append(prompt)
        return "hello"

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(readline, "__doc__", backend)
    monkeypatch.setattr(readline, "parse_and_bind", lambda binding: None)
    monkeypatch.setattr("builtins.input", capture)
    monkeypatch.setattr(Repl, "_read_libedit", lambda self, prompt: capture(prompt))
    assert Repl.__new__(Repl)._read(Theme(enabled=True).accent("» ")) == "hello"
    assert prompts == [expected]


class FakeTTY:
    """The parts of a terminal that the libedit hook touches."""

    def __init__(self, fd):
        self._fd = fd
        self.encoding = "utf-8"

    def fileno(self):
        return self._fd

    def isatty(self):
        return True


def as_libedit(monkeypatch):
    """Make `Repl._read` take the libedit path, as it does on macOS."""
    import readline

    monkeypatch.setattr(
        readline,
        "__doc__",
        "Importing this module enables command line editing using libedit.",
    )


def libedit_repl():
    from ly.settings import Settings

    repl = Repl.__new__(Repl)
    repl._pastes = {}
    repl.settings = Settings.load(env={"LY_MODEL": "openai/gpt-4.1", "LY_API_KEY": "k"})
    repl._models = [repl.settings.model]
    return repl


def ask_completer(text, answers, states=3):
    """Ask the installed completer the way readline/libedit does."""
    import readline

    completer = readline.get_completer()
    assert completer is not None  # the read installs one before asking
    for state in range(states):
        answer = completer(text, state)
        if answer is None:
            break
        answers.append(answer)


def test_libedit_binds_tab_to_completion(monkeypatch):
    """Regression: libedit binds Tab to nothing on its own, so removing the
    paste hook's `ed-insert` binding left Tab doing nothing on macOS."""
    import readline

    bindings = []
    # Captured during the read: the restore after it binds Tab too, so asking
    # afterwards would pass whether or not Tab was bound while typing.
    during = []
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(readline, "parse_and_bind", bindings.append)
    as_libedit(monkeypatch)
    monkeypatch.setattr(
        "builtins.input", lambda prompt: during.append(list(bindings)) or "hello"
    )

    repl = libedit_repl()
    assert repl._read("") == "hello"
    assert 'bind "^I" rl_complete' in during[0]


def test_libedit_tab_reaches_completion(monkeypatch):
    import readline

    readable, writable = os.pipe()
    answers = []

    def fake_input(prompt):
        ask_completer("de", answers)
        return "x"

    as_libedit(monkeypatch)
    monkeypatch.setattr(sys, "stdin", FakeTTY(readable))
    monkeypatch.setattr(readline, "get_line_buffer", lambda: "/model de")
    monkeypatch.setattr("builtins.input", fake_input)

    libedit_repl()._read("")

    assert answers == ["deepseek/"]

    os.close(readable)
    os.close(writable)


def test_libedit_paste_still_becomes_a_placeholder(monkeypatch):
    import readline

    readable, writable = os.pipe()
    os.write(writable, "first\nsecond\x1b[201~".encode())
    answers = []

    def fake_input(prompt):
        ask_completer("", answers)
        return "x"

    as_libedit(monkeypatch)
    monkeypatch.setattr(sys, "stdin", FakeTTY(readable))
    monkeypatch.setattr(readline, "get_line_buffer", lambda: "/model ")
    monkeypatch.setattr("builtins.input", fake_input)

    repl = libedit_repl()
    repl._read("")

    assert len(answers) == 1
    assert answers[0].startswith("[paste ")
    assert list(repl._pastes.values()) == ["first\nsecond"]

    os.close(readable)
    os.close(writable)


def test_a_waiting_paste_is_told_apart_from_a_key_press(monkeypatch):
    """Libedit sends both Tab and pastes to one hook; only a paste arrives
    with bytes already buffered, which is what decides the branch."""
    repl = Repl.__new__(Repl)
    monkeypatch.setattr(select, "select", lambda readers, w, x, timeout: [readers])
    assert repl._paste_pending()
    monkeypatch.setattr(select, "select", lambda readers, w, x, timeout: [[], [], []])
    assert not repl._paste_pending()


def test_terminal_tab_completes_a_model_reference():
    """Tab reaches completion on the readline in use.

    On libedit the same entry point serves bracket pastes, so this also covers
    the dispatch that keeps Tab out of the paste reader.
    """
    code = """
import json
from ly.cli import Repl
from ly.settings import Settings
repl = Repl.__new__(Repl)
repl._pastes = {}
repl.settings = Settings.load(env={'LY_MODEL': 'openai/gpt-4.1', 'LY_API_KEY': 'k'})
repl._models = [repl.settings.model]
text = repl._read('PROMPT> ')
print('RESULT=' + json.dumps(text), flush=True)
"""
    pid, fd = pty.fork()
    if pid == 0:
        os.environ["TERM"] = "xterm-256color"
        os.execv(sys.executable, [sys.executable, "-c", code])

    output = bytearray()

    def until(marker: bytes) -> None:
        while marker not in output:
            assert select.select([fd], [], [], 5)[0], bytes(output)
            chunk = os.read(fd, 65536)
            assert chunk, bytes(output)
            output.extend(chunk)

    try:
        until(b"PROMPT> ")
        os.write(fd, b"/model dee\t")
        until(b"deepseek/")
        os.write(fd, b"deepseek-chat\r")
        until(b"RESULT=")
        result = bytes(output).split(b"RESULT=", 1)[1].splitlines()[0]
        assert json.loads(result) == "/model deepseek/deepseek-chat"
    finally:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        os.close(fd)


def test_terminal_tab_completes_a_slash_command():
    code = """
import json
from ly.cli import Repl
repl = Repl.__new__(Repl)
repl._pastes = {}
repl.commands = {}
repl.settings = None
text = repl._read('PROMPT> ')
print('RESULT=' + json.dumps(text), flush=True)
"""
    pid, fd = pty.fork()
    if pid == 0:
        os.environ["TERM"] = "xterm-256color"
        os.execv(sys.executable, [sys.executable, "-c", code])

    output = bytearray()

    def until(marker: bytes) -> None:
        while marker not in output:
            assert select.select([fd], [], [], 5)[0], bytes(output)
            chunk = os.read(fd, 65536)
            assert chunk, bytes(output)
            output.extend(chunk)

    try:
        until(b"PROMPT> ")
        os.write(fd, b"/mod\t")
        until(b"/model")
        os.write(fd, b"\r")
        until(b"RESULT=")
        result = bytes(output).split(b"RESULT=", 1)[1].splitlines()[0]
        assert json.loads(result) == "/model"
    finally:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        os.close(fd)


def test_terminal_ctrl_a_and_ctrl_e_with_colored_prompt():
    code = """
import json
from ly.cli import Repl
from ly.theme import Theme
repl = Repl.__new__(Repl)
repl._pastes = {}
text = repl._read(Theme(enabled=True).accent('» '))
print('RESULT=' + json.dumps(text), flush=True)
"""
    pid, fd = pty.fork()
    if pid == 0:
        os.environ["TERM"] = "xterm-256color"
        os.execv(sys.executable, [sys.executable, "-c", code])

    output = bytearray()

    def until(marker):
        while marker not in output:
            assert select.select([fd], [], [], 5)[0], bytes(output)
            chunk = os.read(fd, 65536)
            assert chunk, bytes(output)
            output.extend(chunk)

    try:
        until("» ".encode())
        os.write(fd, b"hello\x01")
        # Libedit uses absolute column positioning; GNU Readline may use
        # relative movements instead. Catch the original display bug on libedit.
        import readline

        if "libedit" in (readline.__doc__ or ""):
            until(b"\x1b[3G")
        os.write(fd, b"X\x05Y\r")
        until(b'RESULT="XhelloY"')
    finally:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        os.close(fd)


@pytest.mark.parametrize("cancel", [False, True])
def test_terminal_paste_waits_for_enter(cancel):
    # A separate interpreter gives readline a real controlling terminal without
    # changing pytest's terminal state. Handshakes split the paste into chunks;
    # no sleep or assumptions about input delivery speed are needed.
    code = """
import json
from ly.cli import Repl
repl = Repl.__new__(Repl)
repl._pastes = {}
try:
    text = repl._read('PROMPT> ')
    print('RESULT=' + json.dumps(text), flush=True)
    recalled = repl._read('RECALL> ')
    print('RECALLED=' + json.dumps(recalled), flush=True)
except KeyboardInterrupt:
    print('CANCELLED', flush=True)
"""
    pid, fd = pty.fork()
    if pid == 0:
        os.execv(sys.executable, [sys.executable, "-c", code])

    output = bytearray()

    def until(marker: bytes) -> None:
        while marker not in output:
            assert select.select([fd], [], [], 5)[0], bytes(output)
            chunk = os.read(fd, 65536)
            assert chunk, bytes(output)
            output.extend(chunk)

    try:
        until(b"PROMPT> ")
        assert b"\x1b[?2004h" in output
        content = "first\n" + "\tlong café line\n" * 1000 + "\nlast\n"
        os.write(fd, b"prefix \x1b[200~first\r")
        # Deliver a long Unicode paste in chunks, including literal tabs and
        # trailing newlines that must not act as editor commands.
        payload = content[len("first\n") :].replace("\n", "\r").encode()
        while payload:
            written = os.write(fd, payload[:1024])
            payload = payload[written:]
        os.write(fd, b"\x1b[201~")
        import readline

        if "libedit" in (readline.__doc__ or ""):
            until(f": {len(content)} chars]".encode())
        assert b"RESULT=" not in output
        os.write(fd, b" suffix")
        until(b"suffix")
        assert b"RESULT=" not in output
        if cancel:
            os.write(fd, b"\x03")
            until(b"CANCELLED")
        else:
            os.write(fd, b"\r")
            until(b"RESULT=")
            until(b' suffix"')
            result = bytes(output).split(b"RESULT=", 1)[1].splitlines()[0]
            assert json.loads(result) == "prefix " + content + " suffix"
            until(b"RECALL> ")
            output.clear()
            os.write(fd, b"\x10")
            if "libedit" in (readline.__doc__ or ""):
                until(f": {len(content)} chars]".encode())
            else:
                until(b"prefix ")
            assert b"RECALLED=" not in output
            # A recalled prompt must remain editable until Enter is pressed.
            os.write(fd, b"\x05 edited\r")
            until(b"RECALLED=")
            until(b' suffix edited"')
            recalled = bytes(output).split(b"RECALLED=", 1)[1].splitlines()[0]
            assert json.loads(recalled) == "prefix " + content + " suffix edited"
        assert b"\x1b[?2004l" in output
    finally:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        os.close(fd)
