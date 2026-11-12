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
    repl._models = [repl.settings.model or ""]
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


class Terminal:
    """A separate interpreter on its own terminal, driven by output markers.

    A pty of its own keeps pytest's terminal state out of the run, and every
    wait is a handshake on output rather than a sleep, so input sent in chunks
    is read as it arrives. The child gets `TERM` because readline decides what
    escapes it may emit from it.
    """

    def __init__(self, code: str):
        self.code = code
        self.output = bytearray()
        self.pid = 0
        self.fd = 0

    def __enter__(self) -> "Terminal":
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            os.environ["TERM"] = "xterm-256color"
            os.execv(sys.executable, [sys.executable, "-c", self.code])
        return self

    def __exit__(self, *_exc: object) -> None:
        os.kill(self.pid, signal.SIGKILL)
        os.waitpid(self.pid, 0)
        os.close(self.fd)

    def _read_more(self) -> None:
        """Take the next chunk, failing with what arrived if none comes."""
        assert select.select([self.fd], [], [], 5)[0], bytes(self.output)
        chunk = os.read(self.fd, 65536)
        assert chunk, bytes(self.output)
        self.output.extend(chunk)

    def until(self, marker: bytes) -> None:
        """Read until `marker` appears."""
        while marker not in self.output:
            self._read_more()

    def send(self, data: bytes) -> int:
        """Write `data`, returning how many bytes the terminal took."""
        return os.write(self.fd, data)

    def clear(self) -> None:
        self.output.clear()

    def json_after(self, marker: bytes) -> object:
        """The JSON value the child printed on the line after `marker`.

        The line is read to its newline: the marker can arrive in a chunk of
        its own, leaving the value it introduces still on its way.
        """
        self.until(marker)
        while b"\n" not in bytes(self.output).split(marker, 1)[1]:
            self._read_more()
        return json.loads(bytes(self.output).split(marker, 1)[1].splitlines()[0])


def on_libedit() -> bool:
    """Whether the readline in this interpreter is libedit, as on macOS."""
    import readline

    return "libedit" in (readline.__doc__ or "")


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
    with Terminal(code) as terminal:
        terminal.until(b"PROMPT> ")
        terminal.send(b"/model dee\t")
        terminal.until(b"deepseek/")
        terminal.send(b"deepseek-chat\r")
        assert terminal.json_after(b"RESULT=") == "/model deepseek/deepseek-chat"


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
    with Terminal(code) as terminal:
        terminal.until(b"PROMPT> ")
        terminal.send(b"/mod\t")
        terminal.until(b"/model")
        terminal.send(b"\r")
        assert terminal.json_after(b"RESULT=") == "/model"


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
    with Terminal(code) as terminal:
        terminal.until("» ".encode())
        terminal.send(b"hello\x01")
        # Libedit uses absolute column positioning; GNU Readline may use
        # relative movements instead. Catch the original display bug on libedit.
        if on_libedit():
            terminal.until(b"\x1b[3G")
        terminal.send(b"X\x05Y\r")
        terminal.until(b'RESULT="XhelloY"')


@pytest.mark.parametrize("cancel", [False, True])
def test_terminal_paste_waits_for_enter(cancel):
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
    with Terminal(code) as terminal:
        terminal.until(b"PROMPT> ")
        assert b"\x1b[?2004h" in terminal.output
        content = "first\n" + "\tlong café line\n" * 1000 + "\nlast\n"
        terminal.send(b"prefix \x1b[200~first\r")
        # Deliver a long Unicode paste in chunks, including literal tabs and
        # trailing newlines that must not act as editor commands.
        payload = content[len("first\n") :].replace("\n", "\r").encode()
        while payload:
            payload = payload[terminal.send(payload[:1024]) :]
        terminal.send(b"\x1b[201~")
        if on_libedit():
            terminal.until(f": {len(content)} chars]".encode())
        assert b"RESULT=" not in terminal.output
        terminal.send(b" suffix")
        terminal.until(b"suffix")
        assert b"RESULT=" not in terminal.output
        if cancel:
            terminal.send(b"\x03")
            terminal.until(b"CANCELLED")
        else:
            terminal.send(b"\r")
            assert terminal.json_after(b"RESULT=") == "prefix " + content + " suffix"
            terminal.until(b"RECALL> ")
            terminal.clear()
            terminal.send(b"\x10")
            if on_libedit():
                terminal.until(f": {len(content)} chars]".encode())
            else:
                terminal.until(b"prefix ")
            assert b"RECALLED=" not in terminal.output
            # A recalled prompt must remain editable until Enter is pressed.
            terminal.send(b"\x05 edited\r")
            terminal.until(b'RECALLED="')
            assert (
                terminal.json_after(b"RECALLED=")
                == "prefix " + content + " suffix edited"
            )
        assert b"\x1b[?2004l" in terminal.output
