"""Prompt input preserves paste boundaries, not input delivery timing."""

import io
import json
import os
import pty
import select
import signal
import sys

import pytest

from tau.cli import Repl


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
    assert bindings == ["set enable-bracketed-paste on"]


@pytest.mark.parametrize(
    "backend, expected",
    [
        ("GNU readline", "\001\x1b[36m\002» \001\x1b[0m\002"),
        ("libedit", "» "),
    ],
)
def test_prompt_color_does_not_count_as_visible_columns(monkeypatch, backend, expected):
    import readline

    from tau.theme import Theme

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


def test_terminal_ctrl_a_and_ctrl_e_with_colored_prompt():
    code = """
import json
from tau.cli import Repl
from tau.theme import Theme
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
from tau.cli import Repl
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
