"""Tests for the built-in tools: read, write, bash process handling, and the
edit tool's offset-based application and unified-diff output."""

import os
import subprocess
import sys
import time

import pytest

from ly.tools.bash import MAX_OUTPUT_BYTES, MAX_OUTPUT_LINES, BashTool
from ly.tools.edit import MAX_DIFF_LINES, EditTool, WriteTool
from ly.tools.read import MAX_LINES, ReadTool
from ly.tools.validation import validate_arguments


@pytest.fixture
def edit(tmp_path):
    return EditTool(cwd=str(tmp_path))


@pytest.fixture
def bash(tmp_path):
    return BashTool(cwd=str(tmp_path))


def write_file(tmp_path, name: str, content: str):
    path = tmp_path / name
    path.write_text(content)
    return path


# -- edit: offset-based application ------------------------------------------


def test_chained_edits_are_applied_to_the_original_text(tmp_path, edit):
    """Regression: edits used to be applied sequentially to mutating content,
    so an earlier replacement could be clobbered by a later oldText match."""
    path = write_file(tmp_path, "f.txt", "A\nB\n")
    result = edit.execute(
        path="f.txt",
        edits=[
            {"oldText": "A", "newText": "B"},
            {"oldText": "B", "newText": "C"},
        ],
    )
    assert not result.is_error
    assert path.read_text() == "B\nC\n"


def test_independent_edits_apply_in_source_order(tmp_path, edit):
    path = write_file(
        tmp_path, "f.py", "def one():\n    return 1\n\ndef two():\n    return 2\n"
    )
    result = edit.execute(
        path="f.py",
        edits=[
            {"oldText": "return 2", "newText": "return 22"},
            {"oldText": "return 1", "newText": "return 11"},
        ],
    )
    assert not result.is_error
    assert (
        path.read_text() == "def one():\n    return 11\n\ndef two():\n    return 22\n"
    )


def test_edits_are_length_shifting_and_order_independent(tmp_path, edit):
    """A long replacement must not shift a later match's resolved offsets."""
    path = write_file(tmp_path, "f.txt", "start middle end")
    result = edit.execute(
        path="f.txt",
        edits=[
            {"oldText": "start", "newText": "start with a much longer prefix"},
            {"oldText": "end", "newText": "finish"},
        ],
    )
    assert not result.is_error
    assert path.read_text() == "start with a much longer prefix middle finish"


def test_overlapping_edits_are_rejected(tmp_path, edit):
    path = write_file(tmp_path, "f.txt", "abcd")
    result = edit.execute(
        path="f.txt",
        edits=[
            {"oldText": "abc", "newText": "x"},
            {"oldText": "bcd", "newText": "y"},
        ],
    )
    assert result.is_error
    assert "overlap" in result.output
    assert path.read_text() == "abcd"  # untouched


def test_ambiguous_match_is_rejected(tmp_path, edit):
    path = write_file(tmp_path, "f.txt", "x = 1\ny = 1\n")
    result = edit.execute(path="f.txt", edits=[{"oldText": "= 1", "newText": "= 2"}])
    assert result.is_error
    assert "matches 2 regions" in result.output
    assert path.read_text() == "x = 1\ny = 1\n"


def test_missing_text_is_rejected(tmp_path, edit):
    write_file(tmp_path, "f.txt", "hello\n")
    result = edit.execute(path="f.txt", edits=[{"oldText": "goodbye", "newText": "hi"}])
    assert result.is_error
    assert "not found" in result.output
    assert "edits[0]" in result.output


def test_empty_old_text_is_rejected(tmp_path, edit):
    write_file(tmp_path, "f.txt", "hello\n")
    result = edit.execute(path="f.txt", edits=[{"oldText": "", "newText": "x"}])
    assert result.is_error
    assert "empty" in result.output


@pytest.mark.parametrize("edits", [[], "not-a-list", [{"oldText": "a"}], ["nope"]])
def test_malformed_edits_are_rejected(tmp_path, edit, edits):
    path = write_file(tmp_path, "f.txt", "abc\n")
    result = edit.execute(path="f.txt", edits=edits)
    assert result.is_error
    assert path.read_text() == "abc\n"


def test_missing_file_reports_error(tmp_path, edit):
    result = edit.execute(path="nope.txt", edits=[{"oldText": "a", "newText": "b"}])
    assert result.is_error
    assert "Error reading" in result.output


# -- edit: diff output -------------------------------------------------------


def test_result_leads_with_summary_then_diff(tmp_path, edit):
    write_file(tmp_path, "f.txt", "one\ntwo\nthree\n")
    result = edit.execute(path="f.txt", edits=[{"oldText": "two", "newText": "TWO"}])

    lines = result.output.splitlines()
    assert lines[0] == "Edited f.txt (1 change)"  # REPL previews the first line
    assert "--- a/f.txt" in result.output
    assert "+++ b/f.txt" in result.output
    assert "-two" in result.output
    assert "+TWO" in result.output
    assert "@@" in result.output


def test_diff_uses_relative_path_for_files_under_cwd(tmp_path):
    nested = tmp_path / "pkg" / "mod.py"
    nested.parent.mkdir()
    nested.write_text("x = 1\n")
    EditTool(cwd=str(tmp_path)).execute(
        path="pkg/mod.py", edits=[{"oldText": "x = 1", "newText": "x = 2"}]
    )
    # re-read via a second edit to inspect the label
    result = EditTool(cwd=str(tmp_path)).execute(
        path="pkg/mod.py", edits=[{"oldText": "x = 2", "newText": "x = 3"}]
    )
    assert "a/pkg/mod.py" in result.output
    assert "+x = 3" in result.output


def test_no_op_edit_omits_diff(tmp_path, edit):
    write_file(tmp_path, "f.txt", "same\n")
    result = edit.execute(path="f.txt", edits=[{"oldText": "same", "newText": "same"}])
    assert not result.is_error
    assert result.output == "Edited f.txt (1 change)"


def test_large_diff_is_truncated(tmp_path, edit):
    before = "\n".join(f"line {i}" for i in range(300)) + "\n"
    after = "\n".join(f"changed {i}" for i in range(300)) + "\n"
    write_file(tmp_path, "big.txt", before)

    result = edit.execute(path="big.txt", edits=[{"oldText": before, "newText": after}])

    assert not result.is_error
    assert "diff truncated" in result.output
    diff_lines = result.output.splitlines()
    assert len(diff_lines) <= MAX_DIFF_LINES + 3  # summary, blank, capped diff


# -- write / read ------------------------------------------------------------


def test_write_creates_parent_directories(tmp_path):
    result = WriteTool(cwd=str(tmp_path)).execute(path="a/b/c.txt", content="hello")
    assert not result.is_error
    assert (tmp_path / "a" / "b" / "c.txt").read_text() == "hello"


def test_write_overwrites(tmp_path):
    path = write_file(tmp_path, "f.txt", "old content that is long")
    WriteTool(cwd=str(tmp_path)).execute(path="f.txt", content="new")
    assert path.read_text() == "new"


def test_read_offset_and_limit(tmp_path):
    write_file(tmp_path, "f.txt", "".join(f"{i}\n" for i in range(1, 11)))
    result = ReadTool(cwd=str(tmp_path)).execute(path="f.txt", offset=3, limit=2)
    assert result.output.startswith("3\n4\n")


def test_read_enforces_the_documented_line_cap(tmp_path):
    write_file(
        tmp_path,
        "f.txt",
        "".join(f"{i}\n" for i in range(1, MAX_LINES + 3)),
    )
    result = ReadTool(cwd=str(tmp_path)).execute(path="f.txt")

    assert result.output.startswith("1\n2\n")
    assert "... (2 more lines)" in result.output
    assert f"{MAX_LINES + 1}\n" not in result.output


def test_read_rejects_binary(tmp_path):
    path = tmp_path / "bin"
    path.write_bytes(b"\x00\x01\x02")
    result = ReadTool(cwd=str(tmp_path)).execute(path="bin")
    assert result.is_error
    assert "not a text file" in result.output


# -- bash: process output ----------------------------------------------------


def test_bash_reports_stdout_stderr_and_exit_code(bash):
    result = bash.execute(command="printf 'out\\n'; printf 'err\\n' >&2; exit 3")

    assert "out" in result.output and "err" in result.output
    assert "[exit code: 3]" in result.output
    assert result.is_error


def test_bash_success_is_not_an_error(bash):
    result = bash.execute(command="printf 'fine\\n'")

    assert result.is_error is False
    assert "[exit code: 0]" in result.output


def test_bash_does_not_leak_env_secrets(bash, monkeypatch):
    """`env` must not print the provider key to the model's context."""
    monkeypatch.setenv("FAKE_API_KEY", "hunter2")
    monkeypatch.setenv("FAKE_VISIBLE", "ok")

    result = bash.execute(command="env")

    assert "hunter2" not in result.output
    assert "FAKE_VISIBLE=ok" in result.output


def test_bash_keeps_git_config_env(bash, monkeypatch):
    """git's env config must survive the secret scrub.

    `GIT_CONFIG_KEY_0` contains "KEY" but is plumbing, not a credential. Losing
    it leaves `GIT_CONFIG_COUNT=1` without a key, and every git command in the
    sandbox dies with "missing config key".
    """
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "safe.directory")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "/workspace")

    result = bash.execute(command="env")

    assert "GIT_CONFIG_KEY_0=safe.directory" in result.output
    assert "GIT_CONFIG_VALUE_0=/workspace" in result.output


def test_bash_survives_non_utf8_output(bash):
    """A latin-1 byte must not throw away the whole result.

    `text=True` alone decodes strictly, so this used to raise
    UnicodeDecodeError and lose the command's output entirely.
    """
    result = bash.execute(command="printf 'caf\\xe9 ok\\n'")

    assert "caf" in result.output and "ok" in result.output
    assert "[exit code: 0]" in result.output
    assert result.is_error is False


def test_bash_returns_while_a_background_child_holds_its_output(bash):
    """A backgrounded server must not hang the call until the timeout.

    With pipes, `subprocess.run` waits for EOF, which only arrives once the
    shell's grandchild exits too. `timeout=1` under the child's own `sleep 3`
    makes a regression fail fast instead of just being slow.
    """
    result = bash.execute(command="sleep 3 & echo started", timeout=1)

    assert "started" in result.output
    assert "timed out" not in result.output
    assert result.is_error is False


def test_bash_timeout_keeps_partial_output(bash):
    result = bash.execute(command="printf 'started\\n'; sleep 5", timeout=1)

    assert "timed out after 1s" in result.output
    assert "started" in result.output
    assert result.is_error


def test_bash_timeout_without_output_says_only_that(bash):
    result = bash.execute(command="sleep 5", timeout=1)

    assert result.output == "Error: command timed out after 1s"
    assert result.is_error


def _gone(pid: int) -> bool:
    """Whether the pid is dead. Polled: a SIGKILLed grandchild is reparented to
    init and reaped a moment later, so one check can still see the zombie."""
    for _ in range(200):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.01)
    return False


def test_bash_timeout_kills_the_whole_process_group(bash, tmp_path):
    """A timed-out call must not leave its grandchild running.

    The shell execs the command, so killing the shell alone (what
    `subprocess.run` did) leaks the command itself; the pid is recorded so the
    assertion can look for it rather than trusting the exit code.
    """
    pidfile = tmp_path / "grandchild.pid"
    result = bash.execute(command=f"sleep 30 & echo $! > {pidfile}; wait", timeout=1)

    assert "timed out after 1s" in result.output
    assert _gone(int(pidfile.read_text()))


#: Stands in for the keyboard: SIGALRM raises the same KeyboardInterrupt that
#: Ctrl-C does, from inside `execute`'s blocking wait.
_INTERRUPTED = """
import signal
from ly.tools.bash import BashTool

def interrupt(*_):
    raise KeyboardInterrupt

signal.signal(signal.SIGALRM, interrupt)
signal.setitimer(signal.ITIMER_REAL, 0.5)
try:
    BashTool().execute("sleep 30 & echo $! > {pidfile}; wait")
except KeyboardInterrupt:
    print("interrupted")
"""


def test_bash_interrupt_kills_the_whole_process_group(tmp_path):
    pidfile = tmp_path / "grandchild.pid"
    run = subprocess.run(
        [sys.executable, "-c", _INTERRUPTED.format(pidfile=pidfile)],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert "interrupted" in run.stdout
    assert _gone(int(pidfile.read_text()))


def test_bash_keeps_the_last_lines_when_truncating(bash):
    over = MAX_OUTPUT_LINES + 20
    result = bash.execute(command=f"seq 1 {over}")

    assert f"... {over + 1 - MAX_OUTPUT_LINES} lines skipped ..." in result.output
    assert f"\n{over}\n[exit code: 0]" in result.output  # newest lines + status kept
    assert "\n1\n" not in result.output  # oldest lines dropped


def test_bash_caps_bytes_on_a_single_huge_line(bash):
    """The line cap cannot bound output on its own: one line can be megabytes."""
    result = bash.execute(
        command="awk 'BEGIN { for (i = 0; i < 60000; i++) printf \"x\" }'"
    )

    assert "bytes skipped ..." in result.output
    assert len(result.output.encode()) < MAX_OUTPUT_BYTES + 100
    assert result.output.endswith("[exit code: 0]")  # status survives the cut
    assert "x" * 1000 in result.output  # the tail, not the head, is kept


def test_bash_empty_output_still_reports_exit_code(bash):
    result = bash.execute(command="true")

    assert result.output == "[exit code: 0]"


# -- schema validation -------------------------------------------------------


ECHO_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
}


def test_validation_accepts_valid_arguments():
    assert validate_arguments(ECHO_SCHEMA, {"text": "hi"}) == []


def test_validation_reports_missing_required():
    assert validate_arguments(ECHO_SCHEMA, {}) == ["arguments.text is required"]


def test_validation_reports_wrong_type():
    assert validate_arguments(ECHO_SCHEMA, {"text": 42}) == [
        "arguments.text must be a string (got integer)"
    ]


def test_validation_treats_booleans_as_booleans_not_integers():
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
    assert validate_arguments(schema, {"n": True}) == [
        "arguments.n must be an integer (got boolean)"
    ]


def test_validation_allows_float_for_number_but_not_integer():
    number = {"type": "object", "properties": {"n": {"type": "number"}}}
    integer = {"type": "object", "properties": {"n": {"type": "integer"}}}
    assert validate_arguments(number, {"n": 1.5}) == []
    assert validate_arguments(integer, {"n": 1.5}) == [
        "arguments.n must be an integer (got number)"
    ]


def test_validation_validates_array_items():
    schema = EditTool.parameters
    problems = validate_arguments(
        schema,
        {
            "path": "f.txt",
            "edits": [{"oldText": "a", "newText": "b"}, {"oldText": "c"}],
        },
    )
    assert problems == ["arguments.edits[1].newText is required"]


def test_validation_rejects_non_list_edits():
    problems = validate_arguments(EditTool.parameters, {"path": "f.txt", "edits": "a"})
    assert problems == ["arguments.edits must be an array (got string)"]


def test_validation_ignores_unknown_properties():
    assert validate_arguments(ECHO_SCHEMA, {"text": "ok", "extra": 1}) == []


def test_validation_handles_non_object_arguments():
    assert validate_arguments(ECHO_SCHEMA, "nope") == [
        "arguments must be an object (got string)"
    ]
