import os
import re
import signal
import subprocess
import tempfile
from typing import Any, BinaryIO, ClassVar

from .base import Tool, ToolResult

MAX_OUTPUT_LINES = 2000
MAX_OUTPUT_BYTES = 50 * 1024
DEFAULT_TIMEOUT = 120

#: Variables a child process never needs, and that must not reach its output.
#: The harness reads its own provider key from the environment
#: (``provider.py:211-214``), so an unscrubbed child leaks it with plain `env`.
#: `GIT_CONFIG_*` is exempt: git takes config through those names
#: (``sandbox.py:127``) and they hold none of our credentials. Scrubbing
#: `GIT_CONFIG_KEY_0` would leave `GIT_CONFIG_COUNT` without its key, and git
#: then refuses every command with "missing config key".
_SCRUB = re.compile(
    r"^(?!GIT_CONFIG_)\w*(?:KEY|TOKEN|SECRET|PASSW|CRED)", re.IGNORECASE
)


def _child_env() -> dict[str, str]:
    """Every variable except credentials. Deliberately not an allowlist: a
    missing PATH/GOPATH/CARGO_HOME breaks toolchains silently, which is worse
    than passing a variable the child did not need."""
    return {k: v for k, v in os.environ.items() if not _SCRUB.search(k)}


def _text(value: str | bytes | None) -> str:
    """Decode command output leniently.

    `text=True` decodes strictly, so a single non-UTF-8 byte (binary output, a
    latin-1 log, a file with junk in it) raises `UnicodeDecodeError` and throws
    away the entire result. Real subprocess output is arbitrary bytes, so
    replacement characters beat losing the command's output.
    """
    if not value:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _read(handle: BinaryIO) -> str:
    """Everything written to a child's output file so far."""
    handle.seek(0)
    return _text(handle.read())


def _combined(stdout: str | bytes | None, stderr: str | bytes | None) -> str:
    out, err = _text(stdout).strip(), _text(stderr).strip()
    if out and err:
        return f"{out}\n{err}"
    return out or err


def _truncate(output: str) -> str:
    """Keep the tail of the output: a command that fails says why in its last
    lines, and the head of a long log is rarely worth the context."""
    lines = output.splitlines()
    if len(lines) > MAX_OUTPUT_LINES:
        skipped = len(lines) - MAX_OUTPUT_LINES
        output = f"... {skipped} lines skipped ...\n" + "\n".join(
            lines[-MAX_OUTPUT_LINES:]
        )
    data = output.encode("utf-8", errors="replace")
    if len(data) <= MAX_OUTPUT_BYTES:
        return output
    # The line cap alone does not bound the result: one line of minified JSON,
    # a base64 blob, or binary junk can be megabytes on its own.
    # The byte cut lands mid-line; the marker's newline keeps the fragment from
    # reading as a continuation of the marker itself.
    kept = data[-MAX_OUTPUT_BYTES:]
    return f"... {len(data) - len(kept)} bytes skipped ...\n" + kept.decode(
        "utf-8", errors="replace"
    )


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    """SIGKILL the call's whole process group, then reap the shell.

    `shell=True` makes the direct child `/bin/sh`, which execs the command as a
    grandchild of ours; killing only the shell leaves that grandchild running
    with nothing left to stop it. `start_new_session` makes the shell a group
    leader, so its pid is the group id and one killpg reaches everything the
    call started.
    """
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:  # the shell already exited
        pass
    proc.wait()


class BashTool(Tool):
    name = "bash"
    description = (
        "Execute a bash command in the current working directory. Returns "
        f"stdout and stderr; output is truncated to the last {MAX_OUTPUT_LINES} "
        f"lines or {MAX_OUTPUT_BYTES // 1024}KB (whichever is hit first)."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (optional)",
            },
        },
        "required": ["command"],
    }

    def __init__(self, cwd: str | None = None):
        self.cwd = cwd

    def execute(self, command: str, timeout: int | None = None, **_: Any) -> ToolResult:
        timeout_seconds = timeout or DEFAULT_TIMEOUT
        # Regular files, not pipes: `wait` on a pipe waits for EOF, so a
        # backgrounded grandchild (`nohup srv &`) inheriting the pipe would keep
        # the call hanging until the timeout. Output files are unlinked at
        # creation, so nothing is left on disk however the call ends.
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            # `start_new_session` puts the shell in its own process group so
            # `_kill_group` can reach the whole tree. The cost is the controlling
            # terminal: a command that opens /dev/tty (an editor, a password
            # prompt) now fails instead of prompting, and the terminal's Ctrl-C no
            # longer reaches the child — the interrupt is forwarded below instead.
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=self.cwd,
                env=_child_env(),
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
            try:
                proc.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                _kill_group(proc)
                # Whatever the command produced before the deadline is still
                # valuable — a build that fails at 95% has already said why.
                partial = _combined(_read(out), _read(err))
                header = f"Error: command timed out after {timeout_seconds}s"
                if not partial:
                    return ToolResult(header, is_error=True)
                return ToolResult(
                    _truncate(f"{header}\nPartial output:\n{partial}"),
                    is_error=True,
                )
            except BaseException:
                # Ctrl-C arrives here as KeyboardInterrupt. Any other exit takes
                # the same path: nothing the call started may outlive it.
                _kill_group(proc)
                raise

            output = _combined(_read(out), _read(err))
        status = f"[exit code: {proc.returncode}]"
        output = f"{output}\n{status}" if output else status
        return ToolResult(_truncate(output), is_error=proc.returncode != 0)
