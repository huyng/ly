"""A basic safety guard for tool calls.

The guard runs in ``AgentSession._execute_one`` before a tool executes, so it
sees every call, including custom tools and sub-agent calls. It is a
guardrail, not a sandbox: ``bash`` runs with the process's permissions and a
string check cannot see through a shell. These checks stop the obvious
accidents and make refusals visible to the model; they do not contain a
determined one.
"""

import os
import shlex
import tempfile
from collections.abc import Callable
from pathlib import PurePath

from .messages import ToolCall

Guard = Callable[[ToolCall], str | None]

#: Commands refused outright. Substring checks are conservative on purpose:
#: a false positive costs one retry, a false negative can cost the tree.
_BASH_DENY = (
    "mkfs",
    "dd if=/dev/zero of=/dev/",
    ":(){",
    "sudo ",
    "shutdown",
    "reboot",
)

#: Filesystem tools whose targets must stay under the working directory.
_WRITE_TOOLS = frozenset({"edit", "write"})

#: Tools whose string arguments name files they will read.
_READ_TOOLS = frozenset({"read", "grep", "glob"})

#: Path components that usually hold credentials. Matched whole, so `~/.ssh`
#: and `id_rsa` are caught without refusing every path containing "ssh".
_SECRETS = frozenset(
    {
        ".aws",
        ".docker",
        ".env",
        ".git-credentials",
        ".gnupg",
        ".kube",
        ".netrc",
        ".npmrc",
        ".pgpass",
        ".pypirc",
        ".ssh",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
    }
)


def _resolve(path: str, cwd: str) -> str:
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(cwd, path)
    return os.path.realpath(path)


def _inside(path: str, cwd: str) -> bool:
    try:
        return os.path.commonpath([path, cwd]) == cwd
    except ValueError:  # different drive on Windows
        return False


def _writable(path: str, roots: tuple[str, ...]) -> bool:
    """True if ``path`` sits strictly inside one of ``roots``.

    Strictly, so that naming a scratch dir does not make ``rm -rf <that dir>``
    legal, and so ``rm -rf .`` is refused from the working directory.
    """
    return any(path != root and _inside(path, root) for root in roots)


def _allowed(roots: tuple[str, ...]) -> str:
    """Name the writable roots for a refusal message.

    A refusal that names the alternative stops the model repeating the call.
    """
    return " or ".join(roots)


def _tokens(command: str) -> list[str] | None:
    """Split a command into words, re-splitting quoted payloads.

    ``bash -c "rm -rf /"`` and ``xargs rm -rf /`` hide the interesting words
    inside one quoted argument, so split each token again on whitespace.
    """
    try:
        return [part for token in shlex.split(command) for part in token.split()]
    except ValueError:  # unbalanced quotes: let the shell report them
        return None


def _credential(path: str) -> bool:
    """True if any component of ``path`` names a usual credential store."""
    parts = PurePath(path).parts
    return bool(parts) and (any(p in _SECRETS for p in parts) or parts[-1].endswith(".pem"))


def _mentions_credentials(command: str, cwd: str) -> str | None:
    """Refuse a command naming a credential path.

    Reading a secret is the first half of exfiltration; the second half is a
    network call this guard cannot follow across tool calls, so refuse here.
    """
    for token in _tokens(command) or ():
        if token.startswith("-"):
            continue
        if _credential(_resolve(token, cwd)):
            return f"command references credentials ({token})"
    return None


def _rm_outside(command: str, roots: tuple[str, ...]) -> str | None:
    """Refuse ``rm`` whose targets resolve outside ``roots``.

    Matching flags as strings cannot enumerate their spellings (``-rf``,
    ``-r -f``, ``--recursive --force``), so look at targets instead. Relative
    targets resolve against ``roots[0]``, the working directory.
    """
    tokens = _tokens(command) or ()
    for i, token in enumerate(tokens):
        if os.path.basename(token).lower() != "rm":
            continue
        for target in tokens[i + 1 :]:
            if target.startswith("-"):
                continue
            resolved = _resolve(target, roots[0])
            if not _writable(resolved, roots):
                return f"rm targets {resolved}, only {_allowed(roots)} may be deleted"
    return None


def basic_guard(
    cwd: str | None = None,
    deny: frozenset[str] = frozenset(),
    scratch: tuple[str, ...] | None = None,
) -> Guard:
    """Refuse denied tools, writes outside the tree, and destructive commands.

    ``scratch`` names extra writable roots, defaulting to the system temp dirs.
    Scratch work belongs there, and refusing it only pushes the same write into
    ``bash``, which this guard cannot follow into a shell. Refusals name the
    writable space for the same reason: a bare "no" invites the retry.
    """
    root = os.path.realpath(cwd or os.getcwd())
    if scratch is None:
        # Both, because $TMPDIR on macOS is not /tmp and the model uses either.
        scratch = (os.path.realpath("/tmp"), os.path.realpath(tempfile.gettempdir()))
    writable_roots = [root]
    for path in scratch:
        resolved = os.path.realpath(path)
        if resolved not in writable_roots:
            writable_roots.append(resolved)
    roots = tuple(writable_roots)

    def guard(call: ToolCall) -> str | None:
        if call.name in deny:
            return f"{call.name} is disabled by policy"

        if call.name in _WRITE_TOOLS:
            path = call.arguments.get("path")
            if isinstance(path, str):
                target = _resolve(path, root)
                if not _writable(target, roots):
                    return f"{call.name} may only write inside {_allowed(roots)} (got {target})"

        elif call.name in _READ_TOOLS:
            for value in call.arguments.values():
                if isinstance(value, str) and _credential(value):
                    return f"{call.name} may not read credentials ({value})"

        elif call.name == "bash":
            command = call.arguments.get("command")
            if isinstance(command, str):
                # Collapse whitespace before matching: spacing variants
                # ("rm    -rf /") and newline-separated commands would
                # otherwise slip past the substring checks below.
                lowered = " ".join(command.lower().split())
                for bad in _BASH_DENY:
                    if bad in lowered:
                        return f"command contains {bad!r}"
                if refusal := _rm_outside(command, roots):
                    return refusal
                if refusal := _mentions_credentials(command, root):
                    return refusal

        return None

    return guard
