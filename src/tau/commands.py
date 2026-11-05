"""Slash commands defined by Markdown files, one command per file.

`~/.agents/commands/review.md` defines `/review` for every project, and
`<cwd>/.agents/commands/review.md` defines it for one project. Only the
working directory is read, never its ancestors, so a command comes from
exactly one of the two places the user chose. Project files win when both
scopes define the same name, mirroring how the two AGENTS.md scopes interact.

A file's text is the prompt sent when the command runs. `$ARGUMENTS` is
replaced by whatever the user typed after the command name; without that
placeholder the arguments are appended as a final paragraph instead.
"""

import os
from dataclasses import dataclass
from pathlib import Path

ARGUMENTS = "$ARGUMENTS"


@dataclass(frozen=True)
class Command:
    """One Markdown-defined slash command."""

    name: str
    path: str
    text: str

    @property
    def description(self) -> str:
        """First non-blank line of the file, without Markdown's leading `#`."""
        for line in self.text.splitlines():
            if line.strip():
                return line.strip().lstrip("#").strip()
        return ""

    def expand(self, arguments: str) -> str:
        """Build the prompt this command sends for the given arguments.

        Args:
            arguments: Text the user typed after the command name, if any.

        Returns:
            The command text with `$ARGUMENTS` substituted, or with the
            arguments appended when the text holds no placeholder.
        """
        if ARGUMENTS in self.text:
            return self.text.replace(ARGUMENTS, arguments)
        if not arguments:
            return self.text
        return f"{self.text}\n\n{arguments}"


def _load(directory: Path) -> dict[str, Command]:
    """Read the `*.md` files in one directory, skipping unreadable and empty ones."""
    commands: dict[str, Command] = {}
    try:
        paths = sorted(directory.glob("*.md"))
    except OSError:
        return commands
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text:
            # Names are matched lowercase, the same way /HELP reaches /help.
            name = path.stem.lower()
            commands[name] = Command(name, str(path), text)
    return commands


def discover(cwd: str | None = None) -> dict[str, Command]:
    """Commands available here, keyed by name, project files overriding user files."""
    commands = _load(Path(os.path.expanduser("~/.agents/commands")))
    project = Path(os.path.abspath(cwd or os.getcwd()), ".agents", "commands")
    return commands | _load(project)
