"""ANSI styling for the interactive terminal UI.

Styling is disabled automatically when stdout is not a TTY, when ``NO_COLOR``
is set, or when ``TERM=dumb``. That keeps ``ly -p`` (and any piped output)
free of escape sequences.

The palette deliberately sticks to the basic 16 ANSI colors so it inherits
the user's terminal theme instead of fighting it.
"""

import os
import shutil
import sys

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"

RED = "\x1b[31m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
MAGENTA = "\x1b[35m"
CYAN = "\x1b[36m"


class Theme:
    """A tiny styling facade; a no-op when colors are disabled."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def paint(self, text: str, *codes: str) -> str:
        if not self.enabled or not text or not codes:
            return text
        return "".join(codes) + text + RESET

    # -- semantic roles ---------------------------------------------------------

    def bold(self, text: str) -> str:
        return self.paint(text, BOLD)

    def dim(self, text: str) -> str:
        return self.paint(text, DIM)

    def accent(self, text: str) -> str:
        return self.paint(text, CYAN)

    def model(self, text: str) -> str:
        return self.paint(text, MAGENTA)

    def heading(self, text: str) -> str:
        return self.paint(text, BOLD, CYAN)

    def code(self, text: str) -> str:
        return self.paint(text, CYAN)

    def success(self, text: str) -> str:
        return self.paint(text, GREEN)

    def error(self, text: str) -> str:
        return self.paint(text, RED)

    def warning(self, text: str) -> str:
        return self.paint(text, YELLOW)

    def add(self, text: str) -> str:
        return self.paint(text, GREEN)

    def remove(self, text: str) -> str:
        return self.paint(text, RED)


def color_enabled(stream=None) -> bool:
    """Whether ANSI styling should be emitted for ``stream``."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if os.environ.get("TERM", "").lower() == "dumb":
        return False
    stream = stream if stream is not None else sys.stdout
    return bool(getattr(stream, "isatty", lambda: False)())


def get_theme(stream=None) -> Theme:
    return Theme(enabled=color_enabled(stream))


def terminal_width(default: int = 80) -> int:
    try:
        return shutil.get_terminal_size((default, 24)).columns
    except OSError:  # pragma: no cover - unusual terminal environments
        return default
