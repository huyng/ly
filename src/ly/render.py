"""Presentation helpers for streamed assistant text and tool activity.

Two renderers live here:

- ``MarkdownRenderer`` turns streamed model text into a terminal-friendly
  transcript. It is line-buffered so styles never wrap a half-arrived token,
  and it keeps fenced code readable with a dim gutter.
- ``render_tool_call`` / ``render_tool_result`` compact tool activity into a
  two-line block, including a colorized unified diff for ``edit``.

Everything degrades to plain text when ``Theme.enabled`` is false.
"""

import re

from .theme import BOLD, CYAN, GREEN, RED, Theme, terminal_width

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})\s*([\w.+#-]*)\s*$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_BULLET_RE = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_NUMBER_RE = re.compile(r"^(\s*)(\d+[.)])\s+(.*)$")
_QUOTE_RE = re.compile(r"^\s*>\s?(.*)$")
_RULE_RE = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$")
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_TOKEN_RE = re.compile(r"\x00(\d+)\x00")

MAX_DIFF_LINES = 18
MAX_PREVIEW_LINES = 12

_DIFF_LANGS = {"diff", "patch", "udiff"}


class MarkdownRenderer:
    """A small, streaming-safe Markdown-to-ANSI renderer."""

    def __init__(self, theme: Theme, width: int | None = None):
        self.theme = theme
        self.width = width or terminal_width()
        self._buffer = ""
        self._open_line = False
        self._in_fence = False
        self._fence_marker = ""
        self._fence_lang = ""

    def feed(self, delta: str) -> str:
        """Consume a streamed chunk and return the text ready to print."""
        self._buffer += delta
        out: list[str] = []
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            out.append(self._render_line(line) + "\n")
            self._open_line = False

        # Keep very long prose lines responsive instead of buffering them
        # until the model finally emits a newline. Code-fence lines are left
        # buffered so the gutter is only emitted once per line.
        if len(self._buffer) > 200 and not self._in_fence:
            cut = self._buffer.rfind(" ")
            if cut < 100:
                cut = len(self._buffer) - 1
            chunk = self._buffer[: cut + 1]
            self._buffer = self._buffer[cut + 1 :]
            out.append(self._inline(chunk))
            self._open_line = True
        return "".join(out)

    def flush(self) -> str:
        """Render any buffered tail and always leave the cursor on a new line."""
        out: list[str] = []
        if self._buffer:
            out.append(self._render_line(self._buffer))
            self._buffer = ""
            self._open_line = True
        if self._in_fence:
            if self._open_line:
                out.append("\n")
            out.append(self.theme.dim("╰─"))
            self._in_fence = False
            self._open_line = True
        if self._open_line:
            out.append("\n")
            self._open_line = False
        return "".join(out)

    # -- line rendering ---------------------------------------------------------

    def _render_line(self, line: str) -> str:
        theme = self.theme

        if self._in_fence:
            closing = _FENCE_RE.match(line)
            if (
                closing
                and closing.group(1)[0] == self._fence_marker[0]
                and len(closing.group(1)) >= len(self._fence_marker)
            ):
                self._in_fence = False
                return theme.dim("╰─")
            if self._fence_lang in _DIFF_LANGS:
                return theme.dim("│ ") + colorize_diff_line(theme, line)
            return theme.dim("│ ") + line

        opening = _FENCE_RE.match(line)
        if opening:
            self._in_fence = True
            self._fence_marker = opening.group(1)
            self._fence_lang = opening.group(2).lower()
            label = self._fence_lang or "code"
            return theme.dim("╭─ " + label)

        if not line.strip():
            return ""
        if _RULE_RE.match(line):
            return theme.dim("─" * min(self.width, 60))

        heading = _HEADING_RE.match(line)
        if heading:
            return theme.heading(self._inline(heading.group(2)))

        quote = _QUOTE_RE.match(line)
        if quote:
            return theme.dim("│ " + self._inline(quote.group(1)))

        bullet = _BULLET_RE.match(line)
        if bullet:
            indent, body = bullet.groups()
            return f"{indent}{theme.accent('•')} {self._inline(body)}"

        numbered = _NUMBER_RE.match(line)
        if numbered:
            indent, marker, body = numbered.groups()
            return f"{indent}{theme.accent(marker)} {self._inline(body)}"

        return self._inline(line)

    def _inline(self, text: str) -> str:
        """Style inline code, bold spans, and links without eating markers."""
        theme = self.theme
        spans: list[str] = []

        def stash(match: re.Match[str]) -> str:
            spans.append(match.group(1))
            return f"\x00{len(spans) - 1}\x00"

        text = _INLINE_CODE_RE.sub(stash, text)
        text = _BOLD_RE.sub(lambda m: theme.paint(m.group(1), BOLD), text)
        text = _LINK_RE.sub(lambda m: m.group(1) + theme.dim(f" ({m.group(2)})"), text)
        if spans:
            text = _TOKEN_RE.sub(lambda m: theme.code(spans[int(m.group(1))]), text)
        return text


class GutterWriter:
    """Prefix every line of a stream with a dim gutter (used for reasoning).

    Lines are wrapped here instead of by the terminal, so the remainder keeps
    its gutter rather than running back to column zero. Text is held until a
    line is full, so breaks land on spaces; reasoning therefore arrives a line
    at a time, the same granularity as ``MarkdownRenderer``.

    A ``label`` frames the stream the way a tool block is framed, so reasoning
    does not read as a continuation of the tool output above it.
    """

    def __init__(
        self,
        theme: Theme,
        prefix: str = "│ ",
        style: str = "",
        width: int | None = None,
        label: str | None = None,
    ):
        self.theme = theme
        self.prefix = theme.dim(prefix)
        self.blank = theme.dim(prefix.rstrip())
        self.style = style
        # One column stays free: filling the last one makes the terminal wrap,
        # and the newline we print would then break the line twice.
        self.room = max((width or terminal_width()) - len(prefix) - 1, 1)
        self.label = label
        self._pending = ""  # the line in progress, not yet printed
        self._started = False

    def feed(self, delta: str) -> str:
        out: list[str] = []
        for index, part in enumerate(delta.split("\n")):
            if index:  # the newline ends whatever the line held
                out.append(self._line(self._pending))
                self._pending = ""
            self._pending += part
            while len(self._pending) > self.room:
                cut = self._pending.rfind(" ", 0, self.room + 1)
                if cut <= 0:  # a single word wider than the margin
                    cut = self.room
                out.append(self._line(self._pending[:cut]))
                self._pending = self._pending[cut:].lstrip(" ")
        return "".join(out)

    def _line(self, text: str) -> str:
        """Prefix and terminate one line. Blank lines keep a bare gutter so
        the column stays continuous; an empty line before any content is
        dropped, which keeps nothing above the frame header."""
        if not text and not self._started:
            return ""
        head = ""
        if self.label and not self._started:
            head = self.theme.dim("╭─ ") + self.theme.bold(self.label) + "\n"
        self._started = True
        body = self.theme.paint(text, self.style) if text and self.style else text
        return head + (self.prefix if text else self.blank) + body + "\n"

    def flush(self) -> str:
        """Print the tail and close the frame. Silent if nothing streamed, so
        callers can flush unconditionally."""
        out = self._line(self._pending) if self._pending else ""
        if self.label and self._started:
            out += self.theme.dim("╰─") + "\n"
        self._pending = ""
        self._started = False
        return out


# -- tool activity ---------------------------------------------------------------


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{max(int(seconds * 1000), 0)}ms"
    return f"{seconds:.1f}s"


def summarize_tool_call(
    theme: Theme, name: str, arguments: dict, width: int | None = None
) -> str:
    """A tool-aware summary of the call's arguments, one line where possible."""
    width = width or terminal_width()
    if not isinstance(arguments, dict):
        return theme.dim(_shorten(str(arguments), max(width - 10, 16)))

    def short(value: object, limit: int) -> str:
        text = " ".join(str(value).split())
        return text if len(text) <= limit else text[: max(limit - 1, 1)] + "…"

    def arg(key: str, default: str = "") -> str:
        return str(arguments.get(key, default))

    if name == "read":
        path = short(arg("path", "?"), max(width - 24, 12))
        offset = _as_int(arguments.get("offset"))
        limit = _as_int(arguments.get("limit"))
        if offset and limit:
            return theme.dim(f"{path}:{offset}-{offset + limit - 1}")
        if offset:
            return theme.dim(f"{path}:{offset}")
        if limit:
            return theme.dim(f"{path}:1-{limit}")
        return theme.dim(path)

    if name == "bash":
        # The script is what the user is deciding to trust, so it is shown
        # whole. A one-liner short enough to sit beside the tool name rides on
        # the header line; anything longer moves below it, where every line
        # shares one gutter and is wrapped, so the terminal never wraps it on
        # its own and leaves a stray fragment at column zero.
        lines = arg("command").splitlines()
        beside = max(width - 10, 1)  # what `╭─ bash ` leaves on the header
        if len(lines) == 1 and len(lines[0]) <= beside:
            return theme.dim(lines[0])
        room = max(width - 3, 1)  # `│ ` plus one column kept free
        wrapped = [piece for line in lines for piece in _wrap(line, room)]
        return "\n".join(theme.dim("│ " + line) for line in wrapped)

    if name == "grep":
        pattern = short(arg("pattern", "?"), max(width - 34, 12))
        scope = arg("glob") or arg("path")
        mode = arg("output_mode", "content")
        suffix = "" if mode == "content" else f" [{mode}]"
        return theme.dim(
            f"/{pattern}/"
            + (f" {short(scope, max(width - 46, 8))}" if scope else "")
            + suffix
        )

    if name == "glob":
        pattern = short(arg("pattern", "?"), max(width - 26, 12))
        root = arg("path")
        return theme.dim(
            pattern + (f" in {short(root, max(width - 40, 8))}" if root else "")
        )

    if name == "subagent":
        label = arg("description") or arg("prompt")
        return theme.dim(short(label, max(width - 10, 16)))

    if name in ("edit", "write"):
        path = short(arg("path", "?"), max(width - 30, 12))
        if name == "edit":
            count = len(arguments.get("edits") or [])
            note = f"{count} edit{'s' if count != 1 else ''}"
        else:
            note = f"{len(arg('content'))} chars"
        return theme.dim(f"{path} · {note}")

    if not arguments:
        return ""
    joined = ", ".join(f"{key}={value}" for key, value in arguments.items())
    return theme.dim(short(joined, max(width - 10, 16)))


def render_tool_result(
    theme: Theme,
    name: str,
    output: str,
    is_error: bool,
    elapsed: float,
    width: int | None = None,
) -> str:
    """Render a tool result as a compact, optionally multi-line block."""
    width = width or terminal_width()
    duration = format_duration(elapsed)
    status = (
        theme.error("✗") + theme.dim(f" error {duration} ↓")
        if is_error
        else theme.success("✓") + theme.dim(f" done {duration} ↓")
    )
    head = theme.dim("├─ ") + status
    foot = "\n" + theme.dim("╰─")
    lines = output.strip().splitlines()
    # `│ ` plus one column kept free, the same rule `GutterWriter` follows:
    # filling the last column makes the terminal wrap the line itself, and the
    # newline this module prints would then move down a second time.
    room = max(width - 3, 1)

    if is_error:
        body = "\n".join(theme.dim("│ " + _shorten(line, room)) for line in lines[:6])
        if len(lines) > 6:
            body += "\n" + theme.dim(f"│ … {len(lines) - 6} more lines")
        return head + (("\n" + body) if body else "") + foot

    summary, separator, diff = output.partition("\n\n")

    if name == "edit" and separator and diff.lstrip().startswith("---"):
        diff_lines = diff.strip().splitlines()
        additions = sum(
            1
            for line in diff_lines
            if line.startswith("+") and not line.startswith("+++")
        )
        deletions = sum(
            1
            for line in diff_lines
            if line.startswith("-") and not line.startswith("---")
        )
        counts = (
            theme.dim("│ ")
            + theme.add(f"+{additions}")
            + theme.dim(" ")
            + theme.remove(f"−{deletions}")
        )
        shown = diff_lines[:MAX_DIFF_LINES]
        body = "\n".join(
            theme.dim("│ ") + colorize_diff_line(theme, line) for line in shown
        )
        if len(diff_lines) > MAX_DIFF_LINES:
            body += "\n" + theme.dim(
                f"│ … {len(diff_lines) - MAX_DIFF_LINES} more lines"
            )
        return head + "\n" + counts + "\n" + body + foot

    if name == "read":
        count = len(output.splitlines())
        return head + "\n" + theme.dim(f"│ {count} lines") + foot

    if name == "bash":
        exit_note = ""
        if lines and lines[-1].startswith("[exit code:"):
            exit_note = lines.pop()
        preview = lines[:MAX_PREVIEW_LINES]
        body = "\n".join(theme.dim("│ " + _shorten(line, room)) for line in preview)
        if len(lines) > MAX_PREVIEW_LINES:
            body += "\n" + theme.dim(f"│ … {len(lines) - MAX_PREVIEW_LINES} more lines")
        if exit_note:
            body += ("\n" if body else "") + theme.dim("│ " + exit_note)
        return head + (("\n" + body) if body else "") + foot

    if name == "write":
        return head + "\n" + theme.dim("│ " + _shorten(summary, room)) + foot

    preview = lines[:6]
    body = "\n".join(theme.dim("│ " + _shorten(line, room)) for line in preview)
    if len(lines) > 6:
        body += "\n" + theme.dim(f"│ … {len(lines) - 6} more lines")
    return head + (("\n" + body) if body else "") + foot


def colorize_diff_line(theme: Theme, line: str) -> str:
    if line.startswith(("+++", "---")):
        return theme.paint(line, BOLD)
    if line.startswith("@@"):
        return theme.paint(line, CYAN)
    if line.startswith("+"):
        return theme.paint(line, GREEN)
    if line.startswith("-"):
        return theme.paint(line, RED)
    return theme.dim(line)


def _as_int(value: object) -> int | None:
    if not isinstance(value, (int, float, str)):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _wrap(text: str, limit: int) -> list[str]:
    """Split one line into chunks of at most `limit` columns.

    Breaks on spaces where it can and mid-word only when a single word is wider
    than the limit. A space the break lands on is dropped, as `GutterWriter`
    does: it would otherwise end up trailing a line.

    Args:
        text: The line to break.
        limit: Widest chunk to return, in columns; at least one.

    Returns:
        The chunks in order, the last one possibly empty.
    """
    chunks: list[str] = []
    rest = text
    while len(rest) > limit:
        cut = rest.rfind(" ", 0, limit + 1)
        if cut <= 0:  # a single word wider than the margin
            cut = limit
        chunks.append(rest[:cut])
        rest = rest[cut:].lstrip(" ")
    chunks.append(rest)
    return chunks


def _shorten(text: str, limit: int) -> str:
    if limit <= 1 or len(text) <= limit:
        return text
    return text[: limit - 1] + "…"
