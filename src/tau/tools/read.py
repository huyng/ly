import os
import stat
from typing import Any, ClassVar

from .base import Tool, ToolResult

MAX_LINES = 2000
MAX_BYTES = 50 * 1024


class ReadTool(Tool):
    name = "read"

    description = (
        f"Read the contents of a text file. Output is truncated to {MAX_LINES} lines "
        "or 50KB (whichever is hit first). Use offset/limit for large files."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file (relative or absolute)",
            },
            "offset": {
                "type": "integer",
                "description": "Line number to start reading from (1-indexed)",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to read",
            },
        },
        "required": ["path"],
    }

    def __init__(self, cwd: str | None = None):
        self.cwd = cwd

    def execute(
        self,
        path: str,
        offset: int | None = None,
        limit: int | None = None,
        **_: Any,
    ) -> ToolResult:
        path = os.path.expanduser(path)
        if not os.path.isabs(path) and self.cwd:
            path = os.path.join(self.cwd, path)
        try:
            if not os.path.isfile(path):
                return ToolResult(f"Error: not a file: {path}", is_error=True)
            if stat.S_ISFIFO(os.stat(path).st_mode) or _looks_binary(path):
                return ToolResult(f"Error: {path} is not a text file", is_error=True)
            selected, truncated = _read_lines(path, offset, limit)
        except OSError as exc:
            return ToolResult(f"Error reading {path}: {exc}", is_error=True)

        return ToolResult("".join(selected) + truncated)


def _read_lines(
    path: str, offset: int | None, limit: int | None
) -> tuple[list[str], str]:
    """Read only the requested window, so huge files are not loaded whole."""
    start = max(offset or 1, 1) - 1
    line_limit = min(limit, MAX_LINES) if limit and limit > 0 else MAX_LINES
    selected: list[str] = []
    used = 0
    truncated = ""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for _ in range(start):
            if not f.readline():
                return selected, truncated
        for line in f:
            if line_limit is not None and len(selected) >= line_limit:
                remaining = 1 + sum(1 for _ in f)
                truncated = f"\n... ({remaining} more lines)"
                break
            size = len(line.encode("utf-8"))
            if used + size > MAX_BYTES:
                truncated = f"\n... output truncated at {MAX_BYTES // 1024}KB"
                break
            selected.append(line)
            used += size
    return selected, truncated


def _looks_binary(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(1024)
        return b"\x00" in chunk
    except OSError:
        return False
