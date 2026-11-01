import difflib
import os
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, ClassVar

from .base import Tool, ToolResult

MAX_DIFF_LINES = 120
DIFF_CONTEXT_LINES = 3

_EDIT_HINT = "edits must be a non-empty list of {oldText, newText} objects"


@dataclass(frozen=True)
class _Region:
    """One edit resolved to a character range in the *original* content."""

    start: int
    end: int
    replacement: str
    index: int


def _locate(content: str, old: str) -> list[int]:
    """Every non-overlapping start offset of `old` in `content`."""
    positions: list[int] = []
    cursor = 0
    while True:
        found = content.find(old, cursor)
        if found < 0:
            return positions
        positions.append(found)
        cursor = found + len(old)


def _apply_regions(content: str, regions: list[_Region]) -> str:
    """Rebuild `content` by splicing every region into the original text.

    Regions are resolved against the untouched original, so one edit's
    replacement can never shift or clobber another's match.
    """
    pieces: list[str] = []
    cursor = 0
    for region in sorted(regions, key=lambda r: (r.start, r.end)):
        pieces.append(content[cursor : region.start])
        pieces.append(region.replacement)
        cursor = region.end
    pieces.append(content[cursor:])
    return "".join(pieces)


def _display_path(path: str, cwd: str | None) -> str:
    """Prefer a cwd-relative path in user-facing output."""
    shown = path
    if cwd:
        try:
            relative = os.path.relpath(path, cwd)
        except ValueError:  # different drive on Windows
            relative = path
        if relative != os.pardir and not relative.startswith(os.pardir + os.sep):
            shown = relative
    return shown.replace(os.sep, "/")


def _unified_diff(before: str, after: str, name: str) -> str:
    """A truncated unified diff, or "" when nothing changed."""
    hunks = list(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{name}",
            tofile=f"b/{name}",
            n=DIFF_CONTEXT_LINES,
        )
    )
    if not hunks:
        return ""
    lines = "".join(hunks).splitlines()
    if len(lines) > MAX_DIFF_LINES:
        extra = len(lines) - MAX_DIFF_LINES
        lines = lines[:MAX_DIFF_LINES] + [f"... diff truncated ({extra} more lines)"]
    return "\n".join(lines)


class EditTool(Tool):
    name = "edit"
    description = (
        "Edit a single file using exact text replacement. Every edits[].oldText "
        "must match a unique, non-overlapping region of the original file. Do "
        "not include large unchanged regions just to connect distant changes. "
        "Returns a unified diff of what changed."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to edit"},
            "edits": {
                "type": "array",
                "description": "One or more targeted replacements",
                "items": {
                    "type": "object",
                    "properties": {
                        "oldText": {
                            "type": "string",
                            "description": "Exact text to replace",
                        },
                        "newText": {
                            "type": "string",
                            "description": "Replacement text",
                        },
                    },
                    "required": ["oldText", "newText"],
                },
            },
        },
        "required": ["path", "edits"],
    }

    def __init__(self, cwd: str | None = None):
        self.cwd = cwd

    def execute(self, path: str, edits: list[dict], **_: Any) -> ToolResult:
        path = os.path.expanduser(path)
        if not os.path.isabs(path) and self.cwd:
            path = os.path.join(self.cwd, path)
        name = _display_path(path, self.cwd)

        if not isinstance(edits, list) or not edits:
            return ToolResult(f"Error: {_EDIT_HINT} for {name}.", is_error=True)
        for index, edit in enumerate(edits):
            if (
                not isinstance(edit, dict)
                or "oldText" not in edit
                or "newText" not in edit
            ):
                return ToolResult(
                    f"Error: {_EDIT_HINT} for {name}; got something else at index {index}.",
                    is_error=True,
                )
            if not isinstance(edit["oldText"], str) or not isinstance(
                edit["newText"], str
            ):
                return ToolResult(
                    f"Error: edits[{index}].oldText and edits[{index}].newText must both be "
                    f"strings for {name}.",
                    is_error=True,
                )

        try:
            with open(path, "r", encoding="utf-8") as f:
                before = f.read()
        except OSError as exc:
            return ToolResult(f"Error reading {name}: {exc}", is_error=True)

        # Resolve every edit against the original text before touching it.
        regions: list[_Region] = []
        for index, edit in enumerate(edits):
            old = edit["oldText"]
            if old == "":
                return ToolResult(
                    f"Error: edits[{index}].oldText is empty in {name}; "
                    "an empty oldText cannot be located.",
                    is_error=True,
                )
            positions = _locate(before, old)
            if not positions:
                return ToolResult(
                    f"Error: edits[{index}].oldText not found in {name}. "
                    "It must match exactly.",
                    is_error=True,
                )
            if len(positions) > 1:
                return ToolResult(
                    f"Error: edits[{index}].oldText matches {len(positions)} regions in "
                    f"{name}; it must be unique. Include more surrounding lines to "
                    "disambiguate.",
                    is_error=True,
                )
            start = positions[0]
            regions.append(_Region(start, start + len(old), edit["newText"], index))

        ordered = sorted(regions, key=lambda r: (r.start, r.end))
        for previous, current in pairwise(ordered):
            if current.start < previous.end:
                return ToolResult(
                    f"Error: edits[{previous.index}] and edits[{current.index}] overlap in "
                    f"{name} (characters {previous.start}-{previous.end} and "
                    f"{current.start}-{current.end}); merge them into a single edit.",
                    is_error=True,
                )

        after = _apply_regions(before, regions)

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(after)
        except OSError as exc:
            return ToolResult(f"Error writing {name}: {exc}", is_error=True)

        summary = f"Edited {name} ({len(edits)} change{'s' if len(edits) != 1 else ''})"
        diff = _unified_diff(before, after, name)
        return ToolResult(f"{summary}\n\n{diff}" if diff else summary)


class WriteTool(Tool):
    name = "write"
    description = (
        "Write content to a file. Creates the file if it doesn't exist, "
        "overwrites if it does. Automatically creates parent directories."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to write"},
            "content": {
                "type": "string",
                "description": "Content to write to the file",
            },
        },
        "required": ["path", "content"],
    }

    def __init__(self, cwd: str | None = None):
        self.cwd = cwd

    def execute(self, path: str, content: str, **_: Any) -> ToolResult:
        path = os.path.expanduser(path)
        if not os.path.isabs(path) and self.cwd:
            path = os.path.join(self.cwd, path)
        try:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError as exc:
            return ToolResult(f"Error writing {path}: {exc}", is_error=True)
        return ToolResult(f"Wrote {path} ({len(content)} bytes)")
