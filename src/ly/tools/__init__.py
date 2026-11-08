"""Built-in tool set: read, grep, glob, bash, edit, write, subagent.

Modeled after pi's defaults, plus the two tools that most change how a turn
is spent: `grep`/`glob` (search without shell plumbing) and `subagent` (delegate
exploration to a sub-agent so its transcript never enters this context).
"""

from .base import Tool, ToolResult
from .bash import BashTool
from .edit import EditTool, WriteTool
from .read import ReadTool
from .search import GlobTool, GrepTool
from .task import SubAgentTool
from .validation import validate_arguments

__all__ = [
    "BashTool",
    "EditTool",
    "GlobTool",
    "GrepTool",
    "ReadTool",
    "SubAgentTool",
    "Tool",
    "ToolResult",
    "WriteTool",
    "default_tools",
    "validate_arguments",
]


def default_tools(cwd: str | None = None) -> list[Tool]:
    return [
        ReadTool(cwd=cwd),
        GrepTool(cwd=cwd),
        GlobTool(cwd=cwd),
        BashTool(cwd=cwd),
        EditTool(cwd=cwd),
        WriteTool(cwd=cwd),
        SubAgentTool(cwd=cwd),
    ]
