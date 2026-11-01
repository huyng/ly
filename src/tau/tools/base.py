"""Tool abstraction. A tool has a name, a description, a JSON schema for its
parameters, and an execute method returning a ToolResult."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass
class ToolResult:
    output: str
    is_error: bool = False


class Tool(ABC):
    name: str = ""
    description: str = ""
    parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> ToolResult:
        """Execute arguments validated against the tool's JSON schema.

        The unconstrained signature lets subclasses define their own parameters.
        """
