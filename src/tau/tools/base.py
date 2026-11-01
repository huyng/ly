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
    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with validated JSON-schema arguments.

        Subclasses narrow this to the arguments they need; callers dispatch
        with arbitrary ``**arguments`` from the model.
        """
