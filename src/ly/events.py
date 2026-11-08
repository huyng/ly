"""Session lifecycle events, modeled after pi's event stream.

Subscribe with `AgentSession.subscribe(callback)` before prompting when the
host needs streamed output.
"""

from dataclasses import dataclass

from .messages import AssistantMessage, ToolCall, ToolResultMessage


@dataclass
class AgentStartEvent:
    """A run began for the given user message."""


@dataclass
class MessageStartEvent:
    """An assistant message began streaming (stop_reason is "pending")."""


@dataclass
class TextDeltaEvent:
    """A chunk of assistant text streamed from the provider."""

    delta: str


@dataclass
class ReasoningDeltaEvent:
    """A chunk of model reasoning streamed from the provider (e.g. DeepSeek
    or GLM reasoning_content). Persisted with the assistant message and sent
    back on later requests, as DeepSeek requires when tools are present."""

    delta: str


@dataclass
class ToolNameEvent:
    """The name of a tool call became known while streaming."""

    name: str


@dataclass
class MessageEndEvent:
    """The authoritative completed assistant message for one turn."""

    message: AssistantMessage


@dataclass
class ToolExecutionStartEvent:
    """A tool call is about to execute."""

    tool_call: ToolCall


@dataclass
class ToolExecutionEndEvent:
    """A tool call finished with this result."""

    tool_call: ToolCall
    result: ToolResultMessage


@dataclass
class TaskActivityEvent:
    """Sub-agent tool activity for display, not the parent's conversation."""

    description: str
    event: ToolExecutionStartEvent | ToolExecutionEndEvent


@dataclass
class AgentEndEvent:
    """The run finished. `aborted` is True if it was interrupted."""

    aborted: bool


@dataclass
class ErrorEvent:
    """A recoverable error occurred inside the run."""

    message: str


@dataclass
class CompactionStartEvent:
    """The oldest turns are about to be summarized away."""

    messages: int
    tokens: int


@dataclass
class CompactionEndEvent:
    """History was replaced by a summary; `messages` were dropped.

    `skipped` is set when the summary would not have been smaller than the
    history it replaced, in which case nothing was rewritten.
    """

    messages: int
    tokens_before: int
    tokens_after: int
    skipped: bool = False


# What a provider yields while streaming, before the turn is complete.
ProviderEvent = TextDeltaEvent | ReasoningDeltaEvent | ToolNameEvent

SessionEvent = (
    AgentStartEvent
    | MessageStartEvent
    | TextDeltaEvent
    | ReasoningDeltaEvent
    | ToolNameEvent
    | MessageEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionEndEvent
    | TaskActivityEvent
    | AgentEndEvent
    | ErrorEvent
    | CompactionStartEvent
    | CompactionEndEvent
)
