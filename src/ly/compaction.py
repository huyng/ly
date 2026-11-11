"""Context compaction: summarize old turns when the branch grows too long.

A long-running session eventually outgrows the model's context window. Instead
of failing at that point, ly summarizes the oldest turns into a single
message once the active branch crosses a token threshold, and keeps the most
recent turns verbatim.

Two properties are deliberate:

- **Cuts land on user-message boundaries.** The kept suffix therefore never
  starts with a tool result whose assistant tool call was summarized away,
  which would be an invalid request for OpenAI-compatible providers.
- **Nothing is destroyed.** Compaction only moves the active leaf of the
  session tree; the original entries stay in the JSONL file, so the full
  transcript is still on disk (and still resumable by hand).
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass

from .messages import (
    AgentMessage,
    AssistantMessage,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from .providers import ProviderError

CHARS_PER_TOKEN = 4
MESSAGE_OVERHEAD_TOKENS = 4

MAX_USER_CHARS = 2_000
MAX_ASSISTANT_CHARS = 1_200
MAX_TOOL_RESULT_CHARS = 2_000
MAX_ARG_CHARS = 200

#: Threshold used when no preset knows the model's window. Roughly half a
#: typical window, and low enough that an unknown model still compacts.
DEFAULT_THRESHOLD_TOKENS = 100_000

SUMMARY_PREFIX = (
    "<conversation_summary>\n"
    "Earlier turns were compacted to save context. This is a summary of them, "
    "not verbatim history; the code and files on disk are the source of truth."
)
SUMMARY_SUFFIX = "</conversation_summary>"

#: A summary earns no more room than a user turn, wrapper included: it replaces
#: turns, and bounding it here once means neither the next batch nor a later
#: compaction has to trim it.
MAX_SUMMARY_CHARS = MAX_USER_CHARS - len(f"{SUMMARY_PREFIX}\n\n{SUMMARY_SUFFIX}")


def summary_message(summary: str) -> UserMessage:
    """Wrap summary text as the user message that re-enters history."""
    return UserMessage(content=f"{SUMMARY_PREFIX}\n{summary}\n{SUMMARY_SUFFIX}")


SUMMARY_SYSTEM_PROMPT = """\
You compress conversation history for a coding agent that will keep working
after the transcript you are given is replaced by your notes. Write terse,
factual notes, never prose or commentary about the task itself."""

SUMMARY_REQUEST = """\
Summarize the transcript below so an agent can continue the same task without
it.

Preserve, with exact paths and identifiers:
- the user's goals, constraints, and stated preferences
- decisions made, and the reasons for them
- files read, created, or edited, and what changed
- key findings, with file:line references where they matter
- commands run and their outcomes, including failures and error text
- what is still unfinished or uncertain

Drop greetings, narration, and raw tool output. Do not invent details; if
something is unknown, say so. Use bullet points. Keep it under 300 words.

Previous summary (may be "(none)"):
{summary}

Transcript:
{transcript}"""


@dataclass(frozen=True)
class CompactionConfig:
    """When and how aggressively to compact the active branch.

    Defaults only: `Settings` supplies the values, since it is the one place that
    reads the environment and `~/.ly/settings.json`.
    """

    enabled: bool = True
    # Estimated tokens in the active branch at which compaction kicks in.
    # Sizes a model's window implies are in `for_window`; override for taste.
    threshold_tokens: int = DEFAULT_THRESHOLD_TOKENS
    # Recent user turns kept verbatim. Everything older is summarized.
    keep_turns: int = 4
    # Upper bound on a single summarization request; longer histories are
    # summarized in batches, each folding in the previous summary. Equal to the
    # threshold by default, since a branch never holds more than that when it
    # triggers, so one pass can read the whole history.
    max_summary_input_tokens: int = DEFAULT_THRESHOLD_TOKENS


def for_window(window: int) -> CompactionConfig:
    """Compaction defaults sized to one model's context window.

    Half the window, so the request and the reply always fit, however large the
    window is: a bigger model is worth a longer verbatim history. The summary
    cap is that same half rather than a second guess at the window, since
    compaction triggers there and the history to summarize cannot be longer, so
    one pass reads all of it.

    Args:
        window: Tokens the model accepts.

    Returns:
        A config that compacts at half the window and summarizes in one pass up
        to the same size. `keep_turns` keeps its own default, since it counts
        turns rather than tokens.
    """
    threshold = window // 2
    return CompactionConfig(
        threshold_tokens=threshold, max_summary_input_tokens=threshold
    )


def estimate_tokens(messages: list[AgentMessage]) -> int:
    """Cheap token estimate (~4 chars/token). Deliberately model-agnostic."""
    total = 0
    for message in messages:
        total += MESSAGE_OVERHEAD_TOKENS
        if isinstance(message, UserMessage):
            total += len(message.content) // CHARS_PER_TOKEN
        elif isinstance(message, ToolResultMessage):
            total += len(message.output) // CHARS_PER_TOKEN
        else:
            text = message.text + (message.reasoning_content or "")
            total += len(text) // CHARS_PER_TOKEN
            for call in message.tool_calls:
                arguments = json.dumps(call.arguments, default=str)
                total += (len(call.name) + len(arguments)) // CHARS_PER_TOKEN
    return total


def split_index(messages: list[AgentMessage], keep_turns: int) -> int | None:
    """Index where the kept suffix begins, or None if there is nothing to drop.

    The cut always lands on a user message so that tool calls and their
    results stay together.
    """
    starts = [
        i for i, message in enumerate(messages) if isinstance(message, UserMessage)
    ]
    keep = max(keep_turns, 1)
    if len(starts) <= keep:
        return None
    index = starts[-keep]
    return index if index > 0 else None


def render_transcript(messages: list[AgentMessage]) -> str:
    """A readable, aggressively truncated view of messages for summarizing."""
    parts: list[str] = []
    for message in messages:
        if isinstance(message, UserMessage):
            parts.append(f"USER: {_clip(message.content, MAX_USER_CHARS)}")
        elif isinstance(message, AssistantMessage):
            if message.text.strip():
                parts.append(
                    f"ASSISTANT: {_clip(message.text.strip(), MAX_ASSISTANT_CHARS)}"
                )
            for call in message.tool_calls:
                arguments = json.dumps(call.arguments, default=str)
                parts.append(
                    f"ASSISTANT called {call.name}({_clip(arguments, MAX_ARG_CHARS)})"
                )
        elif isinstance(message, ToolResultMessage):
            status = "ERROR" if message.is_error else "OK"
            parts.append(
                f"RESULT {message.tool_name} [{status}]: "
                f"{_clip(message.output.strip(), MAX_TOOL_RESULT_CHARS)}"
            )
    return "\n\n".join(parts)


def summarize(
    backend,
    messages: list[AgentMessage],
    config: CompactionConfig,
    abort_event=None,
    usage: Usage | None = None,
) -> str:
    """Summarize messages into terse notes; raises on provider failure.

    Histories larger than `max_summary_input_tokens` are folded in batches,
    each request carrying the summary produced so far. Token usage of the
    summary calls is added to `usage` when given, so it counts toward the
    session's cumulative totals.
    """
    summary = ""
    for batch in _batches(messages, config.max_summary_input_tokens):
        if abort_event is not None and abort_event.is_set():
            raise RuntimeError("compaction aborted")
        request = SUMMARY_REQUEST.format(
            summary=summary.strip() or "(none)",
            transcript=render_transcript(batch),
        )
        try:
            message = _complete(backend, request, abort_event)
        except ProviderError as exc:
            if usage is not None and exc.usage is not None:
                usage.add(exc.usage)
            raise
        if usage is not None:
            usage.add(message.usage)
        if message.stop_reason == "aborted" and abort_event is not None:
            abort_event.set()
        if abort_event is not None and abort_event.is_set():
            raise RuntimeError("compaction aborted")
        if message.stop_reason != "stop":
            raise RuntimeError(
                message.error_message
                or f"summary did not complete: {message.stop_reason}"
            )
        text = message.text.strip()
        if not text:
            raise RuntimeError("provider returned an empty summary")
        summary = _clip(text, MAX_SUMMARY_CHARS)
    return summary


def _batches(
    messages: list[AgentMessage], max_tokens: int
) -> Iterator[list[AgentMessage]]:
    batch: list[AgentMessage] = []
    size = 0
    for message in messages:
        cost = estimate_tokens([message])
        if batch and size + cost > max_tokens:
            yield batch
            batch, size = [], 0
        batch.append(message)
        size += cost
    if batch:
        yield batch


def _complete(backend, request: str, abort_event) -> AssistantMessage:
    """One non-streamed (but still stream-parsed) completion with no tools."""
    stream = backend.stream_response(
        SUMMARY_SYSTEM_PROMPT,
        [UserMessage(content=request)],
        [],
        abort_event=abort_event,
    )
    while True:
        try:
            next(stream)
        except StopIteration as stop:
            message = stop.value
            if not isinstance(message, AssistantMessage):  # defensive
                raise TypeError("backend did not return an assistant message")
            return message


def _clip(text: str, limit: int) -> str:
    if limit <= 1 or len(text) <= limit:
        return text
    return text[: limit - 1] + "…"
