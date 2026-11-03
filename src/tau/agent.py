"""The agent loop, modeled after pi.

A submitted message is added to the active branch. The agent builds a model
request from the system prompt, active branch, and available tools, then
sends it through the provider. The provider streams an assistant response,
which can contain text and tool calls. The agent records the response,
executes each tool call, and records the results. That completes one turn.
If tool results require another model request, the agent starts another
turn; otherwise the run ends.
"""

import threading
from collections.abc import Callable

from .compaction import (
    SUMMARY_PREFIX,
    CompactionConfig,
    estimate_tokens,
    split_index,
    summarize,
)
from .events import (
    AgentEndEvent,
    AgentStartEvent,
    CompactionEndEvent,
    CompactionStartEvent,
    ErrorEvent,
    MessageEndEvent,
    MessageStartEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from .messages import (
    AgentMessage,
    AssistantMessage,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from .policy import basic_guard
from .providers import ProviderError
from .providers.backend import ModelBackend
from .session import SessionManager
from .tools import Tool, ToolResult, validate_arguments

MAX_RETRIES = 3
RETRY_DELAYS = (1, 2, 4)


class AgentSession:
    """Drives one conversation: prompts in, assistant messages and tool results out.

    It keeps no transcript of its own — history lives in the `SessionManager`,
    so a session restored from disk resumes on whatever branch was active.
    """

    def __init__(
        self,
        backend: ModelBackend,
        tools: list[Tool],
        session_manager: SessionManager,
        system_prompt: str,
        cwd: str | None = None,
        abort_event: threading.Event | None = None,
        compaction: CompactionConfig | None = None,
        tool_guard: Callable[[ToolCall], str | None] | None = None,
    ):
        self.backend = backend
        self.tools = {tool.name: tool for tool in tools}  # keyed for tool-call dispatch
        self.session_manager = session_manager
        self.system_prompt = system_prompt
        self.cwd = cwd
        self.compaction = (
            compaction if compaction is not None else CompactionConfig.from_env()
        )
        self.tool_guard = tool_guard if tool_guard is not None else basic_guard(cwd)
        self._subscribers: list[Callable] = []
        # A shared event lets a parent and its sub-agents abort together.
        self._abort_event = (
            abort_event if abort_event is not None else threading.Event()
        )
        self._busy = False
        self._compaction_blocked = False  # at most one *failed* compaction per run
        self.usage = Usage()  # cumulative totals across runs

        for tool in self.tools.values():
            # Tools that need to reach back into the session (e.g. `subagent` spawning
            # sub-agents) opt in by defining this; the rest are left alone.
            bind = getattr(tool, "bind_session", None)
            if callable(bind):
                bind(self)

        if session_manager.file is None:
            # A fresh manager has no path yet; pick one so this run is persisted.
            session_manager.new_session()

    # -- event subscription ----------------------------------------------------

    def subscribe(self, callback: Callable) -> Callable[[], None]:
        """Subscribe to session events. Returns an unsubscribe function."""
        self._subscribers.append(callback)

        def unsubscribe() -> None:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

        return unsubscribe

    def _emit(self, event) -> None:
        """Fan an event out to subscribers, over a copy so they may unsubscribe."""
        for callback in list(self._subscribers):
            callback(event)

    # -- state -----------------------------------------------------------------

    @property
    def messages(self) -> list[AgentMessage]:
        """The active branch: conversation history for the next request."""
        return self.session_manager.active_branch()

    @property
    def busy(self) -> bool:
        """True while a run is in flight."""
        return self._busy

    @property
    def abort_event(self) -> threading.Event:
        """The run's abort flag, shared with sub-agent sessions."""
        return self._abort_event

    def last_assistant_text(self) -> str:
        """Text of the latest assistant message, or "" if there is none.

        Read off the branch rather than a local buffer, so it also answers for a
        session loaded from disk and for a run that just compacted.
        """
        for message in reversed(self.messages):
            if isinstance(message, AssistantMessage):
                return message.text
        return ""

    def set_model(self, model: str) -> None:
        """Switch the model for subsequent requests; recorded history is untouched."""
        self.backend.provider.model = model

    # -- prompting -------------------------------------------------------------

    def prompt(self, text: str) -> AssistantMessage:
        """Run the agent loop for a user message; returns the final assistant message.

        Loop: compact if over threshold, send one request, and if the response
        carries tool calls, execute them and repeat — results always need a
        follow-up request. Anything else ends the run. Messages are appended as
        they happen, so an interrupted run still reads as a transcript.

        Returns an `aborted` stub if nothing landed before the run stopped.
        """
        self._abort_event.clear()
        self._busy = True
        self._compaction_blocked = False
        self._emit(AgentStartEvent())
        self.session_manager.append(UserMessage(content=text))

        asst_message: AssistantMessage | None = None
        aborted = False
        try:
            while True:
                self._maybe_compact()
                if self._abort_event.is_set():
                    aborted = True
                    break
                asst_message = self._run_turn()
                if asst_message is None:
                    aborted = True
                    break
                if asst_message.tool_calls and asst_message.stop_reason in (
                    "toolUse",
                    "aborted",
                ):
                    # Even partial calls from an aborted stream need results
                    # before the branch can be sent again.
                    if not self._execute_tool_calls(asst_message):
                        aborted = True
                        break
                    continue  # tool results require another model request
                if asst_message.stop_reason == "aborted" or self._abort_event.is_set():
                    aborted = True
                break
        except KeyboardInterrupt:
            aborted = True
        finally:
            self._busy = False
            self._emit(AgentEndEvent(aborted=aborted))

        if asst_message is None:
            # Aborted before the first response landed. Still record an assistant
            # message so the branch never ends on a user turn with no reply.
            asst_message = AssistantMessage(
                content=[],
                model=self.backend.provider.model,
                api=self.backend.api,
                stop_reason="aborted",
            )
            self.session_manager.append(asst_message)
        return asst_message

    def abort(self) -> None:
        """Stop the active run after the current provider chunk / tool call."""
        self._abort_event.set()

    # -- compaction ------------------------------------------------------------

    @property
    def context_tokens(self) -> int:
        """Size of the next request in tokens, as far as we can tell.

        The provider reports the real prompt size (`usage.input`) for the last
        request, and everything appended since then is a handful of messages, so
        that number plus a cheap estimate of the tail beats estimating the whole
        branch: chars-per-token drifts by 15-50% on code-heavy transcripts, and
        the estimate also counts reasoning blocks the request no longer carries
        for finished turns. Falls back to the estimate in a fresh session or one
        resumed from a log without usage.
        """
        messages = self.messages
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if isinstance(message, AssistantMessage) and message.usage.input:
                return message.usage.input + estimate_tokens(messages[index + 1 :])
        return estimate_tokens(messages)

    def _maybe_compact(self) -> bool:
        """Compact before a turn if the branch is over the token threshold."""
        if not self.compaction.enabled or self._compaction_blocked:
            return False
        tokens = self.context_tokens
        if tokens < self.compaction.threshold_tokens:
            return False
        return self.compact(tokens)

    def compact(self, tokens_before: int | None = None) -> bool:
        """Summarize the oldest turns into one message. True if it compacted.

        Failure is not fatal: history is left untouched and the run continues,
        so a flaky summary request cannot lose the conversation. Either way
        compaction is not attempted again during this run.
        """
        messages = self.messages
        split = split_index(messages, self.compaction.keep_turns)
        if split is None:
            return False
        older, keep = messages[:split], messages[split:]
        if tokens_before is None:
            tokens_before = estimate_tokens(messages)
        self._emit(
            CompactionStartEvent(messages=len(older), tokens=estimate_tokens(older))
        )
        try:
            summary = summarize(
                self.backend, older, self.compaction, self._abort_event, self.usage
            )
        except Exception as exc:
            self._compaction_blocked = True
            self._emit(ErrorEvent(message=f"compaction failed: {exc}"))
            return False

        summary_message = UserMessage(
            content=f"{SUMMARY_PREFIX}\n{summary}\n</conversation_summary>"
        )
        tokens_after = estimate_tokens([summary_message, *keep])
        if tokens_after >= tokens_before:
            # A branch that is short, or mostly one huge tool result the model
            # summarized at length, can summarize to something *bigger* than
            # itself. Keep the verbatim history: it is strictly better.
            self._compaction_blocked = True
            self._emit(
                CompactionEndEvent(
                    messages=0,
                    tokens_before=tokens_before,
                    tokens_after=tokens_before,
                    skipped=True,
                )
            )
            return False

        if self._abort_event.is_set():
            return False
        self.session_manager.compact(summary_message, keep)
        self._emit(
            CompactionEndEvent(
                messages=len(older),
                tokens_before=tokens_before,
                tokens_after=tokens_after,
            )
        )
        return True

    def dispose(self) -> None:
        """Lifecycle hook for embedders: stop whatever is running."""
        self.abort()

    # -- turns -------------------------------------------------------------------

    def _run_turn(self) -> AssistantMessage | None:
        """One model request: stream a response, record it, return it."""
        last_error: ProviderError | None = None
        for attempt in range(MAX_RETRIES):
            # Emitted per attempt, not once per turn: a retry re-streams the
            # response from its first token, so hosts must reset their
            # renderers instead of appending to a half-rendered message.
            self._emit(MessageStartEvent())
            messages, tools = self.messages, list(self.tools.values())
            stream = None
            try:
                while True:
                    try:
                        if stream is None:
                            stream = self.backend.stream_response(
                                self.system_prompt,
                                messages,
                                tools,
                                abort_event=self._abort_event,
                            )
                        event = next(stream)
                    except StopIteration as stop:
                        asst_message: AssistantMessage = stop.value
                        self.usage.add(asst_message.usage)
                        self.session_manager.append(asst_message)
                        self._emit(MessageEndEvent(message=asst_message))
                        return asst_message
                    except ProviderError as exc:
                        last_error = exc
                        break
                    # Subscriber errors must never become provider retries.
                    self._emit(event)
            finally:
                if stream is not None:
                    stream.close()
            if last_error.usage is not None:
                self.usage.add(last_error.usage)
            if not last_error.retryable:
                break
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                self._emit(ErrorEvent(message=f"{last_error}; retrying in {delay}s"))
                if self._abort_event.wait(delay):
                    return None
        error = AssistantMessage(
            content=[],
            model=self.backend.provider.model,
            api=self.backend.api,
            stop_reason="error",
            error_message=str(last_error),
        )
        self._emit(ErrorEvent(message=str(last_error)))
        self.session_manager.append(error)
        self._emit(MessageEndEvent(message=error))
        return error

    def _execute_tool_calls(self, assistant: AssistantMessage) -> bool:
        """Execute each tool call and record the results. False if aborted."""
        calls = assistant.tool_calls
        for index, call in enumerate(calls):
            # Emitted before the abort check so every announced call is closed on
            # the far side too, even when it never runs.
            self._emit(ToolExecutionStartEvent(tool_call=call))
            if assistant.stop_reason == "aborted" or self._abort_event.is_set():
                self._abandon(calls[index:])
                return False
            try:
                result = self._execute_one(call)
            except KeyboardInterrupt:
                # Ctrl-C inside a tool: the call that was running must still be
                # answered before the interrupt is allowed to unwind.
                self._abandon(calls[index:])
                raise
            message = ToolResultMessage(
                tool_call_id=call.id,
                tool_name=call.name,
                output=result.output,
                is_error=result.is_error,
            )
            # Appended per call, so an abort between two calls still leaves the
            # earlier result beside its tool call rather than orphaned.
            self.session_manager.append(message)
            self._emit(ToolExecutionEndEvent(tool_call=call, result=message))
        return True

    def _abandon(self, calls: list[ToolCall]) -> None:
        """Record a result for calls that will never run.

        Every tool call needs a result on the branch: OpenAI-compatible providers
        reject the next request when an assistant message's tool calls are not all
        answered, so a single interrupt would otherwise end the session for good.
        """
        for call in calls:
            message = ToolResultMessage(
                tool_call_id=call.id,
                tool_name=call.name,
                output="Error: aborted before the tool ran",
                is_error=True,
            )
            self.session_manager.append(message)
            self._emit(ToolExecutionEndEvent(tool_call=call, result=message))

    def _execute_one(self, call: ToolCall) -> ToolResult:
        """Run a single tool call, converting every failure into an error result.

        Unknown tool, malformed arguments, schema violations, a policy block and a
        raising tool all come back as `is_error=True` output instead of an
        exception: the model gets to see what went wrong and try again.
        """
        tool = self.tools.get(call.name)
        if tool is None:
            return ToolResult(output=f"Error: unknown tool: {call.name}", is_error=True)
        if not isinstance(call.arguments, dict):
            return ToolResult(
                output=(
                    f"Error: bad arguments for {call.name}: expected a JSON object, "
                    f"got {type(call.arguments).__name__}"
                ),
                is_error=True,
            )
        problems = validate_arguments(tool.parameters, call.arguments)
        if problems:
            return ToolResult(
                output=(
                    f"Error: invalid arguments for {call.name}: "
                    + "; ".join(problems)
                    + ". Fix the arguments and call the tool again."
                ),
                is_error=True,
            )
        if self.tool_guard is not None:
            reason = self.tool_guard(call)
            if reason:
                return ToolResult(
                    output=f"Error: blocked by policy: {reason}", is_error=True
                )
        try:
            return tool.execute(**call.arguments)
        except TypeError as exc:
            return ToolResult(
                output=f"Error: bad arguments for {call.name}: {exc}", is_error=True
            )
        except Exception as exc:  # noqa: BLE001 — tool failures must reach the model
            return ToolResult(
                output=f"Error executing {call.name}: {exc}", is_error=True
            )
