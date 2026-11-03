"""Interactive terminal interface, modeled after pi's REPL.

Modes:
- interactive (default): stream responses, use slash commands
- print: `tau -p "prompt"` runs once and writes the final response
"""

import argparse
import os
import readline  # noqa: F401 — side effect: line editing for input(), see _read
import select
import sys
import time
from collections.abc import Callable
from datetime import datetime

from . import __version__
from .agent import AgentSession
from .events import (
    AgentEndEvent,
    AgentStartEvent,
    CompactionEndEvent,
    CompactionStartEvent,
    ErrorEvent,
    MessageEndEvent,
    MessageStartEvent,
    ReasoningDeltaEvent,
    TaskActivityEvent,
    TextDeltaEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolNameEvent,
)
from .prompts import build_system_prompt, context_files
from .providers import Provider, make_backend
from .providers.errors import MissingCredentialsError
from .render import (
    GutterWriter,
    MarkdownRenderer,
    format_duration,
    render_tool_result,
    summarize_tool_call,
)
from .session import SessionManager
from .spinner import Spinner
from .theme import DIM, Theme, get_theme, terminal_width
from .tools import default_tools

HELP = """\
/help              Show this help
/model [name]      Show or switch the model
/tools             List active tools
/system            Show the current system prompt
/plain             Toggle Markdown rendering
/thinking          Toggle streamed model reasoning
/new               Start a new session
/sessions          List recent sessions
/resume <n>        Resume the n-th session from /sessions
/compact           Summarize old turns now (automatic near the limit)
/quit              Exit (also Ctrl-D)
"""


def _flag(name: str) -> bool | None:
    """Read a tri-state environment flag; None when the variable is unset."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.strip().lower() not in ("0", "false", "no", "off")


def make_session(args: argparse.Namespace) -> AgentSession:
    provider = Provider(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        pass_reasoning_content=_flag("TAU_PASS_REASONING_CONTENT"),
    )
    manager = SessionManager()
    if args.resume:
        sessions = SessionManager.list_sessions()
        if not sessions:
            sys.exit("No sessions found.")
        manager.load(sessions[args.resume - 1])
    else:
        manager.new_session()
    return AgentSession(
        backend=make_backend(provider, args.api),
        tools=default_tools(cwd=os.getcwd()),
        session_manager=manager,
        system_prompt=build_system_prompt(os.getcwd()),
        cwd=os.getcwd(),
    )


class Repl:
    def __init__(self, session: AgentSession, theme: Theme | None = None):
        self.session = session
        self.theme = theme if theme is not None else get_theme()
        self.width = terminal_width()
        self.markdown = True
        self.show_thinking = True

        self.session.subscribe(self._on_event)
        self._text_started = False
        self._text_ends_newline = False
        self._renderer = MarkdownRenderer(self.theme, self.width)
        self._reasoning = GutterWriter(
            self.theme, style=DIM, width=self.width, label="thinking"
        )
        self._tool_started: dict[str, float] = {}
        self._spinner = Spinner()
        self._handlers: dict[type, Callable[..., None]] = {
            AgentStartEvent: self._on_agent_start,
            MessageStartEvent: self._on_message_start,
            ReasoningDeltaEvent: self._on_reasoning_delta,
            TextDeltaEvent: self._on_text_delta,
            ToolNameEvent: self._on_tool_name,
            MessageEndEvent: self._on_message_end,
            ToolExecutionStartEvent: self._on_tool_start,
            ToolExecutionEndEvent: self._on_tool_end,
            TaskActivityEvent: self._on_task_activity,
            ErrorEvent: self._on_error,
            CompactionStartEvent: self._on_compaction_start,
            CompactionEndEvent: self._on_compaction_end,
            AgentEndEvent: self._on_agent_end,
        }

    # -- event rendering ---------------------------------------------------------

    def _on_event(self, event) -> None:
        handler = self._handlers.get(type(event))
        if handler is not None:
            handler(event)

    def _on_agent_start(self, _event: AgentStartEvent) -> None:
        pass

    def _on_tool_name(self, _event: ToolNameEvent) -> None:
        pass

    def _on_message_start(self, _event: MessageStartEvent) -> None:
        self._text_started = False
        self._text_ends_newline = False
        # A resize between turns should land here rather than mid-line.
        self.width = terminal_width()
        self._renderer = MarkdownRenderer(self.theme, self.width)
        self._reasoning = GutterWriter(
            self.theme, style=DIM, width=self.width, label="thinking"
        )
        self._spinner.start(f"{self.session.backend.provider.model} · thinking")

    def _on_reasoning_delta(self, event: ReasoningDeltaEvent) -> None:
        self._spinner.stop()
        if self.show_thinking:
            print(self._reasoning.feed(event.delta), end="", flush=True)

    def _on_text_delta(self, event: TextDeltaEvent) -> None:
        self._spinner.stop()
        self._finish_reasoning()
        self._text_started = True
        self._text_ends_newline = event.delta.endswith("\n")
        if self.markdown:
            print(self._renderer.feed(event.delta), end="", flush=True)
        else:
            print(event.delta, end="", flush=True)

    def _on_message_end(self, event: MessageEndEvent) -> None:
        self._spinner.stop()
        self._finish_reasoning()
        self._finish_text(event.message.text)

    def _on_tool_start(self, event: ToolExecutionStartEvent) -> None:
        self._spinner.stop()
        self._finish_reasoning()
        self._finish_text()
        call = event.tool_call
        summary = summarize_tool_call(self.theme, call.name, call.arguments, self.width)
        head = self.theme.dim("╭─ ") + self.theme.bold(call.name)
        if summary:
            # Multi-line summaries (bash scripts) carry their own gutter, so
            # they start on the line below the tool name instead of beside it.
            head += ("\n" if "\n" in summary else " ") + summary
        print(head, flush=True)
        self._tool_started[call.id] = time.monotonic()
        self._spinner.start(f"{call.name} …")

    def _on_tool_end(self, event: ToolExecutionEndEvent) -> None:
        self._spinner.stop()
        started = self._tool_started.pop(event.tool_call.id, None)
        elapsed = time.monotonic() - started if started is not None else 0.0
        print(
            render_tool_result(
                self.theme,
                event.tool_call.name,
                event.result.output,
                event.result.is_error,
                elapsed,
                self.width,
            ),
            flush=True,
        )

    def _on_task_activity(self, event: TaskActivityEvent) -> None:
        self._spinner.stop()
        activity = event.event
        call = activity.tool_call
        if isinstance(activity, ToolExecutionStartEvent):
            summary = summarize_tool_call(
                self.theme, call.name, call.arguments, self.width - 4
            )
            line = self.theme.dim("→ ") + self.theme.bold(call.name)
            if summary:
                line += " " + summary
        else:
            status = (
                self.theme.error("✗")
                if activity.result.is_error
                else self.theme.dim("✓")
            )
            line = status + self.theme.dim(f" {call.name}")
        print(
            "\n".join(self.theme.dim("│  ") + part for part in line.splitlines()),
            flush=True,
        )
        self._spinner.start(f"subagent · {event.description or 'sub-agent'} …")

    def _on_error(self, event: ErrorEvent) -> None:
        self._spinner.stop()
        print(self.theme.warning(event.message), file=sys.stderr, flush=True)

    def _on_compaction_start(self, event: CompactionStartEvent) -> None:
        self._spinner.stop()
        print(
            self.theme.dim(
                f"compacting {event.messages} messages (~{event.tokens:,} tokens)…"
            ),
            flush=True,
        )

    def _on_compaction_end(self, event: CompactionEndEvent) -> None:
        self._spinner.stop()
        if event.skipped:
            print(
                self.theme.dim(
                    "compaction skipped: the summary would not be smaller "
                    "than the history it replaces"
                ),
                flush=True,
            )
            return
        print(
            self.theme.dim(
                f"compacted {event.messages} messages · "
                f"~{event.tokens_before:,} → ~{event.tokens_after:,} tokens · "
                "full history kept in the session file"
            ),
            flush=True,
        )

    def _on_agent_end(self, event: AgentEndEvent) -> None:
        self._spinner.stop()
        self._finish_reasoning()
        self._finish_text()
        if event.aborted:
            print(self.theme.warning("[aborted]"), flush=True)

    def _finish_reasoning(self) -> None:
        print(self._reasoning.flush(), end="", flush=True)

    def _finish_text(self, text: str | None = None) -> None:
        if not self._text_started:
            return
        if self.markdown:
            print(self._renderer.flush(), end="", flush=True)
        else:
            ended = text.endswith("\n") if text is not None else self._text_ends_newline
            if not ended:
                print(flush=True)
        self._text_started = False

    # -- main loop ----------------------------------------------------------------

    def run(self) -> None:
        theme = self.theme
        print(
            f"{theme.bold('tau')} {theme.dim('v' + __version__)} {theme.dim('·')} "
            f"{theme.model(self.session.backend.provider.model)}"
        )
        print(theme.dim(f"  • session {self.session.session_manager.file}"))
        for path, scope, _ in context_files(self.session.cwd):
            print(theme.dim(f"  • {scope.lower()} AGENTS.md {path}"))
        while True:
            try:
                line = self._read(theme.accent("» "))
            except KeyboardInterrupt:
                print(theme.dim("(use /quit to exit)"))
                continue
            if line is None:
                print()
                break

            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                if self._command(line):
                    break
                continue

            self._submit(line)

    def _read(self, prompt: str) -> str | None:
        """Read one prompt, sending a pasted block as a single message.

        `input()` returns a single line, so a multi-line paste would turn into
        one prompt per line. What is left of the paste is already queued in the
        tty, so drain it while data is ready and join on newlines. Only a real
        terminal behaves that way: with piped stdin the leftover lines sit in
        Python's buffer, where the peek would miss them.

        The line is editable (Ctrl-A/Ctrl-E, arrow-key history) because
        `readline` is imported at module level; `input()` only gets that
        behaviour when the module has been imported.
        """
        try:
            first = input(prompt)
        except EOFError:
            return None
        if not sys.stdin.isatty():
            return first
        lines = [first]
        while select.select([sys.stdin], [], [], 0)[0]:
            try:
                lines.append(input())
            except EOFError:
                break
        return "\n".join(lines)

    def _submit(self, text: str) -> None:
        started = time.monotonic()
        output_before = self.session.usage.output
        try:
            self.session.prompt(text)
        except KeyboardInterrupt:
            pass  # abort path emits [aborted]
        finally:
            self._spinner.stop()  # e.g. Ctrl-C surfaced outside event handling
        print()  # blank line after each run
        elapsed = time.monotonic() - started
        self._print_usage(elapsed, self.session.usage.output - output_before)

    def _print_usage(self, elapsed: float, output_tokens: int) -> None:
        # `output_tokens` is this run's delta: session usage is cumulative,
        # so a rate built from it would shrink with every prompt.
        usage = self.session.usage
        if usage.total_tokens:
            parts = [self.session.backend.provider.model, format_duration(elapsed)]
            if elapsed and output_tokens:
                parts.append(f"{output_tokens / elapsed:.0f} tok/s")
            # The cumulative `in` below cannot say how close the next request is
            # to the compaction threshold; this is the number that does.
            context = f"ctx {self.session.context_tokens:,}"
            if self.session.compaction.enabled:
                context += f" / {self.session.compaction.threshold_tokens:,}"
            parts += [
                context,
                f"{usage.input:,} in",
                f"{usage.output:,} out",
            ]
            print(
                self.theme.dim("⟨ " + " · ".join(parts) + " ⟩"),
                flush=True,
            )

    # -- slash commands -------------------------------------------------------------

    def _command(self, line: str) -> bool:
        """Handle a slash command. Returns True to exit."""
        parts = line.split(maxsplit=1)
        command, arg = parts[0].lower(), (parts[1].strip() if len(parts) > 1 else "")
        theme = self.theme

        if command in ("/quit", "/exit"):
            return True
        if command == "/help":
            print(HELP, end="")
        elif command == "/model":
            if arg:
                self.session.set_model(arg)
            print(
                f"{theme.dim('model:')} {theme.model(self.session.backend.provider.model)}"
            )
        elif command == "/tools":
            for tool in self.session.tools.values():
                first = tool.description.split(". ")[0]
                print(f"  {theme.bold(tool.name)}{theme.dim(': ' + first)}")
        elif command == "/system":
            print(self.session.system_prompt)
        elif command == "/plain":
            self.markdown = not self.markdown
            state = "on" if self.markdown else "off"
            print(theme.dim(f"markdown rendering: {state}"))
        elif command == "/thinking":
            self.show_thinking = not self.show_thinking
            state = "shown" if self.show_thinking else "hidden"
            print(theme.dim(f"reasoning: {state}"))
        elif command == "/new":
            self.session.session_manager.new_session()
            print(theme.dim(f"new session: {self.session.session_manager.file}"))
        elif command == "/sessions":
            for i, path in enumerate(SessionManager.list_sessions(), 1):
                marker = "→" if path == self.session.session_manager.file else " "
                marker = theme.accent(marker)
                stamp = datetime.fromtimestamp(path.stat().st_mtime)
                when = stamp.strftime("%Y-%m-%d %H:%M")
                print(f"{marker} {i}. {path.stem}  {theme.dim(when)}")
        elif command == "/resume":
            if not arg.isdigit():
                print("usage: /resume <n>")
                return False
            sessions = SessionManager.list_sessions()
            index = int(arg) - 1
            if not 0 <= index < len(sessions):
                print(theme.error("no such session"))
                return False
            self.session.session_manager.load(sessions[index])
            print(theme.dim(f"resumed {sessions[index]}"))
        elif command == "/compact":
            if not self.session.compact():
                print(theme.dim("nothing to compact (too few turns, or disabled)"))
        else:
            print(
                theme.error(f"unknown command: {command}") + theme.dim(" (try /help)")
            )
        return False


def print_mode(args: argparse.Namespace) -> None:
    """Like `pi -p`: run one prompt, write the final response to stdout.

    Deliberately plain: no styling, no tool chatter, no spinners. Failures go
    to stderr with a non-zero exit, so `tau -p` is usable in scripts.
    """
    session = make_session(args)
    failures: list[str] = []

    def on_event(event) -> None:
        if isinstance(event, TextDeltaEvent):
            print(event.delta, end="", flush=True)
        elif isinstance(event, ErrorEvent):
            failures.append(event.message)

    unsubscribe = session.subscribe(on_event)
    try:
        session.prompt(args.prompt)
    finally:
        unsubscribe()
    print()
    if failures:
        print(f"tau: {failures[-1]}", file=sys.stderr)
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    """The command line, with `TAU_*` variables filled in as argument defaults."""
    parser = argparse.ArgumentParser(
        prog="tau", description="A minimal coding agent, modeled after pi"
    )
    parser.add_argument("prompt", nargs="?", help="starting prompt (interactive mode)")
    parser.add_argument(
        "-p",
        "--print",
        action="store_true",
        help="print mode: run once, write final response",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=os.environ.get("TAU_MODEL") or "gpt-4.1",
        help="model to use (default: $TAU_MODEL or gpt-4.1)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("TAU_BASE_URL") or "",
        help="OpenAI-compatible API base URL (default: $TAU_BASE_URL)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("TAU_API_KEY"),
        help="API key (default: $TAU_API_KEY)",
    )
    parser.add_argument(
        "--api",
        choices=("chat", "responses"),
        default=os.environ.get("TAU_API") or "chat",
        help="OpenAI endpoint flavour (default: $TAU_API or chat); the newest "
        "models need `responses` to use tools",
    )
    parser.add_argument(
        "--resume", type=int, metavar="N", help="resume the n-th recent session"
    )
    parser.add_argument("--version", action="version", version=f"tau {__version__}")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    try:
        if args.print:
            if not args.prompt:
                sys.exit("-p requires a prompt")
            print_mode(args)
            return

        session = make_session(args)
    except MissingCredentialsError as exc:
        sys.exit(str(exc))

    repl = Repl(session)
    if args.prompt:
        repl._submit(args.prompt)
    repl.run()


if __name__ == "__main__":
    main()
