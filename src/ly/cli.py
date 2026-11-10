"""Interactive terminal interface, modeled after pi's REPL.

Modes:
- interactive (default): stream responses, use slash commands
- print: `ly -p "prompt"` runs once and writes the final response

Both modes run inside a Docker container by default; `--nosandbox` stays on
the host (see sandbox.py).
"""

import argparse
import os
import re
import readline
import sys
import textwrap
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from . import __version__, sandbox
from .agent import AgentSession
from .commands import discover
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
from .providers.errors import (
    MissingBaseURLError,
    MissingCredentialsError,
    UnknownProviderError,
)
from .render import (
    GutterWriter,
    MarkdownRenderer,
    format_duration,
    render_tool_result,
    summarize_tool_call,
)
from .session import SYNOPSIS_CHARS, SessionManager
from .settings import Settings
from .spinner import Spinner
from .theme import BAND, DIM, RESET, SHADE, Theme, get_theme, terminal_width
from .tools import default_tools

HELP = """\
/help              Show this help
/model [ref]       Show the model, or switch to `provider/model`
/tools             List active tools
/system            Show the current system prompt
/config            Show the settings in effect
/plain             Toggle Markdown rendering
/thinking          Toggle streamed model reasoning
/new               Start a new session
/sessions          List recent sessions by their last prompt
/resume <n>        Resume the n-th session from /sessions
/compact           Summarize old turns now (automatic near the limit)
/<name> [args]     Run a command defined in .agents/commands (listed below)
/quit              Exit (also Ctrl-D)
"""


def _reasoning_mode(value: bool | None) -> str:
    """Name the tri-state reasoning setting; None means the provider decides."""
    if value is None:
        return "auto (from base URL)"
    return "on" if value else "off"


def use_container(args: argparse.Namespace) -> bool:
    """Return True when this process should hand off to `docker run` first."""
    # LY_DOCKER is not read here: it reached us through `Settings`, which turned
    # it into the parser's default for `--nosandbox`.
    return not (sandbox.in_container() or args.nosandbox)


def make_session(
    args: argparse.Namespace, provider: Provider, sessions_dir: Path
) -> AgentSession:
    """Build the session this run uses, in the directory `main` resolved."""
    manager = SessionManager(sessions_dir)
    if args.resume:
        sessions = SessionManager.list_sessions(sessions_dir)
        if not sessions:
            sys.exit("No sessions found.")
        manager.load(sessions[args.resume - 1])
    else:
        manager.new_session()
    return AgentSession(
        backend=make_backend(provider),
        tools=default_tools(cwd=os.getcwd()),
        session_manager=manager,
        system_prompt=build_system_prompt(os.getcwd()),
        cwd=os.getcwd(),
        compaction=args.compaction,
    )


class Repl:
    def __init__(
        self,
        session: AgentSession,
        theme: Theme | None = None,
        settings: Settings | None = None,
    ):
        self.session = session
        self.theme = theme if theme is not None else get_theme()
        self.width = terminal_width()
        preferences = settings or Settings()
        self.settings = preferences
        self.markdown = preferences.markdown
        self.show_thinking = preferences.thinking
        self.commands = discover(self.session.cwd)
        self._pastes: dict[str, str] = {}

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
        # A retry re-enters here mid-stream. Close the failed attempt's reasoning
        # frame first, or the retry nests a second header inside the first.
        self._finish_reasoning()
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

        # print startup banner
        theme = self.theme
        print(
            f"{theme.bold('ly')} {theme.dim('v' + __version__)} {theme.dim('·')} "
            f"{theme.model(self.settings.model)}"
        )
        print(theme.dim(f"  • session {self.session.session_manager.file}"))
        if sandbox.in_container():
            # The container's own cwd is always /workspace, so name the host
            # directory too: that is the one the user can type in another shell.
            host = sandbox.host_dir()
            where = f"{host} → {self.session.cwd}" if host else self.session.cwd
            detail = f"  • sandboxed docker {sandbox.image_name()} · {where}"
            print(theme.dim(detail))
        for path, scope, _ in context_files(self.session.cwd):
            print(theme.dim(f"  • {scope.lower()} AGENTS.md {path}"))

        # blank line after banner
        print()

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

            # Pick up a resize made while the prompt was being typed; the next
            # turn refreshes it again in _on_message_start.
            self.width = terminal_width()
            print()  # space the block off from what the user just typed
            self._print_prompt(line)
            self._submit(line)

    def _print_prompt(self, text: str) -> None:
        """Echo a submitted prompt on a dark green band.

        Each row paints a single space and erases to the end of the line: most
        terminals fill erased cells with the current background, so the band
        still reaches the window edge while only a couple of cells are written.
        That keeps a resized window from reflowing the padding into extra rows.
        Long lines wrap short of the edge so the terminal never has to.

        Args:
            text: The prompt the user submitted.
        """
        width = self.width
        # The glyph echoes the `» ` the user typed, repeated on every row so a
        # wrapped prompt still reads as one block.
        glyph = " » "

        def row(text: str) -> str:
            """Paint one band row and fill the rest of it."""
            if not self.theme.enabled:
                return text  # no band without colour; keep the output clean
            return f"{SHADE}{text}{BAND}{RESET}"

        print(row(" "))  # one empty band row above the text
        for line in text.splitlines():
            for line_no, part in enumerate(
                textwrap.wrap(line, width - len(glyph) - 1) or [""]
            ):
                if line_no == 0:
                    print(row(f"{glyph}{part}"))
                else:
                    print(row(f"   {part}"))

        print(row(" "))  # and one below
        print()  # an unpainted line closes the band

    def _read(self, prompt: str) -> str | None:
        """Read a prompt, keeping bracketed pastes together until Enter."""
        try:
            if not sys.stdin.isatty():
                return input(prompt)
            if "libedit" in (readline.__doc__ or ""):
                # Use libedit
                # Libedit misrenders color even with nonprinting markers.
                prompt = re.sub(r"\x1b\[[0-9;]*m", "", prompt)
                return self._read_libedit(prompt)

            else:
                # Use GNU Readline
                # GNU Readline must not count color sequences as visible columns.
                prompt = re.sub(r"\x1b\[[0-9;]*m", "\001\\g<0>\002", prompt)
                readline.parse_and_bind("set enable-bracketed-paste on")
                return input(prompt)
        except EOFError:
            return None

    def _read_libedit(self, prompt: str) -> str:
        """Capture literal pastes through libedit's completion hook.

        Libedit has no native bracketed-paste handler. Show a placeholder rather
        than inserting newlines into its single-line editor, where they submit
        the prompt. Expand placeholders only after the user presses Enter.
        """
        # History keeps placeholders: literal newlines can submit on recall.
        # Keep their contents across reads so recalled pastes still expand.
        pastes = self._pastes
        failures: list[BaseException] = []

        def paste(text: str, state: int) -> str | None:
            if state:
                return None
            try:
                data = bytearray()
                while not data.endswith(b"\x1b[201~"):
                    chunk = os.read(sys.stdin.fileno(), 1)
                    if not chunk:
                        raise EOFError
                    data.extend(chunk)
                content = data[:-6].decode(sys.stdin.encoding, errors="replace")
                content = content.replace("\r\n", "\n").replace("\r", "\n")
                marker = f"[paste {uuid4().hex[:8]}: {len(content)} chars]"
                pastes[marker] = content
                # Let completion insert and paint the placeholder immediately.
                return text + marker
            except BaseException as exc:
                # Readline swallows callback exceptions; propagate after input.
                failures.append(exc)
            return None

        previous = readline.get_completer()
        readline.set_completer(paste)
        readline.parse_and_bind(r'bind "^[[200~" rl_complete')
        # Tab must not invoke the paste callback and wait for an end marker.
        readline.parse_and_bind(r'bind "^I" ed-insert')
        try:
            sys.stdout.write("\x1b[?2004h")
            sys.stdout.flush()
            line = input(prompt)
            if failures:
                raise failures[0]
            expanded = line
            for marker, content in pastes.items():
                expanded = expanded.replace(marker, content)
            return expanded
        finally:
            sys.stdout.write("\x1b[?2004l")
            sys.stdout.flush()
            readline.set_completer(previous)
            readline.parse_and_bind(r'bind -r "^[[200~"')
            readline.parse_and_bind(r'bind "^I" rl_complete')

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

    def _switch_model(self, text: str) -> None:
        """Switch to a `provider/model` reference, replacing the backend.

        The transcript is untouched: only the endpoint, key, wire format and
        model change, and the next request picks them up.

        Args:
            text: A model reference; the provider half names a preset and must
                be there, since nothing says where a bare model goes.
        """
        # Resolved again rather than patched: the new provider brings its own
        # endpoint, wire format and key variable, and `Settings` is where those
        # are read. An unknown provider raises inside `make_provider`.
        fresh = Settings.load(model=text)
        try:
            provider = fresh.make_provider()
        except (
            MissingCredentialsError,
            MissingBaseURLError,
            UnknownProviderError,
        ) as exc:
            print(self.theme.error(str(exc)))
            return
        # The wire format is baked into the backend class, so the backend is
        # replaced. The reader carries over, so a test's stand-in survives.
        self.session.backend = make_backend(provider, self.session.backend.reader)
        self.settings = fresh

    def _print_config(self) -> None:
        """Print the settings in effect, reading live values where they can change.

        The provider, model and the two toggles can be switched mid-session, so
        they come from the session and this object rather than from the
        `Settings` that started the run. The API key is reported as set or
        unset, never printed.
        """
        provider = self.session.backend.provider
        compaction = self.session.compaction
        if compaction.enabled:
            trigger = f"on, {compaction.threshold_tokens:,} tokens"
        else:
            trigger = "off"
        rows = [
            ("provider", self.settings.provider or "custom"),
            ("model", provider.model),
            ("base_url", provider.base_url or "unset"),
            ("api", provider.api),
            ("api key", "set (redacted)" if provider.api_key else "none set"),
            ("pass reasoning", _reasoning_mode(provider.pass_reasoning_content)),
            ("markdown", "on" if self.markdown else "off"),
            ("thinking", "shown" if self.show_thinking else "hidden"),
            ("sandbox", "container" if sandbox.in_container() else "host"),
            ("compaction", trigger),
            ("keep turns", str(compaction.keep_turns)),
            ("summary cap", f"{compaction.max_summary_input_tokens:,} tokens"),
        ]
        width = max(len(label) for label, _ in rows)
        for label, value in rows:
            print(f"{self.theme.dim(label.ljust(width) + ':')} {value}")

    def _command(self, line: str) -> bool:
        """Handle a slash command. Returns True to exit."""
        parts = line.split(maxsplit=1)
        command, arg = parts[0].lower(), (parts[1].strip() if len(parts) > 1 else "")
        theme = self.theme

        if command in ("/quit", "/exit"):
            return True
        if command == "/help":
            print(HELP, end="")
            if self.commands:
                print(theme.dim("custom commands:"))
            for name, custom in sorted(self.commands.items()):
                label = theme.bold("/" + name)
                print(f"  {label}{theme.dim(': ' + custom.description)}")
        elif command == "/model":
            if arg:
                self._switch_model(arg)
            print(f"{theme.dim('model:')} {theme.model(self.settings.model)}")
        elif command == "/tools":
            for tool in self.session.tools.values():
                first = tool.description.split(". ")[0]
                print(f"  {theme.bold(tool.name)}{theme.dim(': ' + first)}")
        elif command == "/system":
            print(self.session.system_prompt)
        elif command == "/config":
            self._print_config()
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
            width = terminal_width()
            directory = self.session.session_manager.directory
            for i, path in enumerate(SessionManager.list_sessions(directory), 1):
                marker = "→" if path == self.session.session_manager.file else " "
                stamp = datetime.fromtimestamp(path.stat().st_mtime)
                when = stamp.strftime("%Y-%m-%d %H:%M")
                # Measure the row unstyled: styles add no width, and the synopsis
                # only gets what the terminal has left, so it cannot wrap onto a
                # second line. SYNOPSIS_CHARS still caps it on a wide window.
                head = f"{marker} {i}. {path.stem}  {when}  "
                room = min(SYNOPSIS_CHARS, width - len(head))
                synopsis = SessionManager.synopsis(path, chars=room)
                styled_head = (
                    f"{theme.accent(marker)} {i}. {path.stem}  {theme.dim(when)}  "
                )
                print(f"{styled_head}{theme.bold(synopsis)}")
        elif command == "/resume":
            if not arg.isdigit():
                print("usage: /resume <n>")
                return False
            manager_directory = self.session.session_manager.directory
            sessions = SessionManager.list_sessions(manager_directory)
            index = int(arg) - 1
            if not 0 <= index < len(sessions):
                print(theme.error("no such session"))
                return False
            self.session.session_manager.load(sessions[index])
            print(theme.dim(f"resumed {sessions[index]}"))
        elif command == "/compact":
            if not self.session.compact():
                print(theme.dim("nothing to compact (too few turns, or disabled)"))
        elif custom := self.commands.get(command.removeprefix("/")):
            self._submit(custom.expand(arg))
        else:
            print(
                theme.error(f"unknown command: {command}") + theme.dim(" (try /help)")
            )
        return False


def print_mode(
    args: argparse.Namespace, provider: Provider, sessions_dir: Path
) -> None:
    """Run one prompt, write the final response to stdout.

    Deliberately plain: no styling, no tool chatter, no spinners. Failures go
    to stderr with a non-zero exit, so `ly -p` is usable in scripts.
    """
    # A script cannot see the sandbox from its output, and mistakes here are
    # silent, so say which one ran. stderr keeps stdout pipeable.
    if sandbox.in_container():
        print(f"ly: docker {sandbox.image_name()}", file=sys.stderr, flush=True)
    session = make_session(args, provider, sessions_dir)
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
        print(f"ly: {failures[-1]}", file=sys.stderr)
        sys.exit(1)


def build_parser(settings: Settings | None = None) -> argparse.ArgumentParser:
    """The command line.

    Provider options are defaults rather than flags, filled in by `Settings`,
    which reads the environment and the settings file and lets the environment
    win. That is also what the sandbox container sees, so the host and the
    container cannot disagree about which endpoint they are talking to.

    Args:
        settings: Preferences from the settings file and the environment,
            supplying the defaults for the sandbox mode and the image to
            start.
    """
    settings = settings or Settings()
    parser = argparse.ArgumentParser(
        prog="ly", description="A minimal coding agent, modeled after pi"
    )
    parser.set_defaults(
        image=settings.image or sandbox.DEFAULT_IMAGE,
        compaction=settings.compaction,
    )
    parser.add_argument("prompt", nargs="?", help="starting prompt (interactive mode)")
    parser.add_argument(
        "-p",
        "--print",
        action="store_true",
        help="print mode: run once, write final response",
    )
    parser.add_argument(
        "--resume", type=int, metavar="N", help="resume the n-th recent session"
    )
    parser.add_argument(
        "-ns",
        "--nosandbox",
        action="store_true",
        default=not settings.sandbox,
        help="run on the host instead of inside a Docker container",
    )
    parser.add_argument(
        "-s",
        "--sandbox",
        dest="image",
        metavar="IMAGE",
        help=f"Docker image to run ly in (default: {sandbox.DEFAULT_IMAGE})",
    )
    parser.add_argument("--version", action="version", version=f"ly {__version__}")
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Read once, here: below this point the values travel as arguments, and the
    # container only sees them through the mounts and flags.
    settings = Settings.load()
    # Parse here rather than in the container: `--help`, `--version` and a bad
    # argument all exit during parsing, so none of them starts a container.
    args = build_parser(settings).parse_args(argv)
    try:
        # Before the container: a stale or misspelled model should fail once,
        # where it was set, rather than inside a container that then exits.
        provider = settings.make_provider()
    except (MissingCredentialsError, MissingBaseURLError, UnknownProviderError) as exc:
        sys.exit(str(exc))
    if use_container(args):
        # The container rebuilds its settings from what it inherits, so a key
        # kept under the provider's own name has to travel with it.
        sandbox.exec_in_docker(
            argv,
            sessions_dir=settings.sessions_dir,
            image=args.image,
            extra_env=settings.key_variables,
        )

    try:
        if args.print:
            if not args.prompt:
                sys.exit("-p requires a prompt")
            print_mode(args, provider, settings.sessions_dir)
            return

        session = make_session(args, provider, settings.sessions_dir)
    except OSError as exc:
        # The sessions directory is the only thing startup writes. Nothing here
        # says why it failed: a refusal is the machine's state, not ly's.
        sys.exit(f"ly: cannot save sessions in {settings.sessions_dir}: {exc}")

    repl = Repl(session, settings=settings)
    if args.prompt:
        repl._submit(args.prompt)
    repl.run()


if __name__ == "__main__":
    main()
