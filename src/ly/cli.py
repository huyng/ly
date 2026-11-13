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
import select
import subprocess
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
    MissingModelError,
    UnknownProviderError,
)
from .render import (
    GutterWriter,
    MarkdownRenderer,
    render_tool_result,
    summarize_tool_call,
)
from .session import SYNOPSIS_CHARS, SessionManager
from .settings import DEFAULT_PATH, Settings
from .spinner import Spinner
from .theme import BAND, DIM, RESET, SHADE, Theme, get_theme, terminal_width
from .tools import default_tools

#: Built-in slash commands, for Tab completion. Prose lives in `HELP`; keep
#: the two in step when adding one.
COMMANDS = (
    "/help",
    "/model",
    "/tools",
    "/system",
    "/config",
    "/plain",
    "/thinking",
    "/new",
    "/sessions",
    "/resume",
    "/compact",
    "/quit",
    "/exit",
)

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
!<command>         Run a shell command here, outside the conversation
/quit, /exit       Exit (also Ctrl-D)
"""


def use_container(args: argparse.Namespace) -> bool:
    """Return True when this process should hand off to `docker run` first."""
    # LY_SANDBOX is not read here: it reached us through `Settings`, which turned
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
        # References this session has used, newest last. The presets list the
        # models they know, but any reference is legal, so what was typed
        # before stays worth offering. Nothing is typed yet when no model is
        # configured, so the list starts empty rather than holding None.
        self._models = [preferences.model] if preferences.model else []

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
            f"{theme.model(self.settings.model or 'no model')}"
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
            if line.startswith("!"):
                self._shell(line[1:].strip())
                continue

            # Pick up a resize made while the prompt was being typed; the next
            # turn refreshes it again in _on_message_start.
            self.width = terminal_width()
            print()  # space the block off from what the user just typed
            self._print_prompt(line)
            self._submit(line)

    def _shell(self, command: str) -> None:
        """Run `command` as the user would in their own shell.

        This is the user's keystroke, not a tool call: the child inherits the
        terminal and the environment, so interactive commands work and nothing
        is captured, truncated or handed to the model. Nothing is scrubbed
        either — `tools/bash.py` hides credentials only because its output
        enters the model's context, which this never does.

        Args:
            command: The text after `!`, already stripped; empty does nothing.
        """
        if not command:
            return
        try:
            subprocess.run(command, shell=True, cwd=self.session.cwd)
        except KeyboardInterrupt:
            pass  # the child took the signal; the REPL must survive it
        except OSError as exc:
            print(self.theme.error(str(exc)))
        print()

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

    def _command_names(self) -> list[str]:
        """Every slash command Tab can offer, built in and from Markdown."""
        return sorted(set(COMMANDS) | {f"/{name}" for name in self.commands})

    def _completions(self, line: str) -> list[str]:
        """Full-line candidates for what has been typed so far.

        A line is completed in two places: the command name itself, and the
        argument of `/model`. Nothing else has an argument worth guessing, so
        a space elsewhere ends completion rather than offering a filename.

        Args:
            line: The text typed so far, as readline reports it.

        Returns:
            Whole lines that begin with `line`, so the caller only has to drop
            the part already typed. Empty when nothing can be completed.
        """
        if not line.startswith("/"):
            return []
        command, space, argument = line.partition(" ")
        if not space:
            return [name for name in self._command_names() if name.startswith(line)]
        if command != "/model" or len(argument.split()) > 1:
            return []
        name, slash, _ = argument.partition("/")
        if not slash:
            # Before the slash, every provider is a candidate.
            options = [f"{name}/" for name in sorted(self.settings.presets)]
        else:
            # After it, the provider's own models, then anything else this
            # session has used, since the preset list is only a guide.
            preset = self.settings.presets.get(name)
            options = [f"{name}/{model}" for model in preset.models] if preset else []
        options += [model for model in self._models if model not in options]
        lines = [f"{command} {option}" for option in options]
        return [candidate for candidate in lines if candidate.startswith(line)]

    def _complete(self, text: str, state: int) -> str | None:
        """Readline's completion protocol: one candidate per call.

        A candidate replaces only the word being completed, and the two
        readlines disagree about where a word starts: GNU readline treats
        `deepseek/deepseek-r` as one word, libedit breaks on the slash and
        completes just `deepseek-r`. The line up to the word is kept and only
        the rest of the candidate is offered, so nothing is doubled.
        """
        line = readline.get_line_buffer()
        start = len(line) - len(text) if line.endswith(text) else len(line)
        options = [option[start:] for option in self._completions(line)]
        return options[state] if state < len(options) else None

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
                # Enabling bracketed paste leaves Tab inserting a literal tab
                # on this readline, so completion needs the binding back.
                readline.parse_and_bind("tab: complete")
                # Set for the read only, so importing ly never takes Tab away
                # from whatever line editor the embedding program installed.
                previous = readline.get_completer()
                readline.set_completer(self._complete)
                try:
                    return input(prompt)
                finally:
                    readline.set_completer(previous)
        except EOFError:
            return None

    def _paste_pending(self) -> bool:
        """Whether a paste, rather than a key press, reached the hook.

        Used on libedit, where Tab and the bracketed-paste sequence share one
        completion entry point: the paste arrives with its introducer still
        buffered, and a key press arrives with nothing waiting.
        """
        return bool(select.select([sys.stdin], [], [], 0)[0])

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
            """Capture the bytes after the introducer and return a placeholder."""
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
                return marker
            except BaseException as exc:
                # Readline swallows callback exceptions; propagate after input.
                failures.append(exc)
            return None

        inserted = False

        def hook(text: str, state: int) -> str | None:
            """Send Tab to completion and a paste to the placeholder hook.

            Libedit has one entry point for both, and the paste hook blocks on
            stdin until it sees the closing marker. Which one arrived is
            decided by whether bytes are already waiting: a Tab leaves the
            input empty, while a paste arrives with its introducer still
            buffered.
            """
            nonlocal inserted
            if self._paste_pending():
                # A paste answers only the first request; libedit asks again
                # for more candidates, and the placeholder is the whole answer.
                if inserted or state:
                    return None
                marker = paste(text, state)
                inserted = marker is not None
                return None if marker is None else text + marker
            if state == 0:
                inserted = False
            return None if inserted else self._complete(text, state)

        previous = readline.get_completer()
        readline.set_completer(hook)
        readline.parse_and_bind(r'bind "^[[200~" rl_complete')
        # Libedit does not bind Tab to completion on its own, unlike GNU
        # readline, so Tab reaches no completer until this is bound.
        readline.parse_and_bind(r'bind "^I" rl_complete')
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
            parts = [self.session.backend.provider.model]
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
            if self.session.cwd:
                # A resumed session can be in another directory than the shell
                # that started it. Inside a container the mount point says
                # nothing about where the user is, so name the host directory.
                host = sandbox.host_dir() if sandbox.in_container() else None
                parts.append(os.path.expanduser(host or self.session.cwd))
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
        # are read. An unknown provider raises inside `make_provider`. The same
        # file is read again, so a --config run keeps its preferences.
        fresh = Settings.load(path=self.settings.path, model=text)
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
        backend = make_backend(provider, self.session.backend.reader)
        # Compaction is sized by the new model's context window, so the old
        # numbers would compact at the wrong point: too late for a smaller one.
        self.session.set_model(backend, fresh.compaction)
        self.settings = fresh
        if text not in self._models:
            self._models.append(text)

    def _print_config(self) -> None:
        """Print the name and value of every setting in effect, one per line.

        Values are read live where they can change mid-session: the model, the
        provider it points at, and the two toggles. Numbers are printed as
        stored rather than formatted, and the API key is reported as set or
        unset, never printed.
        """

        def flag(value: bool) -> str:
            return "true" if value else "false"

        provider = self.session.backend.provider
        compaction = self.session.compaction
        window = self.settings.context_window
        rows = [
            # Both halves, as `/model` writes them: a bare model name is one
            # provider's catalogue and says nothing about the endpoint.
            ("model", f"{self.settings.provider or 'custom'}/{provider.model}"),
            ("context_window", str(window) if window else "unknown"),
            ("base_url", provider.base_url or "unset"),
            ("api_format", provider.api_format),
            ("api_key", "set" if provider.api_key else "unset"),
            ("pass_reasoning_content", flag(provider.pass_reasoning_content)),
            ("markdown", flag(self.markdown)),
            ("thinking", flag(self.show_thinking)),
            ("sandbox", flag(sandbox.in_container())),
            ("compaction.enabled", flag(compaction.enabled)),
            ("compaction.threshold_tokens", str(compaction.threshold_tokens)),
            ("compaction.keep_turns", str(compaction.keep_turns)),
            (
                "compaction.max_summary_input_tokens",
                str(compaction.max_summary_input_tokens),
            ),
        ]
        # Padded, so the values line up in one column, but plain text: no
        # colons and no colour, so the output stays copy-pasteable.
        width = max(len(name) for name, _ in rows)
        for name, value in rows:
            print(f"{name.ljust(width)} {value}")

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
            print(
                f"{theme.dim('model:')} {theme.model(self.settings.model or 'unset')}"
            )
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
    parser.add_argument(
        "--config",
        metavar="PATH",
        help=f"settings file to read (default: {DEFAULT_PATH})",
    )
    return parser


def _config_path(argv: list[str]) -> str | None:
    """Read `--config` before the rest of the command line is parsed.

    The parser's own defaults come from the settings, so the file has to be
    chosen first. Every other flag is left to the real parser.

    Args:
        argv: The command line, without the program name.

    Returns:
        The path given, or None when the flag is absent.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config")
    known, _ = parser.parse_known_args(argv)
    return known.config


def _without_config(argv: list[str]) -> list[str]:
    """Return the command line with `--config` and its value removed.

    The container reads the file where the host mounted it, at the container's
    own default path, so a host path forwarded inside would name nothing.

    Args:
        argv: The command line, without the program name.

    Returns:
        The arguments to forward to the container-side ly.
    """
    kept: list[str] = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
        elif arg == "--config":
            skip = True
        elif not arg.startswith("--config="):
            kept.append(arg)
    return kept


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Read once, here: below this point the values travel as arguments, and the
    # container only sees them through the mounts and flags.
    settings = Settings.load(path=_config_path(argv))
    # Parse here rather than in the container: `--help`, `--version` and a bad
    # argument all exit during parsing, so none of them starts a container.
    args = build_parser(settings).parse_args(argv)
    try:
        # Before the container: a stale or misspelled model should fail once,
        # where it was set, rather than inside a container that then exits.
        provider = settings.make_provider()
    except (
        MissingCredentialsError,
        MissingModelError,
        MissingBaseURLError,
        UnknownProviderError,
    ) as exc:
        sys.exit(str(exc))
    if use_container(args):
        # The container rebuilds its settings from what it inherits, so a key
        # kept under the provider's own name has to travel with it.
        sandbox.exec_in_docker(
            _without_config(argv),
            sessions_dir=settings.sessions_dir,
            image=args.image,
            settings_file=settings.path,
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
