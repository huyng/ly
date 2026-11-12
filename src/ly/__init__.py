"""ly: a minimal Python coding agent harness, modeled after pi.

Exposes an SDK-style factory, mirroring pi's `createAgentSession()`:

    from ly import create_agent_session

    session = create_agent_session(model="openai/gpt-4.1")
    session.prompt("What files are in the current directory?")
    print(session.last_assistant_text())
    session.dispose()
"""

from pathlib import Path

from .agent import AgentSession
from .session import SessionManager
from .tools import Tool, default_tools

__version__ = "0.1.0"

__all__ = ["AgentSession", "SessionManager", "create_agent_session"]


def create_agent_session(
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    api_format: str | None = None,
    cwd: str | None = None,
    tools: list[Tool] | None = None,
    persistent: bool = True,
    tool_guard=None,
    sessions_dir: str | Path | None = None,
) -> AgentSession:
    """Create an agent session with sensible defaults.

    Mirrors pi's SDK factory: each boundary (model runtime, tools, session
    persistence, system prompt) can be supplied explicitly, or defaults are
    used.

    Args:
        model: A `provider/model` reference, such as `deepseek/deepseek-chat`.
            The provider half names a preset, which supplies the endpoint.
        sessions_dir: Where session JSONL files go. Left unset, a persistent
            session uses `DEFAULT_SESSIONS_DIR`; `persistent=False` writes
            nothing at all and keeps the transcript in memory.
    """
    from .prompts import build_system_prompt
    from .providers import make_backend
    from .settings import DEFAULT_SESSIONS_DIR, Settings

    # One resolution path: the reference names its endpoint, wire format and
    # key variable, so nothing here restates a default the CLI would not use.
    # A caller that passes no model gets `MissingModelError` from
    # `make_provider`, exactly as the CLI would, rather than a model nobody
    # asked for.
    settings = Settings.load(
        env={},
        model=model,
        base_url=base_url,
        api_key=api_key,
        api_format=api_format,
    )
    provider = settings.make_provider()
    # The SDK takes its configuration as arguments, so the default lives here
    # rather than in `SessionManager`, which holds no path of its own. Only the
    # CLI reads `LY_SESSIONS_DIR`, through `Settings`.
    directory = None
    if persistent:
        directory = Path(sessions_dir or DEFAULT_SESSIONS_DIR).expanduser()
    session_manager = SessionManager(directory)
    return AgentSession(
        backend=make_backend(provider),
        tools=tools if tools is not None else default_tools(cwd=cwd),
        session_manager=session_manager,
        system_prompt=build_system_prompt(cwd),
        cwd=cwd,
        tool_guard=tool_guard,
    )
