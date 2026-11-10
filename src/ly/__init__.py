"""ly: a minimal Python coding agent harness, modeled after pi.

Exposes an SDK-style factory, mirroring pi's `createAgentSession()`:

    from ly import create_agent_session

    session = create_agent_session()
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
    api: str | None = None,
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
        sessions_dir: Where session JSONL files go. Left unset, a persistent
            session uses `DEFAULT_SESSIONS_DIR`; `persistent=False` writes
            nothing at all and keeps the transcript in memory.
    """
    from .prompts import build_system_prompt
    from .providers import Provider, make_backend
    from .providers.provider import DEFAULT_MODEL
    from .settings import DEFAULT_SESSIONS_DIR

    provider = Provider(
        model=model or DEFAULT_MODEL,
        base_url=base_url or "",
        api_key=api_key or "",
        api=api or "chat",
    )
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
