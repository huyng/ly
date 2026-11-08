"""ly: a minimal Python coding agent harness, modeled after pi.

Exposes an SDK-style factory, mirroring pi's `createAgentSession()`:

    from ly import create_agent_session

    session = create_agent_session()
    session.prompt("What files are in the current directory?")
    print(session.last_assistant_text())
    session.dispose()
"""

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
) -> AgentSession:
    """Create an agent session with sensible defaults.

    Mirrors pi's SDK factory: each boundary (model runtime, tools, session
    persistence, system prompt) can be supplied explicitly, or defaults are
    used.
    """
    from .prompts import build_system_prompt
    from .providers import Provider, make_backend
    from .providers.provider import DEFAULT_MODEL

    provider = Provider(
        model=model or DEFAULT_MODEL,
        base_url=base_url or "",
        api_key=api_key or "",
    )
    in_memory = not persistent
    session_manager = SessionManager(in_memory=in_memory)
    return AgentSession(
        backend=make_backend(provider, api),
        tools=tools if tools is not None else default_tools(cwd=cwd),
        session_manager=session_manager,
        system_prompt=build_system_prompt(cwd),
        cwd=cwd,
        tool_guard=tool_guard,
    )
