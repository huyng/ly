"""Where and how to talk to a model: config, plus the backend it drives.

`Provider` is the only place configuration lives. It builds the backend named
by `api` and forwards `stream` to it, passing the configuration along on every
call — so `provider.model = ...` takes effect on the next request instead of
being shadowed by a copy inside the backend.
"""

import threading
from collections.abc import Generator

from ..events import ProviderEvent
from ..messages import AgentMessage, AssistantMessage
from ..tools import Tool
from .backend import ChatCompletionsBackend, ModelBackend, ResponsesBackend
from .errors import MissingBaseURLError, MissingCredentialsError
from .sse import DEFAULT_TIMEOUT, Transport

#: Used when the caller does not name a model.
DEFAULT_MODEL = "gpt-4.1"


class Provider:
    """Model choice and credentials; owns the backend that speaks to the API."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = "",
        api_key: str = "",
        timeout: float = DEFAULT_TIMEOUT,
        pass_reasoning_content: bool | None = None,
        api: str | None = None,
        reader: Transport | None = None,
    ):
        if not api_key:
            # Raised here, at construction, so callers fail before any work.
            raise MissingCredentialsError()

        if not base_url:
            raise MissingBaseURLError()

        # DeepSeek thinking mode requires reasoning_content to be echoed back
        # when tools are present. Other providers (e.g. z.ai) keep the previous
        # behavior unless the caller opts in.
        if pass_reasoning_content is None:
            pass_reasoning_content = "deepseek" in base_url

        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.pass_reasoning_content = pass_reasoning_content

        if (api or "chat").strip().lower() == "responses":
            self.backend: ModelBackend = ResponsesBackend(reader)
        else:
            self.backend = ChatCompletionsBackend(reader)

    def stream(
        self,
        system_prompt: str,
        messages: list[AgentMessage],
        tools: list[Tool],
        abort_event: threading.Event | None = None,
    ) -> Generator[ProviderEvent, None, AssistantMessage]:
        """Stream one turn from the configured backend."""
        return self.backend.stream(
            system_prompt,
            messages,
            tools,
            abort_event,
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            pass_reasoning_content=bool(self.pass_reasoning_content),
        )
