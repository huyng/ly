"""What a model backend is: the interface the agent streams through, and the
endpoint configuration both implementations inherit.

`ModelBackend` is the whole contract — one `stream()` call per turn. `Endpoint`
is the part every OpenAI-flavoured backend shares: model, base URL, key,
timeout, headers, and the single POST that `sse` performs.
"""

import threading
from collections.abc import Generator
from typing import Protocol

from ..events import ProviderEvent
from ..messages import AgentMessage, AssistantMessage
from ..tools import Tool
from .errors import MissingBaseURLError, MissingCredentialsError
from .sse import DEFAULT_TIMEOUT, Chunk, Transport, stream_chunk_transport

#: Used when the caller does not name a model.
DEFAULT_MODEL = "gpt-4.1"


class ModelBackend(Protocol):
    """What the agent needs from a model runtime; the backends implement it.

    Tests and other hosts can satisfy it with a stand-in instead of a real
    transport. ``stream`` yields events and returns the completed message via
    StopIteration.value. Endpoint failures raise `ProviderError` with explicit
    retryability; unexpected implementation errors propagate without retries.
    """

    api: str
    model: str

    def stream(
        self,
        system_prompt: str,
        messages: list[AgentMessage],
        tools: list[Tool],
        abort_event: threading.Event | None = None,
    ) -> Generator[ProviderEvent, None, AssistantMessage]: ...


class Endpoint:
    """Configuration and headers for one OpenAI-flavoured endpoint.

    Every value comes from the caller; nothing here reads the environment. The
    CLI is responsible for turning `TAU_*` variables into these arguments.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = "",
        api_key: str = "",
        reader: Transport | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        pass_reasoning_content: bool | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.reader = reader or stream_chunk_transport
        self.timeout = timeout

        if not self.api_key:
            # Raised here, at construction, so callers fail before any work.
            raise MissingCredentialsError()

        if not self.base_url:
            raise MissingBaseURLError()

        # DeepSeek thinking mode requires reasoning_content to be echoed back
        # when tools are present. Other providers (e.g. z.ai) keep the previous
        # behavior unless the caller opts in.
        if pass_reasoning_content is None:
            pass_reasoning_content = "deepseek" in self.base_url
        self.pass_reasoning_content = pass_reasoning_content

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Accept-Encoding": "identity",
        }

    def _post(self, path: str, body: bytes) -> Generator[Chunk, None, None]:
        """POST `body` under the base URL and stream the decoded chunks.

        Closing the returned stream closes the connection underneath it, so
        abandoning a turn drops the response instead of draining it.
        """
        chunks = self.reader(
            self.base_url.rstrip("/") + path,
            self._headers(),
            body,
            timeout=self.timeout,
        )
        try:
            yield from chunks
        finally:
            close = getattr(chunks, "close", None)
            if callable(close):
                close()
