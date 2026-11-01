"""Provider plumbing: the protocol, errors, configuration, and the SSE transport.

The backends differ in how they shape a request and how they read the stream,
so each lives in its own module. What they share is an endpoint's configuration
(model, base URL, key, timeout), its headers, and the wire-level POST that
turns a body into SSE chunks — that is `HTTPBackend` and
`stream_chunk_transport` here.
"""

import http.client
import json
import os
import threading
import urllib.error
import urllib.request
from collections.abc import Generator, Iterator, Mapping
from contextlib import closing
from typing import Any, Protocol

from ..events import ProviderEvent
from ..messages import AgentMessage, AssistantMessage, Usage
from ..tools import Tool

#: A parsed `data:` payload from the event stream.
Chunk = Mapping[str, Any]

DEFAULT_TIMEOUT = 600.0


class Transport(Protocol):
    """`(url, headers, body) -> iterator of decoded chunks`, in request order."""

    def __call__(
        self,
        url: str,
        headers: dict[str, str],
        body: bytes,
        timeout: float = ...,
    ) -> Iterator[Chunk]: ...


class ProviderError(RuntimeError):
    """A failure talking to the endpoint.

    ``retryable`` distinguishes "try again in a moment" (rate limits, 5xx,
    dropped connections) from "this will never succeed" (bad key, bad request,
    content filtered). Permanent failures end the turn without another request.
    ``usage`` preserves billed tokens when a streamed response ends in failure.
    """

    def __init__(
        self, message: str, *, retryable: bool = True, usage: Usage | None = None
    ):
        super().__init__(message)
        self.retryable = retryable
        self.usage = usage


class MissingCredentialsError(ProviderError):
    """No API key was configured for the endpoint.

    Built at construction time, so the message must say how to supply a key.
    """

    def __init__(self):
        super().__init__(
            "Missing credentials. Set TAU_API_KEY or pass --api-key.",
            retryable=False,
        )


#: Status codes worth retrying: rate limits and server-side failures.
RETRYABLE_STATUSES = frozenset({408, 409, 425, 429})


class ModelBackendProtocol(Protocol):
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


class HTTPBackend:
    """Configuration and headers for one OpenAI-flavoured endpoint.

    Configuration precedence: explicit arguments, then TAU_* environment
    variables, then defaults.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        transport: Transport | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.model = model or os.environ.get("TAU_MODEL", "gpt-4.1")
        self.api_key = api_key or os.environ.get("TAU_API_KEY")
        self.base_url = base_url or os.environ.get(
            "TAU_BASE_URL", "https://api.openai.com/v1"
        )
        self.transport = transport or stream_chunk_transport
        self.timeout = timeout

        if not self.api_key:
            # Raised here, at construction, so callers fail before any work.
            raise MissingCredentialsError()

        # DeepSeek thinking mode requires reasoning_content to be echoed back
        # when tools are present. Other providers (e.g. z.ai) should keep the
        # previous behavior unless explicitly opted in.
        self.pass_reasoning_content = "deepseek" in self.base_url
        override = os.environ.get("TAU_PASS_REASONING_CONTENT")
        if override is not None:
            self.pass_reasoning_content = override.strip().lower() not in (
                "0",
                "false",
                "no",
                "off",
            )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Accept-Encoding": "identity",
        }


def stream_chunk_transport(
    url: str,
    headers: dict[str, str],
    body: bytes,
    timeout: float = DEFAULT_TIMEOUT,
) -> Iterator[Chunk]:
    """POST `body` and yield each SSE event's JSON payload.

    Lines that are not `data:` (blank separators, `:` heartbeat comments,
    `event:` fields) are skipped, as is the `[DONE]` terminator. Multiple
    `data:` lines would form one event per the SSE spec, but OpenAI-style APIs
    send exactly one JSON object per event, so each line is decoded alone.

    The response object is closed when the caller stops iterating, so abandoning
    the stream (Ctrl-C, abort) drops the connection instead of draining it.
    """
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise ProviderError(
            _describe_http_error(exc), retryable=_is_retryable_status(exc.code)
        ) from exc
    except urllib.error.URLError as exc:
        # DNS failure, refused connection, TLS problem, read timeout.
        raise ProviderError(f"cannot reach {url}: {exc.reason}") from exc

    try:
        with closing(response):
            for raw in response:  # line-buffered by http.client; one SSE line each
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    return
                if not payload:
                    continue
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError as exc:  # never lose a turn to a stray line
                    raise ProviderError(
                        f"malformed stream event: {payload[:200]}"
                    ) from exc
                if isinstance(chunk, dict):
                    yield chunk
    except (http.client.HTTPException, OSError) as exc:
        # A connection dropped or timed out mid-stream: the turn is incomplete,
        # and a fresh request is the remedy. ProviderErrors raised above pass
        # through untouched.
        raise ProviderError(f"stream interrupted: {exc!r}") from exc


def _describe_http_error(exc: urllib.error.HTTPError) -> str:
    """Fold an HTTP error into one readable line, preferring the API's message."""
    detail = ""
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except (OSError, http.client.HTTPException, ValueError):
        # The body is a courtesy; never mask the status.
        body = ""
    if body:
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            error = parsed.get("error")
            if isinstance(error, dict):
                detail = str(error.get("message") or "")
            elif isinstance(error, str):
                detail = error
            elif parsed.get("message"):
                detail = str(parsed["message"])
        if not detail:
            detail = " ".join(body.split())[:300]
    return f"HTTP {exc.code} {exc.reason}" + (f": {detail}" if detail else "")


def _is_retryable_status(code: int | None) -> bool:
    """429 and 5xx are transient; other 4xx are the caller's fault."""
    if code is None:
        return True
    return code in RETRYABLE_STATUSES or 500 <= code <= 599
