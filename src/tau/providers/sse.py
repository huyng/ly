"""The wire layer: one HTTP POST, decoded into a stream of JSON chunks.

Everything here is about bytes and status codes, not about which API is being
spoken. The backends build a body and read the chunks; this module is the only
place that knows the request is urllib and the response is server-sent events.
"""

import http.client
import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from contextlib import closing
from typing import Any, Generator, Protocol

from .errors import ProviderError

#: A parsed `data:` payload from the event stream.
Chunk = Mapping[str, Any]

#: Status codes worth retrying: rate limits and server-side failures.
RETRYABLE_STATUSES = frozenset({408, 409, 425, 429})
DEFAULT_TIMEOUT = 600.0


class StreamReader:
    """Reads the response to one POST as a stream of decoded chunks.

    `read_chunks` is a generator, so closing it drops the connection instead of
    draining the rest of the response.
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str],
        body: bytes,
        timeout: float | None = None,
    ):
        self.url = url
        self.headers = headers
        self.body = body
        self.timeout = timeout if timeout else DEFAULT_TIMEOUT

    def read_chunks(self) -> Generator[Chunk, None, None]:
        """POST `body` and yield each SSE event's JSON payload.

        Lines that are not `data:` (blank separators, `:` heartbeat comments,
        `event:` fields) are skipped, as is the `[DONE]` terminator. Multiple
        `data:` lines would form one event per the SSE spec, but OpenAI-style
        APIs send exactly one JSON object per event, so each line is decoded
        alone.

        The response object is closed when the caller stops iterating, so
        abandoning the stream (Ctrl-C, abort) drops the connection instead of
        draining it.
        """

        request = urllib.request.Request(
            self.url, data=self.body, headers=self.headers, method="POST"
        )
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as err:
            err_text = _describe_http_error(err)
            raise ProviderError(
                err_text, retryable=_is_retryable_status(err.code)
            ) from err
        except urllib.error.URLError as err:
            # DNS failure, refused connection, TLS problem, read timeout.
            raise ProviderError(f"cannot reach {self.url}: {err.reason}") from err

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
                    except (
                        json.JSONDecodeError
                    ) as err:  # never lose a turn to a stray line
                        raise ProviderError(
                            f"malformed stream event: {payload[:200]}"
                        ) from err
                    if isinstance(chunk, dict):
                        yield chunk

        except (http.client.HTTPException, OSError) as err:
            # A connection dropped or timed out mid-stream: the turn is incomplete,
            # and a fresh request is the remedy. ProviderErrors raised above pass
            # through untouched.
            raise ProviderError(f"stream interrupted: {err!r}") from err


def _describe_http_error(err: urllib.error.HTTPError) -> str:
    """Fold an HTTP error into one readable line, preferring the API's message."""
    detail = ""
    try:
        body = err.read().decode("utf-8", errors="replace")
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
    return f"HTTP {err.code} {err.reason}" + (f": {detail}" if detail else "")


def _is_retryable_status(code: int | None) -> bool:
    """429 and 5xx are transient; other 4xx are the caller's fault."""
    if code is None:
        return True
    return code in RETRYABLE_STATUSES or 500 <= code <= 599
