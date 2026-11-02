"""The model backends, and the factory that picks one.

Both are OpenAI-flavoured endpoints that stream SSE, so neither is a special
case of the other:

- ``chat.ChatCompletionsBackend`` — ``/chat/completions``, the default and the only
  flavour DeepSeek, z.ai and friends speak.
- ``responses.ResponsesBackend`` — ``/responses``, required by OpenAI's newest
  models, which refuse function tools on chat-completions.

Supporting modules, from the wire upward: ``sse`` (one POST, decoded chunks),
``endpoint`` (the shared interface and configuration), ``errors`` (failures).
"""

from typing import Any

from .chat import ChatCompletionsBackend
from .endpoint import Endpoint, ModelBackend
from .errors import MissingBaseURLError, MissingCredentialsError, ProviderError
from .responses import ResponsesBackend
from .sse import stream_chunk_transport

__all__ = [
    "ChatCompletionsBackend",
    "Endpoint",
    "MissingBaseURLError",
    "MissingCredentialsError",
    "ModelBackend",
    "ProviderError",
    "ResponsesBackend",
    "make_provider",
    "stream_chunk_transport",
]


def make_provider(api: str | None = None, **kwargs: Any) -> ModelBackend:
    """Build the backend named by `api`: chat-completions by default.

    Only OpenAI's newest models need `responses`, and no other endpoint speaks
    it, so it never becomes the default.
    """
    if (api or "chat").strip().lower() == "responses":
        return ResponsesBackend(**kwargs)
    return ChatCompletionsBackend(**kwargs)
