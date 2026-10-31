"""The model backends, and the factory that picks one.

Both are OpenAI-flavoured endpoints that stream SSE, so neither is a special
case of the other:

- ``chat.OpenAIProvider`` — ``/chat/completions``, the default and the only
  flavour DeepSeek, z.ai and friends speak.
- ``responses.ResponsesProvider`` — ``/responses``, required by OpenAI's newest
  models, which refuse function tools on chat-completions.
"""

import os
from typing import Any

from .base import (
    MissingCredentialsError,
    Provider,
    ProviderError,
    Transport,
    stream_chunk_transport,
)
from .chat import OpenAIProvider
from .responses import ResponsesProvider

__all__ = [
    "MissingCredentialsError",
    "OpenAIProvider",
    "Provider",
    "ProviderError",
    "ResponsesProvider",
    "Transport",
    "make_provider",
    "stream_chunk_transport",
]


def make_provider(api: str | None = None, **kwargs: Any) -> Provider:
    """Build the backend named by `api` (or TAU_API): chat-completions by default.

    Only OpenAI's newest models need `responses`, and no other endpoint speaks
    it, so it never becomes the default.
    """
    if (api or os.environ.get("TAU_API", "")).strip().lower() == "responses":
        return ResponsesProvider(**kwargs)
    return OpenAIProvider(**kwargs)
