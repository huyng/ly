"""The model backends, and the provider that drives them.

Both are OpenAI-flavoured endpoints that stream SSE, so neither is a special
case of the other:

- ``backend.ChatCompletionsBackend`` — ``/chat/completions``, the default and the
  only flavour DeepSeek, z.ai and friends speak.
- ``backend.ResponsesBackend`` — ``/responses``, required by OpenAI's newest
  models, which refuse function tools on chat-completions.

Supporting modules, from the wire upward: ``sse`` (one POST, decoded chunks),
``backend`` (the transport and every backend that speaks it), ``provider``
(config, and the choice of backend), ``errors`` (failures).
"""

from .backend import ChatCompletionsBackend, ModelBackend, ResponsesBackend
from .errors import MissingBaseURLError, MissingCredentialsError, ProviderError
from .provider import DEFAULT_MODEL, Provider
from .sse import stream_chunk_transport

__all__ = [
    "ChatCompletionsBackend",
    "DEFAULT_MODEL",
    "MissingBaseURLError",
    "MissingCredentialsError",
    "ModelBackend",
    "Provider",
    "ProviderError",
    "ResponsesBackend",
    "stream_chunk_transport",
]
