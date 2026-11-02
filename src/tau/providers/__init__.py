"""The model backends, and the provider that drives them.

Both are OpenAI-flavoured endpoints that stream SSE, so neither is a special
case of the other:

- ``backend.ChatCompletionsBackend`` — ``/chat/completions``, the default and the
  only flavour DeepSeek, z.ai and friends speak.
- ``backend.ResponsesBackend`` — ``/responses``, required by OpenAI's newest
  models, which refuse function tools on chat-completions.

Supporting modules, from the wire upward: ``sse`` (one POST, decoded chunks),
``backend`` (the transport and every backend that speaks it), ``provider``
(config, and the choice of backend), ``errors`` (failures). Import anything
else from those modules directly: this package exposes only ``Provider`` and
``ProviderError``.
"""

from .errors import ProviderError
from .provider import Provider

__all__ = ["Provider", "ProviderError"]
