"""The failures a model endpoint can raise, and which of them are worth retrying.

Nothing here knows about HTTP or SSE: the transport classifies its own errors
using these types, and the agent loop decides how many times to try again.
"""

from ..messages import Usage


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
            "Missing credentials. Set TAU_API_KEY or pass --api-key.", retryable=False
        )


class MissingBaseURLError(ProviderError):
    def __init__(self):
        super().__init__("Missing Base URL for api. Set TAU_BASE_URL", retryable=False)
