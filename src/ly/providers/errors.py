"""The failures a model endpoint can raise, and which of them are worth retrying.

Nothing here knows about HTTP or SSE: the reader classifies its own errors
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

    def __init__(self, variables: tuple[str, ...] = ()):
        # The provider's own variable comes first, then the generic one, which
        # wins when both are set. Naming both is the point: a key under only
        # one of them is enough to start.
        names = [*dict.fromkeys(variables), "LY_API_KEY"]
        super().__init__(
            f"Missing credentials. Set {' or '.join(names)}.", retryable=False
        )


class MissingModelError(ProviderError):
    """No model was configured at all.

    Raised before an endpoint is resolved, so the message names the ways to
    supply a reference rather than complaining about one that was never
    written.
    """

    def __init__(self):
        super().__init__(
            'No model configured. Set LY_MODEL, or "model" in '
            "~/.ly/settings.json (or the file --config names), as "
            "provider/model.",
            retryable=False,
        )


class MissingBaseURLError(ProviderError):
    def __init__(self):
        super().__init__("Missing base URL. Set LY_BASE_URL.", retryable=False)


class UnknownProviderError(ProviderError):
    """The model reference names no provider ly knows.

    Raised where a model is resolved, so the message must say what to write
    instead. A model with no provider in front of it lands here too.
    """

    def __init__(self, model: str, known: str):
        super().__init__(
            f"Model {model!r} names no known provider. "
            f"Write provider/model, such as one of: {known}",
            retryable=False,
        )
