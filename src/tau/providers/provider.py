"""Where to talk to a model: the configuration one backend reads.

`Provider` is a plain record — model, endpoint, credentials, timeout — with no
behaviour. A backend holds the provider and reads these fields on every
request, so `provider.model = ...` takes effect on the next one instead of
being shadowed by a copy taken inside the backend.
"""

from .errors import MissingBaseURLError, MissingCredentialsError
from .sse import DEFAULT_TIMEOUT

#: Used when the caller does not name a model.
DEFAULT_MODEL = "gpt-4.1"


class Provider:
    """Model choice and credentials; read by a backend on every request."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = "",
        api_key: str = "",
        timeout: float = DEFAULT_TIMEOUT,
        pass_reasoning_content: bool | None = None,
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
