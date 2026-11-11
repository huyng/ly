"""Where to talk to a model: the configuration one backend reads.

`Provider` is a plain record — model, endpoint, credentials, wire format,
timeout — with no behaviour. A backend holds the provider and reads these
fields on every request, so `provider.model = ...` takes effect on the next one
instead of being shadowed by a copy taken inside the backend.
"""

from .errors import MissingBaseURLError, MissingCredentialsError


class Provider:
    """Model choice and credentials; read by a backend on every request."""

    def __init__(
        self,
        model: str,
        base_url: str = "",
        api_key: str = "",
        api_format: str = "openai-completions",
        pass_reasoning_content: bool = False,
    ):
        if not api_key:
            # Raised here, at construction, so callers fail before any work.
            raise MissingCredentialsError()

        if not base_url:
            raise MissingBaseURLError()

        self.model = model
        self.base_url = base_url
        # "openai-completions" for `/chat/completions`, "openai-responses" for
        # `/responses`; the latter is the only route to tools on OpenAI's newest
        # models.
        self.api_format = (api_format or "openai-completions").strip().lower()
        self.api_key = api_key
        # Which endpoint needs `reasoning_content` echoed back is a fact about
        # the provider, so it arrives here resolved rather than guessed from the
        # URL: a preset pointed at a proxy still speaks its own dialect.
        self.pass_reasoning_content = pass_reasoning_content
