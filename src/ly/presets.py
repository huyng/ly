"""Named providers: the endpoint facts a model name alone cannot carry.

Every OpenAI-compatible service differs only in where it lives, which wire
format it speaks, and which extra field it needs back. This module is that
table. It reads nothing and holds no state: `settings.Settings` resolves a
model reference against it and fills in whatever the user did not set.

A model is always written `provider/model`, so no request can be sent without
naming where it goes. A preset may name the variables that hold its key, so
switching providers does not mean re-exporting one generic `LY_API_KEY`.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Model:
    """What ly knows about one model name.

    Args:
        context_window: Tokens the model accepts, which sizes compaction. None
            when the name was listed without one, leaving the global
            compaction settings in charge.
        keep_turns: Recent turns to keep verbatim when this model compacts.
            None defers to the settings file and environment.
    """

    context_window: int | None = None
    keep_turns: int | None = None


@dataclass(frozen=True)
class Preset:
    """One provider's defaults.

    Args:
        base_url: Endpoint root, without the trailing `/chat/completions`.
        api_format: Wire format, `openai-completions` or `openai-responses`.
        key_env: Variables that may hold this provider's key, in order, after
            `LY_API_KEY`.
        pass_reasoning_content: Whether tool calls carry `reasoning_content`
            back to the endpoint, which DeepSeek needs and others reject.
        models: What the provider's models are called, offered by Tab after the
            slash. A guide, not a gate: any name is sent as typed.
    """

    base_url: str
    api_format: str = "openai-completions"
    key_env: tuple[str, ...] = ()
    #: Whether a tool call must echo `reasoning_content` back. DeepSeek requires
    #: it; anything else rejecting the field is why this is not always on.
    pass_reasoning_content: bool = False
    #: Written out by hand because no provider publishes a list ly can read
    #: without a key. `GET /models` with a key is where these come from; the
    #: window is from the provider's documentation.
    models: dict[str, Model] = field(default_factory=dict)


#: Providers ly knows without being told. A settings file may add more under
#: `providers`, or override one of these to point it at a proxy.
BUILT_IN: dict[str, Preset] = {
    "openai": Preset(
        base_url="https://api.openai.com/v1",
        key_env=("OPENAI_API_KEY",),
        models={
            "gpt-6.1-sol": Model(context_window=1_047_576),
            "gpt-6.1-terra": Model(context_window=1_047_576),
            "gpt-6-astra": Model(context_window=1_047_576),
        },
    ),
    "deepseek": Preset(
        base_url="https://api.deepseek.com",
        key_env=("DEEPSEEK_API_KEY",),
        pass_reasoning_content=True,
        models={
            "deepseek-flash": Model(context_window=1_047_576, keep_turns=4),
            "deepseek-v4-pro": Model(context_window=1_047_576, keep_turns=4),
        },
    ),
}


def split(reference: str) -> tuple[str, str]:
    """Split a `provider/model` reference into its two halves.

    Only the first slash separates, because model names may contain slashes of
    their own (`openai/moonshotai/kimi-k2`). A reference with no slash names no
    provider, and one of the halves is then empty.

    Args:
        reference: A model reference, such as `deepseek/deepseek-chat`.

    Returns:
        The provider name and the model name to send. The provider name comes
        back empty when the reference does not name one.
    """
    name, _, model = reference.partition("/")
    if not (name and model):
        return "", reference
    return name, model
