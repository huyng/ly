"""Named providers: the endpoint facts a model name alone cannot carry.

Every OpenAI-compatible service differs only in where it lives, which wire
format it speaks, and which extra field it needs back. This module is that
table. It reads nothing and holds no state: `settings.Settings` resolves a
model reference against it and fills in whatever the user did not set.

A model is always written `provider/model`, so no request can be sent without
naming where it goes. A preset may name the variables that hold its key, so
switching providers does not mean re-exporting one generic `LY_API_KEY`.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    """One provider's defaults.

    Args:
        base_url: Endpoint root, without the trailing `/chat/completions`.
        api: Wire format, `chat` or `responses`.
        key_env: Variables that may hold this provider's key, in order, after
            `LY_API_KEY`.
        pass_reasoning_content: Whether tool calls carry `reasoning_content`
            back to the endpoint, which DeepSeek needs and others reject.
        models: Model names the provider offers, offered by Tab after the
            slash. A guide, not a gate: any name is sent as typed.
    """

    base_url: str
    api: str = "chat"
    key_env: tuple[str, ...] = ()
    #: Whether a tool call must echo `reasoning_content` back. DeepSeek requires
    #: it; anything else rejecting the field is why this is not always on.
    pass_reasoning_content: bool = False
    #: Written out by hand because no provider publishes a list ly can read
    #: without a key. `GET /models` with a key is where these come from.
    models: tuple[str, ...] = ()


#: What ly asks for when nothing says otherwise, so a key and nothing else is
#: enough to start. The provider is part of it: a bare model names nowhere.
DEFAULT_MODEL = "openai/gpt-4.1"

#: Providers ly knows without being told. A settings file may add more under
#: `providers`, or override one of these to point it at a proxy.
BUILT_IN: dict[str, Preset] = {
    "openai": Preset(
        base_url="https://api.openai.com/v1",
        key_env=("OPENAI_API_KEY",),
        models=("gpt-4.1",),
    ),
    "deepseek": Preset(
        base_url="https://api.deepseek.com",
        key_env=("DEEPSEEK_API_KEY",),
        pass_reasoning_content=True,
        models=("deepseek-chat", "deepseek-reasoner"),
    ),
}


def split(reference: str) -> tuple[str, str]:
    """Split a `provider/model` reference into its two halves.

    Only the first slash separates, because model names may contain slashes of
    their own (`openai/moonshotai/kimi-k2`). A reference with no slash names no
    provider, and one of the halves is then empty.

    Args:
        reference: A model reference, such as `deepseek/deepseek-flash`.

    Returns:
        The provider name and the model name to send. The provider name comes
        back empty when the reference does not name one.
    """
    name, _, model = reference.partition("/")
    if not (name and model):
        return "", reference
    return name, model
