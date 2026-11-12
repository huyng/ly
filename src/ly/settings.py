"""User preferences, read from `~/.ly/settings.json` and the environment.

This module is the only place that reads configuration out of the environment.
`load()` merges two sources, environment variables over the file, and everything
below `main()` then takes a `Settings` as an argument, so nothing in the core
reads global state.

The file is optional and so is every key. An unreadable or malformed file counts
as no settings at all rather than an error: a typo in a preferences file must not
stop ly from starting. Unknown keys and values of the wrong type are ignored one
by one, so the rest of the file still applies.

Precedence is command-line flag, environment variable, this file, built-in
default. The API key is the exception to the file: it is read from `LY_API_KEY`
or the selected provider's own variable, so a copy of the file can be shared
without leaking it.

The model is written `provider/model`, and the provider half names a preset in
`ly/presets.py`, which supplies the endpoint, the wire format and the variables
its key hides in. Everything named above still wins over that preset, so the
provider is a default, not a mode. There is no built-in model: with none
configured, `make_provider` raises rather than aim a request somewhere nobody
chose. A model without a provider names nowhere, and is rejected the same way
when it is resolved, not when it is read.
"""

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from .compaction import CompactionConfig, for_window
from .presets import BUILT_IN, Model, Preset, split
from .providers import Provider
from .providers.errors import (
    MissingCredentialsError,
    MissingModelError,
    UnknownProviderError,
)

DEFAULT_PATH = "~/.ly/settings.json"

#: Where session JSONL files go when nothing overrides it.
DEFAULT_SESSIONS_DIR = "~/.ly/sessions"

# Only these values reach the provider; argparse used to enforce them, but an
# environment variable or a settings file bypasses that check.
API_FORMATS = ("openai-completions", "openai-responses")

# (setting name, environment variable) for each key that is a plain integer.
COMPACTION_INTS = (
    ("threshold_tokens", "LY_COMPACT_THRESHOLD"),
    ("keep_turns", "LY_COMPACT_KEEP_TURNS"),
    ("max_summary_input_tokens", "LY_COMPACT_INPUT_TOKENS"),
)


def _tri_state(raw: str | None) -> bool | None:
    """Read an on/off environment value; None when it is unset."""
    if raw is None:
        return None
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _integer(raw: str | None) -> int | None:
    """Read an integer environment value; None when it is absent or not a number."""
    try:
        return int(raw) if raw else None
    except ValueError:
        return None


def _compaction_from_file(data: dict) -> dict:
    """Read the usable keys out of the `compaction` block of the file."""
    block = data.get("compaction")
    if not isinstance(block, dict):
        return {}

    values: dict = {}
    # `isinstance(True, int)` is True, so `enabled` needs the boolean check and
    # the integers need to exclude booleans explicitly.
    if isinstance(block.get("enabled"), bool):
        values["enabled"] = block["enabled"]
    for name, _ in COMPACTION_INTS:
        value = block.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            values[name] = value
    return values


def _models_from_file(entry: dict) -> dict[str, Model]:
    """Read a provider's `models` table: names, windows, or both.

    A list of names says what Tab may offer without saying how big the models
    are. A number is a context window. An object carries a window and, when it
    differs from the global default, the turns to keep verbatim.
    """
    block = entry.get("models")
    if isinstance(block, list):
        return {name: Model() for name in block if isinstance(name, str)}
    if not isinstance(block, dict):
        return {}

    models: dict[str, Model] = {}
    for name, value in block.items():
        window = (
            value if isinstance(value, int) and not isinstance(value, bool) else None
        )
        keep = None
        if isinstance(value, dict):
            window = value.get("context_window")
            keep = value.get("keep_turns")
        # A window has to be a positive count of tokens to say anything.
        models[name] = Model(
            context_window=window if isinstance(window, int) and window > 0 else None,
            keep_turns=keep if isinstance(keep, int) and keep > 0 else None,
        )
    return models


def _presets_from_file(data: dict) -> dict[str, Preset]:
    """Read the `providers` block: named endpoints added or overridden by hand.

    An entry merges over the built-in provider of the same name, so pointing a
    known provider at a proxy is one key and its models and key variable stay
    as they were.
    """
    block = data.get("providers")
    if not isinstance(block, dict):
        return {}

    presets: dict[str, Preset] = {}
    for name, entry in block.items():
        if not isinstance(entry, dict):
            continue
        given: dict = {}
        base_url = entry.get("base_url")
        if isinstance(base_url, str) and base_url:
            given["base_url"] = base_url
        if entry.get("api_format") in API_FORMATS:
            given["api_format"] = entry["api_format"]
        # A declared provider that speaks DeepSeek's dialect needs this false
        # flag turned on; the environment can still override it per run.
        if isinstance(entry.get("pass_reasoning_content"), bool):
            given["pass_reasoning_content"] = entry["pass_reasoning_content"]
        key_env = entry.get("key_env")
        if isinstance(key_env, list):
            given["key_env"] = tuple(v for v in key_env if isinstance(v, str))
        if "models" in entry:
            given["models"] = _models_from_file(entry)
        # A new provider without an endpoint says nothing a default could fill
        # in; a known one keeps the endpoint it already has.
        if "base_url" not in given and name not in BUILT_IN:
            continue
        presets[name] = replace(BUILT_IN.get(name, Preset(base_url="")), **given)
    return presets


def _from_file(path: str | Path | None) -> dict:
    """Read the usable keys out of the settings file, ignoring the rest."""
    try:
        text = Path(path or DEFAULT_PATH).expanduser().read_text(encoding="utf-8")
        data = json.loads(text)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}

    values: dict = {
        "compaction": _compaction_from_file(data),
        "presets": _presets_from_file(data),
    }
    for name in ("model", "base_url"):
        value = data.get(name)
        if isinstance(value, str) and value:
            values[name] = value
    if data.get("api_format") in API_FORMATS:
        values["api_format"] = data["api_format"]
    for name in ("markdown", "thinking", "sandbox"):
        value = data.get(name)
        # `isinstance(1, bool)` is False, so a JSON number is not read as a flag.
        if isinstance(value, bool):
            values[name] = value
    return values


def _model_of(presets: Mapping[str, Preset], reference: str | None) -> Model | None:
    """What the reference's preset says about that model, if anything."""
    if not reference:
        return None
    name, model = split(reference)
    preset = presets.get(name)
    return preset.models.get(model) if preset else None


def _compaction(
    explicit: dict, presets: Mapping[str, Preset], reference: str | None
) -> CompactionConfig:
    """Size compaction to the model in use, then let explicit keys win.

    Args:
        explicit: Keys from the settings file and environment.
        presets: The presets in force, built-in plus declared.
        reference: The `provider/model` reference in use, or None when none is
            configured.

    Returns:
        A context window's worth of defaults, unless a model or the user said
        otherwise. A model that no preset sizes keeps the plain defaults.
    """
    model = _model_of(presets, reference)
    sized = CompactionConfig()
    if model is not None and model.context_window:
        sized = for_window(model.context_window)
    if model is not None and model.keep_turns is not None:
        sized = replace(sized, keep_turns=model.keep_turns)
    return replace(sized, **explicit)


def _apply_preset(values: dict, preset: Preset, env: Mapping[str, str]) -> None:
    """Fill the values the user did not set from the preset the model names."""
    values.setdefault("base_url", preset.base_url)
    values.setdefault("api_format", preset.api_format)
    values.setdefault("pass_reasoning_content", preset.pass_reasoning_content)
    if values.get("api_key"):
        return
    # Nothing generic was exported, so fall back to the provider's own variable.
    for name in preset.key_env:
        if env.get(name):
            values["api_key"] = env[name]
            return


def _from_env(env) -> dict:
    """Read the usable keys out of the environment, ignoring the rest."""
    values: dict = {}
    for name, variable in (
        ("model", "LY_MODEL"),
        ("base_url", "LY_BASE_URL"),
        ("api_key", "LY_API_KEY"),
        ("image", "LY_SANDBOX_IMAGE"),
    ):
        text = env.get(variable)
        if text:
            values[name] = text
    if env.get("LY_API_FORMAT") in API_FORMATS:
        values["api_format"] = env["LY_API_FORMAT"]

    # `LY_SB_SESSIONS_DIR` is where the sandbox mounted the host's sessions, so
    # it wins over the preference: the mount, not the setting, decides where the
    # files really are. Resolved here, the one place that reads the environment,
    # so nothing below takes a default of its own.
    sessions = (
        env.get("LY_SB_SESSIONS_DIR")
        or env.get("LY_SESSIONS_DIR")
        or DEFAULT_SESSIONS_DIR
    )
    values["sessions_dir"] = Path(sessions).expanduser()

    reasoning = _tri_state(env.get("LY_PASS_REASONING_CONTENT"))
    if reasoning is not None:
        values["pass_reasoning_content"] = reasoning
    sandbox = _tri_state(env.get("LY_SANDBOX"))
    if sandbox is not None:
        # `LY_SANDBOX=0` means "do not use a container", which is exactly what
        # `sandbox: false` says, so the value carries over as it is.
        values["sandbox"] = sandbox

    compaction: dict = {}
    for name, variable in COMPACTION_INTS:
        number = _integer(env.get(variable))
        if number is not None:
            compaction[name] = number
    values["compaction"] = compaction
    return values


@dataclass(frozen=True)
class Settings:
    """Preferences from the settings file and the environment."""

    # Built-in presets plus whatever the file added, keyed by name. The model's
    # provider half is looked up here.
    presets: dict[str, Preset] = field(default_factory=lambda: dict(BUILT_IN))
    # A `provider/model` reference, or None when nothing configured one. The
    # provider half is what aims a request; without it `make_provider` refuses.
    model: str | None = None
    # The file these values came from, or None when nothing was read. Kept so
    # a later `load` (a `/model` switch, the docker mount) reads the same file
    # rather than silently falling back to the default path. Excluded from
    # equality: two settings objects are the same preferences however the
    # files holding them are named.
    path: Path | None = field(default=None, compare=False, repr=False)
    base_url: str | None = None
    api_format: str | None = None
    api_key: str | None = None
    # Resolved by `load` from the preset; `LY_PASS_REASONING_CONTENT` overrides.
    # A provider named by hand gets False, which is what every non-DeepSeek
    # endpoint wants.
    pass_reasoning_content: bool = False
    # None asks the CLI for `sandbox.DEFAULT_IMAGE`; the settings file does not
    # carry this key, because the image belongs to the host that starts Docker.
    image: str | None = None
    # Where session JSONL files are written. Asked for rather than read from the
    # environment by `SessionManager`, which keeps no default of its own.
    sessions_dir: Path = field(
        default_factory=lambda: Path(DEFAULT_SESSIONS_DIR).expanduser()
    )
    markdown: bool = True
    thinking: bool = True
    sandbox: bool = True
    compaction: CompactionConfig = field(default_factory=CompactionConfig)

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        env: dict | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_format: str | None = None,
    ) -> "Settings":
        """Merge the settings file and the environment, environment winning.

        Args:
            path: File to read, defaulting to `~/.ly/settings.json`. Recorded on
                the result, so a later `load` can name the same file.
            env: Variables to read, defaulting to `os.environ`. Passed in so
                tests need not patch the process environment.
            model: Model reference to use instead of the one the file or
                environment names. A reference naming no known provider is
                kept, and refused later by `make_provider`.
            base_url: Endpoint to use instead of the one the model's provider
                supplies, which is how a preset is pointed at a proxy. The
                provider half still has to name a preset: an endpoint alone
                does not say which wire format or key variable to use.
            api_key: Key to use instead of the one the environment supplies.
            api_format: Wire format; ignored unless it is one of the known ones.

        Returns:
            The merged settings, with defaults for keys neither source supplies
            or for values of the wrong type.
        """
        environment = os.environ if env is None else env
        values = _from_file(path)
        env_values = _from_env(environment)
        presets = BUILT_IN | values.pop("presets", {})
        # Compaction is a nested object, so it merges key by key rather than as
        # a unit: LY_COMPACT_THRESHOLD can win while a file's keep_turns stays.
        compaction = values.pop("compaction", {}) | env_values.pop("compaction", {})
        values |= env_values
        # These are what a caller passes; `None` means it passed nothing and
        # the merged sources stand. They win over both, because the caller is
        # nearer than either.
        given = {
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
            "api_format": api_format if api_format in API_FORMATS else None,
        }
        values |= {name: value for name, value in given.items() if value}
        # The provider half of the model is what picks the endpoint and the
        # variable the key hides in. A half that matches no preset is not an
        # error yet: the endpoint can still have been set by hand.
        reference = values.get("model")
        preset = presets.get(split(reference)[0]) if reference else None
        if preset is not None:
            _apply_preset(values, preset, environment)
        return cls(
            compaction=_compaction(compaction, presets, reference),
            presets=presets,
            path=Path(path).expanduser() if path else None,
            **values,
        )

    @property
    def provider(self) -> str | None:
        """Name of the preset the model names, None when none is set or named."""
        if not self.model:
            return None
        name, _ = split(self.model)
        return name if name in self.presets else None

    @property
    def model_name(self) -> str:
        """The half of the model reference sent to the endpoint; "" if unset."""
        return split(self.model)[1] if self.model else ""

    @property
    def context_window(self) -> int | None:
        """Tokens the model in use accepts, None when its preset does not say."""
        model = _model_of(self.presets, self.model)
        return model.context_window if model else None

    @property
    def key_variables(self) -> tuple[str, ...]:
        """Variables that may hold a key for any provider ly can switch to.

        The sandbox forwards `LY_*` and nothing else, so a key kept in e.g.
        `DEEPSEEK_API_KEY` has to travel by name; see `sandbox.forwarded_env`.
        Every preset contributes, not just the active one: `/model` can switch
        provider mid-session, and that switch happens inside the container.

        Returns:
            The key variables of every known preset, in a stable order.
        """
        names = {
            variable for preset in self.presets.values() for variable in preset.key_env
        }
        return tuple(sorted(names))

    def make_provider(self) -> Provider:
        """Build the live provider record these settings describe.

        Every value is concrete once `load()` returns, so this is a field
        mapping, and the flattening of "unset" to the endpoint's own default
        stays inside `Provider`.

        Returns:
            A `Provider` the backends can read on each request.

        Raises:
            MissingModelError: No model was configured at all.
            UnknownProviderError: The model reference names no known provider,
                which includes naming no provider at all.
            MissingCredentialsError: No API key is configured.
            MissingBaseURLError: No endpoint is configured.
        """
        if not self.model:
            raise MissingModelError()
        if self.provider is None:
            known = ", ".join(f"{name}/*" for name in sorted(self.presets))
            raise UnknownProviderError(self.model, known)
        # Checked here rather than left to `Provider`, which cannot say which
        # variables the preset reads its key from.
        if not self.api_key:
            preset = self.presets.get(self.provider)
            raise MissingCredentialsError(preset.key_env if preset else ())
        return Provider(
            model=self.model_name,
            base_url=self.base_url or "",
            api_key=self.api_key or "",
            api_format=self.api_format or "openai-completions",
            pass_reasoning_content=self.pass_reasoning_content,
        )
