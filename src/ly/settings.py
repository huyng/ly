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
only, so a copy of the file can be shared without leaking it.
"""

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from .compaction import CompactionConfig

DEFAULT_PATH = "~/.ly/settings.json"

# Only these values reach the provider; argparse used to enforce them, but an
# environment variable or a settings file bypasses that check.
API_KINDS = ("chat", "responses")

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


def _from_file(path: str | Path | None) -> dict:
    """Read the usable keys out of the settings file, ignoring the rest."""
    try:
        text = Path(path or DEFAULT_PATH).expanduser().read_text(encoding="utf-8")
        data = json.loads(text)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}

    values: dict = {"compaction": _compaction_from_file(data)}
    for name in ("model", "base_url"):
        value = data.get(name)
        if isinstance(value, str) and value:
            values[name] = value
    if data.get("api") in API_KINDS:
        values["api"] = data["api"]
    for name in ("markdown", "thinking", "sandbox"):
        value = data.get(name)
        # `isinstance(1, bool)` is False, so a JSON number is not read as a flag.
        if isinstance(value, bool):
            values[name] = value
    return values


def _from_env(env) -> dict:
    """Read the usable keys out of the environment, ignoring the rest."""
    values: dict = {}
    for name, variable in (
        ("model", "LY_MODEL"),
        ("base_url", "LY_BASE_URL"),
        ("api_key", "LY_API_KEY"),
        ("image", "LY_DOCKER_IMAGE"),
    ):
        text = env.get(variable)
        if text:
            values[name] = text
    if env.get("LY_API") in API_KINDS:
        values["api"] = env["LY_API"]

    reasoning = _tri_state(env.get("LY_PASS_REASONING_CONTENT"))
    if reasoning is not None:
        values["pass_reasoning_content"] = reasoning
    docker = _tri_state(env.get("LY_DOCKER"))
    if docker is not None:
        # `LY_DOCKER=0` means "do not use a container", which is exactly what
        # `sandbox: false` says, so the value carries over as it is.
        values["sandbox"] = docker

    compaction: dict = {}
    enabled = _tri_state(env.get("LY_COMPACT"))
    if enabled is not None:
        compaction["enabled"] = enabled
    for name, variable in COMPACTION_INTS:
        number = _integer(env.get(variable))
        if number is not None:
            compaction[name] = number
    values["compaction"] = compaction
    return values


@dataclass(frozen=True)
class Settings:
    """Preferences from the settings file and the environment."""

    model: str | None = None
    base_url: str | None = None
    api: str | None = None
    api_key: str | None = None
    # None lets the provider decide from the base URL; see `Provider`.
    pass_reasoning_content: bool | None = None
    # None asks the CLI for `sandbox.DEFAULT_IMAGE`; the settings file does not
    # carry this key, because the image belongs to the host that starts Docker.
    image: str | None = None
    markdown: bool = True
    thinking: bool = True
    sandbox: bool = True
    compaction: CompactionConfig = field(default_factory=CompactionConfig)

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        env: dict | None = None,
    ) -> "Settings":
        """Merge the settings file and the environment, environment winning.

        Args:
            path: File to read, defaulting to `~/.ly/settings.json`.
            env: Variables to read, defaulting to `os.environ`. Passed in so
                tests need not patch the process environment.

        Returns:
            The merged settings, with defaults for keys neither source supplies
            or for values of the wrong type.
        """
        values = _from_file(path)
        env_values = _from_env(os.environ if env is None else env)
        # Compaction is a nested object, so it merges key by key rather than as
        # a unit: LY_COMPACT_THRESHOLD can win while a file's keep_turns stays.
        compaction = values.pop("compaction", {}) | env_values.pop("compaction", {})
        values |= env_values
        return cls(compaction=replace(CompactionConfig(), **compaction), **values)
