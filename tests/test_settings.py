"""The settings file: what it supplies, and what it must never break."""

import json
from pathlib import Path

import pytest

from ly.agent import AgentSession
from ly.cli import Repl, build_parser
from ly.compaction import CompactionConfig
from ly.providers.backend import ModelBackend
from ly.providers.errors import (
    MissingCredentialsError,
    MissingModelError,
    UnknownProviderError,
)
from ly.providers.provider import Provider
from ly.session import SessionManager
from ly.settings import Settings


def write(tmp_path, data) -> str:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def read(path) -> Settings:
    """Load the file on its own; the machine running the tests exports LY_* too."""
    return Settings.load(path, env={})


def test_a_missing_file_means_no_model_and_no_endpoint(tmp_path):
    """Every key is optional, and there is no built-in model to fall back to:
    nothing aims a request until something names a provider."""
    settings = read(tmp_path / "absent.json")

    assert (settings.model, settings.base_url) == (None, None)
    with pytest.raises(MissingModelError):
        settings.make_provider()


def test_a_malformed_file_does_not_stop_ly(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not json", encoding="utf-8")

    assert read(path) == read(tmp_path / "absent.json")


def test_a_file_that_is_not_an_object_means_defaults(tmp_path):
    assert read(write(tmp_path, ["model", "x"])) == read(tmp_path / "absent.json")


def test_known_keys_are_read(tmp_path):
    settings = read(
        write(
            tmp_path,
            {
                "model": "gpt-5",
                "base_url": "https://example.test/v1",
                "api_format": "openai-responses",
                "markdown": False,
                "thinking": False,
                "sandbox": False,
            },
        )
    )

    assert settings == Settings(
        model="gpt-5",
        base_url="https://example.test/v1",
        api_format="openai-responses",
        markdown=False,
        thinking=False,
        sandbox=False,
    )


def test_unknown_keys_and_bad_types_are_ignored_one_by_one(tmp_path):
    settings = read(
        write(
            tmp_path,
            {
                "model": "kept",
                "unknown": "ignored",
                "markdown": "yes",  # a string is not a flag
                "thinking": 1,  # a number is not a flag either
                "api_format": "grpc",  # the two OpenAI formats are all there is
                "base_url": "",  # empty means unset
            },
        )
    )

    assert settings == Settings(model="kept")


def test_environment_beats_the_file(tmp_path):
    path = write(tmp_path, {"model": "from-file", "api_format": "openai-completions"})

    settings = Settings.load(
        path, env={"LY_MODEL": "from-env", "LY_API_FORMAT": "openai-responses"}
    )

    assert (settings.model, settings.api_format) == ("from-env", "openai-responses")


def test_the_file_fills_in_what_the_environment_leaves_alone(tmp_path):
    path = write(tmp_path, {"model": "kept", "base_url": "https://file.test"})

    settings = Settings.load(path, env={"LY_MODEL": "from-env"})

    assert settings.model == "from-env"
    assert settings.base_url == "https://file.test"


def test_an_unusable_environment_value_is_ignored(tmp_path):
    path = write(tmp_path, {"api_format": "openai-responses"})

    # A wrong LY_API_FORMAT falls back rather than reaching the provider, which
    # is what argparse's choices used to enforce for a flag.
    assert (
        Settings.load(path, env={"LY_API_FORMAT": "grpc"}).api_format
        == "openai-responses"
    )


def test_the_api_key_comes_from_the_environment_only(tmp_path):
    path = write(tmp_path, {"api_key": "in-the-file"})

    assert Settings.load(path, env={}).api_key is None
    assert Settings.load(path, env={"LY_API_KEY": "sk-env"}).api_key == "sk-env"


def test_docker_off_in_the_environment_means_no_sandbox():
    assert Settings.load(env={"LY_DOCKER": "0"}).sandbox is False
    assert Settings.load(env={"LY_DOCKER": "1"}).sandbox is True
    assert Settings.load(env={}).sandbox is True  # sandboxed unless told otherwise


def test_the_docker_image_comes_from_the_environment():
    assert Settings.load(env={}).image is None
    assert Settings.load(env={"LY_DOCKER_IMAGE": "python:3.13"}).image == (
        "python:3.13"
    )


def test_the_sessions_directory_is_resolved_here_and_the_mount_wins():
    default = Settings.load(env={}).sessions_dir
    preferred = Settings.load(env={"LY_SESSIONS_DIR": "/srv/sessions"}).sessions_dir
    # Where the sandbox mounted the host's sessions is the truth: a stale
    # preference must not describe a directory the container cannot see.
    mounted = Settings.load(
        env={"LY_SB_SESSIONS_DIR": "/sessions", "LY_SESSIONS_DIR": "/srv/x"}
    ).sessions_dir

    assert default == Path("~/.ly/sessions").expanduser()
    assert preferred == Path("/srv/sessions")
    assert mounted == Path("/sessions")


def test_pass_reasoning_content_comes_from_the_provider():
    """DeepSeek needs the field echoed back and OpenAI rejects it, so which
    one is in play decides it. The environment can override either way."""
    deepseek = {"LY_MODEL": "deepseek/deepseek-chat"}
    openai = {"LY_MODEL": "openai/gpt-4.1"}

    assert Settings.load(env=deepseek).pass_reasoning_content
    assert not Settings.load(env=openai).pass_reasoning_content
    assert not Settings.load(
        env={**deepseek, "LY_PASS_REASONING_CONTENT": "off"}
    ).pass_reasoning_content
    assert Settings.load(
        env={**openai, "LY_PASS_REASONING_CONTENT": "yes"}
    ).pass_reasoning_content


def test_compaction_comes_from_the_file(tmp_path):
    path = write(
        tmp_path,
        {"compaction": {"enabled": False, "threshold_tokens": 8000, "keep_turns": 1}},
    )

    assert read(path).compaction == CompactionConfig(
        enabled=False,
        threshold_tokens=8000,
        keep_turns=1,
        max_summary_input_tokens=100_000,  # untouched keys keep their default
    )


def test_compaction_comes_from_the_environment(tmp_path):
    settings = Settings.load(
        write(tmp_path, {}),
        env={
            "LY_COMPACT": "0",
            "LY_COMPACT_THRESHOLD": "8000",
            "LY_COMPACT_KEEP_TURNS": "2",
            "LY_COMPACT_INPUT_TOKENS": "500",
        },
    )

    assert settings.compaction == CompactionConfig(
        enabled=False, threshold_tokens=8000, keep_turns=2, max_summary_input_tokens=500
    )


def test_compaction_merges_key_by_key(tmp_path):
    path = write(tmp_path, {"compaction": {"threshold_tokens": 8000, "keep_turns": 1}})

    settings = Settings.load(path, env={"LY_COMPACT_THRESHOLD": "2000"})

    # The variable wins for its own key only; the file still sets keep_turns.
    assert (settings.compaction.threshold_tokens, settings.compaction.keep_turns) == (
        2000,
        1,
    )


def test_bad_compaction_values_are_ignored(tmp_path):
    path = write(
        tmp_path,
        {"compaction": {"enabled": "yes", "threshold_tokens": True}},
    )

    # `true` is not a token count, and a string is not a flag.
    assert read(path).compaction == CompactionConfig()
    assert Settings.load(path, env={"LY_COMPACT_THRESHOLD": "lots"}).compaction == (
        CompactionConfig()
    )


def test_a_compaction_block_that_is_not_an_object_is_ignored(tmp_path):
    assert read(write(tmp_path, {"compaction": 5})).compaction == CompactionConfig()


def test_compaction_becomes_an_argument_default():
    config = CompactionConfig(threshold_tokens=1234)

    # `make_session` passes args.compaction on; this covers the default it reads.
    assert build_parser(Settings(compaction=config)).parse_args([]).compaction == config


def test_the_repl_starts_with_the_file_toggles(tmp_path):
    session = AgentSession(
        backend=ModelBackend(Provider(model="m", base_url="u", api_key="k")),
        tools=[],
        session_manager=SessionManager(),
        system_prompt="system",
        cwd=str(tmp_path),
    )
    repl = Repl(session, settings=Settings(markdown=False, thinking=False))

    assert (repl.markdown, repl.show_thinking) == (False, False)


# -- the model reference ------------------------------------------------------


def absent(tmp_path) -> str:
    """A path with no file at it, so only the environment is under test."""
    return str(tmp_path / "absent.json")


def test_a_model_without_a_provider_names_nowhere(tmp_path):
    settings = Settings.load(absent(tmp_path), env={"LY_MODEL": "gpt-4.1"})

    assert settings.provider is None
    with pytest.raises(UnknownProviderError) as raised:
        settings.make_provider()

    # The message has to say what to type instead, since nothing else does.
    assert "provider/model" in str(raised.value)
    assert "deepseek/*" in str(raised.value)


def test_the_provider_half_supplies_the_endpoint(tmp_path):
    settings = Settings.load(
        absent(tmp_path), env={"LY_MODEL": "deepseek/deepseek-chat"}
    )

    assert (settings.provider, settings.model_name, settings.base_url) == (
        "deepseek",
        "deepseek-chat",
        "https://api.deepseek.com",
    )


def test_only_the_first_slash_separates(tmp_path):
    """Providers that name models with a slash (`moonshotai/kimi`) keep them."""
    settings = Settings.load(
        absent(tmp_path), env={"LY_MODEL": "openai/moonshotai/kimi-k2"}
    )

    assert settings.model_name == "moonshotai/kimi-k2"
    assert settings.base_url == "https://api.openai.com/v1"


def test_explicit_values_beat_the_preset(tmp_path):
    settings = Settings.load(
        absent(tmp_path),
        env={
            "LY_MODEL": "deepseek/deepseek-reasoner",
            "LY_BASE_URL": "https://proxy.test/v1",
            "LY_API_FORMAT": "openai-responses",
            "LY_API_KEY": "sk-env",
        },
    )

    assert (settings.model_name, settings.base_url, settings.api_format) == (
        "deepseek-reasoner",
        "https://proxy.test/v1",
        "openai-responses",
    )


def test_the_preset_supplies_its_own_key_variable(tmp_path):
    settings = Settings.load(
        absent(tmp_path), env={"LY_MODEL": "openai/gpt-4.1", "OPENAI_API_KEY": "sk-o"}
    )

    assert settings.api_key == "sk-o"
    # Every preset's, because /model can switch provider inside the container.
    assert settings.key_variables == ("DEEPSEEK_API_KEY", "OPENAI_API_KEY")


def test_a_generic_key_wins_over_the_preset_variable(tmp_path):
    settings = Settings.load(
        absent(tmp_path),
        env={
            "LY_MODEL": "openai/gpt-4.1",
            "LY_API_KEY": "sk-env",
            "OPENAI_API_KEY": "sk-openai",
        },
    )

    assert settings.api_key == "sk-env"


def test_a_model_override_beats_the_environment(tmp_path):
    """`--model` and `/model` hand the reference in, and its provider wins.

    A pinned `LY_MODEL` used to survive a switch, which sent one provider's
    model name to another provider's endpoint.
    """
    env = {
        "LY_MODEL": "openai/gpt-4.1",
        "OPENAI_API_KEY": "sk-o",
        "DEEPSEEK_API_KEY": "sk-d",
    }

    settings = Settings.load(absent(tmp_path), env=env)
    switched = Settings.load(absent(tmp_path), env=env, model="deepseek/deepseek-chat")

    assert (settings.model_name, settings.api_key) == ("gpt-4.1", "sk-o")
    assert (switched.model_name, switched.api_key) == ("deepseek-chat", "sk-d")


def test_a_file_provider_can_be_named(tmp_path):
    path = write(
        tmp_path,
        {
            "model": "mygateway/house-model",
            "providers": {
                "mygateway": {
                    "base_url": "https://gw.test/v1",
                    "key_env": ["MYGW_API_KEY"],
                }
            },
        },
    )

    settings = Settings.load(path, env={"MYGW_API_KEY": "sk-gw"})

    assert (settings.base_url, settings.model_name, settings.api_key) == (
        "https://gw.test/v1",
        "house-model",
        "sk-gw",
    )
    # A gateway that speaks DeepSeek's dialect says so once, here.
    assert not settings.pass_reasoning_content


def test_a_file_provider_can_declare_the_reasoning_field(tmp_path):
    path = write(
        tmp_path,
        {
            "model": "mygateway/house-model",
            "providers": {
                "mygateway": {
                    "base_url": "https://gw.test/v1",
                    "pass_reasoning_content": True,
                }
            },
        },
    )

    settings = Settings.load(path, env={"LY_API_KEY": "sk"})

    assert settings.make_provider().pass_reasoning_content


def test_a_file_provider_can_point_a_built_in_at_a_proxy(tmp_path):
    path = write(
        tmp_path,
        {"providers": {"openai": {"base_url": "https://proxy.test/v1"}}},
    )

    settings = Settings.load(
        path, env={"LY_MODEL": "openai/gpt-4.1", "OPENAI_API_KEY": "sk-openai"}
    )

    assert settings.base_url == "https://proxy.test/v1"
    # The key variable stays the built-in one, which the model's provider names.
    assert settings.api_key == "sk-openai"


def test_a_proxied_provider_keeps_its_own_dialect(tmp_path):
    """Regression: the reasoning flag was guessed from the base URL, so
    pointing the deepseek preset at a proxy silently turned it off."""
    path = write(
        tmp_path,
        {"providers": {"deepseek": {"base_url": "https://proxy.corp.test/v1"}}},
    )

    settings = Settings.load(
        path, env={"LY_MODEL": "deepseek/deepseek-chat", "LY_API_KEY": "sk"}
    )

    assert settings.make_provider().pass_reasoning_content


def test_a_file_provider_can_list_its_models(tmp_path):
    path = write(
        tmp_path,
        {
            "providers": {
                "mygateway": {
                    "base_url": "https://gw.test/v1",
                    "models": ["house-model", "house-fast"],
                }
            }
        },
    )

    settings = Settings.load(path, env={})

    assert set(settings.presets["mygateway"].models) == {"house-model", "house-fast"}
    # Listed without sizes, so a model here compacts on the global defaults.
    assert settings.presets["mygateway"].models["house-model"].context_window is None


def test_a_file_provider_can_size_its_models(tmp_path):
    path = write(
        tmp_path,
        {
            "providers": {
                "mygateway": {
                    "base_url": "https://gw.test/v1",
                    "models": {
                        "house-model": 32768,
                        "house-fast": {"context_window": 8192},
                    },
                }
            }
        },
    )

    models = Settings.load(path, env={}).presets["mygateway"].models

    assert models["house-model"].context_window == 32768
    assert models["house-fast"].context_window == 8192


def test_a_nonsense_window_is_ignored(tmp_path):
    path = write(
        tmp_path,
        {
            "providers": {
                "mygateway": {
                    "base_url": "https://gw.test/v1",
                    "models": {"a": -5, "b": True, "c": "lots", "d": 0},
                }
            }
        },
    )

    models = Settings.load(path, env={}).presets["mygateway"].models

    assert [model.context_window for model in models.values()] == [None] * 4


def sized(tmp_path, models: dict, **extra) -> str:
    """A settings file declaring one provider whose models have these windows.

    The windows ly ships are the user's own data, so sizing tests state the
    windows they mean rather than reading the built-in catalogue. Extra
    top-level keys, such as `compaction`, are written alongside.

    Args:
        tmp_path: Directory to write the file in.
        models: Model name to window, or to a `{context_window, keep_turns}` map.
        **extra: Further top-level settings.

    Returns:
        The path of the settings file.
    """
    return write(
        tmp_path,
        {
            "providers": {
                "house": {"base_url": "https://house.test/v1", "models": models}
            },
            **extra,
        },
    )


def test_compaction_is_sized_to_the_models_context_window(tmp_path):
    path = sized(tmp_path, {"small": 65_536})

    settings = Settings.load(path, env={"LY_MODEL": "house/small"})

    assert settings.context_window == 65_536
    # Half the window, and the summary cap follows the threshold.
    assert settings.compaction.threshold_tokens == 32_768
    assert settings.compaction.max_summary_input_tokens == 32_768


def test_the_summary_cap_never_exceeds_the_threshold(tmp_path):
    """A branch cannot hold more than the threshold when compaction triggers,
    so a cap equal to it reads the whole history in one pass."""
    settings = Settings.load(
        sized(tmp_path, {"big": 4_000_000}), env={"LY_MODEL": "house/big"}
    )

    assert settings.compaction.threshold_tokens == 2_000_000
    assert settings.compaction.max_summary_input_tokens == 2_000_000


def test_a_large_window_raises_the_threshold(tmp_path):
    """A million tokens is worth a long verbatim history, so the threshold
    scales with the window rather than stopping at a fixed ceiling."""
    settings = Settings.load(
        sized(tmp_path, {"big": 1_047_576}), env={"LY_MODEL": "house/big"}
    )

    assert settings.context_window == 1_047_576
    assert settings.compaction.threshold_tokens == 523_788


def test_a_model_can_keep_its_own_number_of_turns(tmp_path):
    path = sized(tmp_path, {"small": {"context_window": 65_536, "keep_turns": 3}})

    settings = Settings.load(path, env={"LY_MODEL": "house/small"})

    assert settings.compaction.keep_turns == 3


def test_a_model_outside_the_catalogue_keeps_the_plain_defaults(tmp_path):
    settings = Settings.load(
        sized(tmp_path, {"small": 65_536}), env={"LY_MODEL": "house/unlisted"}
    )

    assert settings.context_window is None
    assert settings.compaction == CompactionConfig()


def test_an_explicit_setting_beats_the_window(tmp_path):
    """The window is a default, not an instruction: what the user wrote wins."""
    path = sized(
        tmp_path,
        {"small": 65_536},
        compaction={"threshold_tokens": 8_000, "keep_turns": 6},
    )

    settings = Settings.load(path, env={"LY_MODEL": "house/small"})

    assert settings.compaction.threshold_tokens == 8_000
    assert settings.compaction.keep_turns == 6
    # Untouched keys still come from the window.
    assert settings.compaction.max_summary_input_tokens == 32_768


def test_a_provider_entry_without_a_base_url_is_ignored(tmp_path):
    path = write(
        tmp_path, {"providers": {"broken": {"api_format": "openai-responses"}}}
    )

    assert set(Settings.load(path, env={}).presets) == {"openai", "deepseek"}


def test_a_missing_key_names_the_variable_the_provider_reads(tmp_path):
    """`/model deepseek/...` should say which key deepseek wants, not just
    LY_API_KEY, which it would also accept."""
    settings = Settings.load(
        absent(tmp_path), env={"LY_MODEL": "deepseek/deepseek-chat"}
    )

    with pytest.raises(MissingCredentialsError) as raised:
        settings.make_provider()

    assert (
        str(raised.value) == "Missing credentials. Set DEEPSEEK_API_KEY or LY_API_KEY."
    )


def test_make_provider_sends_the_half_after_the_slash(tmp_path):
    settings = Settings.load(
        absent(tmp_path),
        env={"LY_MODEL": "deepseek/deepseek-chat", "DEEPSEEK_API_KEY": "k"},
    )

    provider = settings.make_provider()

    assert (provider.api_format, provider.model) == (
        "openai-completions",
        "deepseek-chat",
    )
