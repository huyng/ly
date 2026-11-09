"""The settings file: what it supplies, and what it must never break."""

import json

from ly.agent import AgentSession
from ly.cli import Repl, build_parser
from ly.compaction import CompactionConfig
from ly.providers.backend import ModelBackend
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


def test_a_missing_file_means_defaults(tmp_path):
    assert read(tmp_path / "absent.json") == Settings()


def test_a_malformed_file_does_not_stop_ly(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not json", encoding="utf-8")

    assert read(path) == Settings()


def test_a_file_that_is_not_an_object_means_defaults(tmp_path):
    assert read(write(tmp_path, ["model", "x"])) == Settings()


def test_known_keys_are_read(tmp_path):
    settings = read(
        write(
            tmp_path,
            {
                "model": "gpt-5",
                "base_url": "https://example.test/v1",
                "api": "responses",
                "markdown": False,
                "thinking": False,
                "sandbox": False,
            },
        )
    )

    assert settings == Settings(
        model="gpt-5",
        base_url="https://example.test/v1",
        api="responses",
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
                "api": "grpc",  # only chat and responses exist
                "base_url": "",  # empty means unset
            },
        )
    )

    assert settings == Settings(model="kept")


def test_the_file_supplies_argument_defaults():
    settings = Settings(
        model="from-file", base_url="https://file.test", api="responses"
    )

    args = build_parser(settings).parse_args([])

    assert (args.model, args.base_url, args.api) == (
        "from-file",
        "https://file.test",
        "responses",
    )


def test_environment_beats_the_file(tmp_path):
    path = write(tmp_path, {"model": "from-file", "api": "chat"})

    settings = Settings.load(path, env={"LY_MODEL": "from-env", "LY_API": "responses"})

    assert (settings.model, settings.api) == ("from-env", "responses")


def test_the_file_fills_in_what_the_environment_leaves_alone(tmp_path):
    path = write(tmp_path, {"model": "kept", "base_url": "https://file.test"})

    settings = Settings.load(path, env={"LY_MODEL": "from-env"})

    assert settings.model == "from-env"
    assert settings.base_url == "https://file.test"


def test_an_unusable_environment_value_is_ignored(tmp_path):
    path = write(tmp_path, {"api": "responses"})

    # A wrong LY_API falls back rather than reaching the provider, which is what
    # argparse's choices used to enforce for a flag.
    assert Settings.load(path, env={"LY_API": "grpc"}).api == "responses"


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


def test_pass_reasoning_content_is_tri_state():
    assert Settings.load(env={}).pass_reasoning_content is None
    yes = Settings.load(env={"LY_PASS_REASONING_CONTENT": "yes"})
    assert yes.pass_reasoning_content
    assert not Settings.load(
        env={"LY_PASS_REASONING_CONTENT": "off"}
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
        max_summary_input_tokens=30_000,  # untouched keys keep their default
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
        session_manager=SessionManager(in_memory=True),
        system_prompt="system",
        cwd=str(tmp_path),
    )
    repl = Repl(session, settings=Settings(markdown=False, thinking=False))

    assert (repl.markdown, repl.show_thinking) == (False, False)
