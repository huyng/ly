"""The docker hand-off: the command it builds and when it hands off at all."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from ly import sandbox
from ly.cli import build_parser, main, use_container
from ly.settings import Settings


def options(*argv: str):
    """Parse flags the way `main` does, without running anything."""
    return build_parser(Settings.load()).parse_args(list(argv))


@pytest.fixture(autouse=True)
def _clean_model_environment(monkeypatch):
    """No provider settings from the machine: only what a test sets applies.

    The model names its provider now, so an exported `LY_MODEL` would decide
    whether `main` reaches docker at all.
    """
    for name in ("LY_MODEL", "LY_BASE_URL", "LY_API_FORMAT", "LY_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _must_not_run(*_args) -> None:
    """Stand-in for exec_in_docker: reaching docker from --help is a bug."""
    raise AssertionError("this command line must not start a container")


def test_command_mounts_the_working_directory_and_forwards_argv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    command = sandbox.docker_command(["-p", "hello"], sessions_dir=tmp_path)

    assert f"{Path.cwd()}:/workspace" in command
    assert command[-7:] == [
        sandbox.DEFAULT_IMAGE,
        "sh",
        "-c",
        sandbox.INTERPRETER,
        "ly",
        "-p",
        "hello",
    ]


def test_the_container_falls_back_to_a_plain_python(tmp_path):
    """Not every image that can run ly names its interpreter `python3`."""
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "python").write_text('#!/bin/sh\necho "$@"\n')
    (fake / "python").chmod(0o755)

    result = subprocess.run(
        ["/bin/sh", "-c", sandbox.INTERPRETER, "ly", "-p", "hello"],
        capture_output=True,
        text=True,
        env={"PATH": str(fake)},
    )

    assert result.stdout.split() == ["-P", "-m", "ly", "-p", "hello"]


def test_the_sandbox_flag_names_the_image():
    assert options("-s", "example/ly:1").image == "example/ly:1"
    assert options("--sandbox", "example/ly:1").image == "example/ly:1"


def test_the_image_defaults_to_the_environment_then_the_builtin(monkeypatch):
    monkeypatch.delenv("LY_SANDBOX_IMAGE", raising=False)
    assert options().image == sandbox.DEFAULT_IMAGE

    monkeypatch.setenv("LY_SANDBOX_IMAGE", "python:3.13")
    assert options().image == "python:3.13"
    assert options("-s", "example/ly:1").image == "example/ly:1"


def test_command_uses_the_image_it_is_given(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    command = sandbox.docker_command([], sessions_dir=tmp_path, image="example/ly:1")

    assert command[-5] == "example/ly:1"


def test_command_runs_as_the_host_user_with_a_writable_home(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    command = sandbox.docker_command([], sessions_dir=tmp_path)

    assert command[command.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert f"--env=HOME={sandbox.CONTAINER_HOME}" in command
    # Personal instructions are found through HOME, so they mount there.
    agents = Path.home() / ".agents"
    if agents.is_dir():
        assert f"{agents}:{sandbox.CONTAINER_HOME}/.agents:ro" in command
    assert not any("/root/" in part for part in command)


def test_podman_keeps_the_host_uid_instead_of_passing_one(monkeypatch, tmp_path):
    """Rootless podman maps container root to us; `--user` asks for a subuid."""
    monkeypatch.chdir(tmp_path)

    command = sandbox.docker_command([], sessions_dir=tmp_path, keep_id=True)

    assert "--userns=keep-id" in command
    assert "--user" not in command
    # HOME still has to move: the host home is not mounted.
    assert f"--env=HOME={sandbox.CONTAINER_HOME}" in command


def test_a_podman_version_banner_is_recognised(monkeypatch):
    class Version:
        stdout = "Emulate Docker CLI using podman. Create /etc/containers/nodocker"
        stderr = ""

    monkeypatch.setattr(sandbox.subprocess, "run", lambda *a, **k: Version)
    assert sandbox.runtime_is_podman() is True


@pytest.mark.parametrize("banner", ["Docker version 27.3.1, build ce12230", ""])
def test_docker_is_not_mistaken_for_podman(monkeypatch, banner):
    class Version:
        stdout = banner
        stderr = ""

    monkeypatch.setattr(sandbox.subprocess, "run", lambda *a, **k: Version)
    assert sandbox.runtime_is_podman() is False


def test_a_missing_docker_does_not_crash_the_probe(monkeypatch):
    def missing(*_args, **_kwargs):
        raise FileNotFoundError(2, "No such file or directory", "docker")

    monkeypatch.setattr(sandbox.subprocess, "run", missing)

    assert sandbox.runtime_is_podman() is False


def test_command_drops_capabilities_and_forbids_new_privileges(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    command = sandbox.docker_command([], sessions_dir=tmp_path)

    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command


def test_command_trusts_the_workspace_for_git(monkeypatch, tmp_path):
    """Docker Desktop may report the mount root as root-owned; git must not
    refuse the repo, and the trust must stay scoped to /workspace."""
    monkeypatch.chdir(tmp_path)

    command = sandbox.docker_command([], sessions_dir=tmp_path)

    assert "--env=GIT_CONFIG_COUNT=1" in command
    assert "--env=GIT_CONFIG_KEY_0=safe.directory" in command
    assert "--env=GIT_CONFIG_VALUE_0=/workspace" in command


def test_command_runs_the_local_ly_and_keeps_its_sessions(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    command = sandbox.docker_command([], sessions_dir=tmp_path)

    assert f"{Path(sandbox.ly.__file__).parent}:/opt/ly/ly:ro" in command
    assert "--env=PYTHONPATH=/opt/ly" in command
    assert "--env=LY_SB_SESSIONS_DIR=/sessions" in command
    assert "--env=LY_SB_IN_CONTAINER=1" in command
    assert f"--env=LY_SB_IMAGE={sandbox.DEFAULT_IMAGE}" in command


def test_command_forwards_ly_and_terminal_variables_only(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LY_MODEL", "gpt-4.1")
    monkeypatch.setenv("LY_SANDBOX", "1")
    monkeypatch.setenv("HOME_UNRELATED", "nope")
    monkeypatch.setenv("TERM", "xterm-256color")

    command = sandbox.docker_command([], sessions_dir=tmp_path)

    assert "--env=LY_MODEL=gpt-4.1" in command
    assert "--env=TERM=xterm-256color" in command
    assert "--env=LY_SANDBOX=1" not in command
    assert not any("HOME_UNRELATED" in part for part in command)


def test_host_only_switches_are_not_forwarded(monkeypatch):
    """`LY_SANDBOX` and `LY_SANDBOX_IMAGE` chose what to start; the container
    has no use for them."""
    monkeypatch.setenv("LY_SANDBOX", "1")
    monkeypatch.setenv("LY_SANDBOX_IMAGE", "python:3.13")

    forwarded = sandbox.forwarded_env()

    assert not any("LY_SANDBOX" in part for part in forwarded)


def test_container_facts_are_never_forwarded(monkeypatch):
    """The LY_SB_ prefix means "about this container", so it is set, not passed."""
    monkeypatch.setenv("LY_SB_IMAGE", "stale/image:1")
    monkeypatch.setenv("LY_SB_HOST_DIR", "/stale/path")

    forwarded = sandbox.forwarded_env()

    assert not any("LY_SB_" in part for part in forwarded)


def test_container_side_does_not_spawn_another_container(monkeypatch):
    monkeypatch.setenv("LY_SB_IN_CONTAINER", "1")

    assert sandbox.in_container() is True
    assert use_container(options()) is False


def test_container_reports_the_image_it_was_started_with(monkeypatch):
    monkeypatch.setenv("LY_SB_IMAGE", "docker.io/library/alpine")
    assert sandbox.image_name() == "docker.io/library/alpine"

    monkeypatch.delenv("LY_SB_IMAGE")
    assert sandbox.image_name() == "unknown"


def test_the_host_directory_travels_into_the_container(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LY_SB_HOST_DIR", raising=False)

    command = sandbox.docker_command([], sessions_dir=tmp_path)

    assert f"--env=LY_SB_HOST_DIR={tmp_path}" in command


def test_host_dir_is_what_the_host_sent(monkeypatch):
    monkeypatch.setenv("LY_SB_HOST_DIR", "/Users/someone/ly")
    assert sandbox.host_dir() == "/Users/someone/ly"

    monkeypatch.delenv("LY_SB_HOST_DIR")
    assert sandbox.host_dir() is None


def test_the_host_dir_is_not_forwarded_twice(monkeypatch):
    """`forwarded_env` must not echo it back from a host that had one set."""
    monkeypatch.setenv("LY_SB_HOST_DIR", "/stale/path")

    assert "--env=LY_SB_HOST_DIR=/stale/path" not in sandbox.forwarded_env()


def test_a_provider_key_travels_by_name(monkeypatch):
    """A preset keeps its key under its own variable, and the container cannot
    guess the name, so the caller passes it in."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    monkeypatch.delenv("LY_API_KEY", raising=False)

    forwarded = sandbox.forwarded_env(("DEEPSEEK_API_KEY",))

    assert "--env=DEEPSEEK_API_KEY=sk-deepseek" in forwarded


def test_an_unset_provider_key_is_not_forwarded(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    assert not any(
        "DEEPSEEK" in part for part in sandbox.forwarded_env(("DEEPSEEK_API_KEY",))
    )


def test_hand_off_is_the_default_but_can_be_turned_off(monkeypatch):
    monkeypatch.delenv("LY_SB_IN_CONTAINER", raising=False)
    monkeypatch.delenv("LY_SANDBOX", raising=False)
    assert use_container(options()) is True
    assert use_container(options("--nosandbox")) is False

    monkeypatch.setenv("LY_SANDBOX", "0")
    assert use_container(options()) is False


def test_help_and_version_never_start_a_container(monkeypatch, capsys):
    monkeypatch.delenv("LY_SB_IN_CONTAINER", raising=False)
    monkeypatch.delenv("LY_SANDBOX", raising=False)
    monkeypatch.setattr(sandbox, "exec_in_docker", _must_not_run)

    for flag in ("--version", "--help"):
        with pytest.raises(SystemExit) as exit:
            main([flag])
        assert exit.value.code == 0
    assert "ly" in capsys.readouterr().out


def test_a_real_run_hands_the_whole_command_line_to_docker(monkeypatch):
    handed_off: list[list[str]] = []

    def record(
        argv: list[str],
        *,
        sessions_dir: Path,
        image: str,
        extra_env=(),
        settings_file=None,
    ) -> None:
        handed_off.append([image, *argv])
        raise SystemExit(0)

    monkeypatch.delenv("LY_SB_IN_CONTAINER", raising=False)
    monkeypatch.delenv("LY_SANDBOX", raising=False)
    monkeypatch.delenv("LY_SANDBOX_IMAGE", raising=False)
    monkeypatch.setenv("LY_MODEL", "openai/gpt-4.1")
    monkeypatch.setenv("LY_API_KEY", "sk-test")
    monkeypatch.setattr(sandbox, "exec_in_docker", record)

    with pytest.raises(SystemExit):
        main(["-p", "hello"])

    assert handed_off == [[sandbox.DEFAULT_IMAGE, "-p", "hello"]]


def test_the_sandbox_flag_reaches_docker(monkeypatch):
    handed_off: list[list[str]] = []

    def record(
        argv: list[str],
        *,
        sessions_dir: Path,
        image: str,
        extra_env=(),
        settings_file=None,
    ) -> None:
        handed_off.append([image, *argv])
        raise SystemExit(0)

    monkeypatch.delenv("LY_SB_IN_CONTAINER", raising=False)
    monkeypatch.delenv("LY_SANDBOX", raising=False)
    monkeypatch.delenv("LY_SANDBOX_IMAGE", raising=False)
    monkeypatch.setenv("LY_MODEL", "openai/gpt-4.1")
    monkeypatch.setenv("LY_API_KEY", "sk-test")
    monkeypatch.setattr(sandbox, "exec_in_docker", record)

    with pytest.raises(SystemExit):
        main(["--sandbox", "example/ly:1", "-p", "hello"])

    # The flag is forwarded as well, so the container-side parser still sees
    # the command line it was given.
    assert handed_off == [["example/ly:1", "--sandbox", "example/ly:1", "-p", "hello"]]


# -- the sessions directory the container mounts -----------------------------


def test_a_sessions_directory_that_cannot_be_created_says_so(monkeypatch, tmp_path):
    """The user owns this path, so a refusal is the machine, not a fixable setup."""
    sessions = tmp_path / "sessions"
    monkeypatch.setattr(sandbox.os, "execvp", _must_not_run)

    def refuse(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied", str(sessions))

    monkeypatch.setattr(sandbox.Path, "mkdir", refuse)

    with pytest.raises(SystemExit) as exited:
        sandbox.exec_in_docker([], sessions_dir=sessions)

    message = str(exited.value)
    assert str(sessions) in message
    assert "Permission denied" in message
    # No guess at a cause: the directory is ours to make.
    assert "chown" not in message


def test_a_writable_sessions_directory_hands_off(monkeypatch, tmp_path):
    sessions = tmp_path / "sessions"
    # Never probe the host's docker from a test: it may not exist, and the
    # answer must not change what this test asserts.
    monkeypatch.setattr(sandbox, "runtime_is_podman", lambda: False)
    handed_off: list[list[str]] = []

    def record(command: list[str]) -> None:
        handed_off.append(command)
        raise SystemExit(0)

    monkeypatch.setattr(sandbox.os, "execvp", lambda name, command: record(command))

    with pytest.raises(SystemExit):
        sandbox.exec_in_docker(["-p", "hello"], sessions_dir=sessions)

    assert sessions.is_dir()  # created for the bind mount
    assert handed_off[0][0] == "docker"


def test_the_hand_off_asks_podman_to_keep_the_host_uid(monkeypatch, tmp_path):
    """Regression: a fresh machine on rootless podman could not write /sessions."""
    sessions = tmp_path / "sessions"
    monkeypatch.setattr(sandbox, "runtime_is_podman", lambda: True)
    handed_off: list[list[str]] = []

    def record(command: list[str]) -> None:
        handed_off.append(command)
        raise SystemExit(0)

    monkeypatch.setattr(sandbox.os, "execvp", lambda name, command: record(command))

    with pytest.raises(SystemExit):
        sandbox.exec_in_docker([], sessions_dir=sessions)

    assert "--userns=keep-id" in handed_off[0]
    assert "--user" not in handed_off[0]


# -- provider presets and the hand-off ----------------------------------------


def clean_providers(monkeypatch, tmp_path) -> None:
    """No settings file, and no `LY_*` sandbox setting from the machine."""
    monkeypatch.setattr("ly.settings.DEFAULT_PATH", str(tmp_path / "absent.json"))
    monkeypatch.delenv("LY_SB_IN_CONTAINER", raising=False)
    monkeypatch.delenv("LY_SANDBOX", raising=False)


def test_every_presets_key_variable_is_carried(monkeypatch, tmp_path):
    """`/model` switches provider inside the container, so a key the host did
    not start with still has to be there when the switch happens."""
    clean_providers(monkeypatch, tmp_path)
    monkeypatch.setenv("LY_MODEL", "deepseek/deepseek-chat")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    handed_off: list[tuple[str, ...]] = []

    def record(
        argv: list[str],
        *,
        sessions_dir: Path,
        image: str,
        extra_env=(),
        settings_file=None,
    ) -> None:
        handed_off.append(extra_env)
        raise SystemExit(0)

    monkeypatch.setattr(sandbox, "exec_in_docker", record)

    with pytest.raises(SystemExit):
        main(["-p", "hello"])

    assert handed_off == [("DEEPSEEK_API_KEY", "OPENAI_API_KEY")]


def test_a_config_file_is_mounted_where_the_container_reads_it(tmp_path):
    """`--config` moves the mount source, not the mount point: the container
    reads its default path, and the host path would mean nothing there."""
    path = tmp_path / "custom.json"
    path.write_text("{}")

    command = sandbox.docker_command([], sessions_dir=tmp_path, settings_file=path)

    assert f"{path}:{sandbox.CONTAINER_HOME}/.ly/settings.json:ro" in command


def test_config_is_stripped_from_the_container_command_line(monkeypatch, tmp_path):
    """The file travels as a mount, so forwarding the flag (a host path) would
    only make the container-side parser reject it."""
    clean_providers(monkeypatch, tmp_path)
    path = tmp_path / "custom.json"
    path.write_text(json.dumps({"model": "deepseek/deepseek-chat"}), encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    handed_off: list[tuple[list[str], Path | None]] = []

    def record(
        argv: list[str],
        *,
        sessions_dir: Path,
        image: str,
        extra_env=(),
        settings_file=None,
    ) -> None:
        handed_off.append((argv, settings_file))
        raise SystemExit(0)

    monkeypatch.setattr(sandbox, "exec_in_docker", record)

    with pytest.raises(SystemExit):
        main(["--config", str(path), "-p", "hello"])

    assert handed_off == [(["-p", "hello"], path)]


def test_a_model_without_a_provider_stops_before_docker(monkeypatch, tmp_path):
    clean_providers(monkeypatch, tmp_path)
    monkeypatch.setenv("LY_MODEL", "gpt-4.1")
    monkeypatch.setenv("LY_API_KEY", "sk-test")
    monkeypatch.setattr(sandbox, "exec_in_docker", _must_not_run)

    with pytest.raises(SystemExit) as exited:
        main(["-p", "hello"])

    assert "provider/model" in str(exited.value)
    assert "deepseek/*" in str(exited.value)  # the names it could have meant
