"""The docker hand-off: the command it builds and when it hands off at all."""

import os
from pathlib import Path

import pytest

from tau import sandbox
from tau.cli import build_parser, main, use_container


def options(*argv: str):
    """Parse flags the way `main` does, without running anything."""
    return build_parser().parse_args(list(argv))


def _must_not_run(*_args) -> None:
    """Stand-in for exec_in_docker: reaching docker from --help is a bug."""
    raise AssertionError("this command line must not start a container")


def test_command_mounts_the_working_directory_and_forwards_argv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    command = sandbox.docker_command(["-p", "hello"])

    assert f"{Path.cwd()}:/workspace" in command
    assert command[-7:] == [
        sandbox.DEFAULT_IMAGE,
        "python3",
        "-P",
        "-m",
        "tau",
        "-p",
        "hello",
    ]


def test_command_runs_as_the_host_user_with_a_writable_home(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    command = sandbox.docker_command([])

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

    command = sandbox.docker_command([], keep_id=True)

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

    command = sandbox.docker_command([])

    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command


def test_command_trusts_the_workspace_for_git(monkeypatch, tmp_path):
    """Docker Desktop may report the mount root as root-owned; git must not
    refuse the repo, and the trust must stay scoped to /workspace."""
    monkeypatch.chdir(tmp_path)

    command = sandbox.docker_command([])

    assert "--env=GIT_CONFIG_COUNT=1" in command
    assert "--env=GIT_CONFIG_KEY_0=safe.directory" in command
    assert "--env=GIT_CONFIG_VALUE_0=/workspace" in command


def test_command_runs_the_local_tau_and_keeps_its_sessions(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    command = sandbox.docker_command([])

    assert f"{Path(sandbox.tau.__file__).parent}:/opt/tau/tau:ro" in command
    assert "--env=PYTHONPATH=/opt/tau" in command
    assert "--env=TAU_SESSIONS_DIR=/sessions" in command
    assert "--env=TAU_IN_CONTAINER=1" in command
    assert f"--env=TAU_IMAGE={sandbox.DEFAULT_IMAGE}" in command


def test_command_forwards_tau_and_terminal_variables_only(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TAU_MODEL", "gpt-4.1")
    monkeypatch.setenv("TAU_DOCKER", "1")
    monkeypatch.setenv("HOME_UNRELATED", "nope")
    monkeypatch.setenv("TERM", "xterm-256color")

    command = sandbox.docker_command([])

    assert "--env=TAU_MODEL=gpt-4.1" in command
    assert "--env=TERM=xterm-256color" in command
    assert "--env=TAU_DOCKER=1" not in command
    assert not any("HOME_UNRELATED" in part for part in command)


def test_container_side_does_not_spawn_another_container(monkeypatch):
    monkeypatch.setenv("TAU_IN_CONTAINER", "1")

    assert sandbox.in_container() is True
    assert use_container(options()) is False


def test_container_reports_the_image_it_was_started_with(monkeypatch):
    monkeypatch.setenv("TAU_IMAGE", "docker.io/library/alpine")
    assert sandbox.image_name() == "docker.io/library/alpine"

    monkeypatch.delenv("TAU_IMAGE")
    assert sandbox.image_name() == "unknown"


def test_the_host_directory_travels_into_the_container(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TAU_HOST_DIR", raising=False)

    command = sandbox.docker_command([])

    assert f"--env=TAU_HOST_DIR={tmp_path}" in command


def test_host_dir_is_what_the_host_sent(monkeypatch):
    monkeypatch.setenv("TAU_HOST_DIR", "/Users/someone/tau")
    assert sandbox.host_dir() == "/Users/someone/tau"

    monkeypatch.delenv("TAU_HOST_DIR")
    assert sandbox.host_dir() is None


def test_the_host_dir_is_not_forwarded_twice(monkeypatch):
    """`forwarded_env` must not echo it back from a host that had one set."""
    monkeypatch.setenv("TAU_HOST_DIR", "/stale/path")

    assert "--env=TAU_HOST_DIR=/stale/path" not in sandbox.forwarded_env()


def test_hand_off_is_the_default_but_can_be_turned_off(monkeypatch):
    monkeypatch.delenv("TAU_IN_CONTAINER", raising=False)
    monkeypatch.delenv("TAU_DOCKER", raising=False)
    assert use_container(options()) is True
    assert use_container(options("--nosandbox")) is False

    monkeypatch.setenv("TAU_DOCKER", "0")
    assert use_container(options()) is False


def test_help_and_version_never_start_a_container(monkeypatch, capsys):
    monkeypatch.delenv("TAU_IN_CONTAINER", raising=False)
    monkeypatch.delenv("TAU_DOCKER", raising=False)
    monkeypatch.setattr(sandbox, "exec_in_docker", _must_not_run)

    for flag in ("--version", "--help"):
        with pytest.raises(SystemExit) as exit:
            main([flag])
        assert exit.value.code == 0
    assert "tau" in capsys.readouterr().out


def test_a_real_run_hands_the_whole_command_line_to_docker(monkeypatch):
    handed_off: list[list[str]] = []

    def record(argv: list[str]) -> None:
        handed_off.append(argv)
        raise SystemExit(0)

    monkeypatch.delenv("TAU_IN_CONTAINER", raising=False)
    monkeypatch.delenv("TAU_DOCKER", raising=False)
    monkeypatch.setattr(sandbox, "exec_in_docker", record)

    with pytest.raises(SystemExit):
        main(["-p", "hello"])

    assert handed_off == [["-p", "hello"]]


# -- the sessions directory the container mounts -----------------------------


def test_a_sessions_directory_that_cannot_be_created_says_so(monkeypatch, tmp_path):
    """The user owns this path, so a refusal is the machine, not a fixable setup."""
    sessions = tmp_path / "sessions"
    monkeypatch.setattr(sandbox, "SESSIONS_DIR", sessions)
    monkeypatch.setattr(sandbox.os, "execvp", _must_not_run)

    def refuse(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied", str(sessions))

    monkeypatch.setattr(sandbox.Path, "mkdir", refuse)

    with pytest.raises(SystemExit) as exited:
        sandbox.exec_in_docker([])

    message = str(exited.value)
    assert str(sessions) in message
    assert "Permission denied" in message
    # No guess at a cause: the directory is ours to make.
    assert "chown" not in message


def test_a_writable_sessions_directory_hands_off(monkeypatch, tmp_path):
    sessions = tmp_path / "sessions"
    monkeypatch.setattr(sandbox, "SESSIONS_DIR", sessions)
    # Never probe the host's docker from a test: it may not exist, and the
    # answer must not change what this test asserts.
    monkeypatch.setattr(sandbox, "runtime_is_podman", lambda: False)
    handed_off: list[list[str]] = []

    def record(command: list[str]) -> None:
        handed_off.append(command)
        raise SystemExit(0)

    monkeypatch.setattr(sandbox.os, "execvp", lambda name, command: record(command))

    with pytest.raises(SystemExit):
        sandbox.exec_in_docker(["-p", "hello"])

    assert sessions.is_dir()  # created for the bind mount
    assert handed_off[0][0] == "docker"


def test_the_hand_off_asks_podman_to_keep_the_host_uid(monkeypatch, tmp_path):
    """Regression: a fresh machine on rootless podman could not write /sessions."""
    monkeypatch.setattr(sandbox, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(sandbox, "runtime_is_podman", lambda: True)
    handed_off: list[list[str]] = []

    def record(command: list[str]) -> None:
        handed_off.append(command)
        raise SystemExit(0)

    monkeypatch.setattr(sandbox.os, "execvp", lambda name, command: record(command))

    with pytest.raises(SystemExit):
        sandbox.exec_in_docker([])

    assert "--userns=keep-id" in handed_off[0]
    assert "--user" not in handed_off[0]
