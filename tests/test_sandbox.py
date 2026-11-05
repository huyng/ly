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
        "python",
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
