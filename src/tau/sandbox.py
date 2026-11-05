"""Hand the CLI off to a Docker container instead of running on the host.

`tau` re-executes itself at startup: the host process becomes a `docker run`
client, and the agent loop, the tools and the session files all live inside the
container. The working directory is mounted at /workspace, so the tools see the
project and little else of the host.
"""

import os
import sys
from pathlib import Path

import tau

from .session import SESSIONS_DIR

DEFAULT_IMAGE = "python:3.12"
WORKSPACE = "/workspace"
#: HOME inside the container. Whatever the process writes to `~` must be
#: writable by a non-root uid, so it cannot stay at the image's `/root`.
CONTAINER_HOME = "/tmp"

#: Variables this process sets or consumes itself; forwarding them would loop.
NOT_FORWARDED = frozenset(
    {
        "TAU_IN_CONTAINER",
        "TAU_DOCKER",
        "TAU_DOCKER_IMAGE",
        "TAU_HOST_DIR",
        "TAU_IMAGE",
        "TAU_SESSIONS_DIR",
    }
)


def in_container() -> bool:
    """Return True when this is the copy of tau running inside the container."""
    return os.environ.get("TAU_IN_CONTAINER") == "1"


def image_name() -> str:
    """The image this container was started from, as the host named it."""
    return os.environ.get("TAU_IMAGE") or "unknown"


def host_dir() -> str | None:
    """The host directory mounted at the workspace, as the host spelled it.

    Returns:
        The host path, or None when this is not a container (where the
        workspace already is the host directory) or the host did not say.
    """
    return os.environ.get("TAU_HOST_DIR")


def forwarded_env() -> list[str]:
    """Build `--env` flags for the host variables the container-side tau reads.

    Returns:
        `--env=NAME=value` arguments for every `TAU_*` variable the host set,
        plus the terminal variables that decide colour and readline behaviour.
    """
    names = [
        name
        for name in os.environ
        if name.startswith("TAU_") and name not in NOT_FORWARDED
    ]
    names += [
        name for name in ("TERM", "NO_COLOR", "FORCE_COLOR") if name in os.environ
    ]
    return [f"--env={name}={os.environ[name]}" for name in names]


def docker_command(argv: list[str]) -> list[str]:
    """Build the `docker run` command that re-runs tau inside a container.

    Args:
        argv: The host command line, forwarded verbatim after the image name.

    Returns:
        A ready-to-exec argv list starting with `docker`.
    """
    image = os.environ.get("TAU_DOCKER_IMAGE") or DEFAULT_IMAGE
    command = ["docker", "run", "--rm", "-i"]
    # The REPL wants a tty, but docker refuses -t when it has none.
    if sys.stdin.isatty() and sys.stdout.isatty():
        command.append("-t")
    command += ["-v", f"{Path.cwd()}:{WORKSPACE}"]
    command += ["--workdir", WORKSPACE]
    # Run as the host user, or a Linux host finds the project files owned by
    # root afterwards. HOME moves off /root, which that uid cannot write.
    command += [
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        f"--env=HOME={CONTAINER_HOME}",
    ]
    # Nothing inside needs a capability, and a setuid binary must not let one
    # back in. Both are no-ops for normal tooling and shrink the kernel surface.
    command += ["--cap-drop=ALL", "--security-opt=no-new-privileges"]
    # Docker Desktop sometimes reports the mount root as root-owned, which makes
    # git refuse the repo as "dubious ownership". Trust exactly /workspace, via
    # command-line config so no file is written and `git -c` still wins.
    command += [
        "--env=GIT_CONFIG_COUNT=1",
        f"--env=GIT_CONFIG_KEY_0=safe.directory",
        f"--env=GIT_CONFIG_VALUE_0={WORKSPACE}",
    ]
    # Sessions sit outside the project so /resume still finds them next run.
    command += ["-v", f"{SESSIONS_DIR}:/sessions", "--env=TAU_SESSIONS_DIR=/sessions"]
    # Personal instructions, or the system prompt quietly loses them. They
    # belong under HOME, because that is where the container-side tau looks.
    user_agents = Path.home() / ".agents"
    if user_agents.is_dir():
        command += ["-v", f"{user_agents}:{CONTAINER_HOME}/.agents:ro"]
    # Run the tau that is on the host, not a copy baked into an image.
    package = Path(tau.__file__).parent
    command += ["-v", f"{package}:/opt/tau/tau:ro", "--env=PYTHONPATH=/opt/tau"]
    command += forwarded_env()
    # The container sees /workspace; only this tells the banner which host
    # directory that is, and the banner is how the mount is made visible.
    command += [f"--env=TAU_HOST_DIR={Path.cwd()}"]
    command += ["--env=TAU_IN_CONTAINER=1", f"--env=TAU_IMAGE={image}", image]
    command += ["python3", "-P", "-m", "tau"]
    return command + list(argv)


def exec_in_docker(argv: list[str]) -> None:
    """Replace this process with `docker run` running tau in the container.

    Args:
        argv: The host command line, forwarded verbatim.

    Raises:
        SystemExit: docker is not installed on the host.
    """
    Path(SESSIONS_DIR).mkdir(parents=True, exist_ok=True)
    command = docker_command(argv)
    try:
        os.execvp(command[0], command)
    except FileNotFoundError:
        raise SystemExit(
            "tau: docker not found; install Docker or run with --nosandbox"
        ) from None
