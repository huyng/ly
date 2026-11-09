"""Hand the CLI off to a Docker container instead of running on the host.

`ly` re-executes itself at startup: the host process becomes a `docker run`
client, and the agent loop, the tools and the session files all live inside the
container. The working directory is mounted at /workspace, so the tools see the
project and little else of the host.
"""

import os
import subprocess
import sys
from pathlib import Path

import ly

from .session import SESSIONS_DIR
from .settings import DEFAULT_PATH

DEFAULT_IMAGE = "python:3.12"
WORKSPACE = "/workspace"
#: How the container starts ly. Images that can run the agent do not all spell
#: the interpreter `python3`, so pick whichever exists; `-P` keeps /workspace
#: off sys.path, so a file in the project cannot shadow ly or the stdlib.
INTERPRETER = (
    'if command -v python3 >/dev/null 2>&1; then exec python3 -P -m ly "$@"; '
    'else exec python -P -m ly "$@"; fi'
)
#: HOME inside the container. Whatever the process writes to `~` must be
#: writable by a non-root uid, so it cannot stay at the image's `/root`.
CONTAINER_HOME = "/tmp"

#: Prefix for the facts a container knows about itself. Anything else under
#: `LY_` is a setting, which travels both ways; these are set by the host for
#: this run and never forwarded, so a stale value cannot describe the wrong box.
SANDBOX_PREFIX = "LY_SB_"

#: Host-only switches: they choose what to start, so the container has no use for
#: them. `LY_SB_` covers the rest of the variables that stay behind.
NOT_FORWARDED = frozenset({"LY_DOCKER", "LY_DOCKER_IMAGE"})


def in_container() -> bool:
    """Return True when this is the copy of ly running inside the container."""
    return os.environ.get(SANDBOX_PREFIX + "IN_CONTAINER") == "1"


def image_name() -> str:
    """The image this container was started from, as the host named it."""
    return os.environ.get(SANDBOX_PREFIX + "IMAGE") or "unknown"


def host_dir() -> str | None:
    """The host directory mounted at the workspace, as the host spelled it.

    Returns:
        The host path, or None when this is not a container (where the
        workspace already is the host directory) or the host did not say.
    """
    return os.environ.get(SANDBOX_PREFIX + "HOST_DIR")


def forwarded_env() -> list[str]:
    """Build `--env` flags for the host settings the container-side ly reads.

    Every `LY_*` variable is a setting and travels, except the `LY_SB_*` ones,
    which describe a container and are set for it rather than inherited, and the
    host-only switches in `NOT_FORWARDED`, which chose what to start.

    Returns:
        `--env=NAME=value` arguments for the `LY_*` variables the host set,
        plus the terminal variables that decide colour and readline behaviour.
    """
    names = [
        name
        for name in os.environ
        if name.startswith("LY_")
        and not name.startswith(SANDBOX_PREFIX)
        and name not in NOT_FORWARDED
    ]
    names += [
        name for name in ("TERM", "NO_COLOR", "FORCE_COLOR") if name in os.environ
    ]
    return [f"--env={name}={os.environ[name]}" for name in names]


def runtime_is_podman() -> bool:
    """Whether `docker` on PATH is podman's emulation shim.

    Returns:
        True when `docker --version` names podman, False for real Docker or
        when the command cannot be run at all. The hand-off reports a missing
        docker itself, so a failed probe is not an error here.
    """
    try:
        result = subprocess.run(
            ["docker", "--version"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return "podman" in (result.stdout + result.stderr).lower()


def docker_command(
    argv: list[str], *, image: str = DEFAULT_IMAGE, keep_id: bool = False
) -> list[str]:
    """Build the `docker run` command that re-runs ly inside a container.

    Args:
        argv: The host command line, forwarded verbatim after the image name.
        image: The image to start. Callers take it from the `--sandbox` flag
            or `LY_DOCKER_IMAGE`, so this function never reads the environment.
        keep_id: Map the container process to this uid instead of passing an
            explicit `--user`. Needed for podman, whose user namespace maps
            container root to the host user and everything else to a subuid.

    Returns:
        A ready-to-exec argv list starting with `docker`.
    """
    command = ["docker", "run", "--rm", "-i"]
    # The REPL wants a tty, but docker refuses -t when it has none.
    if sys.stdin.isatty() and sys.stdout.isatty():
        command.append("-t")
    command += ["-v", f"{Path.cwd()}:{WORKSPACE}"]
    command += ["--workdir", WORKSPACE]
    # Run as the host user, or a Linux host finds the project files owned by
    # root afterwards. HOME moves off /root, which that uid cannot write.
    if keep_id:
        # podman's namespace already maps our uid onto the container, so mounts
        # show up owned by us; `--user` would ask for a subuid that owns nothing.
        command += ["--userns=keep-id"]
    else:
        command += ["--user", f"{os.getuid()}:{os.getgid()}"]
    command += [f"--env=HOME={CONTAINER_HOME}"]
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
    command += [
        "-v",
        f"{SESSIONS_DIR}:/sessions",
        f"--env={SANDBOX_PREFIX}SESSIONS_DIR=/sessions",
    ]
    # Personal instructions, or the system prompt quietly loses them. They
    # belong under HOME, because that is where the container-side ly looks.
    user_agents = Path.home() / ".agents"
    if user_agents.is_dir():
        command += ["-v", f"{user_agents}:{CONTAINER_HOME}/.agents:ro"]
    # The settings file is read through HOME too. Without it the host applies
    # the user's preferences and the container silently falls back to defaults.
    settings_file = Path(DEFAULT_PATH).expanduser()
    if settings_file.is_file():
        command += ["-v", f"{settings_file}:{CONTAINER_HOME}/.ly/settings.json:ro"]
    # Run the ly that is on the host, not a copy baked into an image.
    package = Path(ly.__file__).parent
    command += ["-v", f"{package}:/opt/ly/ly:ro", "--env=PYTHONPATH=/opt/ly"]
    command += forwarded_env()
    # The container sees /workspace; only this tells the banner which host
    # directory that is, and the banner is how the mount is made visible.
    command += [f"--env={SANDBOX_PREFIX}HOST_DIR={Path.cwd()}"]
    command += [
        f"--env={SANDBOX_PREFIX}IN_CONTAINER=1",
        f"--env={SANDBOX_PREFIX}IMAGE={image}",
        image,
    ]
    # `sh -c` consumes a name for $0 before the host arguments reach "$@".
    command += ["sh", "-c", INTERPRETER, "ly"]
    return command + list(argv)


def exec_in_docker(argv: list[str], *, image: str = DEFAULT_IMAGE) -> None:
    """Replace this process with `docker run` running ly in the container.

    Args:
        argv: The host command line, forwarded verbatim.
        image: The image to start, already resolved by the CLI.

    Raises:
        SystemExit: docker is not installed, or the session directory cannot
            be created.
    """
    try:
        Path(SESSIONS_DIR).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        # Nothing here diagnoses why: this directory is ours to create, so a
        # refusal means the filesystem or $HOME is in a state we cannot guess.
        raise SystemExit(f"ly: cannot create {SESSIONS_DIR}: {exc}") from exc
    command = docker_command(argv, image=image, keep_id=runtime_is_podman())
    try:
        os.execvp(command[0], command)
    except FileNotFoundError:
        raise SystemExit(
            "ly: docker not found; install Docker or run with --nosandbox"
        ) from None
