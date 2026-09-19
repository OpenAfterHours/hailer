"""The docker kernel runtime: marimo in a Linux container that sees only the notebooks folder
(read-write) and the data folder (read-only), with no network and none of the host's secrets.

Everything here drives the ``docker`` CLI through a :class:`DockerRunner` (no Docker SDK), so the
tests check argument lists against a scripted fake. It fails closed: every check raises a
:class:`~hailer.errors.KernelRuntimeError` with a hint and nothing ever falls back to the local
runtime. The shared pieces (``kernel.json``, :class:`~hailer.kernel.PathMap`, the start guard)
live in :mod:`hailer.kernel`; ``runtime_for`` imports this module only when docker is asked for.

Containers and networks are removed by the ids Docker gave them when they were created (recorded
in ``kernel.json``), never by name: an old handle (a chat started before a restart) then cannot
remove a newer kernel that reuses the names. A start never removes a *running* kernel container
it has no record of; ``uvx hailer kernel stop`` does, on request.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from hailer import __version__, kernel_image
from hailer.errors import HailerError, KernelRuntimeError
from hailer.kernel import (
    KERNEL_DATA_DIR,
    KERNEL_NOTEBOOKS_DIR,
    KERNEL_WORKDIR,
    START_TIMEOUT_SEC,
    KernelState,
    LocalProcesses,
    LocalRuntime,
    Probe,
    RunningKernel,
    _host_parts,
    _same_parts,
    delete_kernel_state,
    describe_runtime,
    docker_paths,
    live_kernel_state,
    new_token,
    note_kernel_start,
    read_kernel_state,
    refuse_live_kernel,
    write_kernel_state,
)
from hailer.marimo_client import answers_with_token, wait_for_health
from hailer.models import KERNEL_RUNTIME_DOCKER, Check, HailerConfig, KernelConfig, MarimoServer

#: marimo's port inside the kernel container; the forwarder listens on the same port in its own.
KERNEL_PORT = 2718
#: The image's user (``analyst``, uid and gid 1000) and its home, a tmpfs owned by the kernel's user.
IMAGE_UID = 1000
IMAGE_GID = 1000
KERNEL_HOME = "/home/analyst"
#: Labels on every container and network Hailer creates, so it finds its own (and strays).
LABEL_PREFIX = "org.openafterhours.hailer"
LABEL_WORKSPACE = f"{LABEL_PREFIX}.workspace"
LABEL_VERSION = f"{LABEL_PREFIX}.version"
LABEL_ROLE = f"{LABEL_PREFIX}.role"
#: How long one docker command may take; ``docker version`` gets less (an engine that is still
#: starting can hang instead of refusing).
DOCKER_TIMEOUT_SEC = 60.0
DOCKER_PROBE_TIMEOUT_SEC = 20.0
#: ``GetDriveTypeW``'s answer for a network drive.
DRIVE_REMOTE = 4
#: How many entries of the data folder the link check looks at before giving up.
LINK_SCAN_LIMIT = 5000
#: How many of docker's error lines a streamed command keeps for the message that explains it.
KEPT_ERROR_LINES = 20
DOCKER_NOT_INSTALLED = "Docker is not installed."
DOCKER_NOT_RUNNING = "Docker is not running."
#: What docker says when an image cannot be downloaded because the registry does not have it (or
#: will not show it to this user): the release that publishes it has not happened yet.
_UNPUBLISHED_SIGNS = ("denied", "not found", "manifest unknown", "unauthorized", "does not exist")
#: What docker says about an object that is not there (a removal that has nothing left to do).
_MISSING_SIGNS = ("no such container", "no such network", "no such object", "not found")
#: Container states that mean "running" for the leftover rule (a paused kernel still holds its work).
_RUNNING_STATES = ("running", "restarting", "paused", "removing")


class DockerRunner(Protocol):
    """Runs the ``docker`` CLI for :class:`DockerRuntime` and :mod:`hailer.kernel_image`; tests pass
    a fake to check the argv. ``args`` never include ``docker`` itself. Both methods raise
    ``KernelRuntimeError`` (:func:`docker_not_installed`) when there is no ``docker`` to run."""

    def run(self, args: Sequence[str], *, timeout: float | None = None, check: bool = False) -> subprocess.CompletedProcess[str]:
        """Run to completion with the output captured (text). ``check`` turns a non-zero exit into
        a ``KernelRuntimeError`` quoting docker; ``subprocess.TimeoutExpired`` after ``timeout``."""
        ...

    def stream(self, args: Sequence[str], *, keep_errors: bool = False) -> subprocess.CompletedProcess[str]:
        """Run with this terminal's output (pull, build, ``logs -f``); the result's ``returncode``
        is docker's exit code. ``keep_errors`` also keeps docker's last error lines in ``stderr``
        (they are still shown as they come), so a failure can be explained."""
        ...


def docker_not_installed() -> KernelRuntimeError:
    return KernelRuntimeError(
        DOCKER_NOT_INSTALLED,
        hint=(
            "Install Docker Desktop (Windows, macOS) or Docker Engine (Linux), then run the command again. "
            'Or set [kernel] runtime = "local" in hailer.toml to run notebook code on this machine without isolation.'
        ),
    )


def docker_not_running(detail: str = "") -> KernelRuntimeError:
    hint = "Start Docker Desktop (or the Docker service), wait until it is running, and run the command again."
    if detail:
        hint += f"\n{detail}"
    return KernelRuntimeError(DOCKER_NOT_RUNNING, hint=hint)


def docker_said(result: subprocess.CompletedProcess[str]) -> str:
    """The last lines docker printed, for a hint ("" when it printed nothing)."""
    text = "\n".join(part for part in (result.stderr, result.stdout) if part)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return ("docker said: " + " / ".join(lines[-3:])) if lines else ""


def _missing(result: subprocess.CompletedProcess[str]) -> bool:
    """True when docker failed because the object is not there (nothing left to remove)."""
    text = f"{result.stderr or ''}\n{result.stdout or ''}".lower()
    return any(sign in text for sign in _MISSING_SIGNS)


def _say(text: str) -> None:
    print(text, flush=True)


def _end_process(proc: Any) -> None:
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:  # noqa: BLE001 - best effort
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


class SubprocessDockerRunner:
    """The ``docker`` CLI found on PATH, run with :mod:`subprocess` (text mode, UTF-8, empty stdin).
    Docker Desktop and Docker Engine both ship it: no SDK needed."""

    def __init__(self, executable: str | None = None) -> None:
        self._executable = executable

    def _argv(self, args: Sequence[str]) -> list[str]:
        executable = self._executable or shutil.which("docker")
        if not executable:
            raise docker_not_installed()
        return [executable, *args]

    def run(self, args: Sequence[str], *, timeout: float | None = None, check: bool = False) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(
                self._argv(args),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as err:
            raise docker_not_installed() from err
        if check and result.returncode != 0:
            raise KernelRuntimeError(f"docker {' '.join(args[:2])} failed (exit code {result.returncode}).", hint=docker_said(result))
        return result

    def stream(self, args: Sequence[str], *, keep_errors: bool = False) -> subprocess.CompletedProcess[str]:
        argv = self._argv(args)
        try:
            proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stderr=subprocess.PIPE if keep_errors else None)
        except FileNotFoundError as err:
            raise docker_not_installed() from err
        kept: list[str] = []
        reader: threading.Thread | None = None
        if keep_errors:

            def pump() -> None:
                for raw in iter(proc.stderr.readline, b""):
                    text = raw.decode("utf-8", "replace")
                    try:
                        sys.stderr.write(text)
                        sys.stderr.flush()
                    except (OSError, ValueError):
                        pass
                    kept.append(text.rstrip("\r\n"))
                    del kept[:-KEPT_ERROR_LINES]

            reader = threading.Thread(target=pump, name="hailer-docker-stderr", daemon=True)
            reader.start()
        try:
            while True:
                try:
                    code = int(proc.wait(timeout=0.5))  # a short timeout keeps Ctrl+C deliverable on Windows
                    break
                except subprocess.TimeoutExpired:
                    continue
        except KeyboardInterrupt:
            _end_process(proc)
            raise
        if reader is not None:
            reader.join(timeout=5)
        return subprocess.CompletedProcess(argv, code, None, "\n".join(kept))


# --------------------------------------------------------------------------- #
# Names, mounts and the folders Docker can (not) see
# --------------------------------------------------------------------------- #


def workspace_id(workspace: Path) -> str:
    """The short id in the container and network names: 10 hex chars of the workspace path's hash."""
    key = os.path.normcase(str(Path(workspace).resolve()))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]


@dataclass(frozen=True)
class DockerNames:
    """The names of one workspace's kernel container, forwarder container and internal network."""

    kernel: str
    forwarder: str
    network: str


def docker_names(workspace: Path) -> DockerNames:
    ident = workspace_id(workspace)
    return DockerNames(f"hailer-kernel-{ident}", f"hailer-fwd-{ident}", f"hailer-net-{ident}")


def _csv_field(value: str) -> str:
    """One field of a ``--mount`` value, which docker parses as CSV."""
    if "," in value or '"' in value:
        return '"' + value.replace('"', '""') + '"'
    return value


def mount_arg(source: Path | str, target: PurePosixPath | str, *, readonly: bool = False) -> str:
    """A ``--mount`` value binding the host folder ``source`` at ``target`` (``--mount`` fails on a
    missing source instead of creating it, unlike ``-v``)."""
    fields = ["type=bind", f"source={source}", f"target={target}", *(["readonly"] if readonly else [])]
    return ",".join(_csv_field(item) for item in fields)


def linux_host_user() -> tuple[int, int] | None:
    """The uid and gid the kernel runs as on a Linux host, so the files marimo saves stay the
    user's; ``None`` (the image's own user) elsewhere, where Docker Desktop maps ownership itself,
    and for root, since the kernel never runs as root."""
    if not sys.platform.startswith("linux"):
        return None
    uid, gid = os.getuid(), os.getgid()  # type: ignore[attr-defined]  # POSIX only
    return None if uid == 0 else (uid, gid)


def _drive_type(root: str) -> int:
    """``GetDriveTypeW(root)`` on Windows (``4`` = network drive); ``0`` when it cannot be asked."""
    if os.name != "nt":
        return 0
    import ctypes

    try:
        return int(ctypes.windll.kernel32.GetDriveTypeW(ctypes.c_wchar_p(root)))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return 0


UNC_PATH = "a network share (UNC path)"


def _on_windows() -> bool:
    return os.name == "nt"


def network_location(path: Path | str, drive_type: Callable[[str], int] = _drive_type) -> str | None:
    """Why the Windows path ``path`` is on the network (:data:`UNC_PATH`, or a mapped network
    drive); ``None`` otherwise."""
    import ntpath

    drive = ntpath.splitdrive(str(path))[0].replace("/", "\\")
    if drive.upper().startswith(("\\\\?\\", "\\\\.\\")):
        drive = drive[4:]
        if drive.upper().startswith("UNC\\"):
            return UNC_PATH
    elif drive.startswith("\\\\"):
        return UNC_PATH
    if len(drive) == 2 and drive[1] == ":" and drive_type(drive + "\\") == DRIVE_REMOTE:
        return f"a mapped network drive ({drive.upper()})"
    return None


def _links_outside(folder: Path, limit: int = LINK_SCAN_LIMIT) -> list[str]:
    """Symlinks and junctions in ``folder`` whose target is outside it (never followed)."""
    try:
        root = os.path.normcase(os.path.realpath(folder))
    except OSError:
        return []
    found: list[str] = []
    seen = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for name in [*dirnames, *filenames]:
            seen += 1
            if seen > limit:
                return found
            path = Path(dirpath) / name
            try:
                linked = path.is_symlink() or path.is_junction()
                target = os.path.normcase(os.path.realpath(path)) if linked else ""
            except OSError:
                continue
            if not linked:
                continue
            if name in dirnames:
                dirnames.remove(name)  # a link is reported, never walked through
            if target != root and not target.startswith(root.rstrip(os.sep) + os.sep):
                found.append(path.relative_to(folder).as_posix())
    return found


def folder_problems(
    folder: Path, what: str, setting: str, *, windows: bool | None = None, drive_type: Callable[[str], int] = _drive_type, links: bool = False
) -> list[str]:
    """Reasons the kernel container may not see ``folder`` (the ``what`` folder, set by
    ``setting``), each with its fix (empty: none).

    On Windows: a UNC path or a mapped network drive, which Docker Desktop usually cannot mount.
    With ``links``: symlinks or junctions inside the folder that point outside it, which do not
    resolve in the container (only the folder itself is mounted).
    """
    text = str(folder)
    if windows if windows is not None else _on_windows():
        where = network_location(text, drive_type)
        if where is not None:  # no link scan: it would walk the share over the network, and the fix is the same
            return [
                f"The {what} folder {text} is on {where}; Docker Desktop usually cannot mount it. "
                f"Copy it to a folder on a local disk and point {setting} at it."
            ]
    found = _links_outside(Path(folder)) if links else []
    if not found:
        return []
    shown = ", ".join(found[:3]) + (f" and {len(found) - 3} more" if len(found) > 3 else "")
    return [
        f"{len(found)} link(s) in the {what} folder point outside it ({shown}); they do not resolve in the "
        f"container. Copy those files into the {what} folder."
    ]


def data_path_problems(data_dir: Path, *, windows: bool | None = None, drive_type: Callable[[str], int] = _drive_type) -> list[str]:
    """:func:`folder_problems` for the data folder, links included."""
    return folder_problems(data_dir, "data", "[hailer].data_dir", windows=windows, drive_type=drive_type, links=True)


# --------------------------------------------------------------------------- #
# What a docker record says about the kernel
# --------------------------------------------------------------------------- #


def containers_gone(state: KernelState, runner: DockerRunner | None = None) -> bool:
    """True when Docker says the kernel container ``state`` records is gone or stopped (a record
    without container ids is not usable either). False whenever Docker cannot be asked."""
    if not state.container_ids:
        return True
    try:
        result = (runner or SubprocessDockerRunner()).run(
            ["inspect", "--type", "container", "--format", "{{.State.Running}}", state.container_ids[0]],
            timeout=DOCKER_PROBE_TIMEOUT_SEC,
        )
    except (HailerError, OSError, subprocess.TimeoutExpired):
        return False
    if result.returncode != 0:
        return _missing(result)
    return (result.stdout or "").strip() != "true"


def _on_off(value: bool) -> str:
    return "on" if value else "off"


def settings_mismatch(state: KernelState, config: HailerConfig) -> list[str]:
    """How the docker kernel ``state`` records was started differently from what ``config`` asks
    for now, one phrase per setting (empty: it matches). The folders always count (the kernel
    sees only what it mounted); image, network, memory and cpus only when ``config`` asks for docker."""
    diffs: list[str] = []
    recorded = {kernel: host for host, kernel in state.mounts}
    for host, kernel_dir, what in (
        (config.notebooks_root, KERNEL_NOTEBOOKS_DIR, "notebooks folder"),
        (config.data_dir, KERNEL_DATA_DIR, "data folder"),
    ):
        started = recorded.get(str(kernel_dir))
        if started is None:
            continue
        a, b = _host_parts(started), _host_parts(host)
        if len(a) != len(b) or not _same_parts(a, b):
            diffs.append(f"{what} {started}, hailer.toml: {host}")
    kernel = config.kernel
    if kernel.runtime != KERNEL_RUNTIME_DOCKER:
        return diffs
    if state.network_access != kernel.network:
        diffs.append(f"network {_on_off(state.network_access)}, hailer.toml: {_on_off(kernel.network)}")
    if state.image and state.image != kernel.effective_image:
        diffs.append(f"image {state.image}, hailer.toml: {kernel.effective_image}")
    if state.memory is not None and state.memory.strip().lower() != kernel.memory.strip().lower():
        diffs.append(f"memory {state.memory}, hailer.toml: {kernel.memory}")
    if state.cpus is not None and state.cpus != kernel.cpus:
        diffs.append(f"cpus {state.cpus:g}, hailer.toml: {kernel.cpus:g}")
    return diffs


def mismatch_error(diffs: Sequence[str]) -> KernelRuntimeError:
    """``KernelRuntimeError`` for reusing a kernel started with other settings (see :func:`settings_mismatch`)."""
    return KernelRuntimeError(
        f"The running kernel was started with other settings ({'; '.join(diffs)}).",
        hint="Stop it with uvx hailer kernel stop, then run this command again to start one with the current settings.",
    )


# --------------------------------------------------------------------------- #
# A running docker kernel
# --------------------------------------------------------------------------- #


@dataclass
class Removal:
    """What removing containers and networks did: removed, already gone, or failed (with why)."""

    removed: list[str] = field(default_factory=list)
    gone: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    def add(self, other: Removal) -> None:
        self.removed += other.removed
        self.gone += other.gone
        self.failed += other.failed


def _remove(runner: DockerRunner, kind: str, ident: str, shown: str) -> Removal:
    """Remove one container (``kind="container"``) or network by id; "not there" counts as gone."""
    args = ["rm", "-f", ident] if kind == "container" else ["network", "rm", ident]
    try:
        result = runner.run(args, timeout=DOCKER_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        return Removal(failed=[f"{shown}: docker did not answer within {int(DOCKER_TIMEOUT_SEC)} s"])
    if result.returncode == 0:
        # `docker rm -f` exits 0 for a container that is not there too; it echoes the ones it removed.
        if kind == "container" and ident not in (result.stdout or ""):
            return Removal(gone=[shown])
        return Removal(removed=[shown])
    if _missing(result):
        return Removal(gone=[shown])
    return Removal(failed=[f"{shown}: {docker_said(result) or f'exit code {result.returncode}'}"])


@dataclass
class DockerKernel(RunningKernel):
    """A docker kernel's containers and network: the ones this Hailer started, or found through
    ``kernel.json``. ``containers`` are names (kernel first), ``container_ids`` the same containers'
    ids; everything is removed and inspected by id."""

    runner: Any = None
    containers: tuple[str, ...] = ()
    container_ids: tuple[str, ...] = ()
    network: str | None = None
    network_id: str | None = None

    def _name(self, ident: str) -> str:
        for name, known in zip(self.containers, self.container_ids):
            if known == ident:
                return name
        return ident[:12]

    def remove(self) -> Removal:
        """Remove this kernel's containers, then its network, by id. Raises only what the runner
        raises (Docker missing)."""
        outcome = Removal()
        for ident in self.container_ids:
            outcome.add(_remove(self.runner, "container", ident, self._name(ident)))
        if self.network_id:
            outcome.add(_remove(self.runner, "network", self.network_id, self.network or self.network_id[:12]))
        return outcome

    def stop(self) -> None:
        try:
            self.remove()
        except Exception:  # noqa: BLE001 - best effort on the way out
            pass
        super().stop()

    def log_tail(self, lines: int = 15, *, container: str | None = None) -> list[str]:
        """The last lines of ``docker logs`` for the kernel (or the container id ``container``), token masked."""
        ident = container or (self.container_ids[0] if self.container_ids else None)
        if ident is None:
            return []
        try:
            result = self.runner.run(["logs", "--tail", str(lines), ident], timeout=DOCKER_PROBE_TIMEOUT_SEC)
        except Exception:  # noqa: BLE001 - the log is a courtesy
            return []
        text = "\n".join(part for part in (result.stdout, result.stderr) if part)
        token = self.server.token
        return [line.replace(token, "<token>") if token else line for line in text.splitlines()][-lines:]

    def wait(self) -> int:
        """Follow the kernel's new log lines in this terminal (``--foreground``) until it stops or
        Ctrl+C. Returns the kernel container's exit code (0 after Ctrl+C); when it stopped by
        itself, ``ended`` says why (removed elsewhere, out of memory, exited)."""
        if not self.container_ids:
            return 1
        ident = self.container_ids[0]
        try:
            self.runner.stream(["logs", "-f", "--tail", "0", ident])
        except KeyboardInterrupt:
            return 0
        except (HailerError, OSError):
            self.ended = "Lost contact with Docker while following the kernel's log."
            return 1
        try:
            result = self.runner.run(
                ["inspect", "--type", "container", "--format", "{{.State.Status}} {{.State.ExitCode}} {{.State.OOMKilled}}", ident],
                timeout=DOCKER_PROBE_TIMEOUT_SEC,
            )
        except KeyboardInterrupt:
            return 0
        except (HailerError, OSError, subprocess.TimeoutExpired):
            self.ended = "The kernel's log ended and Docker could not say why."
            return 1
        removed = "The kernel container was removed from outside this terminal (uvx hailer kernel stop, or docker rm)."
        if result.returncode != 0:
            self.ended = removed if _missing(result) else f"The kernel's log ended and Docker could not say why. {docker_said(result)}".strip()
            return 1
        status, code, oom = ((result.stdout or "").split() + ["", "", ""])[:3]
        try:
            exit_code = int(code)
        except ValueError:
            exit_code = 1
        if status == "removing":
            self.ended = removed
        elif status in _RUNNING_STATES:
            self.ended = "Stopped following the kernel's log (docker logs ended while the kernel still runs)."
        elif oom == "true":
            self.ended = "The kernel ran out of memory and Docker stopped it; [kernel].memory in hailer.toml raises the limit."
        elif exit_code == 137:
            self.ended = "The kernel container was stopped from outside this terminal (docker stop, docker kill or docker rm)."
        else:
            self.ended = f"The kernel container exited (code {exit_code})."
        return exit_code


# --------------------------------------------------------------------------- #
# The runtime
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Labelled:
    """A container Docker lists with this workspace's label."""

    id: str
    name: str
    state: str
    role: str

    @property
    def running(self) -> bool:
        return self.state in _RUNNING_STATES


class DockerRuntime:
    """marimo in a Linux container that sees only the notebooks folder (read-write) and the data
    folder (read-only), without network access or any of the host's secrets.

    One workspace gets (names from :func:`docker_names`, every one labelled with the workspace):

    - ``hailer-net-<id>``: an ``--internal`` network, with no route out.
    - ``hailer-kernel-<id>``: ``marimo edit`` on that network only, hardened (read-only root,
      no capabilities, pid, memory (no swap on top) and CPU limits), no port published (Docker
      ignores ``-p`` on an internal network).
    - ``hailer-fwd-<id>``: :mod:`hailer._forward` from the same image, published on
      ``127.0.0.1:<port>`` and joined to both networks; it only ever connects to the kernel.

    ``[kernel] network = true`` runs the kernel on Docker's default network with its port
    published directly, and no forwarder or internal network; notebook code can then reach the
    internet, services on this machine (``host.docker.internal``) and other containers.

    Every check fails closed with a :class:`KernelRuntimeError`; nothing here ever falls back to
    the local runtime. All docker calls go through ``runner`` (:class:`DockerRunner`).
    """

    name = KERNEL_RUNTIME_DOCKER

    def __init__(
        self,
        config: HailerConfig,
        *,
        runner: DockerRunner | None = None,
        token_factory: Callable[[], str] = new_token,
        start_timeout: float = START_TIMEOUT_SEC,
        health: Callable[..., bool] | None = None,
        user: Callable[[], tuple[int, int] | None] = linux_host_user,
        probe: Probe | None = None,
    ) -> None:
        self.config = config
        self.paths = docker_paths(config)
        self.runner: DockerRunner = runner if runner is not None else SubprocessDockerRunner()
        self.names = docker_names(config.workspace)
        self._token_factory = token_factory
        self.start_timeout = start_timeout
        self._health = health
        self._user = user
        self._probe = probe
        self._engine_version: str | None = None
        self._cpus = config.kernel.cpus
        self._prepared = False

    # -- what it is ---------------------------------------------------------- #

    @property
    def image(self) -> str:
        return self.config.kernel.effective_image

    @property
    def workspace_label(self) -> str:
        return str(Path(self.config.workspace))

    @property
    def _kernel_config(self) -> KernelConfig:
        return replace(self.config.kernel, runtime=KERNEL_RUNTIME_DOCKER)

    def describe(self) -> str:
        return describe_runtime(self._kernel_config)

    # -- checks -------------------------------------------------------------- #

    def check(self) -> list[Check]:
        """Rows ``kernel``, ``docker`` (CLI, engine, version), ``image`` (present, version),
        ``data`` (path warnings) and, when Docker may not see it, ``notebooks``. An image that is
        not downloaded yet is only a warning: ``prepare`` downloads it."""
        rows = [Check("kernel", True, self.describe(), fatal=False)]
        try:
            version = self.engine_version()
        except KernelRuntimeError as err:
            rows.append(Check("docker", False, str(err), hint=err.hint))
        else:
            rows.append(Check("docker", True, f"Docker {version} (Linux engine)", fatal=False))
            rows.append(self._image_check())
        rows.append(self._data_check())
        notebooks = folder_problems(self.config.notebooks_root, "notebooks", "[hailer].notebooks_dir")
        if notebooks:
            rows.append(Check("notebooks", False, "the container may not see the notebooks folder", hint="\n".join(notebooks), fatal=False))
        return rows

    def _image_check(self) -> Check:
        try:
            found = self._image_version()
        except KernelRuntimeError as err:
            return Check("image", False, str(err), hint=err.hint)
        if found is None:
            return Check(
                "image",
                False,
                f"{self.image} is not on this machine yet; the first start downloads it",
                hint="uvx hailer kernel pull downloads it now; uvx hailer kernel build builds it on this machine instead.",
                fatal=False,
            )
        if found != __version__:
            err = self._mismatch(found)
            return Check("image", False, str(err), hint=err.hint)
        return Check("image", True, f"{self.image} (Hailer {found})", fatal=False)

    def _data_check(self) -> Check:
        problems = data_path_problems(self.config.data_dir)
        if problems:
            return Check("data", False, "the container may not see all of the data folder", hint="\n".join(problems), fatal=False)
        return Check("data", True, f"{self.config.data_dir} (read-only in the kernel, as {KERNEL_DATA_DIR})", fatal=False)

    def engine_version(self) -> str:
        """The Docker engine's version. ``KernelRuntimeError`` when Docker is not installed, not
        running, or runs Windows containers."""
        if self._engine_version is not None:
            return self._engine_version
        try:
            result = self.runner.run(["version", "--format", "{{.Server.Version}} {{.Server.Os}}"], timeout=DOCKER_PROBE_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            raise docker_not_running(f"docker version did not answer within {int(DOCKER_PROBE_TIMEOUT_SEC)} s.") from None
        version, _, engine_os = (result.stdout or "").strip().partition(" ")
        if result.returncode != 0 or not version or version.startswith("<"):  # "<no value>": no server
            raise docker_not_running(docker_said(result))
        engine_os = engine_os.strip().lower()
        if engine_os != "linux":
            raise KernelRuntimeError(
                f"Docker runs {engine_os or 'non-Linux'} containers; Hailer's kernel image needs Linux containers.",
                hint=(
                    "Switch Docker Desktop to Linux containers (its tray icon menu: Switch to Linux containers), "
                    'or set [kernel] runtime = "local" in hailer.toml.'
                ),
            )
        self._engine_version = version
        return version

    def engine_cpus(self) -> int | None:
        """How many CPUs the Docker engine has (``None`` when it does not say)."""
        try:
            result = self.runner.run(["info", "--format", "{{.NCPU}}"], timeout=DOCKER_PROBE_TIMEOUT_SEC)
        except (HailerError, OSError, subprocess.TimeoutExpired):
            return None
        try:
            count = int((result.stdout or "").strip()) if result.returncode == 0 else 0
        except ValueError:
            return None
        return count if count > 0 else None

    def _image_version(self) -> str | None:
        try:
            return kernel_image.image_version(self.image, self.runner)
        except subprocess.TimeoutExpired as err:
            raise KernelRuntimeError(
                f"Docker did not answer within {int(err.timeout or 0)} s when asked about {self.image}.",
                hint="Check that Docker Desktop is running and responsive, then run the command again.",
            ) from err

    def _mismatch(self, found: str) -> KernelRuntimeError:
        return KernelRuntimeError(
            f"The kernel image {self.image} is for Hailer {found or '(no version label)'}; this is Hailer {__version__}.",
            hint=(
                "The image must match this Hailer (another marimo would break code mode). Download the matching "
                "image with `uvx hailer kernel pull`, or build it on this machine with `uvx hailer kernel build`. "
                "[kernel].image in hailer.toml (or HAILER_KERNEL_IMAGE) picks another image."
            ),
        )

    def _pull_failed(self, detail: str = "") -> KernelRuntimeError:
        if any(sign in detail.lower() for sign in _UNPUBLISHED_SIGNS):
            return KernelRuntimeError(
                f"The kernel image for Hailer {__version__} is not published (or not visible to you): {self.image}.",
                hint=(
                    "Build it on this machine with: uvx hailer kernel build (it downloads the base image and the "
                    "Python packages once). Or set [kernel].image in hailer.toml (or HAILER_KERNEL_IMAGE) to a copy "
                    "you can reach, such as a company mirror."
                ),
            )
        return KernelRuntimeError(
            f"Could not download the kernel image {self.image}.",
            hint=(
                "docker's own error is above. Without access to that registry, build the image on this machine "
                "with `uvx hailer kernel build` (it downloads the base image and Python packages once), or point "
                "[kernel].image in hailer.toml (or HAILER_KERNEL_IMAGE) at a copy you can reach, such as a "
                "company mirror."
            ),
        )

    def _download(self) -> str:
        result = kernel_image.pull(self.image, self.runner)
        if result.returncode != 0:
            raise self._pull_failed(result.stderr or "")
        found = self._image_version()
        if found is None:
            raise self._pull_failed()
        if found != __version__:
            raise self._mismatch(found)
        return found

    def pull(self, say: Callable[[str], None] | None = None) -> str:
        """``uvx hailer kernel pull``: download the image (docker's progress in this terminal) even
        when a copy is here, and check its version. Returns that version."""
        self.engine_version()
        (say or _say)(f"Downloading {self.image} ...")
        return self._download()

    def mount_problems(self) -> list[str]:
        """Why the notebooks or data folder must not be mounted (empty: both are fine): folders
        that expose Hailer's own files, the home folder or a drive (see
        :func:`hailer.config.docker_mount_problems`), and UNC paths, which Docker cannot mount."""
        from hailer.config import docker_mount_problems  # lazy: config does not import the runtimes

        problems = docker_mount_problems(self.config)
        if _on_windows():
            for folder, what, setting in (
                (self.config.notebooks_root, "notebooks", "[hailer].notebooks_dir"),
                (self.config.data_dir, "data", "[hailer].data_dir"),
            ):
                if network_location(folder, lambda root: 0) == UNC_PATH:
                    problems.append(
                        f"The {what} folder {folder} is on a network share (UNC path), which Docker cannot mount. "
                        f"Copy it to a folder on a local disk and point {setting} at it."
                    )
        return problems

    def _check_mounts(self) -> None:
        problems = self.mount_problems()
        if problems:
            raise KernelRuntimeError(problems[0], hint="\n".join(problems[1:]))

    def _gone(self, state: KernelState) -> bool:
        return containers_gone(state, self.runner)

    def prepare(self, say: Callable[[str], None] | None = None) -> None:
        """Docker runs a Linux engine, the folders are safe to mount, no kernel Hailer started for
        this workspace is still running, and the image for this Hailer is here: downloaded now
        (with docker's progress in this terminal) when it is missing. A ``[kernel].cpus`` above
        what Docker has is lowered, with a note. Raises ``KernelRuntimeError``."""
        out = say or _say
        self.engine_version()
        self._check_mounts()
        refuse_live_kernel(Path(self.config.workspace), probe=self._probe, gone=self._gone)
        found = self._image_version()
        if found is None:
            out(f"Downloading the kernel image {self.image} (first use; this can take a few minutes) ...")
            found = self._download()
        if found != __version__:
            raise self._mismatch(found)
        wanted = self.config.kernel.cpus
        available = self.engine_cpus()
        if available is not None and wanted > available:
            self._cpus = float(available)
            out(f"Note: [kernel].cpus = {wanted:g} is more than the {available} CPUs Docker has; the kernel gets {available}.")
        self._prepared = True

    # -- docker commands ----------------------------------------------------- #

    def _docker(self, args: Sequence[str], *, what: str, timeout: float = DOCKER_TIMEOUT_SEC) -> subprocess.CompletedProcess[str]:
        """Run one docker command; ``KernelRuntimeError`` (with docker's message) when it fails."""
        try:
            result = self.runner.run(list(args), timeout=timeout)
        except subprocess.TimeoutExpired as err:
            raise KernelRuntimeError(
                f"Docker did not {what} within {int(timeout)} s.",
                hint="Check that Docker Desktop is running and responsive, then run the command again.",
            ) from err
        if result.returncode != 0:
            raise KernelRuntimeError(f"Docker could not {what}.", hint=docker_said(result))
        return result

    @staticmethod
    def _created_id(result: subprocess.CompletedProcess[str], fallback: str) -> str:
        """The id ``docker run -d`` / ``create`` / ``network create`` printed (its last line)."""
        lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
        return lines[-1] if lines else fallback

    def _labels(self, role: str) -> list[str]:
        return [
            "--label", f"{LABEL_WORKSPACE}={self.workspace_label}",
            "--label", f"{LABEL_VERSION}={__version__}",
            "--label", f"{LABEL_ROLE}={role}",
        ]  # fmt: skip

    def network_command(self) -> list[str]:
        return ["network", "create", "--internal", *self._labels("network"), self.names.network]

    def kernel_command(self, port: int, token: str) -> list[str]:
        """``docker run`` for the kernel. Nothing is mounted but the notebooks folder (read-write)
        and the data folder (read-only): not the workspace root, ``.hailer/``, ``.config/hailer/``,
        the home folder or the Docker socket. ``--memory-swap`` equal to ``--memory``: no swap on top."""
        kernel = self.config.kernel
        user = self._user()
        uid, gid = user if user is not None else (IMAGE_UID, IMAGE_GID)
        cmd = ["run", "-d", "--pull", "never", "--name", self.names.kernel]
        if kernel.network:
            cmd += ["-p", f"127.0.0.1:{port}:{KERNEL_PORT}"]
        else:
            cmd += ["--network", self.names.network]
        cmd += ["--init", "--read-only", "--tmpfs", "/tmp", "--tmpfs", f"{KERNEL_HOME}:uid={uid},gid={gid}"]
        if user is not None:
            cmd += ["--user", f"{uid}:{gid}", "-e", f"HOME={KERNEL_HOME}"]
        cmd += [
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--pids-limit", "256",
            "--memory", kernel.memory,
            "--memory-swap", kernel.memory,
            "--cpus", f"{self._cpus:g}",
            "-w", str(KERNEL_WORKDIR),
            *self._labels("kernel"),
            "--mount", mount_arg(self.config.notebooks_root, KERNEL_NOTEBOOKS_DIR),
            "--mount", mount_arg(self.config.data_dir, KERNEL_DATA_DIR, readonly=True),
            self.image,
            "marimo", "edit", KERNEL_NOTEBOOKS_DIR.name,
            "--host", "0.0.0.0",
            "--port", str(KERNEL_PORT),
            "--headless",
            "--skip-update-check",
            "--token-password", token,
        ]  # fmt: skip
        return cmd

    def forwarder_command(self, port: int) -> list[str]:
        """``docker create`` for the forwarder: published on 127.0.0.1 only, and it only ever
        connects to the kernel."""
        return [
            "create", "--pull", "never", "--name", self.names.forwarder,
            "-p", f"127.0.0.1:{port}:{KERNEL_PORT}",
            "--init", "--read-only",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--pids-limit", "64",
            "--memory", "64m",
            *self._labels("forwarder"),
            self.image,
            "python", "-m", "hailer._forward", self.names.kernel, str(KERNEL_PORT), str(KERNEL_PORT),
        ]  # fmt: skip

    def _listed(self, args: Sequence[str], *, what: str) -> list[str]:
        return [line.strip() for line in (self._docker(args, what=what).stdout or "").splitlines() if line.strip()]

    def labelled_containers(self) -> list[Labelled]:
        """Every container, running or not, labelled with this workspace."""
        label = f"label={LABEL_WORKSPACE}={self.workspace_label}"
        rows = self._listed(
            ["ps", "-a", "--filter", label, "--format", f'{{{{.ID}}}}\t{{{{.Names}}}}\t{{{{.State}}}}\t{{{{.Label "{LABEL_ROLE}"}}}}'],
            what="list this workspace's containers",
        )
        found = []
        for row in rows:
            parts = (row.split("\t") + ["", "", ""])[:4]
            found.append(Labelled(id=parts[0], name=parts[1], state=parts[2].lower(), role=parts[3]))
        return found

    def remove_leftovers(self, *, running: bool = False) -> Removal:
        """Remove the containers and networks labelled with this workspace (left by a crash, a
        ``kill``, a failed start or ``--keep-marimo``).

        ``running=False`` (a start): a running kernel container is never removed (a liveness probe
        that got it wrong must not destroy a kernel in use): ``KernelRuntimeError`` instead. A
        running forwarder whose kernel is not running is removed. ``running=True`` (``kernel
        stop``): everything goes.
        """
        containers = self.labelled_containers()
        live = [c for c in containers if c.running and c.role != "forwarder"]
        if live and not running:
            raise KernelRuntimeError(
                f"A kernel container for this workspace is still running, but Hailer cannot reach it: {', '.join(c.name for c in live)}.",
                hint=(
                    "Hailer never removes a running kernel on its own (it may still be in use). Stop it with "
                    "uvx hailer kernel stop, then run this command again."
                ),
            )
        outcome = Removal()
        for container in containers:
            outcome.add(_remove(self.runner, "container", container.id, container.name))
        label = f"label={LABEL_WORKSPACE}={self.workspace_label}"
        for row in self._listed(["network", "ls", "--filter", label, "--format", "{{.ID}}\t{{.Name}}"], what="list this workspace's networks"):
            ident, _, name = row.partition("\t")
            outcome.add(_remove(self.runner, "network", ident, name or ident))
        return outcome

    def _stopped(self, containers: Sequence[str]) -> dict[str, str]:
        """The container ids among ``containers`` that are not running, with their exit codes ("?"
        when unknown, e.g. the container is gone). Empty when docker cannot be asked."""
        try:
            result = self.runner.run(
                ["inspect", "--type", "container", "--format", "{{.Id}} {{.State.Running}} {{.State.ExitCode}}", *containers],
                timeout=DOCKER_PROBE_TIMEOUT_SEC,
            )
        except (HailerError, OSError, subprocess.TimeoutExpired):
            return {}
        running: set[str] = set()
        stopped: dict[str, str] = {}
        for line in (result.stdout or "").splitlines():
            parts = line.split()
            if len(parts) != 3:
                continue
            ident = next((c for c in containers if parts[0].startswith(c) or c.startswith(parts[0])), parts[0])
            if parts[1] == "true":
                running.add(ident)
            else:
                stopped[ident] = parts[2]
        for ident in containers:
            if ident not in running and ident not in stopped:
                stopped[ident] = "?"
        return stopped

    def _labelled_ours(self, container: str) -> bool:
        try:
            result = self.runner.run(
                ["inspect", "--type", "container", "--format", "{{json .Config.Labels}}", container], timeout=DOCKER_PROBE_TIMEOUT_SEC
            )
            labels = json.loads((result.stdout or "").strip() or "null") if result.returncode == 0 else None
        except (HailerError, OSError, ValueError, subprocess.TimeoutExpired):
            return False
        return isinstance(labels, dict) and labels.get(LABEL_WORKSPACE) == self.workspace_label

    # -- lifecycle ------------------------------------------------------------ #

    def _kernel(self, server: MarimoServer, state: KernelState | None = None) -> DockerKernel:
        names = state.containers if state is not None else ()
        return DockerKernel(
            server=server,
            log_hint=f"docker logs {names[0] if names else self.names.kernel}",
            stop_hint="stop it with uvx hailer kernel stop",
            workspace=Path(self.config.workspace),
            runner=self.runner,
            containers=names,
            container_ids=state.container_ids if state is not None else (),
            network=state.network if state is not None else None,
            network_id=state.network_id if state is not None else None,
        )

    def find_running(self) -> RunningKernel | None:
        """The docker kernel ``kernel.json`` records, when it answers with its token and its
        container carries this workspace's label."""
        state = live_kernel_state(Path(self.config.workspace), probe=self._probe, gone=self._gone)
        if state is None or state.runtime != self.name or not state.container_ids:
            return None
        if not self._labelled_ours(state.container_ids[0]):
            return None
        return self._kernel(state.server(), state)

    def start(self, port: int, *, foreground: bool = False) -> RunningKernel:
        """Create the network, the kernel and the forwarder, wait for ``/health`` through the
        forwarder and write ``kernel.json`` (with the ids Docker gave them). ``foreground``
        changes nothing here: the containers run detached either way, and
        :meth:`DockerKernel.wait` follows the kernel's log."""
        del foreground
        config = self.config
        workspace = Path(config.workspace)
        if not self._prepared:
            self.prepare()
        self._check_mounts()
        refuse_live_kernel(workspace, probe=self._probe, gone=self._gone)
        delete_kernel_state(workspace)  # not live (checked just now): a stale record at most
        leftovers = self.remove_leftovers()
        if leftovers.failed:
            raise KernelRuntimeError("Docker could not remove what an earlier kernel left behind.", hint="\n".join(leftovers.failed))
        for folder in (config.notebooks_root, config.data_dir):
            try:
                folder.mkdir(parents=True, exist_ok=True)  # before Docker creates a missing one owned by root
            except OSError as err:
                raise KernelRuntimeError(f"Could not create {folder}: {err}", hint="Check the folder's permissions.") from err
        names = self.names
        network_access = config.kernel.network
        token = self._token_factory()
        url = f"http://127.0.0.1:{port}"
        server = MarimoServer(
            url=url, server_id=f"127.0.0.1:{port}", source="kernel", token=token, runtime=self.name, paths=self.paths,
            network_access=network_access,
        )  # fmt: skip
        kernel = self._kernel(server)
        health = self._health if self._health is not None else wait_for_health
        try:
            if not network_access:
                created = self._docker(self.network_command(), what="create the kernel's internal network")
                kernel.network, kernel.network_id = names.network, self._created_id(created, names.network)
            created = self._docker(self.kernel_command(port, token), what="start the kernel container")
            kernel.containers, kernel.container_ids = (names.kernel,), (self._created_id(created, names.kernel),)
            if not network_access:
                created = self._docker(self.forwarder_command(port), what="create the forwarder container")
                forwarder = self._created_id(created, names.forwarder)
                kernel.containers, kernel.container_ids = (*kernel.containers, names.forwarder), (*kernel.container_ids, forwarder)
                self._docker(["network", "connect", kernel.network_id or names.network, forwarder], what="connect the forwarder to the kernel's network")
                self._docker(["start", forwarder], what="start the forwarder container")
            ids = kernel.container_ids
            healthy = health(url, self.start_timeout, should_stop=lambda: bool(self._stopped(ids)))
        except BaseException:  # a docker error, or Ctrl+C while waiting: never leave containers behind
            kernel.stop()
            raise
        if not healthy:
            stopped = self._stopped(kernel.container_ids)
            kernel_id = kernel.container_ids[0]
            if kernel_id in stopped:
                message = f"The kernel container exited early (code {stopped[kernel_id]})."
                shown = kernel_id
            elif stopped:
                shown = next(iter(stopped))
                message = f"The forwarder container exited early (code {stopped[shown]})."
            else:
                message = f"Marimo did not answer on {url} within {int(self.start_timeout)} s."
                shown = kernel_id
            tail = kernel.log_tail(container=shown)
            kernel.stop()
            name = kernel._name(shown)
            hint = f"Last lines of docker logs {name}:" if tail else f"docker logs {name} was empty."
            if tail:
                hint += "\n" + "\n".join(f"    {line}" for line in tail)
            raise KernelRuntimeError(message, hint=hint)
        write_kernel_state(
            workspace,
            KernelState(
                runtime=self.name,
                url=url,
                port=port,
                token=token,
                image=self.image,
                containers=kernel.containers,
                container_ids=kernel.container_ids,
                network=kernel.network,
                network_id=kernel.network_id,
                network_access=network_access,
                mounts=tuple((str(host), str(kernel_path)) for host, kernel_path in self.paths.mounts),
                memory=config.kernel.memory,
                cpus=config.kernel.cpus,
            ),
        )
        note_kernel_start(workspace, self.name, config.notebooks_root)
        return kernel


# --------------------------------------------------------------------------- #
# uvx hailer kernel stop
# --------------------------------------------------------------------------- #


@dataclass
class StopReport:
    """What ``uvx hailer kernel stop`` did (``done``) and what it could not do (``failed``)."""

    done: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


def _removed_line(outcome: Removal, url: str) -> str:
    if outcome.removed:
        return f"Stopped the docker kernel at {url} (removed {', '.join(outcome.removed)})."
    return f"Removed the record of a docker kernel whose containers were already gone ({url})."


def stop_workspace_kernels(
    config: HailerConfig,
    runner: DockerRunner | None = None,
    *,
    procs: LocalProcesses | None = None,
    probe: Probe | None = None,
) -> StopReport:
    """``uvx hailer kernel stop``: stop the server ``kernel.json`` records (either runtime) and
    remove every container and network labelled with this workspace, running or not. It reports
    what it actually removed; failures are reported, never raised, once anything was cleaned up.

    A local server is stopped by its pid only when it answers with its token (so a stale record
    never kills an unrelated process that reused the pid). Raises ``KernelRuntimeError`` only when
    a docker kernel still answers but Docker cannot be reached to remove it (``kernel.json`` is
    kept then).
    """
    workspace = Path(config.workspace)
    docker = DockerRuntime(config, runner=runner, probe=probe)
    try:
        docker.engine_version()
        docker_error: KernelRuntimeError | None = None
    except KernelRuntimeError as err:
        docker_error = err
    report = StopReport()
    state = read_kernel_state(workspace)
    if state is not None and state.runtime == KERNEL_RUNTIME_DOCKER:
        if docker_error is None:
            outcome = docker._kernel(state.server(), state).remove()
            if not outcome.failed:
                delete_kernel_state(workspace, token=state.token)
            report.done.append(_removed_line(outcome, state.url))
            report.failed += [f"Could not remove {failure}" for failure in outcome.failed]
        elif (probe or answers_with_token)(state.url, state.token):
            raise docker_error
        else:
            delete_kernel_state(workspace)
            report.done.append(f"Removed the record of a docker kernel that no longer answers ({state.url}).")
    elif state is not None:
        running = LocalRuntime(config, procs=procs, probe=probe).find_running()
        if running is not None:
            running.stop()
            process = f" (process {state.pid})" if state.pid is not None else ""
            report.done.append(f"Stopped the local marimo server at {state.url}{process}.")
        else:
            delete_kernel_state(workspace)
            report.done.append(f"Removed the record of a marimo server that no longer answers ({state.url}).")
    if docker_error is None:
        try:
            leftovers = docker.remove_leftovers(running=True)
        except KernelRuntimeError as err:
            report.failed.append(f"{err} {err.hint or ''}".strip())
        else:
            if leftovers.removed:
                report.done.append("Removed " + ", ".join(leftovers.removed) + ".")
            report.failed += [f"Could not remove {failure}" for failure in leftovers.failed]
    elif str(docker_error) == DOCKER_NOT_RUNNING:
        report.done.append(
            "Docker is not running, so containers an earlier docker kernel may have left were not checked "
            "(start Docker Desktop and run uvx hailer kernel stop again to remove them)."
        )
    return report


__all__ = [
    "LABEL_WORKSPACE",
    "DockerKernel",
    "DockerRunner",
    "DockerRuntime",
    "Removal",
    "StopReport",
    "SubprocessDockerRunner",
    "containers_gone",
    "data_path_problems",
    "docker_names",
    "mismatch_error",
    "settings_mismatch",
    "stop_workspace_kernels",
]
