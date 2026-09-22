"""The docker kernel runtime: marimo in a Linux container with a notebooks folder of its own (in
memory) and the data folder (read-only), with no network and none of the host's secrets.

Everything here drives the ``docker`` CLI through a :class:`DockerRunner` (no Docker SDK), so the
tests check argument lists against a scripted fake. It fails closed: every check raises a
:class:`~hailer.errors.KernelRuntimeError` with a hint and nothing ever falls back to the local
runtime. The sandbox a start returns (:class:`DockerKernel`) is a
:class:`~hailer.sandbox.MarimoSandbox` whose kernel knows the folders as ``/work/notebooks`` and
``/work/data``; ``runtime_for`` imports this module only when docker is asked for. Nothing in the
container can write to this machine: the workspace's notebooks are copied in when the kernel starts
and the ones it changed are copied back through marimo's file API (:mod:`hailer.notebook_sync`,
marimo notebooks with plain names only).

Each Hailer process starts its own kernel: names are unique per start (the workspace id plus a
random suffix) and every container and network carries labels for the workspace, its role, the
kernel contract and its owner. The owner is an :class:`OwnerLock`: a ``.hailer/owner-<id>.lock``
file the starting process holds an OS lock on for as long as it runs, so the OS itself says when
the owner has gone (however it ended) and a reused pid can never make a dead owner look alive.
The process removes what it started by the ids Docker gave it. A start also removes this
workspace's objects whose owner is not alive; it never removes anything of a live owner.
``uvx hailer kernel stop`` removes every labelled object of the workspace.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from hailer import kernel_image
from hailer.config import is_unc_path
from hailer.errors import HailerError, KernelRuntimeError, NotebookPathError
from hailer.kernel import (
    KERNEL_DATA_DIR,
    KERNEL_NOTEBOOKS_DIR,
    KERNEL_WORKDIR,
    START_TIMEOUT_SEC,
    describe_runtime,
    mask_token,
    new_token,
    note_kernel_start,
)
from hailer.marimo_client import wait_for_health
from hailer.models import KERNEL_RUNTIME_DOCKER, Check, HailerConfig, KernelConfig, MarimoServer
from hailer.notebook_sync import NotebookSync, sync_refusal
from hailer.notebooks import check_notebook_name
from hailer.sandbox import MarimoSandbox, NotebookFile
from hailer.statedir import ensure_state_dir, state_dir

#: marimo's port inside the kernel container; the forwarder listens on the same port in its own.
KERNEL_PORT = 2718
#: The image's uid and gid (1000) and its home, a tmpfs owned by the kernel's user.
IMAGE_UID = 1000
IMAGE_GID = 1000
KERNEL_HOME = "/home/analyst"
#: The size cap of the kernel's notebooks folder, a tmpfs owned by the kernel's user (it counts
#: against the container's memory as it fills; marimo keeps its ``__marimo__`` caches there too).
NOTEBOOKS_TMPFS_SIZE = "512m"
#: Where the kernel reads its marimo token (``--token-password-file``): a read-only single-file
#: mount, so the token is never in ``docker run``'s arguments or environment (``docker inspect``).
KERNEL_TOKEN_FILE = PurePosixPath("/run/secrets/hailer-token")
#: The folders under ``.hailer`` that hold a token file while a kernel starts.
TOKEN_FOLDER_PREFIX = "kernel-token-"
#: ``.hailer/owner-<id>.lock``: one per docker kernel start, locked by its Hailer process while it runs.
OWNER_LOCK_PREFIX = "owner-"
#: A lock file younger than this is never swept: its owner may not have locked it yet.
OWNER_LOCK_GRACE_SEC = 10.0
_OWNER_ID_RE = re.compile(r"[0-9a-f]{16}")
#: Labels on every container and network Hailer creates, so it finds its own (and strays).
LABEL_PREFIX = "org.openafterhours.hailer"
LABEL_WORKSPACE = f"{LABEL_PREFIX}.workspace"
LABEL_ROLE = f"{LABEL_PREFIX}.role"
LABEL_CONTRACT = f"{LABEL_PREFIX}.contract"
#: The :class:`OwnerLock` id of the Hailer process that started the object.
LABEL_OWNER = f"{LABEL_PREFIX}.owner"
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
#: What docker says about an object that is not there (a removal that has nothing left to do):
#: object-specific wording only. A bare "not found" would also match a broken docker context
#: ("context not found"), which must never count as "already removed".
_MISSING_SIGNS = ("no such container", "no such network", "no such object")
_MISSING_NETWORK_RE = re.compile(r"\bnetwork \S+ not found")
#: What docker says when another command is removing the same container right now.
_REMOVAL_IN_PROGRESS = "already in progress"
#: What docker says when a network still has containers attached (they are being removed).
_ACTIVE_ENDPOINTS = "active endpoints"
#: How often, and how far apart, ``network rm`` is retried while containers are still detaching.
NETWORK_RM_RETRIES = 6
NETWORK_RM_RETRY_SEC = 0.5
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
    if "context" in detail.lower():  # DOCKER_CONTEXT or `docker context use` names a context that does not work
        hint = (
            "The docker command cannot reach an engine through its current context: check DOCKER_CONTEXT and "
            "`docker context ls` (`docker context use default` switches back), then run the command again."
        )
    if detail:
        hint += f"\n{detail}"
    return KernelRuntimeError(DOCKER_NOT_RUNNING, hint=hint)


def docker_said(result: subprocess.CompletedProcess[str]) -> str:
    """The last lines docker printed, for a hint ("" when it printed nothing)."""
    text = "\n".join(part for part in (result.stderr, result.stdout) if part)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return ("docker said: " + " / ".join(lines[-3:])) if lines else ""


def _said(result: subprocess.CompletedProcess[str]) -> str:
    return f"{result.stderr or ''}\n{result.stdout or ''}".lower()


def _missing(result: subprocess.CompletedProcess[str]) -> bool:
    """True when docker failed because the object is not there (nothing left to remove)."""
    text = _said(result)
    return any(sign in text for sign in _MISSING_SIGNS) or _MISSING_NETWORK_RE.search(text) is not None


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
    """The names of one start's kernel container, forwarder container and internal network."""

    kernel: str
    forwarder: str
    network: str


def docker_names(workspace: Path, suffix: str | None = None) -> DockerNames:
    """``hailer-kernel-<workspace id>-<suffix>`` and friends; ``suffix`` defaults to 6 random hex
    characters, so every start (and every Hailer process in the workspace) gets names of its own."""
    ident = f"{workspace_id(workspace)}-{suffix if suffix is not None else secrets.token_hex(3)}"
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


def write_token_file(workspace: Path, token: str) -> Path:
    """Write ``token`` where the kernel container can read it and return the file.

    The file sits in a new ``.hailer/kernel-token-<random>`` folder. On POSIX that folder is
    ``0700``, so no other user of this machine reaches the file; on Windows the mode is ignored and
    the folder inherits the workspace's ACL. The file itself is ``0644``: the kernel's user in the
    container can be another uid (a root host, rootless Docker), and Docker mounts the file, not its
    folder. The start deletes it (:func:`remove_token_folder`) once marimo has read it, in a
    ``finally``, whatever ends the wait.
    """
    folder = Path(tempfile.mkdtemp(prefix=TOKEN_FOLDER_PREFIX, dir=ensure_state_dir(workspace)))
    path = folder / "token"
    path.write_text(token, encoding="utf-8")
    os.chmod(path, 0o644)  # whatever the umask took away
    return path


def remove_token_folder(folder: Path) -> str | None:
    """Delete a token folder; a warning line when that fails (it still holds a kernel token)."""
    try:
        shutil.rmtree(folder)
    except FileNotFoundError:
        return None
    except OSError as err:
        return f"Warning: could not delete {folder} ({err}). It holds a kernel token: delete it yourself."
    return None


# --------------------------------------------------------------------------- #
# Owners: a lock file each docker kernel's Hailer holds while it runs
# --------------------------------------------------------------------------- #


def _try_lock(fd: int) -> bool:
    """An exclusive, non-blocking OS lock on ``fd`` (``msvcrt.locking`` / ``fcntl.flock``); False
    when another open handle holds it (another process, or another handle in this one)."""
    try:
        if os.name == "nt":
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[attr-defined]
    except OSError:
        return False
    return True


def _unlock(fd: int) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_UN)  # type: ignore[attr-defined]
    except OSError:
        pass


def _open_existing(path: Path) -> int | None:
    try:
        return os.open(path, os.O_RDWR | getattr(os, "O_BINARY", 0))
    except OSError:
        return None


def _owner_lock_path(workspace: Path, owner_id: str) -> Path:
    return state_dir(workspace) / f"{OWNER_LOCK_PREFIX}{owner_id}.lock"


@dataclass
class OwnerLock:
    """The lock a Hailer process holds for the lifetime of a docker kernel it started. ``id`` is the
    kernel's owner label. The OS drops the lock when the process ends, however it ends."""

    id: str
    path: Path
    fd: int | None = field(default=None, repr=False)

    def release(self) -> None:
        """Unlock, close and delete the lock file. Idempotent, never raises."""
        if self.fd is None:
            return
        fd, self.fd = self.fd, None
        _unlock(fd)
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            self.path.unlink()
        except OSError:
            pass


def acquire_owner_lock(workspace: Path) -> OwnerLock:
    """Create ``.hailer/owner-<random id>.lock`` and lock it; ``OSError`` when that fails."""
    ident = secrets.token_hex(8)
    path = ensure_state_dir(workspace) / f"{OWNER_LOCK_PREFIX}{ident}.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    if not _try_lock(fd):
        os.close(fd)
        raise OSError(f"could not lock {path}")
    return OwnerLock(ident, path, fd)


def owner_alive(workspace: Path, owner_id: str) -> bool:
    """True when ``owner_id``'s lock file exists and cannot be locked: the Hailer that started the
    object still runs. False for a missing file, a lock anyone can take, or a label that is not an
    owner id (an object without one)."""
    if not _OWNER_ID_RE.fullmatch(owner_id or ""):
        return False
    fd = _open_existing(_owner_lock_path(workspace, owner_id))
    if fd is None:
        return False
    try:
        if _try_lock(fd):
            _unlock(fd)
            return False
        return True
    finally:
        os.close(fd)


def sweep_owner_locks(workspace: Path, *, grace: float = OWNER_LOCK_GRACE_SEC) -> None:
    """Delete the ``.hailer/owner-*.lock`` files nobody holds (left by a Hailer that ended), except
    ones younger than ``grace`` seconds (their owner may be about to lock them). Never raises."""
    for path in state_dir(workspace).glob(f"{OWNER_LOCK_PREFIX}*.lock"):
        try:
            if time.time() - path.stat().st_mtime < grace:
                continue
        except OSError:
            continue
        fd = _open_existing(path)
        if fd is None:
            continue
        try:
            free = _try_lock(fd)
            if free:
                _unlock(fd)
        finally:
            os.close(fd)
        if free:
            try:
                path.unlink()
            except OSError:
                pass


def linux_host_user() -> tuple[int, int] | None:
    """The uid and gid the kernel runs as on a Linux host, so it reads the data folder the way the
    user can (files and folders only their owner may read, such as ``0600`` exports, stay readable
    through the read-only mount); ``None`` (the image's own user) elsewhere, where Docker Desktop
    maps ownership itself, and for root, since the kernel never runs as root."""
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


def data_path_problems(data_dir: Path, *, windows: bool | None = None, drive_type: Callable[[str], int] | None = None) -> list[str]:
    """Reasons the kernel container may not see all of the data folder, each with its fix (empty:
    none). Warnings only.

    On Windows: a mapped network drive, which Docker Desktop usually cannot mount. (A UNC path is
    fatal, not a warning: :func:`hailer.config.docker_mount_problems` reports it, so nothing is
    said here.) Everywhere: symlinks or junctions inside the folder that point outside it, which do
    not resolve in the container (only the folder itself is mounted).
    """
    text = str(data_dir)
    if windows if windows is not None else _on_windows():
        where = network_location(text, drive_type or _drive_type)
        if where == UNC_PATH:  # fatal elsewhere, and never walked over the network
            return []
        if where is not None:  # no link scan: it would walk the share over the network, and the fix is the same
            return [
                f"The data folder {text} is on {where}; Docker Desktop usually cannot mount it. "
                "Copy it to a folder on a local disk and point [hailer].data_dir at it."
            ]
    found = _links_outside(Path(data_dir))
    if not found:
        return []
    shown = ", ".join(found[:3]) + (f" and {len(found) - 3} more" if len(found) > 3 else "")
    return [
        f"{len(found)} link(s) in the data folder point outside it ({shown}); they do not resolve in the "
        "container. Copy those files into the data folder."
    ]


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


#: Waits between ``network rm`` retries; tests replace it.
_sleep: Callable[[float], None] = time.sleep


def _still_there(runner: DockerRunner, kind: str, ident: str) -> bool:
    """Whether the container or network ``ident`` still exists (True when docker cannot say)."""
    try:
        result = runner.run(["inspect", "--type", kind, "--format", "{{.Id}}", ident], timeout=DOCKER_PROBE_TIMEOUT_SEC)
    except (HailerError, OSError, subprocess.TimeoutExpired):
        return True
    return not (result.returncode != 0 and _missing(result))


def _remove(runner: DockerRunner, kind: str, ident: str, shown: str) -> Removal:
    """Remove one container (``kind="container"``) or network by id. "Not there" and "someone else
    is removing it" count as gone (``kernel stop`` racing the cleanup of the terminal that started
    the kernel); a network whose containers are still detaching is retried for a few seconds.
    Before anything is reported as failed, docker is asked whether it is still there."""
    args = ["rm", "-f", ident] if kind == "container" else ["network", "rm", ident]
    for attempt in range(NETWORK_RM_RETRIES if kind == "network" else 1):
        if attempt:
            _sleep(NETWORK_RM_RETRY_SEC)
        try:
            result = runner.run(args, timeout=DOCKER_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            return Removal(failed=[f"{shown}: docker did not answer within {int(DOCKER_TIMEOUT_SEC)} s"])
        if result.returncode == 0:
            # `docker rm -f` exits 0 for a container that is not there too; it echoes the ones it removed.
            if kind == "container" and ident not in (result.stdout or ""):
                return Removal(gone=[shown])
            return Removal(removed=[shown])
        if _missing(result) or _REMOVAL_IN_PROGRESS in _said(result):
            return Removal(gone=[shown])
        if _ACTIVE_ENDPOINTS not in _said(result):
            break
    if not _still_there(runner, kind, ident):
        return Removal(gone=[shown])
    return Removal(failed=[f"{shown}: {docker_said(result) or f'exit code {result.returncode}'}"])


@dataclass
class DockerKernel(MarimoSandbox):
    """The containers and network this Hailer started. ``containers`` are names (kernel first),
    ``container_ids`` the same containers' ids; everything is removed and inspected by id.
    ``sync`` copies notebooks between the workspace and the container's own notebooks folder
    (``None`` until the start set it up, and again once :meth:`stop` made the last copy)."""

    runner: Any = None
    containers: tuple[str, ...] = ()
    container_ids: tuple[str, ...] = ()
    network: str | None = None
    network_id: str | None = None
    stop_error: str = ""
    owner: OwnerLock | None = None
    sync: NotebookSync | None = None

    def _name(self, ident: str) -> str:
        for name, known in zip(self.containers, self.container_ids):
            if known == ident:
                return name
        return ident[:12]

    def write_notebook(self, name: str, source: str, *, replace: bool = True) -> NotebookFile:
        """:meth:`MarimoSandbox.write_notebook` for a name that is copied back to the workspace: one
        :func:`~hailer.notebook_sync.sync_refusal` refuses (``conftest.py``, ``test_*.py``, deeper
        than four folders) would stay in the container and be lost when it stops, so it is a
        ``NotebookPathError`` before anything is written."""
        clean = check_notebook_name(name)
        reason = sync_refusal(clean)
        if reason is not None:
            raise NotebookPathError(
                f"{clean} would stay in the docker kernel and be lost when it stops ({reason}).",
                hint="Pick a plain notebook name such as sales.py or q3/review.py (not conftest.py, test_*.py or *_test.py).",
            )
        return super().write_notebook(clean, source, replace=replace)

    def sync_soon(self) -> None:
        if self.sync is not None:
            self.sync.soon()

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
        """Copy the notebooks back a last time (after the periodic copy stopped), remove the
        containers and the network, then release the owner lock; ``stop_error`` says what could not
        be removed (ownerless now, the next start or ``uvx hailer kernel stop`` removes it). Ctrl+C
        during the last copy skips it, with a warning, and the containers are still removed."""
        self.stop_error = ""
        sync, self.sync = self.sync, None
        if sync is not None:
            try:
                sync.close(final=True)
            except KeyboardInterrupt:
                self.notify("Warning: stopped before the notebooks were copied back; the kernel's changes since the last copy are lost.")
        try:
            outcome = self.remove()
        except Exception as err:  # noqa: BLE001 - best effort on the way out
            self.stop_error = f"Could not stop the Docker kernel: {err}. Run uvx hailer kernel stop to retry."
        else:
            if outcome.failed:
                self.stop_error = "Could not finish stopping the Docker kernel: " + "; ".join(outcome.failed) + ". Run uvx hailer kernel stop to retry."
        if self.owner is not None:
            self.owner.release()

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
        return mask_token(text.splitlines()[-lines:], self.server.token)

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
    """A container (or, with ``state`` empty, a network) Docker lists with this workspace's label."""

    id: str
    name: str
    state: str
    role: str
    owner: str = ""  # the OwnerLock id of the Hailer that started it ("" without the label)

    @property
    def running(self) -> bool:
        return self.state in _RUNNING_STATES


class DockerRuntime:
    """marimo in a Linux container with its own notebooks folder (a tmpfs the workspace's notebooks
    are copied into and back from) and the data folder (read-only), without network access or any
    of the host's secrets.

    Each start gets (names from :func:`docker_names`, unique per runtime object; every one
    labelled with the workspace, its role, the kernel contract and its :class:`OwnerLock` id):

    - ``hailer-net-<id>-<suffix>``: an ``--internal`` network, with no route out.
    - ``hailer-kernel-<id>-<suffix>``: ``marimo edit`` on that network only, hardened (read-only
      root, no capabilities, pid, memory (no swap on top) and CPU limits), no port published
      (Docker ignores ``-p`` on an internal network). Its notebooks folder is a size-capped tmpfs.
    - ``hailer-fwd-<id>-<suffix>``: :mod:`hailer._forward` from the same image, published on
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
        suffix: str | None = None,
    ) -> None:
        self.config = config
        self.runner: DockerRunner = runner if runner is not None else SubprocessDockerRunner()
        self.names = docker_names(config.workspace, suffix)
        self._token_factory = token_factory
        self.start_timeout = start_timeout
        self._health = health
        self._user = user
        self._engine_version: str | None = None
        self._cpus = config.kernel.cpus
        self._prepared = False
        #: The :class:`OwnerLock` id of the current start (its owner label); set by :meth:`start`.
        self.owner_id = ""

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
        """Rows ``kernel``, ``docker`` (CLI, engine, version), ``image`` (present, version) and
        ``data`` (path warnings). An image that is not downloaded yet is only a warning: ``prepare``
        downloads it. The data folders that are never mounted are ``config`` problems
        (:func:`hailer.config.docker_mount_problems`)."""
        rows = [Check("kernel", True, self.describe(), fatal=False)]
        try:
            version = self.engine_version()
        except KernelRuntimeError as err:
            rows.append(Check("docker", False, str(err), hint=err.hint))
        else:
            rows.append(Check("docker", True, f"Docker {version} (Linux engine)", fatal=False))
            rows.append(self._image_check())
        data = self._data_check()
        if data is not None:
            rows.append(data)
        return rows

    def _image_check(self) -> Check:
        try:
            found = self._image_contract()
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
        if found != kernel_image.contract_tag():
            err = self._mismatch(found)
            return Check("image", False, str(err), hint=err.hint)
        return Check("image", True, f"{self.image} (kernel contract {found})", fatal=False)

    def _data_check(self) -> Check | None:
        if _on_windows() and is_unc_path(self.config.data_dir):
            return None  # fatal, and said once: the config row (docker_mount_problems) refuses it
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

    def _image_contract(self) -> str | None:
        try:
            return kernel_image.image_contract(self.image, self.runner)
        except subprocess.TimeoutExpired as err:
            raise KernelRuntimeError(
                f"Docker did not answer within {int(err.timeout or 0)} s when asked about {self.image}.",
                hint="Check that Docker Desktop is running and responsive, then run the command again.",
            ) from err

    def _mismatch(self, found: str) -> KernelRuntimeError:
        return KernelRuntimeError(
            f"The kernel image {self.image} has kernel contract {found or '(no contract label)'}; "
            f"this Hailer needs {kernel_image.contract_tag()}.",
            hint=(
                "The image must match this Hailer's kernel contract (another marimo would break code mode). "
                "Download the matching image with `uvx hailer kernel pull`, or build it on this machine with "
                "`uvx hailer kernel build`. "
                "[kernel].image in hailer.toml (or HAILER_KERNEL_IMAGE) picks another image."
            ),
        )

    def _pull_failed(self, detail: str = "") -> KernelRuntimeError:
        if any(sign in detail.lower() for sign in _UNPUBLISHED_SIGNS):
            return KernelRuntimeError(
                f"The kernel image {kernel_image.contract_tag()} is not published (or not visible to you): {self.image}.",
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
        found = self._image_contract()
        if found is None:
            raise self._pull_failed()
        if found != kernel_image.contract_tag():
            raise self._mismatch(found)
        return found

    def pull(self, say: Callable[[str], None] | None = None) -> str:
        """``uvx hailer kernel pull``: download the image (docker's progress in this terminal) even
        when a copy is here, and check its contract label. Returns that contract."""
        self.engine_version()
        (say or _say)(f"Downloading {self.image} ...")
        return self._download()

    def mount_problems(self) -> list[str]:
        """Why the data folder must not be mounted (empty: it is fine). Every rule lives in :func:`hailer.config.docker_mount_problems`, which ``validate()`` reports too;
        this runs on every start path, ``--foreground`` included (it never validates)."""
        from hailer.config import docker_mount_problems  # lazy: config does not import the runtimes

        return docker_mount_problems(self.config, windows=_on_windows())

    def _check_mounts(self) -> None:
        problems = self.mount_problems()
        if problems:
            raise KernelRuntimeError(problems[0], hint="\n".join(problems[1:]))

    def prepare(self, say: Callable[[str], None] | None = None) -> None:
        """Docker runs a Linux engine, the data folder is safe to mount, and the image of this Hailer's
        kernel contract is here: downloaded now (with docker's progress in this terminal) when it is
        missing. A ``[kernel].cpus`` above
        what Docker has is lowered, with a note. Raises ``KernelRuntimeError``."""
        out = say or _say
        self._check_mounts()
        self.engine_version()
        found = self._image_contract()
        if found is None:
            out(f"Downloading the kernel image {self.image} (first use; this can take a few minutes) ...")
            found = self._download()
        if found != kernel_image.contract_tag():
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
            "--label", f"{LABEL_CONTRACT}={kernel_image.contract_tag()}",
            "--label", f"{LABEL_ROLE}={role}",
            "--label", f"{LABEL_OWNER}={self.owner_id}",
        ]  # fmt: skip

    def network_command(self) -> list[str]:
        return ["network", "create", "--internal", *self._labels("network"), self.names.network]

    def kernel_command(self, port: int, token_file: Path) -> list[str]:
        """``docker run`` for the kernel. Nothing of this machine is mounted but the data folder
        (read-only) and ``token_file`` (read-only, at :data:`KERNEL_TOKEN_FILE`, where marimo
        reads its token): not the notebooks folder (the kernel's own is a tmpfs owned by its user,
        at most :data:`NOTEBOOKS_TMPFS_SIZE`), the workspace root, ``.hailer/``,
        ``.config/hailer/``, the home folder or the Docker socket. The token itself is never an
        argument or an environment variable. ``--memory-swap`` equal to ``--memory``: no swap on
        top."""
        kernel = self.config.kernel
        user = self._user()
        uid, gid = user if user is not None else (IMAGE_UID, IMAGE_GID)
        cmd = ["run", "-d", "--pull", "never", "--name", self.names.kernel]
        if kernel.network:
            cmd += ["-p", f"127.0.0.1:{port}:{KERNEL_PORT}"]
        else:
            cmd += ["--network", self.names.network]
        cmd += [
            "--init", "--read-only", "--tmpfs", "/tmp",
            "--tmpfs", f"{KERNEL_HOME}:uid={uid},gid={gid}",
            "--tmpfs", f"{KERNEL_NOTEBOOKS_DIR}:uid={uid},gid={gid},mode=0700,size={NOTEBOOKS_TMPFS_SIZE}",
        ]  # fmt: skip
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
            "--mount", mount_arg(self.config.data_dir, KERNEL_DATA_DIR, readonly=True),
            "--mount", mount_arg(token_file, KERNEL_TOKEN_FILE, readonly=True),
            self.image,
            "marimo", "edit", KERNEL_NOTEBOOKS_DIR.name,
            "--host", "0.0.0.0",
            "--port", str(KERNEL_PORT),
            "--headless",
            "--skip-update-check",
            "--token-password-file", str(KERNEL_TOKEN_FILE),
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

    def _listed(self, args: Sequence[str], *, what: str) -> list[list[str]]:
        """The tab-separated rows a ``--format`` listing printed (every row padded to 5 fields)."""
        lines = [line.strip() for line in (self._docker(args, what=what).stdout or "").splitlines() if line.strip()]
        return [(line.split("\t") + [""] * 5)[:5] for line in lines]

    def labelled_containers(self) -> list[Labelled]:
        """Every container, running or not, labelled with this workspace."""
        label = f"label={LABEL_WORKSPACE}={self.workspace_label}"
        fmt = "\t".join(["{{.ID}}", "{{.Names}}", "{{.State}}", *(f'{{{{.Label "{key}"}}}}' for key in _LISTED_LABELS)])
        rows = self._listed(["ps", "-a", "--filter", label, "--format", fmt], what="list this workspace's containers")
        return [Labelled(id=row[0], name=row[1], state=row[2].lower(), role=row[3], owner=row[4]) for row in rows]

    def labelled_networks(self) -> list[Labelled]:
        """Every network labelled with this workspace (``state`` is empty)."""
        label = f"label={LABEL_WORKSPACE}={self.workspace_label}"
        fmt = "\t".join(["{{.ID}}", "{{.Name}}", "", *(f'{{{{.Label "{key}"}}}}' for key in _LISTED_LABELS)])
        rows = self._listed(["network", "ls", "--filter", label, "--format", fmt], what="list this workspace's networks")
        return [Labelled(id=row[0], name=row[1] or row[0], state="", role=row[3], owner=row[4]) for row in rows]

    def remove_leftovers(self, *, every: bool = False) -> Removal:
        """Remove containers and networks labelled with this workspace, containers first.

        ``every=False`` (a start): only those whose owner is not alive (:func:`owner_alive`: its
        Hailer ended, however, or the object has no owner label), running or not; anything of a
        live owner stays. ``every=True`` (``uvx hailer kernel stop``): everything goes. Raises
        only ``KernelRuntimeError`` when Docker cannot list them.
        """
        workspace = Path(self.config.workspace)
        alive: dict[str, bool] = {}

        def gone(item: Labelled) -> bool:
            if item.owner not in alive:
                alive[item.owner] = owner_alive(workspace, item.owner)
            return every or not alive[item.owner]

        outcome = Removal()
        for container in self.labelled_containers():
            if gone(container):
                outcome.add(_remove(self.runner, "container", container.id, container.name))
        for network in self.labelled_networks():
            if gone(network):
                outcome.add(_remove(self.runner, "network", network.id, network.name))
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

    # -- lifecycle ------------------------------------------------------------ #

    def start(self, port: int, *, foreground: bool = False) -> MarimoSandbox:
        """Take an :class:`OwnerLock`, remove what owners that are gone left (:meth:`remove_leftovers`;
        a failure there is a warning, since the new names never clash), then create the network, the
        kernel and the forwarder and wait, through the forwarder, until marimo answers with this
        start's token (the containers are checked first). The token reaches marimo in a file
        (:func:`write_token_file`) that is deleted once the wait is over. Then the workspace's
        notebooks are copied in (before anything can open one) and the periodic copy back starts
        (:class:`~hailer.notebook_sync.NotebookSync`); Ctrl+C meanwhile stops the kernel.
        ``foreground`` changes nothing here: the containers run detached either way, and
        :meth:`DockerKernel.wait` follows the kernel's log."""
        del foreground
        config = self.config
        workspace = Path(config.workspace)
        if not self._prepared:
            self.prepare()
        self._check_mounts()
        try:
            owner = acquire_owner_lock(workspace)
        except OSError as err:
            raise KernelRuntimeError(
                f"Could not create the kernel's owner lock in {workspace / '.hailer'}: {err}",
                hint="Check the permissions and free space in .hailer, then try again.",
            ) from err
        self.owner_id = owner.id
        names = self.names
        network_access = config.kernel.network
        token = self._token_factory()
        url = f"http://127.0.0.1:{port}"
        server = MarimoServer(url=url, token=token, runtime=self.name, network_access=network_access)
        kernel = DockerKernel(
            server=server,
            notebooks_path=str(KERNEL_NOTEBOOKS_DIR),
            data_path=str(KERNEL_DATA_DIR),
            data_dir=config.data_dir,
            log_hint=f"docker logs {names.kernel}",
            runner=self.runner,
            owner=owner,
        )
        health = self._health if self._health is not None else wait_for_health
        try:
            sweep_owner_locks(workspace)
            leftovers = self.remove_leftovers()
            if leftovers.failed:
                _say("Warning: Docker could not remove what an earlier kernel left behind: " + "; ".join(leftovers.failed))
            try:
                config.data_dir.mkdir(parents=True, exist_ok=True)  # before Docker creates a missing one owned by root
            except OSError as err:
                raise KernelRuntimeError(f"Could not create {config.data_dir}: {err}", hint="Check the folder's permissions.") from err
            try:
                token_file = write_token_file(workspace, token)
            except OSError as err:
                raise KernelRuntimeError(
                    f"Could not write the kernel's token file in {workspace / '.hailer'}: {err}",
                    hint="Check the permissions and free space in .hailer, then try again.",
                ) from err
        except BaseException:
            owner.release()
            raise
        try:
            if not network_access:
                created = self._docker(self.network_command(), what="create the kernel's internal network")
                kernel.network, kernel.network_id = names.network, self._created_id(created, names.network)
            created = self._docker(self.kernel_command(port, token_file), what="start the kernel container")
            kernel.containers, kernel.container_ids = (names.kernel,), (self._created_id(created, names.kernel),)
            if not network_access:
                created = self._docker(self.forwarder_command(port), what="create the forwarder container")
                forwarder = self._created_id(created, names.forwarder)
                kernel.containers, kernel.container_ids = (*kernel.containers, names.forwarder), (*kernel.container_ids, forwarder)
                self._docker(["network", "connect", kernel.network_id or names.network, forwarder], what="connect the forwarder to the kernel's network")
                self._docker(["start", forwarder], what="start the forwarder container")
            ids = kernel.container_ids
            healthy = health(url, self.start_timeout, token=token, should_stop=lambda: bool(self._stopped(ids)))
        except BaseException:  # a docker error, or Ctrl+C while waiting: never leave containers behind
            kernel.stop()
            raise
        finally:  # marimo reads the file as it starts: answering, or gone, it no longer needs it
            warning = remove_token_folder(token_file.parent)
            if warning:
                _say(warning)
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
        sync = NotebookSync(kernel, config.notebooks_root, workspace)
        try:
            sync.sync_in()
            sync.start()
        except BaseException:  # Ctrl+C while copying in: the caller never got the kernel
            sync.close(final=False)
            kernel.stop()
            raise
        kernel.sync = sync
        note_kernel_start(workspace, self.name, config.notebooks_root)
        return kernel


#: The labels ``ps`` and ``network ls`` list after id, name and state.
_LISTED_LABELS = (LABEL_ROLE, LABEL_OWNER)


# --------------------------------------------------------------------------- #
# uvx hailer status and uvx hailer kernel stop
# --------------------------------------------------------------------------- #


def workspace_kernels(config: HailerConfig, runner: DockerRunner | None = None) -> list[str]:
    """``uvx hailer status``: this workspace's running docker kernels, one line each (empty: none),
    whatever ``[kernel] runtime`` says, whenever the docker CLI is there. A Docker that cannot be
    asked is one line saying so. Local kernels are not recorded anywhere. Never raises."""
    if runner is None and shutil.which("docker") is None:
        return []
    docker = DockerRuntime(config, runner=runner)
    try:
        docker.engine_version()
        containers = docker.labelled_containers()
    except KernelRuntimeError as err:
        return [] if str(err) == DOCKER_NOT_INSTALLED else [f"docker: not checked ({err})"]
    workspace = Path(config.workspace)
    return [
        f"docker {c.name} (" + ("its Hailer session is running" if owner_alive(workspace, c.owner) else "its Hailer session has ended; the next start removes it") + ")"
        for c in containers
        if c.running and c.role == "kernel"
    ]


@dataclass
class StopReport:
    """What ``uvx hailer kernel stop`` did (``done``) and what it could not do (``failed``)."""

    done: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


def stop_workspace_kernels(config: HailerConfig, runner: DockerRunner | None = None) -> StopReport:
    """``uvx hailer kernel stop``: remove every container and network labelled with this workspace,
    running or not, whichever session started it. It reports what it actually removed; failures
    are reported, never raised. Nothing in ``.hailer`` is touched (a session still starting keeps
    its owner lock and token folder). A local kernel is recorded nowhere: the session that
    started it stops it."""
    report = StopReport()
    docker = DockerRuntime(config, runner=runner)
    docker_error: KernelRuntimeError | None = None
    try:
        docker.engine_version()
        leftovers = docker.remove_leftovers(every=True)
    except KernelRuntimeError as err:
        docker_error = err
    else:
        if leftovers.removed:
            report.done.append("Removed " + ", ".join(leftovers.removed) + ".")
        report.failed += [f"Could not remove {failure}" for failure in leftovers.failed]
    if docker_error is not None and str(docker_error) == DOCKER_NOT_RUNNING:
        report.done.append(
            "Docker is not running, so containers a docker kernel may have left were not checked "
            "(start Docker Desktop and run uvx hailer kernel stop again to remove them)."
        )
    elif docker_error is not None and str(docker_error) != DOCKER_NOT_INSTALLED:
        report.failed.append(f"{docker_error} {docker_error.hint or ''}".strip())
    return report


__all__ = [
    "LABEL_WORKSPACE",
    "DockerKernel",
    "DockerRunner",
    "DockerRuntime",
    "Removal",
    "StopReport",
    "OwnerLock",
    "SubprocessDockerRunner",
    "acquire_owner_lock",
    "data_path_problems",
    "docker_names",
    "owner_alive",
    "remove_token_folder",
    "stop_workspace_kernels",
    "workspace_kernels",
    "write_token_file",
]
