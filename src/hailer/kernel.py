"""Where the notebook kernel runs: the kernel runtimes and the state they share.

A *runtime* starts, finds and stops the marimo server whose kernels run the model's Python:

- :class:`LocalRuntime` (here) runs marimo in Hailer's own Python, as the user (the default).
  The server gets a random token and an environment without the secrets Hailer can recognise.
- :class:`~hailer.kernel_docker.DockerRuntime` runs it in a Linux container that sees only the
  notebooks folder and, read-only, the data folder, with no network. It fails closed: it never
  falls back to the local runtime.

Pieces every runtime shares:

- :class:`PathMap`: host folders and the kernel folders they are mounted at. Only
  :class:`~hailer.marimo_client.MarimoClient` uses it, at the HTTP boundary.
- ``<workspace>/.hailer/kernel.json`` (:class:`KernelState`): written when Hailer starts a
  server and deleted when it stops it, so chat-only ``uvx hailer``, ``hailer exec`` and a second
  ``hailer notebook`` can attach. It holds the server's token; ``.hailer/`` is git-ignored and is
  never mounted into a container. :func:`live_kernel_state` is the one way to read it *and* prove
  the server it names is up; every caller (``find_server``, ``find_running``, the start guard,
  ``kernel stop``) goes through it.
- :func:`kernel_environment`: the environment a local marimo server gets.
- :func:`runtime_prompt_notes`: what the model is told about the kernel it works in.

Import order: this module imports :mod:`hailer.marimo_client`; the client reaches back here only
lazily (``find_server`` reads ``kernel.json``) and names :class:`PathMap` in annotations only.
:mod:`hailer.kernel_docker` imports this module; this one imports it lazily.
"""

from __future__ import annotations

import json
import os
import posixpath
import secrets
import signal
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.parse import urlsplit

from hailer import __version__
from hailer.errors import ConfigError, KernelRuntimeError
from hailer.log import SECRET_ENV_SUFFIXES
from hailer.marimo_client import (
    answers_with_token,
    marimo_server_command,
    notebook_file_key,
    remove_registry_entry,
    wait_for_health,
)
from hailer.models import (
    KERNEL_RUNTIME_DOCKER,
    KERNEL_RUNTIME_LOCAL,
    Check,
    HailerConfig,
    KernelConfig,
    MarimoServer,
)
from hailer.statedir import STATE_DIRNAME as SESSION_DIRNAME, ensure_state_dir

KERNEL_STATE_FILENAME = "kernel.json"
#: Which runtime last started a server on which notebooks folder (``.hailer/last-kernel.json``).
LAST_KERNEL_FILENAME = "last-kernel.json"
MARIMO_LOG_NAME = "marimo.log"
#: How long a start waits for ``/health`` (counted after any image pull in docker mode).
START_TIMEOUT_SEC = 60.0
#: How long ``--foreground`` gives marimo to shut itself down after Ctrl+C before it is killed.
FOREGROUND_GRACE_SEC = 10.0
#: Where the docker runtime mounts the notebooks folder (read-write) and the data folder (read-only).
#: The image's working directory, ``/work``, holds marimo's user configuration (``.marimo.toml``:
#: a notebook's cells run when it opens) and an empty ``hailer.toml``, so the starter notebook
#: finds ``WORKSPACE = /work`` and ``DATA_DIR = /work/data``.
KERNEL_WORKDIR = PurePosixPath("/work")
KERNEL_NOTEBOOKS_DIR = PurePosixPath("/work/notebooks")
KERNEL_DATA_DIR = PurePosixPath("/work/data")

#: Names that mark an environment variable as a secret the local kernel does not get: the log
#: redaction suffixes plus these (any case), and the exact names below.
KERNEL_SECRET_SUFFIXES: tuple[str, ...] = (
    *SECRET_ENV_SUFFIXES, "_PASSWD", "_PWD", "_CREDENTIALS", "_CONNECTION_STRING", "APIKEY",
)  # fmt: skip
KERNEL_SECRET_NAMES: tuple[str, ...] = ("PGPASSWORD", "MYSQL_PWD", "PASSWORD", "SECRET", "TOKEN")

#: The package-install rule for a kernel that can reach the internet (the local runtime).
LOCAL_PROMPT_NOTES = (
    "- The kernel runs on the user's machine as the user (not isolated).\n"
    "- Do not install packages unless the task truly needs it; if so use `ctx.packages.add()` and say so."
)
#: The docker runtime, as the model needs to know it (the Workspace block gives the two folders).
_DOCKER_ISOLATION_NOTE = (
    "- The kernel runs in an isolated Linux container that sees only the notebooks folder and the data "
    "folder above; always use those paths in code.\n"
)
_DOCKER_TMP_NOTE = (
    "- /tmp is scratch space in memory, shared by every notebook in the container and wiped when the kernel stops."
)
DOCKER_PROMPT_NOTES = (
    _DOCKER_ISOLATION_NOTE
    + "- No internet access from notebook code: packages cannot be installed (ctx.packages.add is "
    "unavailable) and DuckDB cannot download extensions (INSTALL fails). marimo, Polars, fastexcel, DuckDB, altair "
    "and plotly are installed; work with those.\n" + _DOCKER_TMP_NOTE
)
DOCKER_NETWORK_PROMPT_NOTES = (
    _DOCKER_ISOLATION_NOTE
    + "- Notebook code has network access: the internet, and also services on the user's machine and in "
    "other containers. marimo, Polars, fastexcel, DuckDB, altair and plotly are installed. Do not install packages "
    "unless the task truly needs it; if so use `ctx.packages.add()` and say so (installs last only until "
    "the kernel stops).\n" + _DOCKER_TMP_NOTE
)


# --------------------------------------------------------------------------- #
# Paths: host <-> kernel
# --------------------------------------------------------------------------- #


def _host_parts(path: Path | str) -> tuple[str, ...]:
    """``path`` split into parts after the ``notebook_file_key`` normalisation (absolute, no ``..``,
    symlinks and junctions not resolved)."""
    absolute = os.path.normpath(str(Path(path).expanduser().absolute()))
    return Path(absolute).parts


def _same_parts(prefix: tuple[str, ...], parts: tuple[str, ...]) -> bool:
    """True when ``parts`` starts with ``prefix`` (each part compared with ``os.path.normcase``)."""
    if len(prefix) > len(parts):
        return False
    return all(os.path.normcase(a) == os.path.normcase(b) for a, b in zip(prefix, parts))


@dataclass(frozen=True)
class PathMap:
    """Host folders and the kernel folders they are mounted at; no mounts means identity.

    :class:`~hailer.marimo_client.MarimoClient` translates at exactly two places: the ``?file=``
    key it sends becomes a kernel path (:meth:`to_kernel`) and the session paths marimo reports
    become host paths (:meth:`to_host`). Everything above the client works in host paths.

    Host paths are normalised like :func:`~hailer.marimo_client.notebook_file_key` and compared
    case-insensitively on Windows. A subfolder maps to the same subfolder; the longest matching
    mount wins.
    """

    mounts: tuple[tuple[Path, PurePosixPath], ...] = ()

    @property
    def identity(self) -> bool:
        return not self.mounts

    def to_kernel(self, host: Path | str) -> str:
        """The path the kernel knows ``host`` by (POSIX). Identity: :func:`notebook_file_key`.

        ``ValueError`` when ``host`` is outside every mounted folder.
        """
        if not self.mounts:
            return notebook_file_key(Path(host))
        parts = _host_parts(host)
        best: tuple[int, PurePosixPath, tuple[str, ...]] | None = None
        for host_root, kernel_root in self.mounts:
            root = _host_parts(host_root)
            if _same_parts(root, parts) and (best is None or len(root) > best[0]):
                best = (len(root), kernel_root, parts[len(root):])
        if best is None:
            raise ValueError(f"{notebook_file_key(Path(host))} is outside every folder mounted into the kernel")
        _length, kernel_root, rest = best
        return str(kernel_root.joinpath(*rest))

    def to_host(self, kernel: str) -> str | None:
        """The host path for a path the kernel reported; ``None`` when it is outside every mount
        (or relative). Identity: ``kernel`` unchanged."""
        if not self.mounts:
            return kernel
        if not kernel or not kernel.startswith("/"):
            return None
        parts = PurePosixPath(posixpath.normpath(kernel)).parts
        best: tuple[int, Path, tuple[str, ...]] | None = None
        for host_root, kernel_root in self.mounts:
            prefix = kernel_root.parts
            if parts[: len(prefix)] == prefix and (best is None or len(prefix) > best[0]):
                best = (len(prefix), host_root, parts[len(prefix):])
        if best is None:
            return None
        _length, host_root, rest = best
        return str(Path(os.path.normpath(str(Path(host_root).expanduser().absolute()))).joinpath(*rest))


def docker_paths(config: HailerConfig) -> PathMap:
    """The docker runtime's mounts: the notebooks folder and the data folder."""
    return PathMap(((config.notebooks_root, KERNEL_NOTEBOOKS_DIR), (config.data_dir, KERNEL_DATA_DIR)))


# --------------------------------------------------------------------------- #
# .hailer/kernel.json
# --------------------------------------------------------------------------- #


def kernel_state_path(workspace: Path) -> Path:
    """``<workspace>/.hailer/kernel.json``."""
    return Path(workspace) / SESSION_DIRNAME / KERNEL_STATE_FILENAME


def new_token() -> str:
    """A random auth token for a server Hailer starts."""
    return secrets.token_urlsafe(32)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


#: Fields only a docker kernel records (a local record leaves them out of the file).
_DOCKER_FIELDS = ("image", "containers", "container_ids", "network", "network_id", "network_access", "mounts", "memory", "cpus")


@dataclass(frozen=True)
class KernelState:
    """The server Hailer started for a workspace, as stored in ``.hailer/kernel.json``.

    A docker record names its containers and network (for messages and ``docker logs``) and
    records their ids: stopping removes by id, so a stale handle can never remove a newer kernel
    that reuses the names. It also records the settings the kernel was started with (image,
    network, mounts, memory, cpus), so a later run can tell when hailer.toml no longer matches.
    """

    runtime: str
    url: str
    port: int
    token: str = field(repr=False)
    hailer_version: str = __version__
    started: str = field(default_factory=_now)
    pid: int | None = None  # local only
    image: str | None = None  # docker only, from here on
    containers: tuple[str, ...] = ()  # names: kernel first
    container_ids: tuple[str, ...] = ()  # the same containers' ids, in the same order
    network: str | None = None
    network_id: str | None = None
    network_access: bool = False
    mounts: tuple[tuple[str, str], ...] = ()  # (host folder, kernel folder)
    memory: str | None = None
    cpus: float | None = None

    @property
    def paths(self) -> PathMap | None:
        if not self.mounts:
            return None
        return PathMap(tuple((Path(host), PurePosixPath(kernel)) for host, kernel in self.mounts))

    def server(self) -> MarimoServer:
        parts = urlsplit(self.url)
        return MarimoServer(
            url=self.url,
            server_id=f"{parts.hostname or '127.0.0.1'}:{parts.port or self.port}",
            pid=self.pid,
            source="kernel",
            token=self.token,
            runtime=self.runtime,
            paths=self.paths,
            network_access=self.runtime == KERNEL_RUNTIME_DOCKER and self.network_access,
        )

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        if self.runtime != KERNEL_RUNTIME_DOCKER:
            for key in _DOCKER_FIELDS:
                data.pop(key)
        else:
            data.pop("pid")
            data["containers"] = list(self.containers)
            data["container_ids"] = list(self.container_ids)
            data["mounts"] = [list(pair) for pair in self.mounts]
        return {key: value for key, value in data.items() if value is not None}

    @classmethod
    def from_json(cls, raw: object) -> KernelState | None:
        """The state in ``raw``; ``None`` when a required field is missing or has the wrong type."""
        if not isinstance(raw, dict):
            return None
        runtime, url, port, token = raw.get("runtime"), raw.get("url"), raw.get("port"), raw.get("token")
        if not (isinstance(runtime, str) and isinstance(url, str) and url.startswith("http") and isinstance(token, str) and token):
            return None
        if isinstance(port, bool) or not isinstance(port, int):
            return None
        pid = raw.get("pid")
        cpus = raw.get("cpus")

        def text(key: str) -> str | None:
            value = raw.get(key)
            return value if isinstance(value, str) else None

        try:
            return cls(
                runtime=runtime,
                url=url.rstrip("/"),
                port=port,
                token=token,
                hailer_version=str(raw.get("hailer_version") or ""),
                started=str(raw.get("started") or ""),
                pid=pid if isinstance(pid, int) and not isinstance(pid, bool) else None,
                image=text("image"),
                containers=tuple(str(c) for c in raw.get("containers") or []),
                container_ids=tuple(str(c) for c in raw.get("container_ids") or []),
                network=text("network"),
                network_id=text("network_id"),
                network_access=raw.get("network_access") is True,
                mounts=tuple((str(host), str(kernel)) for host, kernel in raw.get("mounts") or []),
                memory=text("memory"),
                cpus=float(cpus) if isinstance(cpus, (int, float)) and not isinstance(cpus, bool) else None,
            )
        except (TypeError, ValueError):  # mounts that are not pairs, containers that are not a list
            return None


def read_kernel_state(workspace: Path) -> KernelState | None:
    """The recorded server, or ``None`` (no file, unreadable or malformed). Liveness is not
    checked: use :func:`live_kernel_state` for a server to talk to."""
    try:
        raw = json.loads(kernel_state_path(workspace).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return KernelState.from_json(raw)


def _write_private(path: Path, data: object) -> Path:
    """Write ``data`` as JSON to ``path`` (atomic; owner-only permissions on POSIX)."""
    ensure_state_dir(path.parent.parent)
    tmp = path.with_suffix(".json.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    if os.name != "nt":
        os.chmod(tmp, 0o600)  # O_CREAT keeps the mode of a leftover file
    os.replace(tmp, path)
    return path


def write_kernel_state(workspace: Path, state: KernelState) -> Path:
    """Record ``state`` (atomic write; owner-only permissions on POSIX)."""
    return _write_private(kernel_state_path(workspace), state.to_json())


def delete_kernel_state(workspace: Path, *, token: str | None = None) -> bool:
    """Delete ``kernel.json``. With ``token``, only when the file still records that server (a newer
    start may have replaced it); a malformed file is deleted either way. Never raises."""
    path = kernel_state_path(workspace)
    if token is not None and path.exists():
        state = read_kernel_state(workspace)
        if state is not None and state.token != token:
            return False
    try:
        path.unlink()
        return True
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# Is the recorded server up? One answer for every caller
# --------------------------------------------------------------------------- #

Probe = Callable[[str, "str | None"], bool]
_STILL_ACTIVE = 259  # GetExitCodeProcess: the process has not exited
_ERROR_INVALID_PARAMETER = 87  # OpenProcess: no process has that id


def process_running(pid: int) -> bool | None:
    """Whether process ``pid`` exists: True or False when the OS says so, ``None`` when it cannot tell."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not handle:
                return False if ctypes.get_last_error() == _ERROR_INVALID_PARAMETER else None  # 5: denied, so it exists
            try:
                code = wintypes.DWORD()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return None
                return code.value == _STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        except (AttributeError, OSError):
            return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


def record_is_gone(state: KernelState, *, docker_runner: Any = None) -> bool:
    """True only when the server ``state`` records is provably not running: its local process has
    ended, or Docker says its kernel container is gone or stopped. Anything uncertain is False."""
    if state.runtime == KERNEL_RUNTIME_DOCKER:
        from hailer.kernel_docker import containers_gone  # lazy: kernel_docker imports this module

        return containers_gone(state, docker_runner)
    return state.pid is not None and process_running(state.pid) is False


def live_kernel_state(
    workspace: Path,
    *,
    probe: Probe | None = None,
    gone: Callable[[KernelState], bool] | None = None,
) -> KernelState | None:
    """The record in ``.hailer/kernel.json`` when its server answers with its token; else ``None``.

    Every caller uses this: ``find_server``, both runtimes' ``find_running`` and start guard, and
    ``kernel stop``. A record whose server does not answer is deleted only when it is provably
    dead (``gone``, default :func:`record_is_gone`), so later calls stop waiting on it; a server
    that is merely slow to answer is never taken for dead. ``probe`` defaults to
    :func:`~hailer.marimo_client.answers_with_token`.
    """
    state = read_kernel_state(workspace)
    if state is None:
        return None
    if (probe or answers_with_token)(state.url, state.token):
        return state
    is_gone = gone if gone is not None else record_is_gone
    try:
        dead = is_gone(state)
    except Exception:  # noqa: BLE001 - uncertain: keep the record
        dead = False
    if dead:
        delete_kernel_state(workspace, token=state.token)
    return None


def live_kernel_error(state: KernelState) -> KernelRuntimeError:
    """Why no new server may start while ``state``'s server runs (it would be orphaned, or clash)."""
    if state.runtime == KERNEL_RUNTIME_DOCKER:
        return KernelRuntimeError(
            f"A docker kernel Hailer started for this workspace is already running at {state.url}.",
            hint=(
                "uvx hailer notebook and uvx hailer in this workspace attach to it; uvx hailer kernel stop "
                "stops it (then run this command again)."
            ),
        )
    return KernelRuntimeError(
        f"A local marimo server Hailer started for this workspace is still running at {state.url}.",
        hint=(
            "Stop it first: end the Hailer session that started it (or its --foreground terminal), or run "
            "uvx hailer kernel stop. Then run this command again."
        ),
    )


def unanswered_kernel_error(state: KernelState) -> KernelRuntimeError:
    """Why no new server may start while ``state``'s server may still run but does not answer."""
    if state.runtime == KERNEL_RUNTIME_DOCKER:
        what = f"A docker kernel Hailer started for this workspace ({', '.join(state.containers) or 'its containers'})"
    else:
        what = "A local marimo server Hailer started for this workspace" + (f" (process {state.pid})" if state.pid is not None else "")
    return KernelRuntimeError(
        f"{what} may still be running, but it does not answer at {state.url}.",
        hint=(
            "It may be busy, stuck or suspended; starting another would orphan it. Wait and run this command "
            "again, or stop it with uvx hailer kernel stop first."
        ),
    )


def refuse_live_kernel(workspace: Path, *, probe: Probe | None = None, gone: Callable[[KernelState], bool] | None = None) -> None:
    """The start guard both runtimes share: ``KernelRuntimeError`` while ``kernel.json`` records a
    server that answers, or one that does not answer but is not provably gone (its process still
    exists, its containers are still there): a busy or suspended server must not be orphaned by
    a new start that replaces its record. A record that is provably gone is deleted."""
    state = read_kernel_state(workspace)
    if state is None:
        return
    if (probe or answers_with_token)(state.url, state.token):
        raise live_kernel_error(state)
    is_gone = gone if gone is not None else record_is_gone
    try:
        dead = is_gone(state)
    except Exception:  # noqa: BLE001 - uncertain: keep the record, refuse
        dead = False
    if not dead:
        raise unanswered_kernel_error(state)
    delete_kernel_state(workspace, token=state.token)


# --------------------------------------------------------------------------- #
# Which runtime last used the notebooks folder (a docker kernel's notebooks, later run locally)
# --------------------------------------------------------------------------- #


def _last_kernel_path(workspace: Path) -> Path:
    return Path(workspace) / SESSION_DIRNAME / LAST_KERNEL_FILENAME


def note_kernel_start(workspace: Path, runtime: str, notebooks_root: Path) -> None:
    """Remember that ``runtime`` started a server on ``notebooks_root`` (never raises)."""
    try:
        _write_private(_last_kernel_path(workspace), {"runtime": runtime, "notebooks": str(notebooks_root), "started": _now()})
    except OSError:
        pass


def docker_wrote_notebooks(workspace: Path, notebooks_root: Path) -> bool:
    """True when the last server Hailer started on ``notebooks_root`` was a docker kernel: code the
    container wrote into those notebooks would now run on this machine."""
    try:
        raw = json.loads(_last_kernel_path(workspace).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(raw, dict) or raw.get("runtime") != KERNEL_RUNTIME_DOCKER or not isinstance(raw.get("notebooks"), str):
        return False
    recorded, current = _host_parts(raw["notebooks"]), _host_parts(notebooks_root)
    return len(recorded) == len(current) and _same_parts(recorded, current)


# --------------------------------------------------------------------------- #
# Environment and descriptions
# --------------------------------------------------------------------------- #


def withheld_variables(config: HailerConfig, environ: Mapping[str, str]) -> list[str]:
    """The names in ``environ`` a local marimo server does not get, sorted.

    Withheld: every declared provider's ``env_key`` and ``OPENAI_API_KEY``, every variable named in
    a provider's ``env_http_headers``, ``HAILER_MARIMO_TOKEN``, the names in
    :data:`KERNEL_SECRET_NAMES`, and every name ending in one of :data:`KERNEL_SECRET_SUFFIXES`
    (``_KEY``, ``_TOKEN``, ``_SECRET``, ``_PASSWORD``, ``_PASSWD``, ``_PWD``, ``_CREDENTIALS``,
    ``_CONNECTION_STRING``, ``APIKEY``; any case). ``[kernel].pass_env`` names are never withheld.
    Exact names compare case-insensitively on Windows, like the environment itself.
    """

    def fold(name: str) -> str:
        return name.upper() if os.name == "nt" else name

    named = {"OPENAI_API_KEY", "HAILER_MARIMO_TOKEN", *KERNEL_SECRET_NAMES}
    for provider in config.providers.values():
        if provider.env_key:
            named.add(provider.env_key)
        named.update(provider.env_http_headers.values())
    dropped = {fold(name) for name in named}
    passed = {fold(name) for name in config.kernel.pass_env}
    return sorted(
        name
        for name in environ
        if fold(name) not in passed and (fold(name) in dropped or name.upper().endswith(KERNEL_SECRET_SUFFIXES))
    )


def kernel_environment(config: HailerConfig, environ: Mapping[str, str]) -> dict[str, str]:
    """``environ`` without the secrets :func:`withheld_variables` names, for a marimo server Hailer
    starts on this machine. Notebook code can still read the OS credential store and files, which
    is why the local runtime is "not isolated"."""
    withheld = set(withheld_variables(config, environ))
    return {name: value for name, value in environ.items() if name not in withheld}


def describe_runtime(kernel: KernelConfig, *, version: str = __version__) -> str:
    """The text after ``Kernel:`` in the startup panel, ``hailer status`` and ``/status``."""
    if kernel.runtime == KERNEL_RUNTIME_LOCAL:
        return "local (runs as you; not isolated)"
    if kernel.runtime == KERNEL_RUNTIME_DOCKER:
        network = "network on: the internet and this machine" if kernel.network else "no network"
        return f"docker (hailer-kernel {version}; {network}; data read-only)"
    return f"{kernel.runtime} (unknown runtime)"


def runtime_prompt_notes(kernel: KernelConfig) -> str:
    """The kernel paragraph of the system prompt.

    A function of the settings in effect (the CLI applies :func:`attach_runtime` first), so
    building the prompt never touches Docker. The docker notes assume the Workspace block names
    the kernel paths.
    """
    if kernel.runtime == KERNEL_RUNTIME_DOCKER:
        return DOCKER_NETWORK_PROMPT_NOTES if kernel.network else DOCKER_PROMPT_NOTES
    return LOCAL_PROMPT_NOTES


def attach_runtime(config: HailerConfig, server: MarimoServer | None) -> HailerConfig:
    """``config`` with the kernel settings of the server a chat attaches to.

    A docker kernel sets the runtime and the network setting, whatever hailer.toml says: the
    prompt, the paths and the ``Kernel:`` line follow the kernel in use. (Asking for ``local``
    may attach to a docker kernel Hailer started for this workspace: it is strictly more
    isolated; ``find_server`` never pairs a docker request with a local server.) Any other server
    returns ``config`` unchanged.
    """
    if server is None or server.runtime != KERNEL_RUNTIME_DOCKER:
        return config
    network = bool(server.network_access)
    if config.kernel.runtime == KERNEL_RUNTIME_DOCKER and config.kernel.network == network:
        return config
    return replace(config, kernel=replace(config.kernel, runtime=KERNEL_RUNTIME_DOCKER, network=network))


# --------------------------------------------------------------------------- #
# Processes (the local runtime)
# --------------------------------------------------------------------------- #


def _feed_stdin(proc: Any, text: str) -> None:
    """Write ``text`` (the token) to the child's stdin and close it; marimo reads up to EOF."""
    try:
        proc.stdin.write(text.encode("utf-8"))
    except (OSError, ValueError):  # the child exited already; the health wait reports it
        pass
    finally:
        try:
            proc.stdin.close()
        except (OSError, ValueError):
            pass


def _fresh_log(log_path: Path) -> Any:
    """``log_path`` emptied and opened for the child (owner-only on POSIX): marimo prints its
    signed-in URL, so an old server's token never stays behind in the log."""
    ensure_state_dir(log_path.parent.parent)
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0), 0o600)
    if os.name != "nt":
        os.chmod(log_path, 0o600)
    return os.fdopen(fd, "wb")


def spawn_marimo(
    cmd: list[str], cwd: Path, log_path: Path, *, env: Mapping[str, str] | None = None, stdin_text: str | None = None
) -> subprocess.Popen:
    """Start marimo as a background child of this process, logging to ``log_path`` (a fresh log).

    No new console window: the chat runs in this terminal and marimo is stopped when it ends. On
    Windows the child gets its own process group so Ctrl+C in the chat is not delivered to marimo.
    ``stdin_text`` (the token for ``--token-password-file -``) is written to its stdin, which is
    then closed; without it stdin is empty.
    """
    log = _fresh_log(log_path)
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=flags,
            env=dict(env) if env is not None else None,
        )
    finally:
        log.close()
    if stdin_text is not None:
        _feed_stdin(proc, stdin_text)
    return proc


def attach_marimo(
    cmd: list[str], cwd: Path, *, env: Mapping[str, str] | None = None, stdin_text: str | None = None
) -> subprocess.Popen:
    """Start marimo attached to this terminal (``--foreground``): its output shows here and Ctrl+C
    reaches it. With the token on stdin, stdin is not a terminal, so marimo quits on Ctrl+C without
    asking."""
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdin=subprocess.PIPE if stdin_text is not None else None,
        env=dict(env) if env is not None else None,
    )
    if stdin_text is not None:
        _feed_stdin(proc, stdin_text)
    return proc


def kill_tree(pid: int) -> None:
    """Windows: kill a process and everything it spawned (marimo runs kernels as children)."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def kill_pid(pid: int) -> None:
    """Stop a server Hailer started in an earlier run (known only by its pid). Never raises."""
    try:
        if os.name == "nt":
            kill_tree(pid)
        else:
            os.kill(pid, signal.SIGTERM)  # marimo shuts its kernels down on SIGTERM
    except OSError:
        pass


def stop_process(proc: Any, timeout: float = 5.0, *, kill_tree: Callable[[int], None] = kill_tree) -> None:
    """terminate → wait up to ``timeout`` → kill; never raises. On Windows the tree goes first."""
    try:
        if proc.poll() is not None:
            return
        if os.name == "nt":
            kill_tree(proc.pid)
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except Exception:  # noqa: BLE001 - subprocess.TimeoutExpired or a fake
            proc.kill()
            try:
                proc.wait(timeout=timeout)
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001 - best effort on the way out
        pass


def log_tail(path: Path, lines: int = 15, *, token: str | None = None) -> list[str]:
    """The last ``lines`` lines of ``path`` with ``token`` masked (``[]`` when it cannot be read)."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [line.replace(token, "<token>") if token else line for line in text.splitlines()[-lines:]]


@dataclass
class LocalProcesses:
    """Everything :class:`LocalRuntime` does to the operating system; tests replace members."""

    spawn: Callable[..., Any] = spawn_marimo
    attach: Callable[..., Any] = attach_marimo
    kill_tree: Callable[[int], None] = kill_tree
    kill_pid: Callable[[int], None] = kill_pid
    wait_for_health: Callable[..., bool] = wait_for_health
    remove_registry_entry: Callable[[str], bool] = remove_registry_entry


# --------------------------------------------------------------------------- #
# Runtimes
# --------------------------------------------------------------------------- #


@dataclass
class RunningKernel:
    """A marimo server a runtime started (or found running) for this workspace.

    ``server`` carries the url, token, runtime and path map every client needs. ``log_hint`` says
    where its log is; ``stop_hint`` how a person stops it when Hailer leaves it running
    (``--keep-marimo``). ``ended`` says why :meth:`wait` returned when the kernel stopped by
    itself (empty after Ctrl+C). Each runtime subclasses it; the base only knows ``kernel.json``.
    """

    server: MarimoServer
    log_hint: str = ""
    stop_hint: str = ""
    workspace: Path | None = None
    ended: str = ""

    def stop(self) -> None:
        """Stop the kernel and delete ``kernel.json`` when it still records it. Idempotent, never raises."""
        if self.workspace is not None:
            delete_kernel_state(self.workspace, token=self.server.token)

    def log_tail(self, lines: int = 15) -> list[str]:
        return []

    def wait(self) -> int:
        """Block while the kernel runs (``--foreground``); Ctrl+C ends the wait. Returns an exit code."""
        raise NotImplementedError


@dataclass
class LocalKernel(RunningKernel):
    """A marimo process on this machine: the one this Hailer started (``proc``) or, found through
    ``kernel.json``, one started by an earlier run (known by ``server.pid``)."""

    proc: Any = None
    log_path: Path | None = None
    procs: LocalProcesses = field(default_factory=LocalProcesses)

    def stop(self) -> None:
        try:
            if self.proc is not None:
                stop_process(self.proc, kill_tree=self.procs.kill_tree)
            elif self.server.pid is not None:
                self.procs.kill_pid(self.server.pid)
            self.procs.remove_registry_entry(self.server.url)
        except Exception:  # noqa: BLE001 - best effort on the way out
            pass
        super().stop()

    def log_tail(self, lines: int = 15) -> list[str]:
        return log_tail(self.log_path, lines, token=self.server.token) if self.log_path is not None else []

    def wait(self) -> int:
        """Until marimo exits or Ctrl+C. When it exits by itself, ``ended`` says so (ended from
        outside this terminal, or failed; its own output is above)."""
        if self.proc is None:
            return 0
        try:
            while True:
                try:
                    code = int(self.proc.wait(timeout=0.5))  # a short timeout keeps Ctrl+C deliverable on Windows
                except subprocess.TimeoutExpired:
                    continue
                self.ended = (
                    "marimo shut itself down."
                    if code == 0
                    else f"marimo stopped (exit code {code}): it was ended from outside this terminal (uvx hailer kernel "
                    "stop, Task Manager or kill), or it failed; its own output is above."
                )
                return code
        except KeyboardInterrupt:
            # marimo got the Ctrl+C too and shuts down by itself (its stdin is not a terminal).
            try:
                return int(self.proc.wait(timeout=FOREGROUND_GRACE_SEC))
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                return 130


class KernelRuntime(Protocol):
    """Starts, finds and describes the marimo server for one workspace."""

    name: str  # "local" | "docker"
    paths: PathMap

    def check(self) -> list[Check]:
        """Rows for ``hailer doctor`` and ``hailer notebook``: can this runtime run a kernel here?"""
        ...

    def prepare(self, say: Callable[[str], None] | None = None) -> None:
        """Get ready to start, with any slow step's progress and any warning in this terminal
        (docker: download a missing image). Call it before a spinner; ``start`` does it when it has
        not been done. Raises ``KernelRuntimeError``."""
        ...

    def start(self, port: int, *, foreground: bool = False) -> RunningKernel:
        """Start marimo on ``127.0.0.1:port``, wait for ``/health`` and write ``kernel.json``.

        Refused (``KernelRuntimeError``) while a server Hailer started for this workspace still
        answers. ``foreground`` shows the server's output in this terminal (``hailer notebook
        --foreground``; the caller then blocks in :meth:`RunningKernel.wait`). Raises
        ``KernelRuntimeError`` (with the log tail in the hint) when the server does not come up;
        nothing is left running then.
        """
        ...

    def find_running(self) -> RunningKernel | None:
        """This runtime's server from ``kernel.json`` when it answers with its token."""
        ...

    def describe(self) -> str:
        """The text after ``Kernel:``."""
        ...


class LocalRuntime:
    """marimo in Hailer's own Python, as the user: not isolated, but the server Hailer starts has
    a random token (passed on stdin, never on the command line) and an environment without the
    secrets :func:`withheld_variables` recognises."""

    name = KERNEL_RUNTIME_LOCAL

    def __init__(
        self,
        config: HailerConfig,
        *,
        procs: LocalProcesses | None = None,
        environ: Mapping[str, str] | None = None,
        token_factory: Callable[[], str] = new_token,
        start_timeout: float = START_TIMEOUT_SEC,
        probe: Probe | None = None,
    ) -> None:
        self.config = config
        self.paths = PathMap()
        self.procs = procs if procs is not None else LocalProcesses()
        self._environ = environ
        self._token_factory = token_factory
        self.start_timeout = start_timeout
        self._probe = probe

    @property
    def log_path(self) -> Path:
        return Path(self.config.workspace) / SESSION_DIRNAME / MARIMO_LOG_NAME

    @property
    def environ(self) -> Mapping[str, str]:
        return self._environ if self._environ is not None else os.environ

    def check(self) -> list[Check]:
        summary = self.describe()
        withheld = withheld_variables(self.config, self.environ)
        if withheld:
            summary += f"; {len(withheld)} secret-looking environment variable(s) withheld from notebook code"
        passed = self.config.kernel.pass_env
        if passed:
            summary += f"; passed through by [kernel].pass_env: {', '.join(passed)}"
        elif withheld:
            summary += " ([kernel].pass_env lets named ones through)"
        return [Check("kernel", True, summary, fatal=False)]

    def prepare(self, say: Callable[[str], None] | None = None) -> None:
        """Nothing to download: marimo runs in Hailer's own Python. Refuses (like :meth:`start`)
        while a server Hailer started for this workspace still answers, and warns when a docker
        kernel was the last to run these notebooks: from here on their code runs on this machine."""
        refuse_live_kernel(self.config.workspace, probe=self._probe)
        if docker_wrote_notebooks(self.config.workspace, self.config.notebooks_root):
            out = say or _print
            out(
                f"Warning: the notebooks in {self.config.notebooks_root} were last run by the isolated docker kernel; "
                "in local mode their code runs on this machine as you. Open only notebooks you trust."
            )
            out('    To keep them isolated: uvx hailer notebook --kernel docker (or [kernel] runtime = "docker").')

    def describe(self) -> str:
        return describe_runtime(KernelConfig(runtime=KERNEL_RUNTIME_LOCAL))

    def _kernel(self, server: MarimoServer, *, proc: Any = None, log_path: Path | None = None) -> LocalKernel:
        pid = server.pid if server.pid is not None else "?"
        return LocalKernel(
            server=server,
            log_hint=str(log_path) if log_path is not None else "this terminal",
            stop_hint=f"stop it with uvx hailer kernel stop, or by ending process {pid}",
            workspace=self.config.workspace,
            proc=proc,
            log_path=log_path,
            procs=self.procs,
        )

    def find_running(self) -> RunningKernel | None:
        state = live_kernel_state(self.config.workspace, probe=self._probe)
        if state is None or state.runtime != self.name:
            return None
        return self._kernel(state.server(), log_path=self.log_path)

    def start(self, port: int, *, foreground: bool = False) -> RunningKernel:
        config = self.config
        workspace = Path(config.workspace)
        refuse_live_kernel(workspace, probe=self._probe)
        url = f"http://127.0.0.1:{port}"
        cmd = marimo_server_command(config.notebooks_root, workspace, port)
        token = self._token_factory()
        env = kernel_environment(config, self.environ)
        log_path = None if foreground else self.log_path
        try:
            if foreground:
                proc = self.procs.attach(cmd, workspace, env=env, stdin_text=token)
            else:
                proc = self.procs.spawn(cmd, workspace, log_path, env=env, stdin_text=token)
        except OSError as err:
            raise KernelRuntimeError(
                f"Could not start marimo: {err}",
                hint="Check that Hailer's Python environment is intact (uvx reinstalls it with `uvx --reinstall hailer`).",
            ) from err
        server = MarimoServer(
            url=url,
            server_id=f"127.0.0.1:{port}",
            pid=getattr(proc, "pid", None),
            source="kernel",
            token=token,
            runtime=self.name,
        )
        kernel = self._kernel(server, proc=proc, log_path=log_path)
        try:
            healthy = self.procs.wait_for_health(url, self.start_timeout, should_stop=lambda: proc.poll() is not None)
        except BaseException:  # Ctrl+C while waiting: never leave a server behind
            kernel.stop()
            raise
        if not healthy:
            if proc.poll() is not None:
                message = f"Marimo exited early (code {proc.returncode})."
            else:
                message = f"Marimo did not answer on {url} within {int(self.start_timeout)} s."
            tail = kernel.log_tail()
            kernel.stop()
            hint = f"Log: {kernel.log_hint}" if log_path is not None else "marimo's own output is above."
            if tail:
                hint += "\n" + "\n".join(f"    {line}" for line in tail)
            raise KernelRuntimeError(message, hint=hint)
        try:
            write_kernel_state(workspace, KernelState(runtime=self.name, url=url, port=port, token=token, pid=server.pid))
        except OSError as err:
            kernel.stop()
            raise KernelRuntimeError(
                f"Could not record the local kernel: {err}", hint="Check the permissions and free space in .hailer, then try again.",
            ) from err
        note_kernel_start(workspace, self.name, config.notebooks_root)
        return kernel


def _print(text: str) -> None:
    print(text, flush=True)


def runtime_for(config: HailerConfig, runner: Any = None) -> KernelRuntime:
    """The runtime ``config.kernel.runtime`` asks for; ``ConfigError`` for an unknown one. Nothing
    is checked yet (``check``, ``prepare`` and ``start`` do that), and docker never falls back to
    local."""
    runtime = config.kernel.runtime
    if runtime == KERNEL_RUNTIME_LOCAL:
        return LocalRuntime(config)
    if runtime == KERNEL_RUNTIME_DOCKER:
        from hailer.kernel_docker import DockerRuntime  # lazy: most runs never touch Docker

        return DockerRuntime(config, runner=runner)
    raise ConfigError(
        f'Unknown kernel runtime "{runtime}".',
        hint='Set [kernel] runtime to "local" or "docker" in hailer.toml (or HAILER_KERNEL).',
    )


__all__ = [
    "KERNEL_DATA_DIR",
    "KERNEL_NOTEBOOKS_DIR",
    "KERNEL_WORKDIR",
    "KernelRuntime",
    "KernelState",
    "LocalRuntime",
    "PathMap",
    "RunningKernel",
    "attach_runtime",
    "describe_runtime",
    "docker_paths",
    "kernel_environment",
    "kernel_state_path",
    "live_kernel_state",
    "read_kernel_state",
    "runtime_for",
    "runtime_prompt_notes",
    "write_kernel_state",
]
