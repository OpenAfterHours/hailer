"""Where the notebook kernel runs: the kernel runtimes and the state they share.

A *runtime* starts, finds and stops the marimo server whose kernels run the model's Python:

- :class:`LocalRuntime` runs marimo in Hailer's own Python, as the user (the default). The
  server gets a random token and an environment without secrets.
- :class:`DockerRuntime` runs it in a Linux container that sees only the notebooks folder and,
  read-only, the data folder, with no network. It drives the ``docker`` CLI through a
  :class:`DockerRunner` and fails closed: it never falls back to the local runtime.

Pieces every runtime shares:

- :class:`PathMap`: host folders and the kernel folders they are mounted at. Only
  :class:`~hailer.marimo_client.MarimoClient` uses it, at the HTTP boundary.
- ``<workspace>/.hailer/kernel.json`` (:class:`KernelState`): written when Hailer starts a
  server and deleted when it stops it, so chat-only ``uvx hailer``, ``hailer exec`` and a second
  ``hailer notebook`` can attach. It holds the server's token; ``.hailer/`` is git-ignored and is
  never mounted into a container.
- :func:`kernel_environment`: the environment a local marimo server gets.
- :func:`runtime_prompt_notes`: what the model is told about the kernel it works in.

Import order: this module imports :mod:`hailer.marimo_client`; the client reaches back here only
lazily (``find_server`` reads ``kernel.json``) and names :class:`PathMap` in annotations only.
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import secrets
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.parse import urlsplit

from hailer import __version__, kernel_image
from hailer.errors import ConfigError, HailerError, KernelRuntimeError
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
from hailer.session import SESSION_DIRNAME

KERNEL_STATE_FILENAME = "kernel.json"
MARIMO_LOG_NAME = "marimo.log"
#: How long a start waits for ``/health`` (counted after any image pull in docker mode).
START_TIMEOUT_SEC = 60.0
#: How long ``--foreground`` gives marimo to shut itself down after Ctrl+C before it is killed.
FOREGROUND_GRACE_SEC = 10.0
#: Where the docker runtime mounts the notebooks folder (read-write) and the data folder (read-only).
#: The image's working directory, ``/work``, holds an empty ``hailer.toml`` so the starter notebook
#: finds ``WORKSPACE = /work`` and ``DATA_DIR = /work/data``.
KERNEL_WORKDIR = PurePosixPath("/work")
KERNEL_NOTEBOOKS_DIR = PurePosixPath("/work/notebooks")
KERNEL_DATA_DIR = PurePosixPath("/work/data")

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
DOCKER_PROMPT_NOTES = (
    _DOCKER_ISOLATION_NOTE
    + "- No internet access from notebook code: packages cannot be installed (ctx.packages.add is "
    "unavailable) and DuckDB cannot download extensions (INSTALL fails). marimo, Polars, DuckDB, altair "
    "and plotly are installed; work with those.\n"
    "- /tmp is writable but wiped when the session ends."
)
DOCKER_NETWORK_PROMPT_NOTES = (
    _DOCKER_ISOLATION_NOTE
    + "- Notebook code can reach the internet. marimo, Polars, DuckDB, altair and plotly are installed. "
    "Do not install packages unless the task truly needs it; if so use `ctx.packages.add()` and say so "
    "(installs last only until the kernel stops).\n"
    "- /tmp is writable but wiped when the session ends."
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


@dataclass(frozen=True)
class KernelState:
    """The server Hailer started for a workspace, as stored in ``.hailer/kernel.json``."""

    runtime: str
    url: str
    port: int
    token: str = field(repr=False)
    hailer_version: str = __version__
    started: str = field(default_factory=_now)
    pid: int | None = None  # local only
    image: str | None = None  # docker only, from here on
    containers: tuple[str, ...] = ()
    network: str | None = None
    network_access: bool = False
    mounts: tuple[tuple[str, str], ...] = ()  # (host folder, kernel folder)

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
        )

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        if self.runtime != KERNEL_RUNTIME_DOCKER:
            for key in ("image", "containers", "network", "network_access", "mounts"):
                data.pop(key)
        else:
            data.pop("pid")
            data["containers"] = list(self.containers)
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
        containers = raw.get("containers") or []
        mounts = raw.get("mounts") or []
        try:
            return cls(
                runtime=runtime,
                url=url.rstrip("/"),
                port=port,
                token=token,
                hailer_version=str(raw.get("hailer_version") or ""),
                started=str(raw.get("started") or ""),
                pid=pid if isinstance(pid, int) and not isinstance(pid, bool) else None,
                image=raw.get("image") if isinstance(raw.get("image"), str) else None,
                containers=tuple(str(c) for c in containers),
                network=raw.get("network") if isinstance(raw.get("network"), str) else None,
                network_access=raw.get("network_access") is True,
                mounts=tuple((str(host), str(kernel)) for host, kernel in mounts),
            )
        except (TypeError, ValueError):  # mounts that are not pairs, containers that are not a list
            return None


def read_kernel_state(workspace: Path) -> KernelState | None:
    """The recorded server, or ``None`` (no file, unreadable or malformed). Liveness is not checked."""
    try:
        raw = json.loads(kernel_state_path(workspace).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return KernelState.from_json(raw)


def write_kernel_state(workspace: Path, state: KernelState) -> Path:
    """Record ``state`` (atomic write; owner-only permissions on POSIX)."""
    path = kernel_state_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(state.to_json(), handle, indent=2)
    if os.name != "nt":
        os.chmod(tmp, 0o600)  # O_CREAT keeps the mode of a leftover file
    os.replace(tmp, path)
    return path


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
# Environment and descriptions
# --------------------------------------------------------------------------- #


def kernel_environment(config: HailerConfig, environ: Mapping[str, str]) -> dict[str, str]:
    """``environ`` without secrets, for a marimo server Hailer starts on this machine.

    Removed: every declared provider's ``env_key`` and ``OPENAI_API_KEY``, every variable named in
    a provider's ``env_http_headers``, ``HAILER_MARIMO_TOKEN``, and every name ending in ``_KEY``,
    ``_TOKEN``, ``_SECRET`` or ``_PASSWORD`` (the rule log redaction uses; any case). Exact names
    compare case-insensitively on Windows, like the environment itself. Notebook code can still
    read the OS credential store, which is why the local runtime is "not isolated".
    """

    def fold(name: str) -> str:
        return name.upper() if os.name == "nt" else name

    named = {"OPENAI_API_KEY", "HAILER_MARIMO_TOKEN"}
    for provider in config.providers.values():
        if provider.env_key:
            named.add(provider.env_key)
        named.update(provider.env_http_headers.values())
    dropped = {fold(name) for name in named}
    return {
        name: value
        for name, value in environ.items()
        if fold(name) not in dropped and not name.upper().endswith(SECRET_ENV_SUFFIXES)
    }


def describe_runtime(kernel: KernelConfig, *, version: str = __version__) -> str:
    """The text after ``Kernel:`` in the startup panel, ``hailer status`` and ``/status``."""
    if kernel.runtime == KERNEL_RUNTIME_LOCAL:
        return "local (runs as you; not isolated)"
    if kernel.runtime == KERNEL_RUNTIME_DOCKER:
        network = "network on" if kernel.network else "no network"
        return f"docker (hailer-kernel {version}; {network}; data read-only)"
    return f"{kernel.runtime} (unknown runtime)"


def runtime_prompt_notes(kernel: KernelConfig) -> str:
    """The kernel paragraph of the system prompt (what each runtime's ``prompt_notes`` returns).

    A function of the settings, not of a runtime object, so building the prompt never touches
    Docker. The docker notes assume the Workspace block names the kernel paths.
    """
    if kernel.runtime == KERNEL_RUNTIME_DOCKER:
        return DOCKER_NETWORK_PROMPT_NOTES if kernel.network else DOCKER_PROMPT_NOTES
    return LOCAL_PROMPT_NOTES


def attach_runtime(config: HailerConfig, server: MarimoServer | None) -> HailerConfig:
    """``config`` with the kernel settings of the server a chat attaches to.

    Asking for ``local`` may attach to a docker kernel Hailer started for this workspace (it is
    strictly more isolated). That session then runs as docker: the prompt, the paths and the
    ``Kernel:`` line follow the kernel in use, with the network setting it was started with. Any
    other combination returns ``config`` unchanged (``find_server`` never pairs a docker request
    with a local server).
    """
    if server is None or server.runtime != KERNEL_RUNTIME_DOCKER or config.kernel.runtime == KERNEL_RUNTIME_DOCKER:
        return config
    state = read_kernel_state(config.workspace)
    network = bool(state is not None and state.url == server.url.rstrip("/") and state.network_access)
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


def spawn_marimo(
    cmd: list[str], cwd: Path, log_path: Path, *, env: Mapping[str, str] | None = None, stdin_text: str | None = None
) -> subprocess.Popen:
    """Start marimo as a background child of this process, logging to ``log_path``.

    No new console window: the chat runs in this terminal and marimo is stopped when it ends. On
    Windows the child gets its own process group so Ctrl+C in the chat is not delivered to marimo.
    ``stdin_text`` (the token for ``--token-password-file -``) is written to its stdin, which is
    then closed; without it stdin is empty.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "ab")  # noqa: SIM115 - handed to the child; closed with it
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


def log_tail(path: Path, lines: int = 15) -> list[str]:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return text.splitlines()[-lines:]


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
    (``--keep-marimo``). Each runtime subclasses it; the base only knows ``kernel.json``.
    """

    server: MarimoServer
    log_hint: str = ""
    stop_hint: str = ""
    workspace: Path | None = None

    def stop(self) -> None:
        """Stop the kernel and delete ``kernel.json`` when it still records it. Idempotent, never raises."""
        if self.workspace is not None:
            delete_kernel_state(self.workspace, token=self.server.token)

    def alive(self) -> bool:
        return answers_with_token(self.server.url, self.server.token)

    def log_tail(self, lines: int = 15) -> list[str]:
        return []

    def wait(self) -> int:
        """Block while the kernel runs (``--foreground``); Ctrl+C ends the wait. Returns an exit code."""
        try:
            while self.alive():
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        return 0


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

    def alive(self) -> bool:
        if self.proc is not None and self.proc.poll() is not None:
            return False
        return super().alive()

    def log_tail(self, lines: int = 15) -> list[str]:
        return log_tail(self.log_path, lines) if self.log_path is not None else []

    def wait(self) -> int:
        if self.proc is None:
            return super().wait()
        try:
            while True:
                try:
                    return int(self.proc.wait(timeout=0.5))  # a short timeout keeps Ctrl+C deliverable on Windows
                except subprocess.TimeoutExpired:
                    continue
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
        """Get ready to start, with any slow step's progress in this terminal (docker: download a
        missing image). Call it before a spinner; ``start`` does it too. Raises ``KernelRuntimeError``."""
        ...

    def start(self, port: int, *, foreground: bool = False) -> RunningKernel:
        """Start marimo on ``127.0.0.1:port``, wait for ``/health`` and write ``kernel.json``.

        ``foreground`` shows the server's output in this terminal (``hailer notebook --foreground``;
        the caller then blocks in :meth:`RunningKernel.wait`). Raises ``KernelRuntimeError`` (with the
        log tail in the hint) when the server does not come up; nothing is left running then.
        """
        ...

    def find_running(self) -> RunningKernel | None:
        """This runtime's server from ``kernel.json`` when it answers with its token."""
        ...

    def describe(self) -> str:
        """The text after ``Kernel:``."""
        ...

    def prompt_notes(self) -> str:
        """What the model is told about this runtime (a paragraph for the system prompt)."""
        ...


class DockerRunner(Protocol):
    """Runs the ``docker`` CLI for :class:`DockerRuntime` and :mod:`hailer.kernel_image`; tests pass
    a fake to check the argv. ``args`` never include ``docker`` itself. Both methods raise
    ``KernelRuntimeError`` (:func:`docker_not_installed`) when there is no ``docker`` to run."""

    def run(
        self,
        args: Sequence[str],
        *,
        input: str | None = None,
        timeout: float | None = None,
        check: bool = True,
        capture: bool = True,
    ) -> subprocess.CompletedProcess[str]: ...

    def stream(self, args: Sequence[str]) -> int:
        """Run with the console inherited (pull, build, ``logs -f``); returns the exit code."""
        ...


class LocalRuntime:
    """marimo in Hailer's own Python, as the user: not isolated, but the server Hailer starts has
    a random token (passed on stdin, never on the command line) and an environment without secrets."""

    name = KERNEL_RUNTIME_LOCAL

    def __init__(
        self,
        config: HailerConfig,
        *,
        procs: LocalProcesses | None = None,
        environ: Mapping[str, str] | None = None,
        token_factory: Callable[[], str] = new_token,
        start_timeout: float = START_TIMEOUT_SEC,
    ) -> None:
        self.config = config
        self.paths = PathMap()
        self.procs = procs if procs is not None else LocalProcesses()
        self._environ = environ
        self._token_factory = token_factory
        self.start_timeout = start_timeout

    @property
    def log_path(self) -> Path:
        return Path(self.config.workspace) / SESSION_DIRNAME / MARIMO_LOG_NAME

    def check(self) -> list[Check]:
        return [Check("kernel", True, self.describe(), fatal=False)]

    def prepare(self, say: Callable[[str], None] | None = None) -> None:
        del say  # nothing to download: marimo runs in Hailer's own Python

    def describe(self) -> str:
        return describe_runtime(KernelConfig(runtime=KERNEL_RUNTIME_LOCAL))

    def prompt_notes(self) -> str:
        return runtime_prompt_notes(KernelConfig(runtime=KERNEL_RUNTIME_LOCAL))

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
        state = read_kernel_state(self.config.workspace)
        if state is None or state.runtime != self.name or not answers_with_token(state.url, state.token):
            return None
        return self._kernel(state.server(), log_path=self.log_path)

    def start(self, port: int, *, foreground: bool = False) -> RunningKernel:
        config = self.config
        workspace = Path(config.workspace)
        stale = read_kernel_state(workspace)
        if stale is not None and not answers_with_token(stale.url, stale.token):
            delete_kernel_state(workspace)
        url = f"http://127.0.0.1:{port}"
        cmd = marimo_server_command(config.notebooks_root, workspace, port)
        token = self._token_factory()
        env = kernel_environment(config, self._environ if self._environ is not None else os.environ)
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
        write_kernel_state(workspace, KernelState(runtime=self.name, url=url, port=port, token=token, pid=server.pid))
        return kernel


# --------------------------------------------------------------------------- #
# The docker runtime
# --------------------------------------------------------------------------- #

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


def docker_not_installed() -> KernelRuntimeError:
    return KernelRuntimeError(
        "Docker is not installed.",
        hint=(
            "Install Docker Desktop (Windows, macOS) or Docker Engine (Linux), then run the command again. "
            'Or set [kernel] runtime = "local" in hailer.toml to run notebook code on this machine without isolation.'
        ),
    )


def docker_not_running(detail: str = "") -> KernelRuntimeError:
    hint = "Start Docker Desktop (or the Docker service), wait until it is running, and run the command again."
    if detail:
        hint += f"\n{detail}"
    return KernelRuntimeError("Docker is not running.", hint=hint)


def _docker_said(result: subprocess.CompletedProcess[str]) -> str:
    """The last lines docker printed, for a hint ("" when it printed nothing)."""
    text = "\n".join(part for part in (result.stderr, result.stdout) if part)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return ("docker said: " + " / ".join(lines[-3:])) if lines else ""


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
    """The ``docker`` CLI found on PATH, run with :mod:`subprocess` (text mode, UTF-8, empty stdin
    unless ``input`` is given). Docker Desktop and Docker Engine both ship it: no SDK needed."""

    def __init__(self, executable: str | None = None) -> None:
        self._executable = executable

    def _argv(self, args: Sequence[str]) -> list[str]:
        executable = self._executable or shutil.which("docker")
        if not executable:
            raise docker_not_installed()
        return [executable, *args]

    def run(
        self,
        args: Sequence[str],
        *,
        input: str | None = None,
        timeout: float | None = None,
        check: bool = True,
        capture: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                self._argv(args),
                input=input,
                stdin=subprocess.DEVNULL if input is None else None,
                capture_output=capture,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=check,
            )
        except FileNotFoundError as err:
            raise docker_not_installed() from err

    def stream(self, args: Sequence[str]) -> int:
        try:
            proc = subprocess.Popen(self._argv(args), stdin=subprocess.DEVNULL)
        except FileNotFoundError as err:
            raise docker_not_installed() from err
        try:
            while True:
                try:
                    return int(proc.wait(timeout=0.5))  # a short timeout keeps Ctrl+C deliverable on Windows
                except subprocess.TimeoutExpired:
                    continue
        except KeyboardInterrupt:
            _end_process(proc)
            raise


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


def _network_location(text: str, drive_type: Callable[[str], int]) -> str | None:
    """Why the Windows path ``text`` is on the network (UNC, or a mapped network drive); else None."""
    import ntpath

    drive = ntpath.splitdrive(text)[0].replace("/", "\\")
    if drive.upper().startswith(("\\\\?\\", "\\\\.\\")):
        drive = drive[4:]
        if drive.upper().startswith("UNC\\"):
            return "a network share (UNC path)"
    elif drive.startswith("\\\\"):
        return "a network share (UNC path)"
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


def data_path_problems(
    data_dir: Path, *, windows: bool | None = None, drive_type: Callable[[str], int] = _drive_type
) -> list[str]:
    """Reasons the kernel container may not see the data folder, each with its fix (empty: none).

    On Windows: a UNC path or a mapped network drive, which Docker Desktop usually cannot mount.
    Anywhere: symlinks or junctions inside the folder that point outside it, which do not resolve
    in the container (only the folder itself is mounted).
    """
    problems: list[str] = []
    text = str(data_dir)
    if windows if windows is not None else os.name == "nt":
        where = _network_location(text, drive_type)
        if where is not None:  # no link scan: it would walk the share over the network, and the fix is the same
            return [
                f"The data folder {text} is on {where}; Docker Desktop usually cannot mount it. "
                "Copy the data to a folder on a local disk and point [hailer].data_dir at it."
            ]
    links = _links_outside(Path(data_dir))
    if links:
        shown = ", ".join(links[:3]) + (f" and {len(links) - 3} more" if len(links) > 3 else "")
        problems.append(
            f"{len(links)} link(s) in the data folder point outside it ({shown}); they do not resolve in the "
            "container. Copy those files into the data folder."
        )
    return problems


@dataclass
class DockerKernel(RunningKernel):
    """A docker kernel's containers and network: the ones this Hailer started, or found through
    ``kernel.json``. ``containers[0]`` is the kernel; stopping removes them all."""

    runner: Any = None
    containers: tuple[str, ...] = ()
    network: str | None = None

    def stop(self) -> None:
        try:
            if self.containers:
                self.runner.run(["rm", "-f", *self.containers], check=False, timeout=DOCKER_TIMEOUT_SEC)
            if self.network:
                self.runner.run(["network", "rm", self.network], check=False, timeout=DOCKER_TIMEOUT_SEC)
        except Exception:  # noqa: BLE001 - best effort on the way out ("no such container" included)
            pass
        super().stop()

    def log_tail(self, lines: int = 15, *, container: str | None = None) -> list[str]:
        """The last lines of ``docker logs`` for the kernel (or ``container``), token masked."""
        name = container or (self.containers[0] if self.containers else None)
        if name is None:
            return []
        try:
            result = self.runner.run(["logs", "--tail", str(lines), name], check=False, timeout=DOCKER_PROBE_TIMEOUT_SEC)
        except Exception:  # noqa: BLE001 - the log is a courtesy
            return []
        text = "\n".join(part for part in (result.stdout, result.stderr) if part)
        token = self.server.token
        return [line.replace(token, "<token>") if token else line for line in text.splitlines()][-lines:]

    def wait(self) -> int:
        """Follow the kernel's new log lines in this terminal (``--foreground``) until it stops or
        Ctrl+C; the kernel container's exit code (0 after Ctrl+C)."""
        if not self.containers:
            return 1
        try:
            self.runner.stream(["logs", "-f", "--tail", "0", self.containers[0]])
            result = self.runner.run(
                ["inspect", "--format", "{{.State.ExitCode}}", self.containers[0]], check=False, timeout=DOCKER_PROBE_TIMEOUT_SEC
            )
            return int((result.stdout or "").strip() or 1)
        except KeyboardInterrupt:
            return 0
        except (HailerError, OSError, ValueError, subprocess.TimeoutExpired):
            return 1


class DockerRuntime:
    """marimo in a Linux container that sees only the notebooks folder (read-write) and the data
    folder (read-only), without network access or any of the host's secrets.

    One workspace gets (names from :func:`docker_names`, every one labelled with the workspace):

    - ``hailer-net-<id>``: an ``--internal`` network, with no route out.
    - ``hailer-kernel-<id>``: ``marimo edit`` on that network only, hardened (read-only root,
      no capabilities, pid, memory and CPU limits), no port published (Docker ignores ``-p`` on
      an internal network).
    - ``hailer-fwd-<id>``: :mod:`hailer._forward` from the same image, published on
      ``127.0.0.1:<port>`` and joined to both networks; it only ever connects to the kernel.

    ``[kernel] network = true`` runs the kernel on Docker's default network with its port
    published directly, and no forwarder or internal network.

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
    ) -> None:
        self.config = config
        self.paths = docker_paths(config)
        self.runner: DockerRunner = runner if runner is not None else SubprocessDockerRunner()
        self.names = docker_names(config.workspace)
        self._token_factory = token_factory
        self.start_timeout = start_timeout
        self._health = health
        self._user = user
        self._engine_version: str | None = None

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

    def prompt_notes(self) -> str:
        return runtime_prompt_notes(self._kernel_config)

    # -- checks -------------------------------------------------------------- #

    def check(self) -> list[Check]:
        """Rows ``kernel``, ``docker`` (CLI, engine, version), ``image`` (present, version) and
        ``data`` (path warnings). An image that is not downloaded yet is only a warning: ``prepare``
        downloads it."""
        rows = [Check("kernel", True, self.describe(), fatal=False)]
        try:
            version = self.engine_version()
        except KernelRuntimeError as err:
            rows.append(Check("docker", False, str(err), hint=err.hint))
        else:
            rows.append(Check("docker", True, f"Docker {version} (Linux engine)", fatal=False))
            rows.append(self._image_check())
        rows.append(self._data_check())
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
            result = self.runner.run(
                ["version", "--format", "{{.Server.Version}} {{.Server.Os}}"], check=False, timeout=DOCKER_PROBE_TIMEOUT_SEC
            )
        except subprocess.TimeoutExpired:
            raise docker_not_running(f"docker version did not answer within {int(DOCKER_PROBE_TIMEOUT_SEC)} s.") from None
        version, _, engine_os = (result.stdout or "").strip().partition(" ")
        if result.returncode != 0 or not version or version.startswith("<"):  # "<no value>": no server
            raise docker_not_running(_docker_said(result))
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

    def _pull_failed(self) -> KernelRuntimeError:
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
        if kernel_image.pull(self.image, self.runner) != 0:
            raise self._pull_failed()
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

    def prepare(self, say: Callable[[str], None] | None = None) -> None:
        """Docker runs a Linux engine, no kernel Hailer started for this workspace is still running,
        and the image for this Hailer is here, downloaded now (with docker's progress in this
        terminal) when it is missing. Raises ``KernelRuntimeError``."""
        self.engine_version()
        self._refuse_live_kernel()
        found = self._image_version()
        if found is None:
            (say or _say)(f"Downloading the kernel image {self.image} (first use; this can take a few minutes) ...")
            found = self._download()
        if found != __version__:
            raise self._mismatch(found)

    # -- docker commands ----------------------------------------------------- #

    def _docker(self, args: Sequence[str], *, what: str, timeout: float = DOCKER_TIMEOUT_SEC) -> subprocess.CompletedProcess[str]:
        """Run one docker command; ``KernelRuntimeError`` (with docker's message) when it fails."""
        try:
            result = self.runner.run(list(args), check=False, timeout=timeout)
        except subprocess.TimeoutExpired as err:
            raise KernelRuntimeError(
                f"Docker did not {what} within {int(timeout)} s.",
                hint="Check that Docker Desktop is running and responsive, then run the command again.",
            ) from err
        if result.returncode != 0:
            raise KernelRuntimeError(f"Docker could not {what}.", hint=_docker_said(result))
        return result

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
        the home folder or the Docker socket."""
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
            "--cpus", f"{kernel.cpus:g}",
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

    def remove_leftovers(self) -> list[str]:
        """Remove every container and network labelled with this workspace (left by a crash, a
        ``kill`` or ``--keep-marimo``); returns their names."""
        label = f"label={LABEL_WORKSPACE}={self.workspace_label}"
        containers = self._listed(["ps", "-a", "--filter", label, "--format", "{{.Names}}"], what="list this workspace's containers")
        networks = self._listed(["network", "ls", "--filter", label, "--format", "{{.Name}}"], what="list this workspace's networks")
        if containers:
            self._docker(["rm", "-f", *containers], what="remove this workspace's old kernel containers")
        for network in networks:
            self._docker(["network", "rm", network], what=f"remove the network {network}")
        return [*containers, *networks]

    def _stopped(self, containers: Sequence[str]) -> dict[str, str]:
        """The containers among ``containers`` that are not running, with their exit codes ("?"
        when unknown, e.g. the container is gone). Empty when docker cannot be asked."""
        try:
            result = self.runner.run(
                ["inspect", "--format", "{{.Name}} {{.State.Running}} {{.State.ExitCode}}", *containers],
                check=False,
                timeout=DOCKER_PROBE_TIMEOUT_SEC,
            )
        except (HailerError, OSError, subprocess.TimeoutExpired):
            return {}
        running: dict[str, str] = {}
        stopped: dict[str, str] = {}
        for line in (result.stdout or "").splitlines():
            parts = line.split()
            if len(parts) == 3:
                name = parts[0].lstrip("/")
                (running if parts[1] == "true" else stopped)[name] = parts[2]
        for name in containers:
            if name not in running and name not in stopped:
                stopped[name] = "?"
        return stopped

    def _labelled_ours(self, container: str) -> bool:
        try:
            result = self.runner.run(
                ["inspect", "--format", "{{json .Config.Labels}}", container], check=False, timeout=DOCKER_PROBE_TIMEOUT_SEC
            )
            labels = json.loads((result.stdout or "").strip() or "null") if result.returncode == 0 else None
        except (HailerError, OSError, ValueError, subprocess.TimeoutExpired):
            return False
        return isinstance(labels, dict) and labels.get(LABEL_WORKSPACE) == self.workspace_label

    # -- lifecycle ------------------------------------------------------------ #

    def _kernel(self, server: MarimoServer, containers: tuple[str, ...], network: str | None) -> DockerKernel:
        return DockerKernel(
            server=server,
            log_hint=f"docker logs {containers[0] if containers else self.names.kernel}",
            stop_hint="stop it with uvx hailer kernel stop",
            workspace=Path(self.config.workspace),
            runner=self.runner,
            containers=containers,
            network=network,
        )

    def find_running(self) -> RunningKernel | None:
        """The docker kernel ``kernel.json`` records, when it answers with its token and its
        container carries this workspace's label."""
        state = read_kernel_state(Path(self.config.workspace))
        if state is None or state.runtime != self.name or not state.containers:
            return None
        if not answers_with_token(state.url, state.token) or not self._labelled_ours(state.containers[0]):
            return None
        return self._kernel(state.server(), state.containers, state.network)

    def _refuse_live_kernel(self) -> None:
        """Never start over a live server Hailer started for this workspace: starting would replace
        its ``kernel.json`` record and orphan it (a local one), or clash with its names (docker)."""
        state = read_kernel_state(Path(self.config.workspace))
        if state is None or not answers_with_token(state.url, state.token):
            return
        if state.runtime == KERNEL_RUNTIME_DOCKER:
            raise KernelRuntimeError(
                f"A docker kernel Hailer started for this workspace is already running at {state.url}.",
                hint="uvx hailer in this workspace attaches to it; uvx hailer kernel stop stops it.",
            )
        raise KernelRuntimeError(
            f"A local marimo server Hailer started for this workspace is still running at {state.url}.",
            hint=(
                "Stop it first: end the Hailer session that started it (or its --foreground terminal), or run "
                "uvx hailer kernel stop. Then run this command again."
            ),
        )

    def _clear_the_way(self) -> None:
        """After :meth:`_refuse_live_kernel`: drop the stale ``kernel.json`` (if any) and remove
        leftover containers and networks labelled with this workspace."""
        delete_kernel_state(Path(self.config.workspace))
        self.remove_leftovers()

    def start(self, port: int, *, foreground: bool = False) -> RunningKernel:
        """Create the network, the kernel and the forwarder, wait for ``/health`` through the
        forwarder and write ``kernel.json``. ``foreground`` changes nothing here: the containers run
        detached either way, and :meth:`DockerKernel.wait` follows the kernel's log."""
        del foreground
        config = self.config
        workspace = Path(config.workspace)
        self.prepare()
        self._clear_the_way()
        for folder in (config.notebooks_root, config.data_dir):
            try:
                folder.mkdir(parents=True, exist_ok=True)  # before Docker creates a missing one owned by root
            except OSError as err:
                raise KernelRuntimeError(f"Could not create {folder}: {err}", hint="Check the folder's permissions.") from err
        names = self.names
        network_access = config.kernel.network
        containers = (names.kernel,) if network_access else (names.kernel, names.forwarder)
        network = None if network_access else names.network
        token = self._token_factory()
        url = f"http://127.0.0.1:{port}"
        server = MarimoServer(
            url=url, server_id=f"127.0.0.1:{port}", source="kernel", token=token, runtime=self.name, paths=self.paths
        )
        kernel = self._kernel(server, containers, network)
        health = self._health if self._health is not None else wait_for_health
        try:
            if network is not None:
                self._docker(self.network_command(), what="create the kernel's internal network")
            self._docker(self.kernel_command(port, token), what="start the kernel container")
            if network is not None:
                self._docker(self.forwarder_command(port), what="create the forwarder container")
                self._docker(["network", "connect", network, names.forwarder], what="connect the forwarder to the kernel's network")
                self._docker(["start", names.forwarder], what="start the forwarder container")
            healthy = health(url, self.start_timeout, should_stop=lambda: bool(self._stopped(containers)))
        except BaseException:  # a docker error, or Ctrl+C while waiting: never leave containers behind
            kernel.stop()
            raise
        if not healthy:
            stopped = self._stopped(containers)
            if names.kernel in stopped:
                message = f"The kernel container exited early (code {stopped[names.kernel]})."
                shown = names.kernel
            elif stopped:
                shown = next(iter(stopped))
                message = f"The forwarder container exited early (code {stopped[shown]})."
            else:
                message = f"Marimo did not answer on {url} within {int(self.start_timeout)} s."
                shown = names.kernel
            tail = kernel.log_tail(container=shown)
            kernel.stop()
            hint = f"Last lines of docker logs {shown}:" if tail else f"docker logs {shown} was empty."
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
                containers=containers,
                network=network,
                network_access=network_access,
                mounts=tuple((str(host), str(kernel_path)) for host, kernel_path in self.paths.mounts),
            ),
        )
        return kernel


def stop_workspace_kernels(
    config: HailerConfig, runner: DockerRunner | None = None, *, procs: LocalProcesses | None = None
) -> list[str]:
    """``uvx hailer kernel stop``: stop the server ``kernel.json`` records (either runtime) and
    remove every container and network labelled with this workspace. One line per thing done.

    A local server is stopped by its pid only when it answers with its token (so a stale record
    never kills an unrelated process that reused the pid). Raises ``KernelRuntimeError`` only when
    a docker kernel still answers but Docker cannot be reached to remove it (``kernel.json`` is
    kept then).
    """
    workspace = Path(config.workspace)
    docker = DockerRuntime(config, runner=runner)
    try:
        docker.engine_version()
        docker_error: KernelRuntimeError | None = None
    except KernelRuntimeError as err:
        docker_error = err
    done: list[str] = []
    state = read_kernel_state(workspace)
    if state is not None and state.runtime == KERNEL_RUNTIME_DOCKER:
        if docker_error is None:
            docker._kernel(state.server(), state.containers, state.network).stop()
            removed = [*state.containers, *([state.network] if state.network else [])]
            done.append(f"Stopped the docker kernel at {state.url} (removed {', '.join(removed)}).")
        elif answers_with_token(state.url, state.token):
            raise docker_error
        else:
            delete_kernel_state(workspace)
            done.append(f"Removed the record of a docker kernel that no longer answers ({state.url}).")
    elif state is not None:
        running = LocalRuntime(config, procs=procs).find_running()
        if running is not None:
            running.stop()
            process = f" (process {state.pid})" if state.pid is not None else ""
            done.append(f"Stopped the local marimo server at {state.url}{process}.")
        else:
            delete_kernel_state(workspace)
            done.append(f"Removed the record of a marimo server that no longer answers ({state.url}).")
    if docker_error is None:
        leftovers = docker.remove_leftovers()
        if leftovers:
            done.append("Removed " + ", ".join(leftovers) + ".")
    return done


def runtime_for(config: HailerConfig, runner: DockerRunner | None = None) -> KernelRuntime:
    """The runtime ``config.kernel.runtime`` asks for. Never falls back from docker to local."""
    runtime = config.kernel.runtime
    if runtime == KERNEL_RUNTIME_LOCAL:
        return LocalRuntime(config)
    if runtime == KERNEL_RUNTIME_DOCKER:
        return DockerRuntime(config, runner=runner)
    raise ConfigError(
        f"Unknown kernel runtime {runtime!r}.",
        hint='Set [kernel] runtime to "local" or "docker" in hailer.toml (or HAILER_KERNEL).',
    )


__all__ = [
    "KERNEL_DATA_DIR",
    "KERNEL_NOTEBOOKS_DIR",
    "KERNEL_STATE_FILENAME",
    "KERNEL_WORKDIR",
    "DockerKernel",
    "DockerNames",
    "DockerRunner",
    "DockerRuntime",
    "KernelRuntime",
    "KernelState",
    "LocalKernel",
    "LocalProcesses",
    "LocalRuntime",
    "PathMap",
    "RunningKernel",
    "SubprocessDockerRunner",
    "attach_runtime",
    "data_path_problems",
    "delete_kernel_state",
    "describe_runtime",
    "docker_names",
    "docker_not_installed",
    "docker_paths",
    "kernel_environment",
    "kernel_state_path",
    "mount_arg",
    "read_kernel_state",
    "runtime_for",
    "runtime_prompt_notes",
    "stop_workspace_kernels",
    "workspace_id",
    "write_kernel_state",
]
