"""Where the notebook kernel runs: the kernel runtimes and the state they share.

A *runtime* starts, finds and stops the marimo server whose kernels run the model's Python:

- :class:`LocalRuntime` runs marimo in Hailer's own Python, as the user (the default). The
  server gets a random token and an environment without secrets.
- ``DockerRuntime`` (not in this build) runs it in a container that sees only the notebooks
  folder and, read-only, the data folder. :func:`runtime_for` says so instead of falling back.

Pieces every runtime shares:

- :class:`PathMap`: host folders and the kernel folders they are mounted at. Only
  :class:`~hailer.marimo_client.MarimoClient` uses it, at the HTTP boundary.
- ``<workspace>/.hailer/kernel.json`` (:class:`KernelState`): written when Hailer starts a
  server and deleted when it stops it, so chat-only ``uvx hailer``, ``hailer exec`` and a second
  ``hailer notebook`` can attach. It holds the server's token; ``.hailer/`` is git-ignored and is
  never mounted into a container.
- :func:`kernel_environment`: the environment a local marimo server gets.

Import order: this module imports :mod:`hailer.marimo_client`; the client reaches back here only
lazily (``find_server`` reads ``kernel.json``) and names :class:`PathMap` in annotations only.
"""

from __future__ import annotations

import json
import os
import posixpath
import secrets
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
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
from hailer.session import SESSION_DIRNAME

KERNEL_STATE_FILENAME = "kernel.json"
MARIMO_LOG_NAME = "marimo.log"
#: How long a start waits for ``/health`` (counted after any image pull in docker mode).
START_TIMEOUT_SEC = 60.0
#: How long ``--foreground`` gives marimo to shut itself down after Ctrl+C before it is killed.
FOREGROUND_GRACE_SEC = 10.0
#: Where the docker runtime mounts the notebooks folder (read-write) and the data folder (read-only).
KERNEL_NOTEBOOKS_DIR = PurePosixPath("/work/notebooks")
KERNEL_DATA_DIR = PurePosixPath("/work/data")

#: The package-install rule for a kernel that can reach the internet (the local runtime).
LOCAL_PROMPT_NOTES = (
    "- The kernel runs on the user's machine as the user (not isolated).\n"
    "- Do not install packages unless the task truly needs it; if so use `ctx.packages.add()` and say so."
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
    """Runs the ``docker`` CLI for ``DockerRuntime`` (stage 2); tests pass a fake to check the argv."""

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

    def describe(self) -> str:
        return describe_runtime(KernelConfig(runtime=KERNEL_RUNTIME_LOCAL))

    def prompt_notes(self) -> str:
        return LOCAL_PROMPT_NOTES

    def _kernel(self, server: MarimoServer, *, proc: Any = None, log_path: Path | None = None) -> LocalKernel:
        pid = server.pid if server.pid is not None else "?"
        return LocalKernel(
            server=server,
            log_hint=str(log_path) if log_path is not None else "this terminal",
            stop_hint=f"stop it by ending process {pid}",
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


def runtime_for(config: HailerConfig, runner: DockerRunner | None = None) -> KernelRuntime:
    """The runtime ``config.kernel.runtime`` asks for. Never falls back from docker to local."""
    del runner  # the docker runtime's CLI runner (stage 2)
    runtime = config.kernel.runtime
    if runtime == KERNEL_RUNTIME_LOCAL:
        return LocalRuntime(config)
    if runtime == KERNEL_RUNTIME_DOCKER:
        raise KernelRuntimeError(
            "The docker kernel runtime is not implemented in this build of Hailer.",
            hint=(
                'Set [kernel] runtime = "local" in hailer.toml (and unset HAILER_KERNEL) to run notebook code '
                "on this machine without isolation."
            ),
        )
    raise ConfigError(
        f"Unknown kernel runtime {runtime!r}.",
        hint='Set [kernel] runtime to "local" or "docker" in hailer.toml (or HAILER_KERNEL).',
    )


__all__ = [
    "KERNEL_DATA_DIR",
    "KERNEL_NOTEBOOKS_DIR",
    "KERNEL_STATE_FILENAME",
    "DockerRunner",
    "KernelRuntime",
    "KernelState",
    "LocalKernel",
    "LocalProcesses",
    "LocalRuntime",
    "PathMap",
    "RunningKernel",
    "attach_runtime",
    "delete_kernel_state",
    "describe_runtime",
    "docker_paths",
    "kernel_environment",
    "kernel_state_path",
    "read_kernel_state",
    "runtime_for",
    "write_kernel_state",
]
