"""Where the notebook kernel runs: the kernel runtimes and the pieces they share.

Every ``uvx hailer`` (and ``uvx hailer notebook``) process starts its own marimo server through a
*runtime*, keeps the :class:`~hailer.sandbox.Sandbox` its start returns in memory for as long as
the chat runs, and stops it on exit. Nothing ever finds or attaches to a server another process
started.

- :class:`~hailer.kernel_docker.DockerRuntime` (the default) runs it in a Linux container with a
  notebooks folder of its own (the workspace's notebooks are copied in and back) that sees only
  the data folder, read-only, with no network. It fails closed: it never falls back to the
  unsafe-local runtime.
- :class:`LocalRuntime` (here; ``runtime = "unsafe-local"``) runs marimo in Hailer's own Python,
  as the user, with their files and network. The server gets a random token and an environment
  without the secrets Hailer can recognise; nothing else is isolated.

Pieces every runtime shares:

- :class:`~hailer.sandbox.MarimoSandbox`: the started kernel and its files; each runtime's kernel
  subclasses it and fills in where the kernel sees the notebooks and data folders.
- :func:`kernel_environment`: the environment an unsafe-local marimo server gets.
- :func:`runtime_prompt_notes`: what the model is told about the kernel it works in.

Import order: this module imports :mod:`hailer.marimo_client` and :mod:`hailer.sandbox`.
:mod:`hailer.kernel_docker` imports this module; this one imports it lazily.
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from hailer.errors import ConfigError, KernelRuntimeError
from hailer.log import SECRET_ENV_SUFFIXES
from hailer.marimo_client import marimo_server_command, notebook_file_key, wait_for_health
from hailer.models import (
    KERNEL_RUNTIME_DOCKER,
    KERNEL_RUNTIME_UNSAFE_LOCAL,
    Check,
    HailerConfig,
    KernelConfig,
    MarimoServer,
)
from hailer.sandbox import MarimoSandbox
from hailer.statedir import ensure_state_dir, state_dir

#: Which runtime last started a server on which notebooks folder (``.hailer/last-kernel.json``).
LAST_KERNEL_FILENAME = "last-kernel.json"
#: ``.hailer/marimo-<pid>.log``: the unsafe-local runtime's log, one per Hailer process, deleted on stop.
LOCAL_LOG_PREFIX = "marimo-"
#: How long a start waits for ``/health`` (counted after any image pull in docker mode).
START_TIMEOUT_SEC = 60.0
#: How long ``--foreground`` gives marimo to shut itself down after Ctrl+C before it is killed.
FOREGROUND_GRACE_SEC = 10.0
#: The docker kernel's notebooks folder (a tmpfs of its own) and where it mounts the data folder
#: (read-only). The image's working directory, ``/work``, holds marimo's user configuration (``.marimo.toml``:
#: a notebook's cells run when it opens) and an empty ``hailer.toml``, so the starter notebook
#: finds ``WORKSPACE = /work`` and ``DATA_DIR = /work/data``.
KERNEL_WORKDIR = PurePosixPath("/work")
KERNEL_NOTEBOOKS_DIR = PurePosixPath("/work/notebooks")
KERNEL_DATA_DIR = PurePosixPath("/work/data")

#: Names that mark an environment variable as a secret the unsafe-local kernel does not get: the log
#: redaction suffixes plus these (any case), and the exact names below.
KERNEL_SECRET_SUFFIXES: tuple[str, ...] = (
    *SECRET_ENV_SUFFIXES, "_PASSWD", "_PWD", "_CREDENTIALS", "_CONNECTION_STRING", "APIKEY",
)  # fmt: skip
KERNEL_SECRET_NAMES: tuple[str, ...] = ("PGPASSWORD", "MYSQL_PWD", "PASSWORD", "SECRET", "TOKEN")

#: How to leave the unsafe-local runtime (its ``doctor`` row's hint).
UNSAFE_LOCAL_HINT = (
    "Notebook code runs as you, with your files and network. To isolate it, install and start Docker, then set "
    '[kernel] runtime = "docker" in hailer.toml (the default; unset HAILER_KERNEL if it says unsafe-local).'
)
#: The unsafe-local runtime, which can reach the internet and the user's files.
LOCAL_PROMPT_NOTES = (
    "- The kernel runs on the user's machine as the user, with their files and network (the unsafe-local "
    "runtime: not isolated). Read and write only what the task needs.\n"
    "- Do not install packages unless the task truly needs it; if so use `ctx.packages.add()` and say so."
)
#: The docker runtime, as the model needs to know it (the Workspace block gives the two folders).
_DOCKER_ISOLATION_NOTE = (
    "- The kernel runs in an isolated Linux container that sees only the notebooks folder and the data "
    "folder above; always use those paths in code.\n"
    "- Only marimo notebooks in the notebooks folder are copied back to the user's machine; any other file "
    "notebook code writes there (or anywhere in the container) is gone when the kernel stops.\n"
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
# Host paths
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


# --------------------------------------------------------------------------- #
# Small state helpers
# --------------------------------------------------------------------------- #


def new_token() -> str:
    """A random auth token for a server Hailer starts."""
    return secrets.token_urlsafe(32)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def local_log_path(workspace: Path, pid: int | None = None) -> Path:
    """``<workspace>/.hailer/marimo-<pid>.log``: this process's unsafe-local marimo log."""
    return state_dir(workspace) / f"{LOCAL_LOG_PREFIX}{os.getpid() if pid is None else pid}.log"


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


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# Which runtime last used the notebooks folder (a docker kernel's notebooks, later run unsafe-local)
# --------------------------------------------------------------------------- #


def _last_kernel_path(workspace: Path) -> Path:
    return state_dir(workspace) / LAST_KERNEL_FILENAME


def note_kernel_start(workspace: Path, runtime: str, notebooks_root: Path) -> None:
    """Remember that ``runtime`` started a server on ``notebooks_root`` (never raises)."""
    try:
        _write_private(_last_kernel_path(workspace), {"runtime": runtime, "notebooks": str(notebooks_root), "started": _now()})
    except OSError:
        pass


def docker_wrote_notebooks(workspace: Path, notebooks_root: Path) -> bool:
    """True when the last server Hailer started on ``notebooks_root`` was a docker kernel: code the
    container wrote into those notebooks (copied back when it stopped) would now run on this
    machine."""
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
    """The names in ``environ`` an unsafe-local marimo server does not get, sorted.

    Withheld: every declared provider's ``env_key`` and ``OPENAI_API_KEY``, every variable named in
    a provider's ``env_http_headers``, the names in :data:`KERNEL_SECRET_NAMES`, and every name ending in one of :data:`KERNEL_SECRET_SUFFIXES`
    (``_KEY``, ``_TOKEN``, ``_SECRET``, ``_PASSWORD``, ``_PASSWD``, ``_PWD``, ``_CREDENTIALS``,
    ``_CONNECTION_STRING``, ``APIKEY``; any case), with no way to let one through. Exact names compare case-insensitively on Windows, like the environment itself.
    """

    def fold(name: str) -> str:
        return name.upper() if os.name == "nt" else name

    named = {"OPENAI_API_KEY", *KERNEL_SECRET_NAMES}
    for provider in config.providers.values():
        if provider.env_key:
            named.add(provider.env_key)
        named.update(provider.env_http_headers.values())
    dropped = {fold(name) for name in named}
    return sorted(name for name in environ if fold(name) in dropped or name.upper().endswith(KERNEL_SECRET_SUFFIXES))


def kernel_environment(config: HailerConfig, environ: Mapping[str, str]) -> dict[str, str]:
    """``environ`` without the secrets :func:`withheld_variables` names, for a marimo server Hailer
    starts on this machine. Notebook code can still read the OS credential store and files, which
    is why the unsafe-local runtime is "not isolated"."""
    withheld = set(withheld_variables(config, environ))
    return {name: value for name, value in environ.items() if name not in withheld}


def describe_runtime(kernel: KernelConfig) -> str:
    """The text after ``Kernel:`` in the startup panel, ``hailer status`` and ``/status``."""
    if kernel.runtime == KERNEL_RUNTIME_UNSAFE_LOCAL:
        return "unsafe-local (runs as you; not isolated)"
    if kernel.runtime == KERNEL_RUNTIME_DOCKER:
        from hailer.kernel_image import contract_tag  # lazy: only the docker runtime needs it

        network = "network on: the internet and this machine" if kernel.network else "no network"
        return f"docker (hailer-kernel {contract_tag()}; {network}; data read-only)"
    return f'{kernel.runtime} (not a kernel runtime: use "docker" or "unsafe-local")'


def runtime_prompt_notes(kernel: KernelConfig) -> str:
    """The kernel paragraph of the system prompt.

    A function of the settings the chat's kernel was started with, so building the prompt never
    touches Docker. The docker notes assume the Workspace block names
    the kernel paths.
    """
    if kernel.runtime == KERNEL_RUNTIME_DOCKER:
        return DOCKER_NETWORK_PROMPT_NOTES if kernel.network else DOCKER_PROMPT_NOTES
    return LOCAL_PROMPT_NOTES


# --------------------------------------------------------------------------- #
# Processes (the unsafe-local runtime)
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


def mask_token(lines: list[str], token: str | None) -> list[str]:
    """``lines`` with ``token`` replaced by ``<token>`` (marimo prints its signed-in URL)."""
    return [line.replace(token, "<token>") for line in lines] if token else list(lines)


def log_tail(path: Path, lines: int = 15, *, token: str | None = None) -> list[str]:
    """The last ``lines`` lines of ``path`` with ``token`` masked (``[]`` when it cannot be read)."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return mask_token(text.splitlines()[-lines:], token)


@dataclass
class LocalProcesses:
    """Everything :class:`LocalRuntime` does to the operating system; tests replace members."""

    spawn: Callable[..., Any] = spawn_marimo
    attach: Callable[..., Any] = attach_marimo
    kill_tree: Callable[[int], None] = kill_tree
    wait_for_health: Callable[..., bool] = wait_for_health


# --------------------------------------------------------------------------- #
# Runtimes
# --------------------------------------------------------------------------- #


@dataclass
class LocalKernel(MarimoSandbox):
    """The marimo process this Hailer started on this machine (``proc``) and its log. Its notebooks
    and data folders are the host's own (``native_paths``)."""

    proc: Any = None
    log_path: Path | None = None
    procs: LocalProcesses = field(default_factory=LocalProcesses)

    def stop(self) -> None:
        """Stop marimo (its process tree on Windows), then delete the log, which holds the signed-in URL."""
        try:
            if self.proc is not None:
                stop_process(self.proc, kill_tree=self.procs.kill_tree)
        except Exception:  # noqa: BLE001 - best effort on the way out
            pass
        if self.log_path is not None:
            _unlink(self.log_path)

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
    """Starts and describes the marimo server of one Hailer process."""

    name: str  # "docker" | "unsafe-local"

    def check(self) -> list[Check]:
        """Rows for ``hailer doctor`` and ``hailer notebook``: can this runtime run a kernel here?"""
        ...

    def prepare(self, say: Callable[[str], None] | None = None) -> None:
        """Get ready to start, with any slow step's progress and any warning in this terminal
        (docker: download a missing image). Call it before a spinner; ``start`` does it when it has
        not been done. Raises ``KernelRuntimeError``."""
        ...

    def start(self, port: int, *, foreground: bool = False) -> MarimoSandbox:
        """Start a new marimo server on ``127.0.0.1:port`` for this process; the sandbox it runs.

        Waits until this start's own server answers with its own token (never another process's
        server on the same port); ``foreground`` shows the server's output in this terminal (``hailer
        notebook --foreground``; the caller then blocks in :meth:`MarimoSandbox.wait`). Raises
        ``KernelRuntimeError`` (with the log tail in the hint) when the server does not come up;
        nothing is left running then.
        """
        ...

    def describe(self) -> str:
        """The text after ``Kernel:``."""
        ...


class LocalRuntime:
    """``runtime = "unsafe-local"``: marimo in Hailer's own Python, as the user, with their files and
    network. Not isolated: the server Hailer starts only has a random token (passed on stdin, never
    on the command line) and an environment without the secrets :func:`withheld_variables`
    recognises. The class keeps its name (it is the runtime on this machine); the ``unsafe-``
    prefix belongs to the setting a user writes."""

    name = KERNEL_RUNTIME_UNSAFE_LOCAL

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
        self.procs = procs if procs is not None else LocalProcesses()
        self._environ = environ
        self._token_factory = token_factory
        self.start_timeout = start_timeout

    @property
    def log_path(self) -> Path:
        """This process's marimo log, ``.hailer/marimo-<pid>.log``."""
        return local_log_path(self.config.workspace)

    @property
    def environ(self) -> Mapping[str, str]:
        return self._environ if self._environ is not None else os.environ

    def check(self) -> list[Check]:
        """One warning row: notebook code runs as the user, with their files and network."""
        summary = self.describe()
        withheld = withheld_variables(self.config, self.environ)
        if withheld:
            summary += f"; {len(withheld)} secret-looking environment variable(s) withheld from notebook code"
        return [Check("kernel", False, summary, hint=UNSAFE_LOCAL_HINT, fatal=False)]

    def prepare(self, say: Callable[[str], None] | None = None) -> None:
        """Nothing to download: marimo runs in Hailer's own Python. Warns when a docker kernel was
        the last to run these notebooks: from here on their code runs on this machine."""
        if docker_wrote_notebooks(self.config.workspace, self.config.notebooks_root):
            out = say or _print
            out(
                f"Warning: the notebooks in {self.config.notebooks_root} were last run by the isolated docker kernel; "
                "with the unsafe-local kernel their code runs on this machine as you. Open only notebooks you trust."
            )
            out('    To keep them isolated: uvx hailer notebook --kernel docker (or [kernel] runtime = "docker").')

    def describe(self) -> str:
        return describe_runtime(KernelConfig(runtime=KERNEL_RUNTIME_UNSAFE_LOCAL))

    def start(self, port: int, *, foreground: bool = False) -> MarimoSandbox:
        """Start marimo and wait until it answers with this start's token. A Hailer that is killed
        leaves its (token-protected) marimo running: end it with Task Manager or ``kill``."""
        config = self.config
        workspace = Path(config.workspace)
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
        server = MarimoServer(url=url, pid=getattr(proc, "pid", None), token=token, runtime=self.name)
        kernel = LocalKernel(
            server=server,
            notebooks_path=notebook_file_key(config.notebooks_root),
            data_path=str(config.data_dir),
            data_dir=config.data_dir,
            native_paths=True,
            log_hint=str(log_path) if log_path is not None else "this terminal",
            proc=proc,
            log_path=log_path,
            procs=self.procs,
        )
        try:
            # Its own process first, then its own token: another server on this port (two starts
            # picked the same free port) must never be taken for this one.
            healthy = self.procs.wait_for_health(url, self.start_timeout, token=token, should_stop=lambda: proc.poll() is not None)
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
            hint = "Last lines of its log:" if tail else ("Its log was empty." if log_path is not None else "marimo's own output is above.")
            if tail:
                hint += "\n" + "\n".join(f"    {line}" for line in tail)
            raise KernelRuntimeError(message, hint=hint)
        note_kernel_start(workspace, self.name, config.notebooks_root)
        return kernel


def _print(text: str) -> None:
    print(text, flush=True)


def runtime_for(config: HailerConfig, runner: Any = None) -> KernelRuntime:
    """The runtime ``config.kernel.runtime`` asks for; ``ConfigError`` for an unknown one (the
    retired ``local`` included). Nothing is checked yet (``check``, ``prepare`` and ``start`` do
    that), and docker never falls back to unsafe-local."""
    runtime = config.kernel.runtime
    if runtime == KERNEL_RUNTIME_UNSAFE_LOCAL:
        return LocalRuntime(config)
    if runtime == KERNEL_RUNTIME_DOCKER:
        from hailer.kernel_docker import DockerRuntime  # lazy: most runs never touch Docker

        return DockerRuntime(config, runner=runner)
    from hailer.config import runtime_problem  # lazy: hailer.config is only needed for this message

    raise ConfigError(runtime_problem(runtime))


__all__ = [
    "KERNEL_DATA_DIR",
    "KERNEL_NOTEBOOKS_DIR",
    "KERNEL_WORKDIR",
    "KernelRuntime",
    "LocalKernel",
    "LocalRuntime",
    "describe_runtime",
    "kernel_environment",
    "runtime_for",
    "runtime_prompt_notes",
]
