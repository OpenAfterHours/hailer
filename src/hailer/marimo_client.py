"""Pure-Python client for a running marimo server (the "marimo pair" protocol).

Talks to a handful of endpoints with the standard library only:

- ``GET  {url}/health``                       liveness (no auth)
- ``GET  {url}/api/sessions``                 ``{session_id: {"filename": ..., "path": ...}}``
- ``POST {url}/api/kernel/execute``           run code in the kernel scratchpad; SSE stream of
                                               ``stdout`` / ``stderr`` / ``done`` events
- ``GET  {url}/``                             the page; carries the server (skew-protection) token
- ``POST {url}/api/home/shutdown_session``    close a kernel session (needs the server token)

One marimo server hosts every notebook: it is started on the notebooks *folder*, and any
existing notebook gets its own kernel session when a browser opens ``?file=<absolute path>``.
A kernel *session* only exists while the notebook is open in a browser. Durable notebook
changes are made from the scratchpad through ``marimo._code_mode``; the snippet constants at
the bottom of this module are the canonical patterns.

Servers Hailer starts carry a random token (``Authorization: Bearer <token>`` for the API,
``&access_token=<token>`` in a browser URL). A kernel in a container knows files by other paths;
:class:`MarimoClient` translates them with a :class:`~hailer.kernel.PathMap` at the HTTP boundary.
``hailer.kernel`` imports this module, so this one reaches back into it only lazily.
"""

from __future__ import annotations

import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote, urlsplit

from hailer.errors import MarimoExecutionError, MarimoUnavailableError, NoSessionError, NotebookPathError
from hailer.models import KERNEL_RUNTIME_LOCAL, ExecResult, HailerConfig, MarimoServer, MarimoSession

if TYPE_CHECKING:  # pragma: no cover - annotations only; hailer.kernel imports this module
    from hailer.kernel import PathMap

SERVER_TOKEN_HEADER = "Marimo-Server-Token"
_HEALTH_TIMEOUT = 1.0
_SERVER_TOKEN_RE = re.compile(r"<marimo-server-token[^>]*\bdata-token=[\"']([^\"']+)[\"']", re.IGNORECASE)

# --------------------------------------------------------------------------- #
# Launch helpers
# --------------------------------------------------------------------------- #


def _relative_to_workspace(notebook: Path, workspace: Path | None) -> str:
    """Notebook path relative to the workspace with forward slashes (falls back to absolute)."""
    nb = Path(notebook)
    if workspace is not None:
        try:
            return nb.resolve().relative_to(Path(workspace).resolve()).as_posix()
        except ValueError:
            pass
    if not nb.is_absolute():
        return nb.as_posix()
    return str(nb)


def launch_command(*, port: int | None = None) -> list[str]:
    """The command a person runs to start marimo in a terminal of its own (no chat).

    ``hailer notebook --foreground`` starts marimo on the notebooks folder with Hailer's own
    interpreter (see :func:`marimo_server_command`), so it works under ``uvx`` without a project
    virtual environment. A bare ``marimo edit`` would need marimo, Polars, DuckDB and Hailer
    installed wherever it runs; ``uvx --from hailer marimo`` works, but uv then suggests
    ``uvx --from marimo marimo``, which gives an environment without Hailer.
    """
    cmd = ["uvx", "hailer", "notebook", "--foreground"]
    if port is not None:
        cmd += ["--port", str(port)]
    return cmd


def marimo_server_command(notebooks_dir: Path, workspace: Path, port: int, *, headless: bool = True) -> list[str]:
    """Argument list ``hailer notebook`` uses to start marimo on the notebooks folder.

    Runs marimo through the current interpreter (the environment that runs Hailer: the ``uvx``
    tool environment or a project venv) rather than through ``uv run``: that needs no project
    ``.venv``, and on Windows terminating a ``uv`` wrapper would not stop the marimo process it
    spawned, while Hailer must be able to stop what it started. Marimo is started on the
    notebooks folder, never on a single file. ``headless=False`` lets marimo open its home page
    in the browser.

    The server requires a token, read from stdin (``--token-password-file -``) so it never
    appears on a command line: the caller writes it to the child's stdin and closes it. Without
    a token any local program could POST code to ``/api/kernel/execute``.
    """
    rel = _relative_to_workspace(notebooks_dir, workspace)
    cmd = [sys.executable, "-m", "marimo", "edit", rel, "--token-password-file", "-"]
    if headless:
        cmd.append("--headless")
    return cmd + ["--port", str(port), "--skip-update-check"]


def launch_hint() -> str:
    """The fix printed under "Marimo is not running." — the one-command route first."""
    return (
        "Start everything in one go:\n\n"
        "    uvx hailer notebook\n\n"
        "Or run marimo on its own in another terminal:\n\n"
        f"    {_format_command(launch_command())}\n\n"
        "Then run Hailer again:\n\n"
        "    uvx hailer"
    )


APP_VIEW_PARAM = "view-as=present"
"""Query parameter that makes marimo's edit server open a notebook in app view (outputs only).

The page is still a normal edit session (the kernel gets a session, code mode works); only the
initial UI mode differs. The reader switches to edit mode with Ctrl+. / Cmd+. ("Toggle app view").
"""


def notebook_file_key(notebook: Path) -> str:
    """The ``?file=`` key marimo receives for ``notebook``: its absolute path with forward slashes.

    An absolute key works whether the server was started on the notebooks folder (keys would
    otherwise resolve against that folder) or on a single file (keys would resolve against the
    server's working directory); verified live on marimo 0.24.2 in both modes.

    The key is normalised the way marimo normalises paths (absolute, ``..`` and ``.`` removed,
    symlinks and junctions NOT resolved), so a workspace reached through a junction still yields
    a key marimo considers inside its folder.
    """
    absolute = os.path.normpath(str(Path(notebook).expanduser().absolute()))
    return Path(absolute).as_posix()


def open_notebook_url(
    server: MarimoServer,
    notebook: Path,
    workspace: Path | None = None,
    *,
    view: str = "app",
    paths: PathMap | None = None,
    with_token: bool = True,
) -> str:
    """URL the user should open so the kernel gets a session.

    ``view="app"`` (default) opens the notebook in app view so the chat user sees results, not code:
    ``http://127.0.0.1:2718/?file=C:/work/notebooks/analysis.py&view-as=present``. ``view="edit"``
    opens the plain editor (no ``view-as`` parameter). ``workspace`` is accepted for
    compatibility; the key is always the absolute path (see :func:`notebook_file_key`).

    ``paths`` (default: ``server.paths``) turns the key into the path the kernel knows the file by
    (``/work/notebooks/...`` in a container). When the server has a token the URL ends with
    ``&access_token=<token>``, which signs the browser in; ``with_token=False`` leaves it out, for
    text that goes to the model endpoint. ``NotebookPathError`` when the kernel cannot see the file.
    """
    del workspace  # the file key no longer depends on the workspace
    if view not in ("app", "edit"):
        raise ValueError(f"view must be 'app' or 'edit', not {view!r}")
    mapping = paths if paths is not None else server.paths
    try:
        key = mapping.to_kernel(notebook) if mapping is not None else notebook_file_key(notebook)
    except ValueError as err:
        raise NotebookPathError(
            f"{notebook_file_key(notebook)} is outside the folders the kernel can see.",
            hint="Keep notebooks in the notebooks folder ([hailer].notebooks_dir).",
        ) from err
    url = f"{server.url.rstrip('/')}/?file={quote(key, safe='/:')}"
    if view == "app":
        url += f"&{APP_VIEW_PARAM}"
    if with_token and server.token:
        url += f"&access_token={quote(server.token, safe='')}"
    return url


def home_url(server: MarimoServer, *, with_token: bool = True) -> str:
    """marimo's home page (the notebooks folder), signed in with the server's token when it has one."""
    url = f"{server.url.rstrip('/')}/"
    if with_token and server.token:
        url += f"?access_token={quote(server.token, safe='')}"
    return url


def _format_command(cmd: Sequence[str]) -> str:
    return " ".join(cmd)


# --------------------------------------------------------------------------- #
# Starting a server: ports and readiness
# --------------------------------------------------------------------------- #


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """True when something already listens on ``host:port`` (bind fails)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return True
    return False


def find_free_port(preferred: int = 2718, host: str = "127.0.0.1") -> int:
    """``preferred`` when it is free, otherwise a free ephemeral port chosen by the OS."""
    if not port_in_use(preferred, host):
        return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def wait_for_health(
    url: str,
    timeout: float = 60.0,
    *,
    interval: float = 0.5,
    should_stop: Callable[[], bool] | None = None,
) -> bool:
    """Poll ``/health`` until it answers, the timeout passes, or ``should_stop()`` returns True."""
    deadline = time.monotonic() + timeout
    while True:
        if _health_ok(url):
            return True
        if should_stop is not None and should_stop():
            return False
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval)


def wait_for_session(
    client: "MarimoClient",
    notebook: Path | None,
    timeout: float = 90.0,
    *,
    interval: float = 0.5,
) -> MarimoSession | None:
    """Poll ``/api/sessions`` until the notebook has a kernel session; ``None`` on timeout."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return client.resolve_session(notebook)
        except NoSessionError:
            pass
        if time.monotonic() >= deadline:
            return None
        time.sleep(interval)


def registry_entry_path(url: str, registry: Path | None = None) -> Path:
    """The registry file marimo writes for a ``--no-token`` server at ``url`` (``<host>_<port>.json``).

    Servers with a token (every server Hailer starts) are never registered; Hailer finds those
    through ``.hailer/kernel.json`` instead.
    """
    parts = urlsplit(url if "://" in url else f"http://{url}")
    host = parts.hostname or "127.0.0.1"
    port = parts.port or 80
    directory = registry if registry is not None else registry_dir()
    return directory / f"{host}_{port}.json".replace(":", "_").replace("/", "_")


def remove_registry_entry(url: str, registry: Path | None = None) -> bool:
    """Delete the registry file for ``url``; marimo only removes it on a clean shutdown."""
    path = registry_entry_path(url, registry)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# Registry discovery
# --------------------------------------------------------------------------- #


def registry_dir() -> Path:
    """Where ``marimo edit --no-token`` servers register themselves."""
    if os.name == "posix":
        base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
        return Path(base) / "marimo" / "servers"
    return Path.home() / ".marimo" / "servers"


def _url_from_entry(entry: dict) -> str | None:
    host = str(entry.get("host") or "127.0.0.1")
    port = entry.get("port")
    if not isinstance(port, int):
        try:
            port = int(port)
        except (TypeError, ValueError):
            return None
    if host in ("0.0.0.0", "", "*"):
        host = "127.0.0.1"
    elif host == "::":
        host = "[::1]"
    elif ":" in host and not host.startswith("["):
        host = f"[{host}]"
    base = str(entry.get("base_url") or "").rstrip("/")
    return f"http://{host}:{port}{base}"


def _health_ok(url: str, timeout: float = _HEALTH_TIMEOUT) -> bool:
    try:
        req = urllib.request.Request(f"{url.rstrip('/')}/health", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - local http only
            body = resp.read(4096).decode("utf-8", "replace")
            return "healthy" in body
    except (urllib.error.URLError, socket.timeout, TimeoutError, OSError, ValueError):
        return False


def _sessions_status(url: str, token: str | None, timeout: float) -> int | None:
    """HTTP status of ``GET /api/sessions`` (with ``token`` as a bearer token); ``None`` when unreachable."""
    headers = {"User-Agent": "hailer"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"{url.rstrip('/')}/api/sessions", method="GET", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - local http only
            resp.read(1)
            return resp.status
    except urllib.error.HTTPError as err:
        return err.code
    except (urllib.error.URLError, socket.timeout, TimeoutError, OSError, ValueError):
        return None


def answers_with_token(url: str, token: str | None, timeout: float = _HEALTH_TIMEOUT) -> bool:
    """True when the server at ``url`` is up and is the one that holds ``token``.

    It must answer ``/health``, refuse ``/api/sessions`` without the token and accept it with
    ``Authorization: Bearer <token>``. A server started with ``--no-token`` accepts anything and
    therefore never passes, so a record in ``.hailer/kernel.json`` only ever leads to the server it
    describes, even when another one now listens on that port. Without a token only ``/health``
    is checked.
    """
    if not _health_ok(url, timeout):
        return False
    if not token:
        return True
    return _sessions_status(url, None, timeout) in (401, 403) and _sessions_status(url, token, timeout) == 200


def discover_servers(registry: Path | None = None) -> list[MarimoServer]:
    """Live servers from the local registry; entries that do not answer ``/health`` are skipped."""
    directory = registry if registry is not None else registry_dir()
    if not directory.is_dir():
        return []
    found: list[MarimoServer] = []
    for path in sorted(directory.glob("*.json")):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(entry, dict):
            continue
        url = _url_from_entry(entry)
        if url is None or not _health_ok(url):
            continue
        pid = entry.get("pid")
        found.append(
            MarimoServer(
                url=url,
                server_id=str(entry.get("server_id") or ""),
                pid=pid if isinstance(pid, int) else None,
                version=str(entry.get("version") or ""),
                source="registry",
            )
        )
    return found


def _attach_allowed(requested: str, runtime: str) -> bool:
    """Fail closed: a docker request never gets a local server; a local one may get either (a docker
    kernel is strictly more isolated)."""
    return requested == KERNEL_RUNTIME_LOCAL or runtime == requested


def find_server(config: HailerConfig, registry: Path | None = None) -> MarimoServer | None:
    """The marimo server Hailer should talk to, or ``None``.

    1. ``[hailer].marimo_url``, for the local runtime. When it is the server recorded in
       ``.hailer/kernel.json``, that record's token, runtime and path map apply (``hailer notebook``
       pins the server it started or reused this way). Not health-checked, as before.
    2. ``.hailer/kernel.json``: the server Hailer started for this workspace, when it answers with
       its token (:func:`answers_with_token`).
    3. marimo's registry of ``--no-token`` servers: the single live one. Local runtime only.

    Fails closed: when ``config.kernel.runtime`` is ``docker`` no local server is returned (not
    from ``marimo_url``, the registry, or a ``kernel.json`` that says local).
    """
    from hailer.kernel import read_kernel_state  # lazy: hailer.kernel imports this module

    requested = config.kernel.runtime
    state = read_kernel_state(config.workspace)
    if config.marimo_url:
        url = config.marimo_url.rstrip("/")
        if state is not None and state.url.lower() == url.lower():
            return state.server() if _attach_allowed(requested, state.runtime) else None
        return MarimoServer(url=url, source="config") if requested == KERNEL_RUNTIME_LOCAL else None
    if state is not None and _attach_allowed(requested, state.runtime) and answers_with_token(state.url, state.token):
        return state.server()
    if requested != KERNEL_RUNTIME_LOCAL:
        return None
    live = discover_servers(registry)
    if len(live) == 1:
        return live[0]
    return None


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


def _normalise_path(value: str | Path) -> str:
    try:
        resolved = Path(value).expanduser().resolve()
    except (OSError, RuntimeError):
        resolved = Path(value)
    return os.path.normcase(os.path.normpath(str(resolved)))


def _session_path(session: MarimoSession, workspace: Path | None) -> str | None:
    """The normalised absolute path of a session's notebook; ``None`` for untitled sessions."""
    raw = session.path or session.filename
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute() and workspace is not None:
        path = Path(workspace) / path
    return _normalise_path(path)


def match_session(sessions: Sequence[MarimoSession], notebook: Path, workspace: Path | None = None) -> MarimoSession | None:
    """The session whose notebook is ``notebook``, or ``None``.

    A session matches only when its ``path`` (or its ``filename`` when marimo reports no path)
    names the same file, compared as normalised absolute paths (case-insensitive on Windows).
    Relative session paths are taken relative to ``workspace``. Untitled sessions never match,
    and a bare filename never matches a notebook of the same name in another folder. When
    several sessions name the same file the most recently created one (last in marimo's
    listing) wins.
    """
    target = _normalise_path(notebook)
    matched = [s for s in sessions if _session_path(s, workspace) == target]
    return matched[-1] if matched else None


def _iter_sse(raw_lines: Iterator[bytes]) -> Iterator[tuple[str, str]]:
    """Yield ``(event, data)`` pairs from an SSE byte stream. Tolerates CRLF and comments."""
    event = "message"
    data_lines: list[str] = []
    for raw in raw_lines:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if line == "":
            if data_lines:
                yield event, "\n".join(data_lines)
            event, data_lines = "message", []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            event = value.strip()
        elif field == "data":
            data_lines.append(value)
    if data_lines:
        yield event, "\n".join(data_lines)


class MarimoClient:
    """HTTP client for one marimo server.

    ``token`` is sent as ``Authorization: Bearer <token>``. ``paths`` translates between host and
    kernel paths at exactly two places: the ``?file=`` key of :meth:`notebook_url` and the session
    paths :meth:`sessions` returns (a kernel path outside every mount is left as it is, so it never
    matches a host notebook). Callers work in host paths only.

    The notebook links in error hints leave the token out unless ``token_in_links`` is set: the
    agent's tools pass those hints to the model endpoint, while the CLI prints them for the user.
    """

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        *,
        timeout: float = 10.0,
        notebook: Path | None = None,
        workspace: Path | None = None,
        paths: PathMap | None = None,
        token_in_links: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.notebook = notebook
        self.workspace = workspace
        self.paths = paths
        self.token_in_links = token_in_links
        self._server_token: str | None = None

    # -- low level ---------------------------------------------------------- #

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {"Accept": "*/*", "User-Agent": "hailer"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if extra:
            headers.update(extra)
        return headers

    def _unavailable(self, reason: str) -> MarimoUnavailableError:
        return MarimoUnavailableError(
            f"Marimo is not running at {self.base_url} ({reason}).",
            hint=launch_hint(),
        )

    def _auth_error(self, status: int) -> MarimoUnavailableError:
        return MarimoUnavailableError(
            f"Marimo at {self.base_url} rejected the request (HTTP {status}).",
            hint=(
                "The server was started with an auth token. If Hailer started it (uvx hailer notebook), "
                "run Hailer in the same workspace so it can read the token from .hailer/kernel.json. "
                "For a server you started yourself, restart it with --no-token or set "
                "HAILER_MARIMO_TOKEN to the token shown when marimo started."
            ),
        )

    def _open(self, method: str, path: str, body: bytes | None = None, headers: dict[str, str] | None = None, timeout: float | None = None):
        req = urllib.request.Request(f"{self.base_url}{path}", data=body, method=method, headers=self._headers(headers))
        try:
            return urllib.request.urlopen(req, timeout=timeout if timeout is not None else self.timeout)  # noqa: S310
        except urllib.error.HTTPError as err:
            if err.code in (401, 403):
                raise self._auth_error(err.code) from err
            raise
        except urllib.error.URLError as err:
            reason = err.reason
            if isinstance(reason, (socket.timeout, TimeoutError)):
                raise self._unavailable("timed out") from err
            raise self._unavailable(str(reason)) from err
        except (socket.timeout, TimeoutError) as err:
            raise self._unavailable("timed out") from err
        except OSError as err:
            raise self._unavailable(str(err)) from err

    # -- public ------------------------------------------------------------- #

    def health(self) -> bool:
        return _health_ok(self.base_url, timeout=min(self.timeout, 3.0))

    def sessions(self) -> list[MarimoSession]:
        try:
            with self._open("GET", "/api/sessions") as resp:
                payload = json.loads(resp.read().decode("utf-8", "replace") or "{}")
        except urllib.error.HTTPError as err:
            # 401/403 are already mapped by _open; anything else (500, 404 on an old marimo) lands here.
            detail = _error_detail(err.read())
            raise MarimoUnavailableError(
                f"Marimo at {self.base_url} answered HTTP {err.code} to /api/sessions" + (f": {detail}" if detail else "."),
                hint=(
                    "Check the marimo server log for errors (0.24.x is expected), or restart it with:\n\n"
                    f"    {_format_command(launch_command())}"
                ),
            ) from err
        if not isinstance(payload, dict):
            return []
        out: list[MarimoSession] = []
        for sid, info in payload.items():
            info = info if isinstance(info, dict) else {}
            out.append(
                MarimoSession(
                    session_id=str(sid),
                    filename=self._host_path(info.get("filename")),
                    path=self._host_path(info.get("path")),
                )
            )
        return out

    def _host_path(self, value: object) -> object:
        """A path the kernel reported, as the host knows it (unchanged when it maps to nothing)."""
        if self.paths is None or self.paths.identity or not isinstance(value, str):
            return value
        host = self.paths.to_host(value)
        return host if host is not None else value

    def server_token(self) -> str:
        """The server's skew-protection token, read once from the page and cached.

        Every POST except ``/api/kernel/execute`` must carry it in the ``Marimo-Server-Token``
        header, otherwise marimo answers 401.
        """
        if self._server_token:
            return self._server_token
        try:
            with self._open("GET", "/") as resp:
                html = resp.read(2_000_000).decode("utf-8", "replace")
        except urllib.error.HTTPError as err:
            raise MarimoUnavailableError(
                f"Marimo at {self.base_url} answered HTTP {err.code} to GET /.",
                hint="Check the marimo server log; the page should be served by marimo 0.24.x.",
            ) from err
        match = _SERVER_TOKEN_RE.search(html)
        if match is None:
            raise MarimoUnavailableError(
                f"Marimo at {self.base_url} did not provide a server token in its page.",
                hint=(
                    "Hailer expects a <marimo-server-token data-token=...> tag (marimo 0.24.x). "
                    "Check that the URL points at a marimo edit server, not a proxy or another app."
                ),
            )
        self._server_token = match.group(1)
        return self._server_token

    def shutdown_session(self, session_id: str) -> None:
        """Close the kernel session ``session_id`` (frees its kernel; the browser tab disconnects)."""
        token = self.server_token()
        body = json.dumps({"sessionId": session_id}).encode("utf-8")
        headers = {"Content-Type": "application/json", SERVER_TOKEN_HEADER: token}
        try:
            with self._open("POST", "/api/home/shutdown_session", body=body, headers=headers) as resp:
                resp.read()
        except MarimoUnavailableError as err:
            cause = err.__cause__
            if isinstance(cause, urllib.error.HTTPError) and cause.code == 401:
                self._server_token = None  # stale after a server restart; the next call refetches it
                raise MarimoUnavailableError(
                    f"Marimo at {self.base_url} rejected the server token (HTTP 401).",
                    hint=(
                        "The marimo server was probably restarted, so its token changed; retry once. "
                        "If the server was started with an auth token, set HAILER_MARIMO_TOKEN as well."
                    ),
                ) from cause
            raise
        except urllib.error.HTTPError as err:
            detail = _error_detail(err.read())
            raise MarimoUnavailableError(
                f"Marimo refused to close session {session_id} (HTTP {err.code})" + (f": {detail}" if detail else "."),
                hint="Check that the session id is current; GET /api/sessions lists the open ones.",
            ) from err

    def notebook_url(self, notebook: Path | None = None) -> str:
        """The URL that opens ``notebook`` (token only with ``token_in_links``; see the class docstring)."""
        nb = notebook or self.notebook
        server = MarimoServer(url=self.base_url, token=self.token if self.token_in_links else None)
        if nb is None:
            return home_url(server)
        try:
            return open_notebook_url(server, nb, self.workspace or Path.cwd(), paths=self.paths)
        except NotebookPathError:
            return home_url(server)

    def resolve_session(self, notebook: Path | None) -> MarimoSession:
        """Pick the session for ``notebook`` (or the only session when ``notebook`` is None)."""
        sessions = self.sessions()
        if not sessions:
            raise NoSessionError(
                "The notebook is not open in a browser, so the kernel has no session.",
                hint=f"Open {self.notebook_url(notebook)} in your browser, then try again.",
            )
        if notebook is None:
            if len(sessions) == 1:
                return sessions[0]
            raise NoSessionError(
                "Several notebooks are open; tell Hailer which one via [hailer].notebook.",
                hint="Open sessions:\n" + self._describe_sessions(sessions),
            )
        session = match_session(sessions, notebook, self.workspace)
        if session is not None:
            return session
        raise NoSessionError(
            f"No open session matches {Path(notebook).name}.",
            hint=f"Open {self.notebook_url(notebook)} in your browser. Open sessions:\n" + self._describe_sessions(sessions),
        )

    @staticmethod
    def _describe_sessions(sessions: Sequence[MarimoSession]) -> str:
        return "\n".join(f"  {s.session_id}  {s.path or s.filename or '(unsaved notebook)'}" for s in sessions) or "  (none)"

    def execute(
        self,
        code: str,
        *,
        session_id: str | None = None,
        notebook: Path | None = None,
        on_stdout: Callable[[str], None] | None = None,
        on_stderr: Callable[[str], None] | None = None,
        timeout: float = 600.0,
    ) -> ExecResult:
        """Run ``code`` in the scratchpad of one session and collect its output."""
        sid = session_id or self.resolve_session(notebook if notebook is not None else self.notebook).session_id
        body = json.dumps({"code": code}).encode("utf-8")
        headers = {"Content-Type": "application/json", "Marimo-Session-Id": sid, "Accept": "text/event-stream"}
        try:
            resp = self._open("POST", "/api/kernel/execute", body=body, headers=headers, timeout=timeout)
        except urllib.error.HTTPError as err:
            detail = _error_detail(err.read())
            raise MarimoExecutionError(
                f"Marimo refused the execute request (HTTP {err.code}): {detail}",
                hint="Check that the notebook is still open in the browser and that the session id is current.",
            ) from err

        stdout: list[str] = []
        stderr: list[str] = []
        done: dict | None = None
        unparsed: list[bytes] = []
        content_type = (resp.headers.get("Content-Type") or "").lower()
        with resp:
            if "text/event-stream" not in content_type:
                raw = resp.read()
                detail = _error_detail(raw)
                raise MarimoExecutionError(
                    f"Marimo returned a non-stream response to the execute request: {detail}",
                    hint="Check that the notebook is still open in the browser and retry.",
                )
            try:
                for event, data in _iter_sse(_read_lines(resp, unparsed)):
                    try:
                        payload = json.loads(data)
                    except ValueError:
                        continue
                    if event == "stdout":
                        chunk = str(payload.get("data", "")) if isinstance(payload, dict) else str(payload)
                        stdout.append(chunk)
                        if on_stdout:
                            on_stdout(chunk)
                    elif event == "stderr":
                        chunk = str(payload.get("data", "")) if isinstance(payload, dict) else str(payload)
                        stderr.append(chunk)
                        if on_stderr:
                            on_stderr(chunk)
                    elif event == "done":
                        done = payload if isinstance(payload, dict) else {"success": False}
                        break
            except (socket.timeout, TimeoutError) as err:
                raise MarimoExecutionError(
                    f"Marimo did not finish executing within {timeout:.0f}s.",
                    hint="The kernel may still be running the code. Interrupt it in the browser or wait and retry.",
                ) from err

        if done is None:
            detail = _error_detail(b"".join(unparsed)) if unparsed else ""
            raise MarimoExecutionError(
                "Execution did not complete: the server ended the stream without a result."
                + (f" {detail}" if detail else ""),
                hint="The session may have changed (browser refresh). Retry; Hailer resolves the session on every call.",
            )
        output = done.get("output") or {}
        if not isinstance(output, dict):
            output = {}
        return ExecResult(
            success=bool(done.get("success", False)),
            stdout="".join(stdout),
            stderr="".join(stderr),
            output=str(output.get("data") or ""),
            mimetype=str(output.get("mimetype") or "text/plain"),
        )


def _read_lines(resp, unparsed: list[bytes]) -> Iterator[bytes]:
    """Yield raw lines from the response; remember lines that are not SSE fields for error reporting."""
    while True:
        line = resp.readline()
        if not line:
            return
        stripped = line.strip()
        if stripped and not (stripped.startswith(b"event:") or stripped.startswith(b"data:") or stripped.startswith(b":") or stripped.startswith(b"id:") or stripped.startswith(b"retry:")):
            unparsed.append(line)
        yield line


def _error_detail(raw: bytes) -> str:
    text = raw.decode("utf-8", "replace").strip()
    if not text:
        return ""
    try:
        payload = json.loads(text)
    except ValueError:
        return text[:500]
    if isinstance(payload, dict):
        for key in ("detail", "error", "message"):
            if payload.get(key):
                return str(payload[key])[:500]
    return text[:500]


# --------------------------------------------------------------------------- #
# Canonical scratchpad snippets (valid in marimo's scratchpad; top-level await allowed)
# --------------------------------------------------------------------------- #

CM_HELP_CODE = "import marimo._code_mode as cm; help(cm)"

LIST_CELLS_CODE = '''import marimo._code_mode as cm
ctx = cm.get_context()
for cell in ctx.cells:
    lines = (cell.code or "").strip().splitlines()
    first = lines[0] if lines else ""
    status = getattr(cell.status, "value", cell.status)
    print(f"{cell.id}\\t{cell.name or ''}\\t{status}\\terrors={len(cell.errors)}\\t{first[:80]}")
'''

NOTEBOOK_GLOBALS_CODE = '''import marimo._code_mode as cm
import types
ctx = cm.get_context()
for name, value in sorted(ctx.globals.items()):
    if name.startswith("_") or isinstance(value, types.ModuleType):
        continue
    shape = getattr(value, "shape", None)
    extra = f" shape={shape}" if shape is not None else ""
    print(f"{name}: {type(value).__name__}{extra}")
'''


def build_create_cell_code(code: str, *, name: str | None = None, hide_code: bool = False, run: bool = True, after: str | None = None) -> str:
    """Return a scratchpad snippet that appends ``code`` as a new cell (and runs it)."""
    kwargs = [f"hide_code={hide_code!r}"]
    if name:
        kwargs.append(f"name={name!r}")
    if after:
        kwargs.append(f"after={after!r}")
    lines = [
        "import marimo._code_mode as cm",
        "async with cm.get_context() as ctx:",
        f"    cid = ctx.create_cell({code!r}, {', '.join(kwargs)})",
    ]
    if run:
        lines.append("    ctx.run_cell(cid)")
    lines.append("print(cid)")
    return "\n".join(lines) + "\n"


__all__ = [
    "CM_HELP_CODE",
    "LIST_CELLS_CODE",
    "NOTEBOOK_GLOBALS_CODE",
    "SERVER_TOKEN_HEADER",
    "MarimoClient",
    "answers_with_token",
    "build_create_cell_code",
    "discover_servers",
    "find_server",
    "home_url",
    "launch_command",
    "launch_hint",
    "marimo_server_command",
    "match_session",
    "notebook_file_key",
    "open_notebook_url",
    "registry_dir",
    "wait_for_health",
    "wait_for_session",
]

if sys.platform == "win32":  # pragma: no cover - documentation only
    __doc__ += "\n\nOn Windows the registry lives in %USERPROFILE%\\.marimo\\servers."
