"""Pure-Python client for a running marimo server (the "marimo pair" protocol).

Talks to a handful of endpoints with the standard library only:

- ``GET  {url}/health``                       liveness (no auth)
- ``GET  {url}/api/sessions``                 ``{session_id: {"filename": ..., "path": ...}}``
- ``POST {url}/api/kernel/execute``           run code in the kernel scratchpad; SSE stream of
                                               ``stdout`` / ``stderr`` / ``done`` events
- ``GET  {url}/``                             the page; carries the server (skew-protection) token
- ``POST {url}/api/home/shutdown_session``    close a kernel session (needs the server token)
- ``POST {url}/api/files/list_files``, ``/file_details``, ``/create`` (multipart), ``/update``,
  ``/delete``                                 marimo's file explorer, in kernel paths (need the
                                               server token)

One marimo server hosts every notebook: it is started on the notebooks *folder*, and any
existing notebook gets its own kernel session when a browser opens ``?file=<absolute path>``.
A kernel *session* only exists while the notebook is open in a browser. Durable notebook
changes are made from the scratchpad through ``marimo._code_mode``; the snippet constants at
the bottom of this module are the canonical patterns.

Hailer only ever talks to the server its own process started (it never looks for others). That
server carries a random token (``Authorization: Bearer <token>`` for the API,
``&access_token=<token>`` in a browser URL). Notebooks are known by *names* relative to the
notebooks folder (``"sales.py"``, ``"q3/review.py"``); :class:`MarimoClient` turns a name into
the kernel's path (``notebooks_path`` + name) for ``?file=`` keys and back for the sessions marimo
reports, so nothing above the runtime layer handles a kernel or host path.
"""

from __future__ import annotations

import json
import os
import posixpath
import re
import secrets
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from hailer.errors import MarimoExecutionError, MarimoUnavailableError, NoSessionError
from hailer.models import ExecResult, MarimoServer, MarimoSession

SERVER_TOKEN_HEADER = "Marimo-Server-Token"
#: The largest notebook Hailer reads or copies (the sandbox's limit; defined here so replies can be
#: sized from it).
MAX_NOTEBOOK_BYTES = 5 * 1024 * 1024
#: The largest reply body Hailer reads from marimo (a notebook's text escaped in JSON, plus room for
#: the listing around it). A longer reply is an error, never read to the end: whatever runs in the
#: kernel can replace the marimo server and answer anything.
MAX_REPLY_BYTES = 3 * MAX_NOTEBOOK_BYTES + 1024 * 1024
#: How much of a reply is read at once (the deadline is checked between reads).
_READ_CHUNK = 64 * 1024
_deadlines = threading.local()


@contextmanager
def request_deadline(seconds: float) -> Iterator[None]:
    """Every marimo request this thread makes inside the block, connecting, headers and the whole
    body, must finish within ``seconds`` in total (``time.monotonic``); one past it fails with
    ``MarimoUnavailableError`` ("out of time"). Nested blocks keep the earlier deadline."""
    previous = getattr(_deadlines, "at", None)
    at = time.monotonic() + seconds
    _deadlines.at = at if previous is None else min(previous, at)
    try:
        yield
    finally:
        _deadlines.at = previous


def _deadline() -> float | None:
    return getattr(_deadlines, "at", None)
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


def marimo_server_command(notebooks_dir: Path, workspace: Path, port: int) -> list[str]:
    """Argument list ``hailer notebook`` uses to start marimo on the notebooks folder.

    Runs marimo through the current interpreter (the environment that runs Hailer: the ``uvx``
    tool environment or a project venv) rather than through ``uv run``: that needs no project
    ``.venv``, and on Windows terminating a ``uv`` wrapper would not stop the marimo process it
    spawned, while Hailer must be able to stop what it started. Marimo is started on the
    notebooks folder, never on a single file, and headless: Hailer opens the signed-in page itself.

    The server requires a token, read from stdin (``--token-password-file -``) so it never
    appears on a command line: the caller writes it to the child's stdin and closes it. Without
    a token any local program could POST code to ``/api/kernel/execute``.
    """
    rel = _relative_to_workspace(notebooks_dir, workspace)
    return [sys.executable, "-m", "marimo", "edit", rel, "--token-password-file", "-", "--headless", "--port", str(port), "--skip-update-check"]


def launch_hint() -> str:
    """The fix printed when the marimo server this Hailer process started does not answer."""
    return (
        "The kernel this Hailer session started has stopped or does not answer. End the chat (/exit) and "
        "run uvx hailer again: every session starts its own kernel and stops it when the chat ends."
    )


APP_VIEW_PARAM = "view-as=present"
"""Query parameter that makes marimo's edit server open a notebook in app view (outputs only).

The page is still a normal edit session (the kernel gets a session, code mode works); only the
initial UI mode differs. The reader switches to edit mode with Ctrl+. / Cmd+. ("Toggle app view").
"""


def notebook_file_key(notebook: Path) -> str:
    """A host path as marimo knows it: absolute, with forward slashes (the unsafe-local runtime's notebooks
    folder is ``notebook_file_key(config.notebooks_root)``).

    An absolute ``?file=`` key works whether the server was started on the notebooks folder (keys would
    otherwise resolve against that folder) or on a single file (keys would resolve against the
    server's working directory); verified live on marimo 0.24.2 in both modes.

    The key is normalised the way marimo normalises paths (absolute, ``..`` and ``.`` removed,
    symlinks and junctions NOT resolved), so a workspace reached through a junction still yields
    a key marimo considers inside its folder.
    """
    absolute = os.path.normpath(str(Path(notebook).expanduser().absolute()))
    return Path(absolute).as_posix()


def kernel_path(folder: str, name: str) -> str:
    """The kernel's path for ``name``, a POSIX path relative to ``folder`` (a kernel path)."""
    return f"{folder.rstrip('/')}/{name}"


def _parts_under(root: tuple[str, ...], parts: tuple[str, ...], fold: Callable[[str], str]) -> str | None:
    if len(parts) <= len(root) or any(fold(a) != fold(b) for a, b in zip(root, parts)):
        return None
    return "/".join(parts[len(root):])


def name_in_folder(folder: str | None, path: object, *, native: bool) -> str | None:
    """``path`` (a path marimo reported) relative to ``folder`` as a POSIX name; ``None`` when it is
    empty, relative or outside the folder.

    ``native``: the kernel's paths are this machine's (the unsafe-local runtime), so both are compared with
    links and junctions resolved (``os.path.realpath``; case-insensitive on Windows, either
    separator): a file reached through a link that leaves the folder is outside it, and the name
    has the file's own spelling where it exists (Windows returns the on-disk case). Otherwise they
    are POSIX paths inside a container.
    """
    if not folder or not isinstance(path, str) or not path:
        return None
    if native:
        if not Path(path).is_absolute():
            return None
        root, target = os.path.realpath(folder), os.path.realpath(path)
        return _parts_under(Path(root).parts, Path(target).parts, os.path.normcase)
    if not path.startswith("/"):
        return None
    root_parts = PurePosixPath(posixpath.normpath(folder)).parts
    return _parts_under(root_parts, PurePosixPath(posixpath.normpath(path)).parts, lambda part: part)


def open_notebook_url(server: MarimoServer, file_key: str, *, view: str = "app", with_token: bool = False) -> str:
    """URL the user should open so the kernel gets a session for the file at ``file_key`` (the
    kernel's path, :func:`kernel_path`).

    ``view="app"`` (default) opens the notebook in app view so the chat user sees results, not code:
    ``http://127.0.0.1:2718/?file=/work/notebooks/sales.py&view-as=present``. ``view="edit"`` opens
    the plain editor (no ``view-as`` parameter). ``with_token=True`` (for a URL the user opens) ends
    it with ``&access_token=<token>`` when the server has one, which signs the browser in; the
    default leaves the token out, so text that goes to the model endpoint never carries it.
    """
    if view not in ("app", "edit"):
        raise ValueError(f"view must be 'app' or 'edit', not {view!r}")
    url = f"{server.url.rstrip('/')}/?file={quote(file_key, safe='/:')}"
    if view == "app":
        url += f"&{APP_VIEW_PARAM}"
    if with_token and server.token:
        url += f"&access_token={quote(server.token, safe='')}"
    return url


def home_url(server: MarimoServer, *, with_token: bool = False) -> str:
    """marimo's home page (the notebooks folder); ``with_token=True`` signs it in with the server's
    token when it has one (only for a URL the user opens)."""
    url = f"{server.url.rstrip('/')}/"
    if with_token and server.token:
        url += f"?access_token={quote(server.token, safe='')}"
    return url


# --------------------------------------------------------------------------- #
# Starting a server: ports and readiness
# --------------------------------------------------------------------------- #


def find_free_port(preferred: int = 2718, host: str = "127.0.0.1") -> int:
    """``preferred`` when it is free (a bind succeeds), otherwise a free ephemeral port chosen by the OS.

    Another process can take the port before marimo binds it; the start then fails, because
    :func:`wait_for_health` only accepts the server that holds this start's token.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, preferred))
            return preferred
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def wait_for_health(
    url: str,
    timeout: float = 60.0,
    *,
    token: str | None = None,
    interval: float = 0.5,
    should_stop: Callable[[], bool] | None = None,
) -> bool:
    """Poll until the server at ``url`` answers with ``token`` (:func:`answers_with_token`; only
    ``/health`` without one), the timeout passes, or ``should_stop()`` (checked first: the
    starting process or container has gone) returns True. Checking the token means a server
    another process runs on the same port is never taken for the one being started."""
    deadline = time.monotonic() + timeout
    while True:
        if should_stop is not None and should_stop():
            return False
        if answers_with_token(url, token):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval)


def wait_for_session(
    client: "MarimoClient",
    notebook: str | None,
    timeout: float = 90.0,
    *,
    interval: float = 0.5,
    should_stop: Callable[[], bool] | None = None,
) -> MarimoSession | None:
    """Poll for a kernel session; return ``None`` on timeout or cooperative cancellation.

    With ``should_stop``, check between requests and at most every 0.1 s while waiting.
    An in-flight request still finishes under the client's own network timeout.
    """
    deadline = time.monotonic() + timeout
    while True:
        if should_stop is not None and should_stop():
            return None
        try:
            session = client.resolve_session(notebook)
        except NoSessionError:
            pass
        else:
            return None if should_stop is not None and should_stop() else session
        if time.monotonic() >= deadline:
            return None
        if should_stop is None:
            time.sleep(interval)
            continue
        next_attempt = min(deadline, time.monotonic() + interval)
        while True:
            if should_stop():
                return None
            remaining = next_attempt - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.1, remaining))


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
    therefore never passes, so a start only ever accepts the server it started, even when
    another one listens on that port. Without a token only ``/health`` is checked.
    """
    if not _health_ok(url, timeout):
        return False
    if not token:
        return True
    return _sessions_status(url, None, timeout) in (401, 403) and _sessions_status(url, token, timeout) == 200


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


def match_session(sessions: Sequence[MarimoSession], notebook: str, *, fold: bool = False) -> MarimoSession | None:
    """The session whose notebook is ``notebook`` (a name in the notebooks folder), or ``None``.

    A session matches only when its ``name`` (its path relative to the notebooks folder, set by
    :meth:`MarimoClient.sessions`) is ``notebook``, ignoring case with ``fold`` (the kernel's paths
    are Windows paths). Untitled sessions and notebooks outside the folder never match, and a bare
    filename never matches a notebook of the same name in another folder. When several sessions
    name the same file the most recently created one (last in marimo's listing) wins.
    """
    wanted = notebook.casefold() if fold else notebook
    matched = [s for s in sessions if s.name is not None and (s.name.casefold() if fold else s.name) == wanted]
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

    ``token`` is sent as ``Authorization: Bearer <token>``. Notebooks are *names* relative to
    ``notebooks_path``, the kernel's path of the notebooks folder: the client turns a name into a
    kernel path for ``?file=`` keys (:meth:`notebook_url`) and a reported session path back into a
    name (:meth:`sessions`); ``native_paths`` says the kernel's paths are this machine's (the local
    runtime) rather than POSIX paths in a container. ``notebook`` is the default name.

    The notebook links in error hints leave the token out unless ``token_in_links`` is set: the
    agent's tools pass those hints to the model endpoint, while the CLI prints them for the user.

    The file methods (:meth:`list_files` ...) are marimo's file explorer, in kernel paths; the
    sandbox builds notebook operations on them.
    """

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        *,
        timeout: float = 10.0,
        notebooks_path: str | None = None,
        native_paths: bool = False,
        notebook: str | None = None,
        token_in_links: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.notebooks_path = notebooks_path
        self.native_paths = native_paths
        self.notebook = notebook
        self.token_in_links = token_in_links
        self._server_token: str | None = None

    # -- names <-> kernel paths ---------------------------------------------- #

    def file_key(self, name: str) -> str:
        """The kernel's path of the notebook ``name``; ``ValueError`` without a notebooks folder."""
        if not self.notebooks_path:
            raise ValueError("this client has no notebooks folder")
        return kernel_path(self.notebooks_path, name)

    def name_of(self, path: object) -> str | None:
        """The name of the file at ``path`` (a kernel path) in the notebooks folder, or ``None``."""
        return name_in_folder(self.notebooks_path, path, native=self.native_paths)

    @property
    def folds_case(self) -> bool:
        """Names ignore case: the kernel's paths are this Windows machine's."""
        return self.native_paths and os.name == "nt"

    # -- low level ---------------------------------------------------------- #

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {"Accept": "*/*", "User-Agent": "hailer"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if extra:
            headers.update(extra)
        return headers

    def _read(self, resp, path: str, limit: int | None = None) -> bytes:
        """The body of ``resp``, at most ``limit`` bytes (a larger ``Content-Length``, or more data,
        is ``MarimoUnavailableError`` without reading on), read in chunks against this thread's
        :func:`request_deadline`, so a server that trickles its answer cannot hold Hailer."""
        limit = MAX_REPLY_BYTES if limit is None else limit
        length = resp.headers.get("Content-Length")
        if length is not None and length.strip().isdigit() and int(length) > limit:
            raise self._too_large(path, limit)
        deadline = _deadline()
        chunks: list[bytes] = []
        size = 0
        while True:
            if deadline is not None and time.monotonic() > deadline:
                raise self._unavailable("out of time")
            try:
                chunk = resp.read1(_READ_CHUNK)
            except (socket.timeout, TimeoutError) as err:
                raise self._unavailable("timed out") from err
            except OSError as err:
                raise self._unavailable(str(err)) from err
            if not chunk:
                return b"".join(chunks)
            size += len(chunk)
            if size > limit:
                raise self._too_large(path, limit)
            chunks.append(chunk)

    def _too_large(self, path: str, limit: int) -> MarimoUnavailableError:
        return MarimoUnavailableError(
            f"Marimo at {self.base_url} answered {path} with more than {limit // (1024 * 1024)} MiB; Hailer stopped reading.",
            hint="A notebook this large is not copied or read. " + launch_hint(),
        )

    def _unavailable(self, reason: str) -> MarimoUnavailableError:
        return MarimoUnavailableError(
            f"Marimo is not running at {self.base_url} ({reason}).",
            hint=launch_hint(),
        )

    def _auth_error(self, status: int) -> MarimoUnavailableError:
        return MarimoUnavailableError(
            f"Marimo at {self.base_url} rejected the request (HTTP {status}).",
            hint=(
                "Hailer sends the token of the server it started, so something else may now answer on that "
                "port. End the chat (/exit) and run uvx hailer again to start a new kernel."
            ),
        )

    def _open(self, method: str, path: str, body: bytes | None = None, headers: dict[str, str] | None = None, timeout: float | None = None):
        req = urllib.request.Request(f"{self.base_url}{path}", data=body, method=method, headers=self._headers(headers))
        timeout = timeout if timeout is not None else self.timeout
        deadline = _deadline()
        if deadline is not None:
            left = deadline - time.monotonic()
            if left <= 0:
                raise self._unavailable("out of time")
            timeout = min(timeout, left)
        try:
            return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310
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
                payload = json.loads(self._read(resp, "/api/sessions").decode("utf-8", "replace") or "{}")
        except urllib.error.HTTPError as err:
            # 401/403 are already mapped by _open; anything else (500, 404 on an old marimo) lands here.
            detail = _error_detail(err.read(65536))
            raise MarimoUnavailableError(
                f"Marimo at {self.base_url} answered HTTP {err.code} to /api/sessions" + (f": {detail}" if detail else "."),
                hint="Check the marimo server log for errors (0.24.x is expected). " + launch_hint(),
            ) from err
        if not isinstance(payload, dict):
            return []
        out: list[MarimoSession] = []
        for sid, info in payload.items():
            info = info if isinstance(info, dict) else {}
            filename, path = info.get("filename"), info.get("path")
            out.append(
                MarimoSession(
                    session_id=str(sid),
                    filename=filename if isinstance(filename, str) else None,
                    path=path if isinstance(path, str) else None,
                    name=self.name_of(path or filename),
                )
            )
        return out

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

    def _post(self, path: str, body: bytes, content_type: str) -> bytes:
        """POST ``body`` with the server token; the reply's body. A 401 means the cached server
        token went stale (the server restarted): it is dropped so the next call refetches it."""
        headers = {"Content-Type": content_type, SERVER_TOKEN_HEADER: self.server_token()}
        try:
            with self._open("POST", path, body=body, headers=headers) as resp:
                return self._read(resp, path)
        except MarimoUnavailableError as err:
            cause = err.__cause__
            if isinstance(cause, urllib.error.HTTPError) and cause.code == 401:
                self._server_token = None
                raise MarimoUnavailableError(
                    f"Marimo at {self.base_url} rejected the server token (HTTP 401).",
                    hint="The marimo server was probably restarted, so its token changed; retry once.",
                ) from cause
            raise

    def _post_json(self, path: str, payload: dict) -> dict:
        """POST ``payload`` as JSON with the server token; the reply as a dict. ``HTTPError`` (not
        401/403) propagates for the caller to explain."""
        raw = self._post(path, json.dumps(payload).encode("utf-8"), "application/json")
        try:
            reply = json.loads(raw.decode("utf-8", "replace") or "{}")
        except ValueError as err:
            raise MarimoUnavailableError(
                f"Marimo at {self.base_url} answered {path} with something that is not JSON.",
                hint="Check that the URL points at a marimo 0.24.x edit server.",
            ) from err
        return reply if isinstance(reply, dict) else {}

    def _file_error(self, path: str, err: urllib.error.HTTPError) -> MarimoUnavailableError:
        detail = _error_detail(err.read(65536))
        return MarimoUnavailableError(
            f"Marimo at {self.base_url} answered HTTP {err.code} to {path}" + (f": {detail}" if detail else "."),
            hint="Check the marimo server log for errors (0.24.x is expected). " + launch_hint(),
        )

    def _file_post(self, path: str, payload: dict) -> dict:
        """:meth:`_post_json` for the file endpoints, with any HTTP error as ``MarimoUnavailableError``."""
        try:
            return self._post_json(path, payload)
        except urllib.error.HTTPError as err:
            raise self._file_error(path, err) from err

    def shutdown_session(self, session_id: str) -> None:
        """Close the kernel session ``session_id`` (frees its kernel; the browser tab disconnects)."""
        try:
            self._post("/api/home/shutdown_session", json.dumps({"sessionId": session_id}).encode("utf-8"), "application/json")
        except urllib.error.HTTPError as err:
            detail = _error_detail(err.read(65536))
            raise MarimoUnavailableError(
                f"Marimo refused to close session {session_id} (HTTP {err.code})" + (f": {detail}" if detail else "."),
                hint="Check that the session id is current; GET /api/sessions lists the open ones.",
            ) from err

    # -- marimo's file explorer (kernel paths) -------------------------------- #

    def list_files(self, folder: str) -> list[dict]:
        """``POST /api/files/list_files``: the entries directly in ``folder`` (marimo's ``FileInfo``:
        ``path``, ``name``, ``isDirectory``, ``isMarimoFile``, ``lastModified``, ``size``). A missing
        folder lists as empty (marimo swallows the error); an HTTP error is ``MarimoUnavailableError``."""
        reply = self._file_post("/api/files/list_files", {"path": folder})
        files = reply.get("files")
        return [entry for entry in files if isinstance(entry, dict)] if isinstance(files, list) else []

    def file_details(self, path: str, max_bytes: int | None = None) -> dict:
        """``POST /api/files/file_details``: ``file`` (FileInfo), ``contents``, ``isBase64`` (not UTF-8)
        and ``isTooLarge`` (over ``max_bytes``; no contents then). ``HTTPError`` when the file cannot
        be read (marimo answers 500 for a missing one)."""
        payload: dict = {"path": path}
        if max_bytes is not None:
            payload["maxBytes"] = max_bytes
        return self._post_json("/api/files/file_details", payload)

    def create_file(self, folder: str, name: str, data: bytes) -> dict:
        """``POST /api/files/create`` (multipart): a new file ``name`` in ``folder`` (created with its
        parents) holding ``data``. marimo never overwrites: when ``name`` is taken it picks
        ``<stem>_1<suffix>``, so the reply's ``info.path`` says where the file went. The reply's
        ``success`` is false (with ``message``) when marimo refused."""
        boundary = f"hailer-{secrets.token_hex(16)}"
        filename = quote(name, safe=" ")  # a header parameter: no quotes, CR or LF
        parts: list[bytes] = []
        for key, value in (("path", folder), ("type", "file"), ("name", name)):
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode("utf-8"))
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n".encode("utf-8") + data + b"\r\n"
        )
        parts.append(f"--{boundary}--\r\n".encode("utf-8"))
        try:
            raw = self._post("/api/files/create", b"".join(parts), f"multipart/form-data; boundary={boundary}")
        except urllib.error.HTTPError as err:
            raise self._file_error("/api/files/create", err) from err
        try:
            reply = json.loads(raw.decode("utf-8", "replace") or "{}")
        except ValueError:
            reply = {}
        return reply if isinstance(reply, dict) else {}

    def update_file(self, path: str, contents: str) -> dict:
        """``POST /api/files/update``: replace the text of the existing file at ``path`` (marimo
        reloads an open session of it). ``success`` is false (with ``message``) when it failed.
        marimo writes it with ``Path.write_text``, so a Windows kernel stores ``\\r\\n`` line endings,
        while :meth:`create_file` stores the bytes as given: compare contents newline-normalised."""
        return self._file_post("/api/files/update", {"path": path, "contents": contents})

    def delete_file(self, path: str) -> dict:
        """``POST /api/files/delete``: remove the file (or folder) at ``path``."""
        return self._file_post("/api/files/delete", {"path": path})

    # -- notebooks and sessions (names) --------------------------------------- #

    def notebook_url(self, notebook: str | None = None) -> str:
        """The URL that opens ``notebook`` (a name; token only with ``token_in_links``, see the class
        docstring); marimo's home page without a name or a notebooks folder."""
        name = notebook or self.notebook
        server = MarimoServer(url=self.base_url, token=self.token if self.token_in_links else None)
        if name is None or not self.notebooks_path:
            return home_url(server, with_token=self.token_in_links)
        return open_notebook_url(server, self.file_key(name), with_token=self.token_in_links)

    def resolve_session(self, notebook: str | None) -> MarimoSession:
        """Pick the session for ``notebook``, a name (or the only session when ``notebook`` is None)."""
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
        session = match_session(sessions, notebook, fold=self.folds_case)
        if session is not None:
            return session
        raise NoSessionError(
            f"No open session matches {notebook}.",
            hint=f"Open {self.notebook_url(notebook)} in your browser. Open sessions:\n" + self._describe_sessions(sessions),
        )

    @staticmethod
    def _describe_sessions(sessions: Sequence[MarimoSession]) -> str:
        return "\n".join(f"  {s.session_id}  {s.name or s.path or s.filename or '(unsaved notebook)'}" for s in sessions) or "  (none)"

    def execute(
        self,
        code: str,
        *,
        session_id: str | None = None,
        notebook: str | None = None,
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
            detail = _error_detail(err.read(65536))
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
    "home_url",
    "launch_hint",
    "marimo_server_command",
    "kernel_path",
    "match_session",
    "name_in_folder",
    "notebook_file_key",
    "open_notebook_url",
    "wait_for_health",
    "wait_for_session",
]
