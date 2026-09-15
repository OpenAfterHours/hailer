"""Pure-Python client for a running marimo server (the "marimo pair" protocol).

Talks to three endpoints with the standard library only:

- ``GET  {url}/health``                  liveness (no auth)
- ``GET  {url}/api/sessions``            ``{session_id: {"filename": ..., "path": ...}}``
- ``POST {url}/api/kernel/execute``      run code in the kernel scratchpad; SSE stream of
                                          ``stdout`` / ``stderr`` / ``done`` events

A kernel *session* only exists while the notebook is open in a browser. Durable
notebook changes are made from the scratchpad through ``marimo._code_mode``; the
snippet constants at the bottom of this module are the canonical patterns.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from urllib.parse import quote, urlsplit

from hailer.errors import MarimoExecutionError, MarimoUnavailableError, NoSessionError
from hailer.models import ExecResult, HailerConfig, MarimoServer, MarimoSession

DEFAULT_LAUNCH_NOTEBOOK = "notebooks/analysis.py"
_HEALTH_TIMEOUT = 1.0

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


def launch_command(notebook: Path | None, workspace: Path | None = None, *, port: int | None = None) -> list[str]:
    """The exact command a user should run to start marimo for this notebook."""
    rel = _relative_to_workspace(notebook, workspace) if notebook is not None else DEFAULT_LAUNCH_NOTEBOOK
    cmd = ["uv", "run", "marimo", "edit", rel, "--no-token"]
    if port is not None:
        cmd += ["--port", str(port)]
    return cmd


def notebook_launch_command(config: HailerConfig, *, port: int | None = None) -> list[str]:
    return launch_command(config.notebook, config.workspace, port=port)


def open_notebook_url(server: MarimoServer, notebook: Path, workspace: Path) -> str:
    """URL the user should open so the kernel gets a session, e.g. ``http://127.0.0.1:2718/?file=notebooks/analysis.py``."""
    rel = _relative_to_workspace(notebook, workspace)
    return f"{server.url.rstrip('/')}/?file={quote(rel, safe='/:')}"


def _format_command(cmd: Sequence[str]) -> str:
    return " ".join(cmd)


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


def find_server(config: HailerConfig, registry: Path | None = None) -> MarimoServer | None:
    """Configured URL first; otherwise the single live registry server; otherwise ``None``."""
    if config.marimo_url:
        return MarimoServer(url=config.marimo_url.rstrip("/"), source="config")
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


def _normalise_name(value: str | Path) -> str:
    return os.path.normcase(Path(value).name)


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
    """HTTP client for one marimo server."""

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        *,
        timeout: float = 10.0,
        notebook: Path | None = None,
        workspace: Path | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.notebook = notebook
        self.workspace = workspace

    # -- low level ---------------------------------------------------------- #

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {"Accept": "*/*", "User-Agent": "hailer"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if extra:
            headers.update(extra)
        return headers

    def _unavailable(self, reason: str) -> MarimoUnavailableError:
        cmd = _format_command(launch_command(self.notebook, self.workspace))
        return MarimoUnavailableError(
            f"Marimo is not running at {self.base_url} ({reason}).",
            hint=f"Start it with:\n\n    {cmd}\n\nThen run Hailer again:\n\n    uv run hailer",
        )

    def _auth_error(self, status: int) -> MarimoUnavailableError:
        return MarimoUnavailableError(
            f"Marimo at {self.base_url} rejected the request (HTTP {status}).",
            hint=(
                "The server was started with an auth token. Either restart it with --no-token "
                "or set HAILER_MARIMO_TOKEN to the token shown when marimo started."
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
        with self._open("GET", "/api/sessions") as resp:
            payload = json.loads(resp.read().decode("utf-8", "replace") or "{}")
        if not isinstance(payload, dict):
            return []
        out: list[MarimoSession] = []
        for sid, info in payload.items():
            info = info if isinstance(info, dict) else {}
            out.append(MarimoSession(session_id=str(sid), filename=info.get("filename"), path=info.get("path")))
        return out

    def notebook_url(self, notebook: Path | None = None) -> str:
        nb = notebook or self.notebook
        if nb is None:
            return f"{self.base_url}/"
        return open_notebook_url(MarimoServer(url=self.base_url), nb, self.workspace or Path.cwd())

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
        target_path = _normalise_path(notebook)
        target_name = _normalise_name(notebook)
        by_path = [s for s in sessions if s.path and _normalise_path(s.path) == target_path]
        if len(by_path) == 1:
            return by_path[0]
        by_file = [s for s in sessions if s.filename and _normalise_path(s.filename) == target_path]
        if len(by_file) == 1:
            return by_file[0]
        by_name = [s for s in sessions if (s.filename and _normalise_name(s.filename) == target_name) or (s.path and _normalise_name(s.path) == target_name)]
        if len(by_name) == 1:
            return by_name[0]
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
    "MarimoClient",
    "build_create_cell_code",
    "discover_servers",
    "find_server",
    "launch_command",
    "notebook_launch_command",
    "open_notebook_url",
    "registry_dir",
]

if sys.platform == "win32":  # pragma: no cover - documentation only
    __doc__ += "\n\nOn Windows the registry lives in %USERPROFILE%\\.marimo\\servers."
