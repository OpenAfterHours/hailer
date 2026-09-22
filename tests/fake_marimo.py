"""A fake marimo edit server on loopback (test helper, standard library only).

Answers what :mod:`hailer.marimo_client` talks to: ``/health``, ``/api/sessions`` (token-checked
when ``token`` is set: 401 without ``Authorization: Bearer <token>``), the page with its
skew-protection token, ``/api/home/shutdown_session``, a scripted SSE ``/api/kernel/execute``
(``mode``) and marimo 0.24.2's file explorer (``/api/files/list_files``, ``/file_details``,
``/create`` (multipart), ``/update``, ``/delete``). Every POST is recorded in ``requests``.

Like marimo, every POST except ``/api/kernel/execute`` needs the skew-protection token in the
``Marimo-Server-Token`` header, and the file endpoints need the bearer token (``@requires("edit")``).
Like marimo, the file endpoints confine nothing: a local-runtime server (``kernel_root`` is
``notebook_file_key(files_root)``) reads and writes any host path it is given, following links and
junctions, and names entries ``os.path.join(folder, name)`` (mixed separators on Windows); a
container-like server (``kernel_root`` ``/work/notebooks``) maps that kernel path to ``files_root``
and has nothing else. Every path a file endpoint was asked about is recorded in ``file_paths``, so
tests assert that Hailer never sends one outside the notebooks folder. Replies have marimo's shapes
(camel-case ``FileInfo``), a missing file in ``file_details`` is a 500, and ``create`` never
overwrites (it picks ``<stem>_1<suffix>``).

:func:`serving` runs one on a daemon thread with a short poll interval, so stopping it takes a
few milliseconds instead of ``serve_forever``'s default half second.
"""

from __future__ import annotations

import base64
import json
import os
import posixpath
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

#: How often a serving FakeMarimo checks for shutdown (the stdlib default, 0.5 s, slowed every test).
POLL_INTERVAL_SEC = 0.05
_FILE_ENDPOINTS = ("/api/files/list_files", "/api/files/file_details", "/api/files/create", "/api/files/update", "/api/files/delete")


def _sse(events: list[tuple[str, dict]], newline: str = "\n") -> bytes:
    return "".join(f"event: {name}{newline}data: {json.dumps(data)}{newline}{newline}" for name, data in events).encode()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # noqa: D401 - silence
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode())

    def _authorised(self) -> bool:
        srv = self.server
        if not srv.token:
            return True
        return self.headers.get("Authorization") == f"Bearer {srv.token}"

    def do_GET(self):
        if self.path == "/health":
            self._send(200, b'{"status":"healthy"}')
        elif self.path == "/":
            self.server.page_hits += 1
            if self.server.mode == "no_token_tag":
                self._send(200, b"<html><body>not marimo</body></html>", "text/html")
                return
            html = f'<html><head><marimo-server-token data-token="{self.server.server_token}"></marimo-server-token></head></html>'
            self._send(200, html.encode(), "text/html")
        elif self.path == "/api/sessions":
            if not self._authorised():
                self._send(401, b'{"detail":"unauthorised"}')
                return
            if self.server.mode == "sessions_500":
                self._send(500, b'{"detail":"kernel manager exploded"}')
                return
            self._send(200, json.dumps(self.server.sessions).encode())
        else:
            self._send(404, b"")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        multipart = (self.headers.get("Content-Type") or "").startswith("multipart/form-data")
        body = self._form(raw) if multipart else json.loads(raw or b"{}")
        self.server.requests.append({"path": self.path, "headers": dict(self.headers.items()), "body": body})
        if not self._authorised():
            self._send(401, b'{"detail":"unauthorised"}')
            return
        if self.path != "/api/kernel/execute":
            # every POST except /api/kernel/execute needs the skew-protection token
            if self.headers.get("Marimo-Server-Token") != self.server.server_token or self.server.mode == "stale_token":
                self._send(401, b'{"error":"Invalid server token"}')
                return
        if self.path == "/api/home/shutdown_session":
            sid = body.get("sessionId")
            if sid not in self.server.sessions:
                self._send(500, b'{"detail":"Session not found"}')
                return
            del self.server.sessions[sid]
            self._send(200, b'{"files":[]}')
            return
        if self.path in _FILE_ENDPOINTS:
            if self.server.mode == "files_500":
                self._send(500, b"Internal Server Error", "text/plain")
                return
            getattr(self, "_" + self.path.rsplit("/", 1)[-1])(body)
            return
        if self.path != "/api/kernel/execute":
            self._send(404, b"")
            return
        mode = self.server.mode
        if mode == "json_error":
            self._send(400, b'{"detail":"Session not found: stale"}')
            return
        if mode == "plain_json_200":
            self._send(200, b'{"detail":"not a stream"}')
            return
        newline = "\r\n" if mode == "crlf" else "\n"
        events: list[tuple[str, dict]]
        if mode in ("success", "crlf"):
            events = [("stdout", {"data": "hello "}), ("stdout", {"data": "world\n"}), ("done", {"success": True, "output": {"mimetype": "text/plain", "data": "42"}})]
        elif mode == "stderr":
            events = [("stderr", {"data": "Traceback: boom\n"}), ("done", {"success": False, "output": {"mimetype": "text/plain", "data": ""}})]
        elif mode == "nodone":
            events = [("stdout", {"data": "partial"})]
        else:
            raise AssertionError(mode)
        self._send(200, _sse(events, newline), "text/event-stream")

    # -- the file explorer -------------------------------------------------- #

    def _form(self, raw: bytes) -> dict:
        """A multipart body as {field: str} plus ``file``: (filename, bytes), parsed like Starlette."""
        message = BytesParser(policy=HTTP).parsebytes(f"Content-Type: {self.headers['Content-Type']}\r\n\r\n".encode() + raw)
        form: dict = {}
        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            filename = part.get_filename()
            data = part.get_payload(decode=True) or b""
            form[name] = (filename, data) if filename is not None else data.decode("utf-8")
        return form

    def _join(self, folder: str, name: str) -> str:
        """The path marimo reports for ``name`` in ``folder``: ``os.path.join`` on the host's paths."""
        return os.path.join(folder, name) if self.server.native else f"{folder.rstrip('/')}/{name}"

    def _info(self, host: Path, kernel: str) -> dict:
        stat = host.stat()
        directory = host.is_dir()
        return {
            "id": kernel,
            "path": kernel,
            "name": host.name,
            "isDirectory": directory,
            "isMarimoFile": not directory and _is_marimo_app(host),
            "lastModified": stat.st_mtime,
            "size": None if directory else stat.st_size,
            "children": [],
        }

    def _list_files(self, body: dict) -> None:
        folder = body.get("path") or self.server.kernel_root
        host = self.server.host_path(folder)
        files, folders = [], []
        if host is not None and host.is_dir():
            for entry in sorted(host.iterdir(), key=lambda p: p.name):
                if entry.name in (".", "..", ".DS_Store"):
                    continue
                info = self._info(entry, self._join(folder, entry.name))
                (folders if info["isDirectory"] else files).append(info)
        self._json({"files": folders + files, "root": folder})

    def _file_details(self, body: dict) -> None:
        path = body["path"]
        host = self.server.host_path(path)
        if host is None or not host.exists():
            self._send(500, b"Internal Server Error", "text/plain")  # marimo: an unhandled FileNotFoundError
            return
        max_bytes = body.get("maxBytes")
        reply = {"file": self._info(host, path), "contents": None, "mimeType": "text/x-python", "isBase64": False, "isTooLarge": False}
        if not host.is_dir():
            data = host.read_bytes()
            if max_bytes is not None and len(data) > max_bytes:
                reply["isTooLarge"] = True
            else:
                try:
                    reply["contents"] = data.decode("utf-8")
                except UnicodeDecodeError:
                    reply["contents"], reply["isBase64"] = base64.b64encode(data).decode(), True
        self._json(reply)

    def _create(self, body: dict) -> None:
        name = body.get("name", "")
        if name in (".", "..") or not name.strip() or "/" in name or "\\" in name or "\x00" in name:
            self._json({"success": False, "message": f"Invalid name {name!r}"})
            return
        folder = body["path"]
        host = self.server.host_path(folder)
        if host is None:
            self._json({"success": False, "message": f"Permission denied: {folder}"})
            return
        host.mkdir(parents=True, exist_ok=True)
        stem, suffix = posixpath.splitext(name)
        target, index = name, 0
        while (host / target).exists():  # marimo never overwrites: <stem>_1<suffix>
            index += 1
            target = f"{stem}_{index}{suffix}"
        _filename, data = body.get("file") or ("", b"")
        (host / target).write_bytes(data)
        self._json({"success": True, "info": self._info(host / target, self._join(folder, target))})

    def _update(self, body: dict) -> None:
        path = body["path"]
        host = self.server.host_path(path)
        if host is None or not host.exists():
            self._json({"success": False, "message": f"No such file: {path}"})
            return
        host.write_text(body["contents"], encoding="utf-8")
        self._json({"success": True, "info": self._info(host, path)})

    def _delete(self, body: dict) -> None:
        host = self.server.host_path(body["path"])
        if host is None or not host.exists():
            self._json({"success": False, "message": f"No such file: {body['path']}"})
            return
        host.unlink()
        self._json({"success": True})


def _is_marimo_app(path: Path) -> bool:
    """marimo's rule for a ``.py`` file: ``import marimo`` and ``marimo.App`` in its first megabyte."""
    if path.suffix != ".py":
        return False
    try:
        head = path.read_bytes()[: 1024 * 1024]
    except OSError:
        return False
    return b"import marimo" in head and b"marimo.App" in head


class FakeMarimo(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.sessions: dict[str, dict] = {}
        self.mode = "success"
        self.token: str | None = None
        self.server_token = "skew-token-123"
        self.page_hits = 0
        self.requests: list[dict] = []
        #: The host folder the file endpoints serve, and the kernel path it appears at.
        self.files_root: Path | None = None
        self.kernel_root = "/work/notebooks"
        self.native = False  # the kernel sees the host's own paths (a local-runtime server)
        #: Every path a file endpoint was asked about, in order.
        self.file_paths: list[str] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def serve_files(self, root: Path, kernel_root: str = "/work/notebooks") -> FakeMarimo:
        """Serve ``root`` through the file endpoints under the kernel path ``kernel_root``; the
        server is a local-runtime one when that is ``root``'s own file key."""
        from hailer.marimo_client import notebook_file_key

        self.files_root, self.kernel_root = Path(root), kernel_root.rstrip("/")
        self.native = self.kernel_root == notebook_file_key(Path(root))
        return self

    def host_path(self, kernel: str) -> Path | None:
        """The host file behind the kernel path ``kernel`` (recorded): any host path for a
        local-runtime server; for a container-like one, ``files_root`` below ``kernel_root`` and
        ``None`` (nothing there) elsewhere."""
        if not isinstance(kernel, str):
            return None
        self.file_paths.append(kernel)
        if self.native:
            return Path(kernel)
        if self.files_root is None:
            return None
        path = posixpath.normpath(kernel.replace("\\", "/"))
        if path == self.kernel_root:
            return self.files_root
        if not path.startswith(self.kernel_root + "/"):
            return None
        return self.files_root.joinpath(*path[len(self.kernel_root) + 1:].split("/"))

    def start(self) -> FakeMarimo:
        """Serve on a daemon thread (stop it with :meth:`stop`)."""
        threading.Thread(target=self.serve_forever, kwargs={"poll_interval": POLL_INTERVAL_SEC}, daemon=True).start()
        return self

    def stop(self) -> None:
        self.shutdown()
        self.server_close()


@contextmanager
def serving(*, token: str | None = None) -> Iterator[FakeMarimo]:
    """A running FakeMarimo (requiring ``token`` when given), stopped afterwards."""
    srv = FakeMarimo()
    srv.token = token
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


running = serving
