"""A fake marimo edit server on loopback (test helper, standard library only).

Answers what :mod:`hailer.marimo_client` talks to: ``/health``, ``/api/sessions`` (token-checked
when ``token`` is set: 401 without ``Authorization: Bearer <token>``), the page with its
skew-protection token, ``/api/home/shutdown_session`` and a scripted SSE ``/api/kernel/execute``
(``mode``). Every POST is recorded in ``requests``.

:func:`serving` runs one on a daemon thread with a short poll interval, so stopping it takes a
few milliseconds instead of ``serve_forever``'s default half second.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

#: How often a serving FakeMarimo checks for shutdown (the stdlib default, 0.5 s, slowed every test).
POLL_INTERVAL_SEC = 0.05


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
        body = self.rfile.read(n) if n else b""
        self.server.requests.append({"path": self.path, "headers": dict(self.headers.items()), "body": json.loads(body or b"{}")})
        if not self._authorised():
            self._send(401, b'{"detail":"unauthorised"}')
            return
        if self.path == "/api/home/shutdown_session":
            # every POST except /api/kernel/execute needs the skew-protection token
            if self.headers.get("Marimo-Server-Token") != self.server.server_token or self.server.mode == "stale_token":
                self._send(401, b'{"error":"Invalid server token"}')
                return
            sid = self.server.requests[-1]["body"].get("sessionId")
            if sid not in self.server.sessions:
                self._send(500, b'{"detail":"Session not found"}')
                return
            del self.server.sessions[sid]
            self._send(200, b'{"files":[]}')
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

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

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
