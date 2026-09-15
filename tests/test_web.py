"""Tests for hailer.web: allow-list matching, HTML-to-text, and allow-listed fetching."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from hailer.errors import HailerError, WebAccessDenied
from hailer.models import HailerConfig, WebConfig
from hailer.web import fetch_page, host_allowed, html_to_text, is_private_host, truncate_text


def make_config(tmp_path: Path, *domains: str, **kw) -> HailerConfig:
    return HailerConfig(
        workspace=tmp_path,
        notebook=tmp_path / "notebooks" / "analysis.py",
        data_dir=tmp_path / "data",
        context_dir=tmp_path / ".config" / "hailer" / "context",
        skills_dir=tmp_path / ".config" / "hailer" / "skills",
        prompts_dir=tmp_path / ".config" / "hailer" / "prompts",
        web=WebConfig(allowed_domains=tuple(domains), max_page_bytes=kw.pop("max_page_bytes", 200_000)),
        **kw,
    )


# --------------------------------------------------------------------------- #
# host_allowed
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "host, allowed, expected",
    [
        ("docs.pola.rs", ["docs.pola.rs"], True),
        ("DOCS.POLA.RS", ["docs.pola.rs"], True),
        ("docs.pola.rs:443", ["docs.pola.rs"], True),
        ("docs.pola.rs.", ["docs.pola.rs"], True),
        ("pola.rs", ["docs.pola.rs"], False),
        ("evil-docs.pola.rs", ["docs.pola.rs"], False),
        # "*.d" = subdomains only
        ("www.bankofengland.co.uk", ["*.bankofengland.co.uk"], True),
        ("a.b.bankofengland.co.uk", ["*.bankofengland.co.uk"], True),
        ("bankofengland.co.uk", ["*.bankofengland.co.uk"], False),
        ("notbankofengland.co.uk", ["*.bankofengland.co.uk"], False),
        # "**.d" = apex plus subdomains
        ("bankofengland.co.uk", ["**.bankofengland.co.uk"], True),
        ("www.bankofengland.co.uk", ["**.bankofengland.co.uk"], True),
        ("xbankofengland.co.uk", ["**.bankofengland.co.uk"], False),
        # global allow never covers private/loopback
        ("example.com", ["*"], True),
        ("127.0.0.1", ["*"], False),
        ("localhost", ["*"], False),
        ("10.1.2.3", ["*"], False),
        # literal private entries are honoured
        ("127.0.0.1", ["127.0.0.1"], True),
        ("127.0.0.1:2718", ["127.0.0.1"], True),
        ("localhost", ["localhost"], True),
        # empties
        ("", ["example.com"], False),
        ("example.com", [], False),
        ("example.com", ["", "  "], False),
    ],
)
def test_host_allowed(host, allowed, expected):
    assert host_allowed(host, allowed) is expected


def test_is_private_host():
    assert is_private_host("127.0.0.1")
    assert is_private_host("localhost")
    assert is_private_host("app.localhost")
    assert is_private_host("192.168.1.10")
    assert is_private_host("[::1]:80")
    assert not is_private_host("example.com")
    assert not is_private_host("8.8.8.8")


# --------------------------------------------------------------------------- #
# html_to_text / truncate_text
# --------------------------------------------------------------------------- #


def test_html_to_text_structure():
    html = """
    <html><head><title>  Capital   Rules </title><style>p{color:red}</style>
    <script>alert('x')</script></head>
    <body><nav><a href="/">Home</a> | <a href="/x">Menu</a></nav>
    <h1>Overview</h1>
    <p>Risk   weighted   assets are <a href="/rwa">explained here</a>.</p>
    <ul><li>First point</li><li>Second point</li></ul>
    <pre>keep   this
  spacing</pre>
    <noscript>Enable JS</noscript>
    <table><tr><th>Class</th><th>RWA</th></tr><tr><td>Corporate</td><td>10</td></tr></table>
    </body></html>
    """
    text = html_to_text(html)
    assert text.startswith("# Capital Rules")
    assert "alert(" not in text and "color:red" not in text
    assert "Menu" not in text and "Enable JS" not in text
    assert "# Overview" in text
    assert "Risk weighted assets are explained here." in text
    assert "- First point" in text and "- Second point" in text
    assert "keep   this" in text and "  spacing" in text
    assert "Class | RWA" in text and "Corporate | 10" in text
    assert "\n\n\n" not in text


def test_truncate_text_keeps_head_and_tail():
    text = "A" * 500 + "B" * 500
    out = truncate_text(text, 200)
    assert len(out) <= 200 + 10  # marker line breaks add a little slack
    assert out.startswith("AAAA")
    assert out.endswith("BBBB")
    assert "truncated" in out
    assert truncate_text("short", 200) == "short"


# --------------------------------------------------------------------------- #
# fetch_page against a local server
# --------------------------------------------------------------------------- #


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # noqa: D401 - silence
        pass

    def _send(self, code: int, body: bytes, ctype: str, extra: dict[str, str] | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/page":
            self._send(200, b"<html><head><title>T</title></head><body><h1>Hello</h1><p>World</p></body></html>", "text/html; charset=utf-8")
        elif self.path == "/plain":
            self._send(200, b"just text\n", "text/plain; charset=utf-8")
        elif self.path == "/big":
            self._send(200, b"<p>" + b"x" * 5000 + b"</p>", "text/html")
        elif self.path == "/binary":
            self._send(200, b"\x00\x01\x02", "application/octet-stream")
        elif self.path == "/redirect-ok":
            self._send(302, b"", "text/plain", {"Location": "/page"})
        elif self.path == "/redirect-bad":
            self._send(302, b"", "text/plain", {"Location": "http://example.com/evil"})
        elif self.path == "/latin":
            self._send(200, "caf\xe9".encode("latin-1"), "text/plain; charset=iso-8859-1")
        else:
            self._send(404, b"nope", "text/plain")


@pytest.fixture(scope="module")
def local_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_fetch_html_page(tmp_path, local_server):
    config = make_config(tmp_path, "127.0.0.1")
    text = fetch_page(f"{local_server}/page", config)
    assert text.startswith(f"Source: {local_server}/page")
    assert "# T" in text and "# Hello" in text and "World" in text


def test_fetch_plain_and_charset(tmp_path, local_server):
    config = make_config(tmp_path, "127.0.0.1")
    assert fetch_page(f"{local_server}/plain", config).endswith("just text")
    assert fetch_page(f"{local_server}/latin", config).endswith("caf\xe9")


def test_fetch_denied_before_any_io(tmp_path):
    config = make_config(tmp_path, "docs.pola.rs")
    with pytest.raises(WebAccessDenied) as exc:
        fetch_page("https://definitely-not-allowed.invalid/x", config)
    assert "definitely-not-allowed.invalid" in str(exc.value)
    assert "docs.pola.rs" in exc.value.hint


def test_fetch_denied_when_no_domains_configured(tmp_path):
    config = make_config(tmp_path)
    with pytest.raises(WebAccessDenied) as exc:
        fetch_page("https://docs.pola.rs/", config)
    assert "No websites are allowed" in exc.value.hint


def test_fetch_rejects_non_http(tmp_path):
    config = make_config(tmp_path, "*")
    with pytest.raises(WebAccessDenied):
        fetch_page("file:///C:/Windows/win.ini", config)
    with pytest.raises(WebAccessDenied):
        fetch_page("ftp://example.com/x", config)


def test_fetch_redirect_same_host_ok(tmp_path, local_server):
    config = make_config(tmp_path, "127.0.0.1")
    text = fetch_page(f"{local_server}/redirect-ok", config)
    assert "# Hello" in text


def test_fetch_redirect_to_disallowed_host_denied(tmp_path, local_server):
    config = make_config(tmp_path, "127.0.0.1")
    with pytest.raises(WebAccessDenied) as exc:
        fetch_page(f"{local_server}/redirect-bad", config)
    assert "example.com" in str(exc.value)


def test_fetch_binary_content_reported(tmp_path, local_server):
    config = make_config(tmp_path, "127.0.0.1")
    text = fetch_page(f"{local_server}/binary", config)
    assert "application/octet-stream" in text and "not text" in text


def test_fetch_respects_max_page_bytes(tmp_path, local_server):
    config = make_config(tmp_path, "127.0.0.1", max_page_bytes=1000)
    text = fetch_page(f"{local_server}/big", config)
    assert "(first 1000 bytes only)" in text
    assert text.count("x") <= 1000


def test_fetch_respects_max_tool_output_chars(tmp_path, local_server):
    config = make_config(tmp_path, "127.0.0.1", max_tool_output_chars=300)
    text = fetch_page(f"{local_server}/big", config)
    assert len(text) <= 320
    assert "truncated" in text


def test_fetch_http_error(tmp_path, local_server):
    config = make_config(tmp_path, "127.0.0.1")
    with pytest.raises(HailerError) as exc:
        fetch_page(f"{local_server}/missing", config)
    assert "HTTP 404" in str(exc.value)


def test_fetch_connection_refused(tmp_path):
    config = make_config(tmp_path, "127.0.0.1")
    with pytest.raises(HailerError) as exc:
        fetch_page("http://127.0.0.1:9/never", config, timeout=2)
    assert "Could not fetch" in str(exc.value)
