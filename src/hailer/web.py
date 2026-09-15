"""Allow-listed web access for context gathering.

The agent may only read pages whose host matches ``[web].allowed_domains`` in
``hailer.toml``. With an empty list nothing can be fetched. Matching rules mirror
Codex's network proxy so both enforcement layers agree:

- ``example.com``      exact host only
- ``*.example.com``    subdomains only (not the apex)
- ``**.example.com``   apex plus all subdomains
- ``*``                any public host (never private/loopback addresses)

Private and loopback addresses are fetched only when listed literally, e.g.
``"127.0.0.1"`` or ``"localhost"``.

Everything here is standard library: ``urllib`` for HTTP and ``html.parser`` for
HTML-to-text conversion.
"""

from __future__ import annotations

import http.client
import ipaddress
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from html.parser import HTMLParser

from .errors import HailerError, WebAccessDenied
from .models import HailerConfig

USER_AGENT = "Hailer/0.1 (+https://github.com/openafterhours/hailer)"
TRUNCATED_MARKER = "[... truncated {n} chars ...]"

_BLOCK_TAGS = {
    "p", "div", "section", "article", "header", "footer", "main", "aside",
    "ul", "ol", "table", "thead", "tbody", "tr", "blockquote", "pre",
    "figure", "figcaption", "form", "fieldset", "details", "summary", "hr",
}
_SKIP_TAGS = {"script", "style", "noscript", "nav", "template", "svg", "iframe"}
_HEADING_RE = re.compile(r"^h([1-6])$")
_PRE_MARK = "\x00"  # internal marker for preformatted lines; never reaches callers


# --------------------------------------------------------------------------- #
# Host matching
# --------------------------------------------------------------------------- #


def _normalise_host(host: str | None) -> str:
    """Lower-case, strip port and brackets and a trailing dot."""
    if not host:
        return ""
    h = host.strip().lower()
    if h.startswith("["):  # [ipv6]:port
        end = h.find("]")
        h = h[1:end] if end != -1 else h[1:]
    elif h.count(":") == 1:  # host:port (not a bare IPv6 literal)
        h = h.split(":", 1)[0]
    return h.rstrip(".")


def _normalise_pattern(pattern: str) -> str:
    return pattern.strip().lower().rstrip(".")


def is_private_host(host: str) -> bool:
    """True for loopback, private, link-local, unspecified addresses and ``localhost``."""
    h = _normalise_host(host)
    if not h:
        return True
    if h == "localhost" or h.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        return False
    return bool(ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_unspecified or ip.is_reserved)


def host_allowed(host: str, allowed: Sequence[str]) -> bool:
    """Return True if ``host`` matches one of the allow-list patterns."""
    h = _normalise_host(host)
    if not h:
        return False
    private = is_private_host(h)
    for raw in allowed:
        p = _normalise_pattern(raw)
        if not p:
            continue
        if p == "*":
            if not private:
                return True
            continue
        if p.startswith("**."):
            base = p[3:]
            if base and (h == base or h.endswith("." + base)):
                return True
        elif p.startswith("*."):
            base = p[2:]
            if base and h != base and h.endswith("." + base):
                return True
        elif h == p:
            return True
    return False


def _deny(host: str, config: HailerConfig, reason: str | None = None) -> WebAccessDenied:
    allowed = list(config.web.allowed_domains)
    msg = reason or f"Web access to '{host}' is not allowed"
    if allowed:
        hint = "Allowed domains: " + ", ".join(allowed) + "\nAdd the domain to [web].allowed_domains in hailer.toml to permit it."
    else:
        hint = (
            "No websites are allowed. Add a [web] section to hailer.toml, e.g.\n"
            '    [web]\n    allowed_domains = ["docs.pola.rs", "**.bankofengland.co.uk"]'
        )
    return WebAccessDenied(msg, hint)


def _check_url(url: str, config: HailerConfig) -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme.lower() not in ("http", "https"):
        raise _deny(parsed.hostname or url, config, f"Only http and https URLs can be fetched (got '{url}')")
    host = _normalise_host(parsed.hostname)
    if not host_allowed(host, config.web.allowed_domains):
        raise _deny(host or url, config)
    return host


# --------------------------------------------------------------------------- #
# HTML to text
# --------------------------------------------------------------------------- #


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0
        self._pre_depth = 0
        self._list_depth = 0
        self._title: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
            return
        if tag == "br":
            self._parts.append("\n")
        elif tag == "li":
            self._parts.append("\n" + "  " * max(self._list_depth - 1, 0) + "- ")
        elif tag in ("ul", "ol"):
            self._list_depth += 1
            self._parts.append("\n")
        elif tag == "pre":
            self._pre_depth += 1
            self._parts.append("\n")
        elif tag in ("td", "th"):
            self._parts.append(" | ")
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")
        else:
            m = _HEADING_RE.match(tag)
            if m:
                self._parts.append("\n\n" + "#" * int(m.group(1)) + " ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth = max(self._skip_depth - 1, 0)
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
        elif tag in ("ul", "ol"):
            self._list_depth = max(self._list_depth - 1, 0)
            self._parts.append("\n")
        elif tag == "pre":
            self._pre_depth = max(self._pre_depth - 1, 0)
            self._parts.append("\n")
        elif tag in _BLOCK_TAGS or _HEADING_RE.match(tag):
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self._title.append(data)
            return
        if self._pre_depth:
            # Mark preformatted lines so text() keeps their leading whitespace.
            self._parts.append(_PRE_MARK + data.replace("\n", "\n" + _PRE_MARK))
        else:
            self._parts.append(re.sub(r"\s+", " ", data))

    def text(self) -> str:
        body = "".join(self._parts)
        lines = [ln.rstrip() for ln in body.splitlines()]
        out: list[str] = []
        blank = 0
        for ln in lines:
            if _PRE_MARK in ln:
                ln = ln.replace(_PRE_MARK, "")
                if not ln.strip():
                    blank += 1
                    if blank <= 1 and out:
                        out.append("")
                    continue
                blank = 0
                out.append(ln)
                continue
            stripped = ln.strip()
            if not stripped:
                blank += 1
                if blank <= 1 and out:
                    out.append("")
                continue
            blank = 0
            out.append(stripped if not ln.startswith("  -") else ln)
        text = "\n".join(out).strip()
        title = re.sub(r"\s+", " ", "".join(self._title)).strip()
        if title:
            text = f"# {title}\n\n{text}" if text else f"# {title}"
        return text


def html_to_text(html: str) -> str:
    """Convert HTML to readable plain text (headings, paragraphs, lists, link text)."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return parser.text()


# --------------------------------------------------------------------------- #
# Truncation (shared with the MCP server)
# --------------------------------------------------------------------------- #


def truncate_text(text: str, max_chars: int, *, head_ratio: float = 0.7) -> str:
    """Keep the head and tail of ``text`` so it fits ``max_chars``, with a marker in between."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker_len = len(TRUNCATED_MARKER.format(n=len(text))) + 2
    budget = max(max_chars - marker_len, 0)
    head = int(budget * head_ratio)
    tail = budget - head
    removed = len(text) - head - tail
    marker = TRUNCATED_MARKER.format(n=removed)
    tail_text = text[-tail:] if tail > 0 else ""
    return f"{text[:head]}\n{marker}\n{tail_text}".rstrip("\n") if head or tail else marker


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #


class _CheckedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-checks the allow-list on every redirect hop."""

    def __init__(self, config: HailerConfig) -> None:
        super().__init__()
        self._config = config

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        _check_url(newurl, self._config)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_page(url: str, config: HailerConfig, *, timeout: float = 20.0) -> str:
    """Fetch an allow-listed page and return readable text for the model.

    Raises ``WebAccessDenied`` before any network I/O when the host is not allowed
    (including on redirects) and ``HailerError`` for transport failures.
    """
    _check_url(url, config)
    max_bytes = max(int(config.web.max_page_bytes), 1)
    opener = urllib.request.build_opener(_CheckedRedirectHandler(config))
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
        },
    )
    try:
        with opener.open(request, timeout=timeout) as resp:
            final_url = resp.geturl()
            content_type = resp.headers.get_content_type()
            charset = resp.headers.get_content_charset() or "utf-8"
            raw = resp.read(max_bytes + 1)
    except WebAccessDenied:
        raise
    except urllib.error.HTTPError as exc:
        raise HailerError(
            f"Could not fetch {url}: HTTP {exc.code} {exc.reason}",
            "Check the URL. If the page needs a login it cannot be read by Hailer.",
        ) from exc
    except urllib.error.URLError as exc:
        raise HailerError(
            f"Could not fetch {url}: {exc.reason}",
            "Check the URL, DNS and any VPN or proxy required to reach that host.",
        ) from exc
    except (TimeoutError, socket.timeout, http.client.HTTPException, OSError) as exc:
        raise HailerError(f"Could not fetch {url}: {exc}", "The site did not answer in time; try again later.") from exc

    truncated_bytes = len(raw) > max_bytes
    raw = raw[:max_bytes]
    try:
        body = raw.decode(charset, errors="replace")
    except LookupError:
        body = raw.decode("utf-8", errors="replace")

    if content_type in ("text/html", "application/xhtml+xml"):
        text = html_to_text(body)
    elif content_type.startswith("text/") or content_type in ("application/json", "application/xml"):
        text = body.strip()
    else:
        return (
            f"Source: {final_url}\n"
            f"The response is '{content_type}' ({len(raw)} bytes), which is not text. "
            "Only HTML and text pages can be read."
        )

    header = f"Source: {final_url}"
    if truncated_bytes:
        header += f" (first {max_bytes} bytes only)"
    return truncate_text(f"{header}\n\n{text}", config.max_tool_output_chars)
