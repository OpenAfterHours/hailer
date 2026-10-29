"""Plain-text helpers shared by the agent side and the terminal.

Outside text (model output, tool names and arguments, file and notebook names, URLs) is made safe to
print here, counts are worded and lines shortened in one way. Standard library only, no Rich or
prompt_toolkit: :mod:`hailer.tool_outcomes` (which the agent imports on any thread) uses it as freely
as :mod:`hailer.steps`, :mod:`hailer.transcript`, :mod:`hailer.completion` and :mod:`hailer.chat_ui`.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from urllib.parse import urlsplit

# A whole escape sequence (CSI, OSC up to its terminator, 8-bit CSI), then any other C0 or C1 control
# character but the newline and the tab.
_ESCAPE_SEQUENCE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?|\x9b[0-?]*[ -/]*[@-~]")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def single_line(text: str) -> str:
    """``text`` on one line: control and formatting characters (an escape sequence's ESC, U+202E)
    dropped, every run of whitespace one space. For labels, names and summaries."""
    return " ".join("".join(c for c in text if c.isprintable() or c.isspace()).split())


def printable(text: str) -> str:
    """``text`` without control or formatting characters, each whitespace character (a newline) a
    space, so its spacing stays as it was."""
    return "".join(char if char.isprintable() else " " if char.isspace() else "" for char in text)


def clean_reply(text: str) -> str:
    """Multi-line text (the model's reply) as a terminal may print it: escape sequences and control
    characters removed, newlines kept, tabs as spaces (every 4 columns, which keeps Markdown's
    indents). Rich's ``Text`` and ``markup=False`` keep ESC, so a reply could recolour, move or
    retitle the terminal."""
    text = _ESCAPE_SEQUENCE.sub("", (text or "").replace("\r\n", "\n"))
    return _CONTROL.sub("", text).expandtabs(4)


def plural(count: int, noun: str, plural_noun: str | None = None, *, grouped: bool = False) -> str:
    """``1 cell``, ``2 cells`` (``plural_noun`` for an irregular one); ``grouped`` writes the count
    with thousands separators: ``12,345 rows``."""
    number = f"{count:,}" if grouped else str(count)
    return f"{number} {noun if count == 1 else (plural_noun or noun + 's')}"


def url_host(url: str, *, port: bool = True) -> str:
    """The host of ``url`` (``host:port`` when it names a port and ``port`` is true), never its
    user, path or query; ``""`` for one without a host or that does not parse."""
    try:
        parts = urlsplit(url.strip())
        host, number = parts.hostname or "", parts.port if port else None
    except ValueError:
        return ""
    return f"{host}:{number}" if host and number else host


def fit(text: str, cells: int, ellipsis: str = "", *, width: Callable[[str], int] = len) -> str:
    """``text`` in at most ``cells`` columns as ``width`` measures them (``len``, or terminal cells
    for the composer), shortened at the end; the shortened text loses trailing spaces and ends with
    ``ellipsis`` when one is given."""
    if width(text) <= cells:
        return text
    room, out, used = cells - width(ellipsis), "", 0
    for char in text:
        used += width(char)
        if used > room:
            break
        out += char
    return out.rstrip() + ellipsis if ellipsis else out
