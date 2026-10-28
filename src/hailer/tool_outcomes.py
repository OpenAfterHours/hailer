"""One-line outcomes of tool calls, for the step rows of the chat terminal.

:func:`tool_outcome` reads the text a tool returned to the model and says whether the call worked
and what it found, in a few words: ``"6 cells"``, ``"4 rows"``, ``"6 months, 25-01 to 25-06"``,
``"growth_by_region, checks clean"``. The agent puts both in its ``tool_result`` events
(:class:`~hailer.models.AgentEvent`); the wording of the call itself (verb and target) is the UI's.

Each parser follows a format written elsewhere: the replies of :mod:`hailer.tools`, the check
report of :func:`hailer.code_checks.report`, :func:`hailer.periods.describe_periods`,
:func:`hailer.context.read_skill`, :func:`hailer.web.fetch_page`, the cell listing of
``marimo_client.LIST_CELLS_CODE`` and the lines marimo's code mode prints when it applies cell
changes (``created and ran cell 'Hbol' (growth_by_region)``). The tests feed them output made by
those code paths. A reply in a format a parser does not know gives an empty summary.

A summary is one line of at most :data:`SUMMARY_LIMIT` characters with counts and names only:
never a data value, never a URL beyond its host (a tool's error may quote the URL it was given,
token and all), never control characters. Nothing here raises.

Standard library only (plus :func:`hailer.log.redact`), so the agent can import it on any thread.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple
from urllib.parse import urlsplit

from hailer.log import get_logger, redact

log = get_logger("hailer.tool_outcomes")

#: The longest summary :func:`tool_outcome` returns.
SUMMARY_LIMIT = 80
#: How many names a summary lists before it counts them instead.
MAX_NAMES = 3


class Outcome(NamedTuple):
    """Whether a tool call worked, and what it found in a few words (``""``: nothing worth saying)."""

    ok: bool
    summary: str = ""


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f  ]")
_URL_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9+.\-]*://[^\s'\"<>]+")
#: The last line of a Python traceback: ``ZeroDivisionError: division by zero``,
#: ``polars.exceptions.ColumnNotFoundError: ...``, a bare ``KeyboardInterrupt``.
_EXCEPTION_RE = re.compile(r"^(?:[A-Za-z_]\w*\.)*([A-Z]\w*(?:Error|Exception|Interrupt|Exit|Warning))(?::|$)")
#: hailer.web.TRUNCATED_MARKER, on its own line.
_TRUNCATED_RE = re.compile(r"^\[\.\.\. truncated \d+ chars \.\.\.\]$", re.MULTILINE)


def _host(url: str) -> str:
    """``host`` or ``host:port`` of ``url``: no user, path or query."""
    try:
        parts = urlsplit(url)
        host, port = parts.hostname or "", parts.port
    except ValueError:
        return ""
    return f"{host}:{port}" if host and port else host


def _one_line(text: str) -> str:
    return " ".join(_CONTROL_RE.sub(" ", text).split())


def _clip(text: str, limit: int = SUMMARY_LIMIT) -> str:
    flat = _one_line(text)
    return flat if len(flat) <= limit else flat[: limit - 3].rstrip() + "..."


def _plural(count: int, noun: str, plural: str | None = None) -> str:
    return f"{count:,} {noun if count == 1 else (plural or noun + 's')}"


def _names(names: list[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _url_host(match: re.Match[str]) -> str:
    url = match.group(0)
    kept = len(url.rstrip(".,;:!?)]"))  # "Could not fetch <url>: ...": the colon is the sentence's
    return _host(url[:kept]) + url[kept:]


def _message(line: str) -> str:
    """An error message as a summary: URLs reduced to their host, secrets masked, one line."""
    text = _URL_RE.sub(_url_host, _one_line(line))
    return _clip(redact(text).strip().removesuffix("."))


def _first_line(text: str) -> str:
    return (text.strip().splitlines() or [""])[0]


def _exception_name(text: str) -> str:
    """The exception named by the last traceback line in ``text`` (``""`` when there is none)."""
    found = ""
    for line in text.splitlines():
        match = _EXCEPTION_RE.match(line.strip())
        if match:
            found = match.group(1)
    return found


def _text(result: Any) -> str:
    """A tool message's content as text (LangChain allows a list of content blocks)."""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts = [item if isinstance(item, str) else str(item.get("text", "")) for item in result if isinstance(item, (str, dict))]
        return "\n".join(parts)
    return "" if result is None else str(result)


# --------------------------------------------------------------------------- #
# Shared formats
# --------------------------------------------------------------------------- #

#: A line of ``LIST_CELLS_CODE``: id, name, status, ``errors=N``, first code line (tab-separated; the
#: last tab goes when the reply's trailing whitespace is stripped).
_CELL_LINE_RE = re.compile(r"^[^\t\n]*\t[^\t\n]*\t[^\t\n]*\terrors=(\d+)(?:\t|$)", re.MULTILINE)
#: The summary line of hailer.code_checks.report.
_CHECKS_RE = re.compile(
    r"^Checks \([^)]*\) on (?:the (?P<changed>\d+) changed cells?|the notebook's (?P<all>\d+) cells?): "
    r"(?:(?P<n>\d+) findings?|no findings)\.$",
    re.MULTILINE,
)
_SKIPPED_RE = re.compile(r"^\(Check skipped: (?P<tool>\w+)", re.MULTILINE)


def _cells(text: str, *, matching: bool = False) -> str:
    """``6 cells``, ``6 cells, 1 with errors`` for a cell listing; ``""`` when ``text`` has none."""
    errors = [int(n) for n in _CELL_LINE_RE.findall(text)]
    if not errors:
        return ""
    noun = "matching cell" if matching else "cell"
    count = f"{len(errors)}+ {noun}s" if _TRUNCATED_RE.search(text) else _plural(len(errors), noun)
    failing = sum(1 for n in errors if n)
    return f"{count}, {failing} with errors" if failing else count


def _checks(text: str, *, clean: str = "checks clean") -> str:
    """``checks clean`` / ``2 findings`` (and which checker was skipped) from a check report."""
    report = _CHECKS_RE.search(text)
    skipped = sorted(set(_SKIPPED_RE.findall(text)))
    if report is None:
        return "checks skipped" if skipped else ""
    findings = int(report.group("n") or 0)
    words = _plural(findings, "finding") if findings else clean
    if skipped:
        words += ", " + " and ".join(skipped) + " skipped"
    return words


def _session_state(text: str) -> str:
    """The kernel-session outcome of notebook_create / notebook_open (hailer.tools._session_lines)."""
    if "marimo is not running (or not reachable)" in text:
        return "marimo not running"
    if "It already has a kernel session (" in text:
        return "already open"
    if "Opened it in the browser; kernel session ready" in text:
        return "opened in your browser"
    if "but no kernel session appeared within" in text:
        return "no session yet"
    if "Could not open a browser from here." in text:
        return "browser did not open"
    return ""


def _join(*parts: str) -> str:
    return ", ".join(part for part in parts if part)


# --------------------------------------------------------------------------- #
# marimo_execute
# --------------------------------------------------------------------------- #

#: What marimo's code mode prints after applying changes (AsyncCodeModeContext._print_summary).
_CELL_OP_RE = re.compile(
    r"^(?P<verb>created and ran|created|edited (?:code|config|code and config) of|deleted|moved|re-ran) "
    r"cell '[^']*'(?: \((?P<name>[^()]*)\))?(?P<ran> and ran)?(?P<error> \(error\))?$"
)
_PACKAGE_RE = re.compile(r"^installed (\S+)$")
#: Polars' ``shape: (4, 3)`` (``1_000`` for large counts; ``(4,)`` for a Series), pandas' ``[4 rows x 3 columns]``.
_SHAPE_RE = re.compile(r"^shape: \((?P<rows>[\d_]+),(?P<cols> [\d_]+)?\)$|^\[(?P<prows>\d+) rows x \d+ columns\]$")


def _cell_ops(text: str) -> tuple[list[tuple[str, str]], bool, list[str]]:
    """``[(kind, name)]`` for the cells code mode changed (name ``""`` when unnamed), whether one of
    them raised, and the packages it installed."""
    ops: list[tuple[str, str]] = []
    errored = False
    packages: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        package = _PACKAGE_RE.match(line)
        if package:
            packages.append(package.group(1))
            continue
        match = _CELL_OP_RE.match(line)
        if match is None:
            continue
        verb, name = match.group("verb"), (match.group("name") or "").strip()
        error = bool(match.group("error"))
        # "created and ran cell 'Hbol' (error)" is an unnamed cell that raised, not a cell named "error".
        if not error and name == "error" and verb in ("created and ran", "re-ran"):
            name, error = "", True
        if name == "_":  # marimo's name for an unnamed cell
            name = ""
        kind = verb.split()[0]  # created, edited, deleted, moved, re-ran
        ops.append(("changed" if kind in ("created", "edited") else kind, name))
        errored = errored or error
    return ops, errored, packages


def _cell_words(ops: list[tuple[str, str]]) -> str:
    groups: list[str] = []
    for kind, prefix in (("changed", ""), ("deleted", "deleted "), ("moved", "moved "), ("re-ran", "re-ran ")):
        names = [name for k, name in ops if k == kind]
        if not names or (kind == "re-ran" and groups):
            continue
        if all(names) and len(names) <= MAX_NAMES:
            groups.append(prefix + _names(names))
        else:
            groups.append(prefix + _plural(len(names), "cell"))
    return ", ".join(groups)


def _rows(text: str) -> str:
    rows = ""
    for line in text.splitlines():
        match = _SHAPE_RE.match(line.strip())
        if match is None:
            continue
        if match.group("prows"):
            rows = _plural(int(match.group("prows")), "row")
        else:
            count = int(match.group("rows").replace("_", ""))
            rows = _plural(count, "row") if match.group("cols") else _plural(count, "value")
    return rows


def _marimo_execute(args: Mapping[str, Any], text: str) -> Outcome:
    failed = text.startswith("Execution failed.")
    body, _sep, report = text.partition("\n\nChecks (")
    report = ("Checks (" + report) if _sep else ""
    if not _sep and "\n\n(Check skipped: " in text:
        body, _sep, rest = text.partition("\n\n(Check skipped: ")
        report = "(Check skipped: " + rest
    ops, cell_error, packages = _cell_ops(body)
    cells = _cell_words(ops)
    if not ops:
        changed = _CHECKS_RE.search(report)
        if changed and changed.group("changed"):
            cells = _plural(int(changed.group("changed")), "cell") + " changed"
    installed = ""
    if packages:
        installed = "installed " + (_names(packages) if len(packages) <= MAX_NAMES else _plural(len(packages), "package"))
    if failed or cell_error:
        problem = _exception_name(body) or ("failed" if failed else "cell error")
        return Outcome(False, _clip(_join(cells, installed, problem, _checks(report))))
    return Outcome(True, _clip(_join(cells, installed, _rows(body), _checks(report))))


# --------------------------------------------------------------------------- #
# The other tools
# --------------------------------------------------------------------------- #

_ACTIVE_READY_RE = re.compile(r"^active notebook: (?P<name>.+?) -> session \S+ \(ready\)$", re.MULTILINE)
_ACTIVE_NONE_RE = re.compile(r"^active notebook: (?P<name>.+?) has NO session\.", re.MULTILINE)


def _marimo_status(args: Mapping[str, Any], text: str) -> Outcome:
    if text.startswith("marimo: not running"):
        return Outcome(True, "not running")
    ready = _ACTIVE_READY_RE.search(text)
    if ready:
        return Outcome(True, _clip(f"{ready.group('name')} ready"))
    missing = _ACTIVE_NONE_RE.search(text)
    if missing:
        return Outcome(True, _clip(f"{missing.group('name')} not open"))
    return Outcome(True, "")


def _notebook_cells(args: Mapping[str, Any], text: str) -> Outcome:
    if text.startswith("Could not list cells."):
        return Outcome(False, "could not list cells")
    if text.startswith("No cells match "):
        return Outcome(True, "no matching cells")
    if text.strip() in ("(no output)", "(no cells)"):
        return Outcome(True, "no cells")
    return Outcome(True, _cells(text, matching=bool(str(args.get("pattern") or "").strip())))


def _notebook_check(args: Mapping[str, Any], text: str) -> Outcome:
    if text.startswith("Could not read the notebook's cells."):
        return Outcome(False, "could not read the cells")
    if text.startswith("Code checks are switched off"):
        return Outcome(True, "checks off")
    if text.startswith("The notebook has no code to check."):
        return Outcome(True, "no code")
    return Outcome(True, _checks(text, clean="clean"))


def _notebook_list(args: Mapping[str, Any], text: str) -> Outcome:
    if text.startswith("No notebooks in "):
        return Outcome(True, "no notebooks")
    if not text.startswith("Notebooks in "):
        return Outcome(True, "")
    entries = [line for line in text.splitlines() if line.startswith("  - ")]
    opened = sum(1 for line in entries if line.endswith("[open]"))
    return Outcome(True, _join(_plural(len(entries), "notebook"), f"{opened} open" if opened else ""))


_CREATED_RE = re.compile(r"^Created (?P<name>.+?) from the \w+ template; it is now the active notebook\.$", re.MULTILINE)
_OPENED_RE = re.compile(r"^(?P<name>.+?) is now the active notebook\.$", re.MULTILINE)
_CELLS_HEADER = "Cells (id, name, status, errors, first line):"


def _notebook_create(args: Mapping[str, Any], text: str) -> Outcome:
    created = _CREATED_RE.search(text)
    if created is None:
        return Outcome(True, "")
    return Outcome(True, _clip(_join(created.group("name"), _session_state(text))))


def _notebook_open(args: Mapping[str, Any], text: str) -> Outcome:
    opened = _OPENED_RE.match(text)
    if opened is None:
        return Outcome(True, "")
    cells = ""
    if _CELLS_HEADER in text:
        listing = text.split(_CELLS_HEADER, 1)[1]
        cells = "no cells" if listing.strip() == "(no cells)" else _cells(listing)
    elif "Could not list its cells." in text:
        cells = "cells not listed"
    return Outcome(True, _clip(_join(opened.group("name"), _session_state(text), cells)))


_CLOSED_RE = re.compile(r"^Closed the kernel session \(\S+\) of (?P<name>.+?); its memory is freed")
_NOT_OPEN_RE = re.compile(r"^(?P<name>.+?) is not open \(no kernel session\), so there is nothing to close\.$")


def _notebook_close(args: Mapping[str, Any], text: str) -> Outcome:
    if text.startswith("marimo is not running, so "):
        return Outcome(True, "marimo not running")
    closed = _CLOSED_RE.match(text)
    if closed:
        return Outcome(True, _clip(f"{closed.group('name')}, closed"))
    not_open = _NOT_OPEN_RE.match(text)
    if not_open:
        return Outcome(True, _clip(f"{not_open.group('name')}, was not open"))
    return Outcome(True, "")


_PERIODS_HEAD_RE = re.compile(
    r"^(?P<files>\d+) period file\(s\) \[(?P<stems>[^\]]*)\]: (?P<first>\d{4}-\d{2}) -> (?P<last>\d{4}-\d{2})$", re.MULTILINE
)
_PERIODS_LINE_RE = re.compile(r"^Periods: (?P<labels>.+)$", re.MULTILINE)
_PARTIAL_RE = re.compile(
    r"^Only in some periods: (?P<col>.+) \(\d+ file\(s\), (?P<a>\d{4}-\d{2})(?:\.\.(?P<b>\d{4}-\d{2}))?\)$", re.MULTILINE
)
_OTHER_FILES_RE = re.compile(r"^Other data files \([^)]*\): (?P<names>.+?)(?: \(and (?P<more>\d+) more\))?$", re.MULTILINE)
#: hailer.periods.DATA_SUFFIXES (that module imports Polars and DuckDB); every listed name ends in one,
#: so the names are counted by their suffixes: a name may itself hold ", ".
DATA_SUFFIXES: tuple[str, ...] = (
    ".csv", ".tsv", ".parquet", ".json", ".jsonl", ".ndjson", ".xlsx", ".xls", ".xlsb", ".arrow", ".feather", ".ipc",
)
_LISTED_FILE_RE = re.compile("(?:" + "|".join(re.escape(s) for s in DATA_SUFFIXES) + r")(?=, |$)", re.IGNORECASE)


def _short(label: str) -> str:
    """``2025-03`` → ``25-03`` (the form in the file names)."""
    return label[2:] if len(label) == 7 else label


def _list_periods(args: Mapping[str, Any], text: str) -> Outcome:
    head = _PERIODS_HEAD_RE.search(text)
    if head is None:
        if not text.startswith("No period files found"):
            return Outcome(True, "")
        others = _OTHER_FILES_RE.search(text)
        count = 0
        if others:
            count = len(_LISTED_FILE_RE.findall(others.group("names"))) + int(others.group("more") or 0)
        return Outcome(True, _join("no period files", _plural(count, "other file") if count else ""))
    first, last = head.group("first"), head.group("last")
    stems = [s for s in head.group("stems").split(", ") if s]
    labels = _PERIODS_LINE_RE.search(text)
    if labels:
        months = _plural(len(set(labels.group("labels").split(", "))), "month")
    elif len(stems) == 1:
        months = _plural(int(head.group("files")), "month")
    else:
        months = _plural(int(head.group("files")), "period file")
    span = _short(first) if first == last else f"{_short(first)} to {_short(last)}"
    words = _join(
        stems[0] if len(stems) == 1 and not str(args.get("name") or "").strip() else "",
        _plural(len(stems), "dataset") if len(stems) > 1 else "",
        months,
        span,
    )
    partial = _PARTIAL_RE.findall(text)
    if len(partial) == 1:
        col, a, b = partial[0]
        b = b or a
        if a == b:
            where = f"in {_short(a)}"
        elif b == last and a != first:
            where = f"from {_short(a)}"
        elif a == first and b != last:
            where = f"until {_short(b)}"
        else:
            where = f"in {_short(a)} to {_short(b)}"
        words += f"; {col.strip()} {where}"
    elif partial:
        words += f"; {len(partial)} columns in some months"
    return Outcome(True, _clip(words))


_SKILL_FILES_RE = re.compile(r"^Files \(read with read_skill_file\('(?P<name>[^']*)', <path>\)\):$", re.MULTILINE)


def _load_skill(args: Mapping[str, Any], text: str) -> Outcome:
    listing = _SKILL_FILES_RE.search(text)
    if listing is None:
        return Outcome(True, "")
    files = [line for line in text[listing.end():].splitlines() if line.startswith("- ") and line != "- (no additional files)"]
    return Outcome(True, _clip(_join(listing.group("name"), _plural(len(files), "file") if files else "")))


def _read_skill_file(args: Mapping[str, Any], text: str) -> Outcome:
    path = str(args.get("path") or "").strip().replace("\\", "/")
    lines = len(text.splitlines())
    return Outcome(True, _clip(_join(path, _plural(lines, "line") if lines else "empty")))


_SOURCE_RE = re.compile(r"^Source: (?P<url>\S+)(?: \(first \d+ bytes only\))?$")
_NOT_TEXT_RE = re.compile(r"^The response is '(?P<type>[^']*)' \(\d+ bytes\), which is not text\.", re.MULTILINE)


def _size(chars: int) -> str:
    if chars < 1024:
        return f"{chars} B"
    if chars < 1024 * 1024:
        return f"{chars / 1024:.1f} KB"
    return f"{chars / (1024 * 1024):.1f} MB"


def _fetch_page(args: Mapping[str, Any], text: str) -> Outcome:
    head, _sep, body = text.partition("\n")
    source = _SOURCE_RE.match(head)
    if source is None:
        return Outcome(True, "")
    host = _host(source.group("url"))
    not_text = _NOT_TEXT_RE.search(body)
    if not_text:
        return Outcome(False, _clip(_join(host, f"not text ({not_text.group('type')})")))
    return Outcome(True, _clip(_join(host, _size(len(body.strip().encode("utf-8"))))))


_PARSERS: dict[str, Callable[[Mapping[str, Any], str], Outcome]] = {
    "marimo_execute": _marimo_execute,
    "marimo_status": _marimo_status,
    "notebook_cells": _notebook_cells,
    "notebook_check": _notebook_check,
    "notebook_list": _notebook_list,
    "notebook_create": _notebook_create,
    "notebook_open": _notebook_open,
    "notebook_close": _notebook_close,
    "list_periods": _list_periods,
    "load_skill": _load_skill,
    "read_skill_file": _read_skill_file,
    "fetch_page": _fetch_page,
}


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def _framework_error(text: str) -> str:
    """A summary of the error text LangGraph's ToolNode writes (``status="error"``): never the
    arguments it quotes, which come from the model."""
    if text.startswith("Error invoking tool "):
        return "invalid arguments"
    if text.startswith("Error executing tool "):
        return "failed"
    if text.startswith("Error: ") and " is not a valid tool, try one of [" in text:
        return "unknown tool"
    match = re.match(r"^Error: ([A-Za-z_]\w*)\(", text)  # Error: ValueError('...')
    return match.group(1) if match else "failed"


def _error_reply(text: str) -> str | None:
    """The summary of a Hailer error reply (``ERROR: <message>\\n<hint>``, see tools._error_text and
    tools._run), ``None`` when ``text`` is not one."""
    if not text.startswith("ERROR: "):
        return None
    line = _first_line(text)[len("ERROR: "):]
    unexpected = re.match(r"^unexpected ([A-Za-z_]\w*): ", line)
    if unexpected:  # the exception's message is anything at all; its type is enough
        return f"unexpected {unexpected.group(1)}"
    return _message(line)


def tool_outcome(name: str, args: Mapping[str, Any] | None, result: Any, *, error: bool = False) -> Outcome:
    """``(ok, summary)`` for a call of the tool ``name`` with ``args`` that returned ``result``.

    ``error`` is a ``status="error"`` tool message (LangChain could not run the tool: bad arguments, an
    unknown tool). ``ok`` is False then, for a Hailer error reply (``ERROR: ...``), for code that failed
    in the kernel and for a tool that reports it could not do its job; the summary is ``""`` when the
    reply's format is not one this module knows. Never raises.
    """
    text = _text(result)
    try:
        if error:
            return Outcome(False, _framework_error(text))
        failure = _error_reply(text)
        if failure is not None:
            return Outcome(False, failure)
        parser = _PARSERS.get(name)
        if parser is None:
            return Outcome(True, "")
        ok, summary = parser(args if isinstance(args, Mapping) else {}, text)
        return Outcome(bool(ok), _clip(summary))
    except Exception as exc:  # noqa: BLE001 - a summary is never worth a failed turn
        log.debug("no outcome for %s: %s", name, type(exc).__name__)
        return Outcome(not error and not text.startswith("ERROR: "), "")


__all__ = ["MAX_NAMES", "SUMMARY_LIMIT", "Outcome", "tool_outcome"]
