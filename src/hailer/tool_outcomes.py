"""One-line outcomes of tool calls, for the step rows of the chat terminal.

:func:`tool_outcome` says whether a tool call worked and what it found, in a few words: ``"6 cells"``,
``"4 rows"``, ``"6 months, 25-01 to 25-06"``, ``"growth_by_region, checks clean"``. The agent puts both
in its ``tool_result`` events (:class:`~hailer.models.AgentEvent`); the wording of the call itself (verb
and target) is the UI's.

Whether the call worked is never read from the reply's text: the text holds whatever the user's code
printed or a file contained. It comes from the reply's status, set where the tool knew (an exception,
the kernel's own success flag; see :class:`hailer.tools.ToolReply`), and a failure is worded by its
error class, never by its message (which may quote a URL with a token, a path or data). The summary of
a call that worked is read from the parts of the text Hailer itself writes, in the formats of
:mod:`hailer.tools`, :func:`hailer.code_checks.report`, :func:`hailer.periods.describe_periods`,
:func:`hailer.context.read_skill`, :func:`hailer.web.fetch_page`, ``marimo_client.LIST_CELLS_CODE`` and
the lines marimo's code mode prints when it applies cell changes; the tests feed them output made by
those code paths. A reply in a format a parser does not know gives an empty summary.

A summary is one line of at most :data:`SUMMARY_LIMIT` printable characters with counts and names
only, and it does not repeat what the call's arguments already say (the notebook or skill the UI
shows as the step's target). Nothing here raises.

Standard library only, so the agent can import it on any thread.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, NamedTuple

from hailer.log import get_logger
from hailer.text import fit, plural, single_line, url_host

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
# Failures: worded by their class
# --------------------------------------------------------------------------- #

#: A failed call's words by ``(tool, error)``, then by ``error`` (the status's error: a class name or
#: a tool's failure code), then by tool for a plain ``HailerError``.
_TOOL_ERROR_WORDS: dict[tuple[str, str], str] = {
    ("notebook_create", "NotebookPathError"): "not a usable notebook name",
}
_ERROR_WORDS: dict[str, str] = {
    "NoSessionError": "notebook not open",
    "MarimoUnavailableError": "marimo not running",
    "MarimoExecutionError": "kernel request failed",
    "KernelRuntimeError": "kernel not available",
    "NotebookNotFoundError": "notebook not found",
    "NotebookExistsError": "notebook already exists",
    "NotebookPathError": "not in the notebooks folder",
    "WebAccessDenied": "host not allowed",
    "MalformedParquetError": "unreadable data file",
    "ConfigError": "configuration problem",
    "cells_unreadable": "could not read the cells",
    "cell_error": "cell error",
    "unknown_template": "unknown template",
    "execution_failed": "failed",
}
_HAILER_ERROR_WORDS: dict[str, str] = {
    "fetch_page": "could not fetch the page",
    "load_skill": "skill not found",
    "read_skill_file": "could not read the file",
    "list_periods": "could not read the data folder",
}
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_]\w*$")


def _failure(name: str, status: Mapping[str, Any]) -> str:
    error = status.get("error")
    error = error if isinstance(error, str) else ""
    if status.get("unexpected"):
        return f"unexpected {error}" if _IDENTIFIER_RE.match(error) else "unexpected error"
    return _TOOL_ERROR_WORDS.get((name, error)) or _ERROR_WORDS.get(error) or _HAILER_ERROR_WORDS.get(name, "failed")


def _framework_error(text: str) -> str:
    """A ``status="error"`` tool message, which LangGraph's ToolNode writes when it could not run the
    tool: worded by its template, never by the arguments it quotes (they come from the model)."""
    if text.startswith("Error invoking tool "):
        return "invalid arguments"
    if text.startswith("Error executing tool "):
        return "failed"
    if text.startswith("Error: ") and " is not a valid tool, try one of [" in text:
        return "unknown tool"
    match = re.match(r"^Error: ([A-Za-z_]\w*)\(", text)  # Error: ValueError('...')
    return match.group(1) if match else "failed"


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #

#: The last line of a Python traceback: ``ZeroDivisionError: division by zero``,
#: ``polars.exceptions.ColumnNotFoundError: ...``, a bare ``KeyboardInterrupt``.
_EXCEPTION_RE = re.compile(r"^(?:[A-Za-z_]\w*\.)*([A-Z]\w*(?:Error|Exception|Interrupt|Exit|Warning))(?::|$)")


def _clip(text: str, limit: int = SUMMARY_LIMIT) -> str:
    return fit(single_line(text), limit, "...")


#: Counts in summaries group thousands: "12,345 rows".
_plural = partial(plural, grouped=True)


def _names(names: list[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _join(*parts: str) -> str:
    return ", ".join(part for part in parts if part)


def _same_notebook(given: Any, name: str) -> bool:
    """Whether the notebook reference in the call's arguments already names ``name`` (``other`` and
    ``other.py``, or a path ending in it)."""
    ref = str(given or "").strip().replace("\\", "/").lower().removesuffix(".py")
    plain = name.strip().lower().removesuffix(".py")
    return bool(ref) and (ref == plain or ref.endswith("/" + plain))


def _text(result: Any) -> str:
    """A tool message's content as text (LangChain allows a list of content blocks)."""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts = [item if isinstance(item, str) else str(item.get("text", "")) for item in result if isinstance(item, (str, dict))]
        return "\n".join(parts)
    return "" if result is None else str(result)


def _strings(value: Any) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, (list, tuple)) else []


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


# --------------------------------------------------------------------------- #
# Shared formats
# --------------------------------------------------------------------------- #

#: A line of ``LIST_CELLS_CODE``: id, name, status, ``errors=N``, first code line (tab-separated; the
#: last tab goes when the reply's trailing whitespace is stripped).
_CELL_LINE_RE = re.compile(r"^[^\t\n]*\t[^\t\n]*\t[^\t\n]*\terrors=(\d+)(?:\t|$)", re.MULTILINE)


def _cells(text: str, *, matching: bool = False, truncated: bool = False) -> str:
    """``6 cells``, ``6 cells, 1 with errors`` for a cell listing; ``""`` when ``text`` has none."""
    errors = [int(n) for n in _CELL_LINE_RE.findall(text)]
    if not errors:
        return ""
    noun = "matching cell" if matching else "cell"
    count = f"{len(errors):,}+ {noun}s" if truncated else _plural(len(errors), noun)
    failing = sum(1 for n in errors if n)
    return f"{count}, {failing} with errors" if failing else count


def _checks(status: Mapping[str, Any], *, clean: str = "checks clean") -> str:
    """``checks clean`` / ``2 findings`` (and which checker was skipped) from a reply's check facts."""
    checked, skipped, findings = _strings(status.get("checked")), _strings(status.get("skipped")), _count(status.get("findings"))
    if not checked:
        return "checks skipped" if skipped else ""
    words = _plural(findings, "finding") if findings else clean
    return words + (", " + " and ".join(skipped) + " skipped" if skipped else "")


def _session_state(text: str) -> str:
    """The kernel-session outcome of notebook_create / notebook_open (hailer.tools._session_lines)."""
    if "\nmarimo is not running (or not reachable)" in text:
        return "marimo not running"
    if "\nIt already has a kernel session (" in text:
        return "already open"
    if "\nOpened it in the browser; kernel session ready" in text:
        return "opened in your browser"
    if "but no kernel session appeared within" in text:
        return "no session yet"
    if "\nCould not open a browser from here." in text:
        return "browser did not open"
    return ""


# --------------------------------------------------------------------------- #
# marimo_execute
# --------------------------------------------------------------------------- #

#: What marimo's code mode prints after applying changes (AsyncCodeModeContext._print_summary); a
#: cell's name is a Python identifier, ``_`` for an unnamed cell.
_CELL_OP_RE = re.compile(
    r"^(?P<verb>created and ran|created|edited (?:code|config|code and config) of|deleted|moved|re-ran) "
    r"cell '[^']*'(?: \((?P<name>[A-Za-z_]\w*)\))?(?P<ran> and ran)?(?P<error> \(error\))?$"
)
_PACKAGE_RE = re.compile(r"^installed ([A-Za-z][A-Za-z0-9._\-\[\],]*)$")
#: Polars' ``shape: (4, 3)`` (``1_000`` for large counts; ``(4,)`` for a Series), pandas' ``[4 rows x 3 columns]``.
_SHAPE_RE = re.compile(r"^shape: \((?P<rows>[\d_]+),(?P<cols> [\d_]+)?\)$|^\[(?P<prows>\d+) rows x \d+ columns\]$")


def _cell_ops(text: str) -> tuple[list[tuple[str, str]], bool, list[str]]:
    """``[(kind, name)]`` for the cells code mode reports it changed (name ``""`` when unnamed), whether
    one it created or edited raised, and the packages it installed."""
    ops: list[tuple[str, str]] = []
    errored = False
    packages: list[str] = []
    for line in text.splitlines():
        package = _PACKAGE_RE.match(line)
        if package:
            packages.append(package.group(1))
            continue
        match = _CELL_OP_RE.match(line)
        if match is None:
            continue
        verb, name = match.group("verb"), match.group("name") or ""
        error = bool(match.group("error"))
        # "created and ran cell 'Hbol' (error)" is an unnamed cell that raised, not a cell named "error".
        if not error and name == "error" and verb in ("created and ran", "re-ran"):
            name, error = "", True
        kind = verb.split()[0]  # created, edited, deleted, moved, re-ran
        ops.append(("changed" if kind in ("created", "edited") else kind, "" if name == "_" else name))
        errored = errored or (error and kind in ("created", "edited"))
    return ops, errored, packages


def _cell_words(changed: list[str] | None, ops: list[tuple[str, str]]) -> str:
    groups: list[str] = []
    for kind, prefix in (("changed", ""), ("deleted", "deleted "), ("moved", "moved "), ("re-ran", "re-ran ")):
        names = changed if kind == "changed" and changed is not None else [name for k, name in ops if k == kind]
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
        match = _SHAPE_RE.match(line)
        if match is None:
            continue
        if match.group("prows"):
            rows = _plural(int(match.group("prows")), "row")
        else:
            count = int(match.group("rows").replace("_", ""))
            rows = _plural(count, "row") if match.group("cols") else _plural(count, "value")
    return rows


def _marimo_execute(args: Mapping[str, Any], text: str, status: Mapping[str, Any]) -> Outcome:
    """The cells the code changed, then what went wrong or the rows it printed, then the checks.

    A cell the code created or edited that raised fails the call: the kernel says so in the status
    (``cell_error``, with the exception classes). Only when there were no cell snapshots (``[checks]``
    all off) does marimo's own ``... (error)`` line say it, read from code that used code mode."""
    body = text
    if "checked" in status:  # the reply ends with the check report: what the code printed comes before it
        header = "\n\nChecks (" if _strings(status.get("checked")) else "\n\n(Check skipped: "
        head, sep, _report = text.rpartition(header)
        body = head if sep else text
    snapshots = "cells" in status  # the cells were compared before and after the code ran
    changed = _strings(status.get("cells")) if snapshots else None
    # Code mode's lines are read only from code that used it: other code may print anything.
    ops, line_error, packages = _cell_ops(body) if "_code_mode" in str(args.get("code") or "") else ([], False, [])
    installed = ""
    if packages:
        installed = "installed " + (_names(packages) if len(packages) <= MAX_NAMES else _plural(len(packages), "package"))
    cells = _join(_cell_words(changed, ops), installed)
    error = status.get("error")
    if status.get("ok", True) is False and error == "cell_error":
        raised = [name for name in _strings(status.get("raised")) if _IDENTIFIER_RE.match(name)][:MAX_NAMES]
        return Outcome(False, _join(cells, _names(raised) if raised else "cell error", _checks(status)))
    if status.get("ok", True) is False:
        return Outcome(False, _join(cells, _exception_name(body) or "failed", _checks(status)))
    if line_error and not snapshots:
        return Outcome(False, _join(cells, _exception_name(body) or "cell error", _checks(status)))
    return Outcome(True, _join(cells, _rows(body), _checks(status)))


def _exception_name(text: str) -> str:
    """The exception named by the last traceback line in ``text`` (``""`` when there is none)."""
    found = ""
    for line in text.splitlines():
        match = _EXCEPTION_RE.match(line.strip())
        if match:
            found = match.group(1)
    return found


# --------------------------------------------------------------------------- #
# The other tools
# --------------------------------------------------------------------------- #

_ACTIVE_READY_RE = re.compile(r"^active notebook: (?P<name>.+?) -> session \S+ \(ready\)$", re.MULTILINE)
_ACTIVE_NONE_RE = re.compile(r"^active notebook: (?P<name>.+?) has NO session\.", re.MULTILINE)


def _marimo_status(args: Mapping[str, Any], text: str, status: Mapping[str, Any]) -> str:
    if text.startswith("marimo: not running\n"):
        return "not running"
    if not text.startswith("marimo: running at "):
        return ""
    ready = _ACTIVE_READY_RE.search(text)
    if ready:
        return f"{ready.group('name')} ready"
    missing = _ACTIVE_NONE_RE.search(text)
    return f"{missing.group('name')} not open" if missing else ""


def _notebook_cells(args: Mapping[str, Any], text: str, status: Mapping[str, Any]) -> str:
    pattern = str(args.get("pattern") or "").strip()
    if pattern and text == f"No cells match '{pattern}'.":
        return "no matching cells"
    if text.strip() in ("(no output)", "(no cells)"):
        return "no cells"
    return _cells(text, matching=bool(pattern), truncated=bool(status.get("truncated")))


def _notebook_check(args: Mapping[str, Any], text: str, status: Mapping[str, Any]) -> str:
    if text.startswith("Code checks are switched off ("):
        return "checks off"
    if text == "The notebook has no code to check.":
        return "no code"
    return _checks(status, clean="clean")


def _notebook_list(args: Mapping[str, Any], text: str, status: Mapping[str, Any]) -> str:
    if text.startswith("No notebooks in "):
        return "no notebooks"
    if not text.startswith("Notebooks in "):
        return ""
    entries = [line for line in text.splitlines() if line.startswith("  - ")]
    opened = sum(1 for line in entries if line.endswith("[open]"))
    return _join(_plural(len(entries), "notebook"), f"{opened} open" if opened else "")


_CREATED_RE = re.compile(r"^Created (?P<name>.+?) from the \w+ template; it is now the active notebook\.$", re.MULTILINE)
_OPENED_RE = re.compile(r"^(?P<name>.+?) is now the active notebook\.$", re.MULTILINE)
_CELLS_HEADER = "\nCells (id, name, status, errors, first line):\n"


def _notebook_create(args: Mapping[str, Any], text: str, status: Mapping[str, Any]) -> str:
    created = _CREATED_RE.match(text)
    if created is None:
        return ""
    name = created.group("name")
    return _join("" if _same_notebook(args.get("name"), name) else name, _session_state(text))


def _notebook_open(args: Mapping[str, Any], text: str, status: Mapping[str, Any]) -> str:
    opened = _OPENED_RE.match(text)
    if opened is None:
        return ""
    head, sep, listing = text.partition(_CELLS_HEADER)  # the listing holds the cells' own code: after the header
    cells = ""
    if sep:
        cells = "no cells" if listing.strip() == "(no cells)" else _cells(listing, truncated=bool(status.get("truncated")))
    elif "\nCould not list its cells.\n" in text:
        cells = "cells not listed"
    name = opened.group("name")
    return _join("" if _same_notebook(args.get("notebook"), name) else name, _session_state(head), cells)


_CLOSED_RE = re.compile(r"^Closed the kernel session \(\S+\) of (?P<name>.+?); its memory is freed")
_NOT_OPEN_RE = re.compile(r"^(?P<name>.+?) is not open \(no kernel session\), so there is nothing to close\.$")


def _notebook_close(args: Mapping[str, Any], text: str, status: Mapping[str, Any]) -> str:
    if text.startswith("marimo is not running, so "):
        return "marimo not running"
    for pattern, state in ((_CLOSED_RE, "closed"), (_NOT_OPEN_RE, "was not open")):
        match = pattern.match(text)
        if match:
            name = match.group("name")
            return _join("" if _same_notebook(args.get("notebook"), name) else name, state)
    return ""


_PERIODS_HEAD_RE = re.compile(
    r"^(?P<files>\d+) period file\(s\) \[(?P<stems>[^\]]*)\]: (?P<first>\d{4}-\d{2}) -> (?P<last>\d{4}-\d{2})$", re.MULTILINE
)
_PERIODS_LINE_RE = re.compile(r"^Periods: (?P<labels>.+)$", re.MULTILINE)
_PARTIAL_RE = re.compile(
    r"^Only in some periods: (?P<col>.+) \(\d+ file\(s\), (?P<a>\d{4}-\d{2})(?:\.\.(?P<b>\d{4}-\d{2}))?\)$", re.MULTILINE
)
_OTHER_FILES_RE = re.compile(r"\nOther data files \([^)]*\): (?P<names>.+?)(?: \(and (?P<more>\d+) more\))?$")
#: hailer.periods.DATA_SUFFIXES (that module imports Polars and DuckDB); every listed name ends in one,
#: so the names are counted by their suffixes: a name may itself hold ", ".
DATA_SUFFIXES: tuple[str, ...] = (
    ".csv", ".tsv", ".parquet", ".json", ".jsonl", ".ndjson", ".xlsx", ".xls", ".xlsb", ".arrow", ".feather", ".ipc",
)
_LISTED_FILE_RE = re.compile("(?:" + "|".join(re.escape(s) for s in DATA_SUFFIXES) + r")(?=, |$)", re.IGNORECASE)


def _short(label: str) -> str:
    """``2025-03`` → ``25-03`` (the form in the file names)."""
    return label[2:] if len(label) == 7 else label


def _list_periods(args: Mapping[str, Any], text: str, status: Mapping[str, Any]) -> str:
    head = _PERIODS_HEAD_RE.match(text)
    if head is None:
        if not text.startswith("No period files found"):
            return ""
        others = _OTHER_FILES_RE.search(text)
        count = 0
        if others:
            count = len(_LISTED_FILE_RE.findall(others.group("names"))) + int(others.group("more") or 0)
        return _join("no period files", _plural(count, "other file") if count else "")
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
    return words


#: The footer read_skill puts after SKILL.md: the last one is Hailer's (the body may quote the same text).
_SKILL_FOOTER = "\n\nFiles (read with read_skill_file('"
_SKILL_FILES_RE = re.compile(r"^(?P<name>[^'\n]*)', <path>\)\):\n(?P<files>(?:- [^\n]*\n?)+)$")


def _load_skill(args: Mapping[str, Any], text: str, status: Mapping[str, Any]) -> str:
    _body, sep, footer = text.rpartition(_SKILL_FOOTER)
    listing = _SKILL_FILES_RE.match(footer) if sep else None
    if listing is None:
        return ""
    files = [line for line in listing.group("files").splitlines() if line != "- (no additional files)"]
    name = listing.group("name")
    given = str(args.get("name") or "").strip()
    return _join("" if given.lower() == name.lower() else name, _plural(len(files), "file") if files else "no files")


def _read_skill_file(args: Mapping[str, Any], text: str, status: Mapping[str, Any]) -> str:
    if status.get("truncated"):
        return "truncated"
    lines = len(text.splitlines())
    return _plural(lines, "line") if lines else "empty"


_SOURCE_RE = re.compile(r"^Source: (?P<url>\S+)(?P<cut> \(first \d+ bytes only\))?$")
_NOT_TEXT_RE = re.compile(r"The response is '(?P<type>[\w.+\-]+/[\w.+\-]+)' \(\d+ bytes\), which is not text\. Only HTML and text pages can be read\.")


def _size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _fetch_page(args: Mapping[str, Any], text: str, status: Mapping[str, Any]) -> str:
    head, _sep, rest = text.partition("\n")
    source = _SOURCE_RE.match(head)
    if source is None:
        return ""
    host = url_host(source.group("url"))
    elsewhere = f"redirected to {host}" if host and host != url_host(str(args.get("url") or "")) else ""
    not_text = _NOT_TEXT_RE.fullmatch(rest)  # the whole reply, never a sentence inside a page
    if not_text:
        return _join(f"not text ({not_text.group('type')})", elsewhere)
    cut = "truncated" if source.group("cut") or status.get("truncated") else ""
    return _join(_size(len(rest.strip().encode("utf-8"))), cut, elsewhere)


_PARSERS: dict[str, Callable[[Mapping[str, Any], str, Mapping[str, Any]], str]] = {
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


def tool_outcome(name: str, args: Mapping[str, Any] | None, content: Any, status: Any = None, *, error: bool = False) -> Outcome:
    """``(ok, summary)`` for a call of the tool ``name`` with ``args`` that replied ``content`` with
    ``status`` (a :class:`hailer.tools.ToolReply` status: the tool message's artifact).

    ``error`` is a ``status="error"`` tool message (LangChain could not run the tool: bad arguments, an
    unknown tool). Without a status (not one of Hailer's tools) the call counts as working and nothing
    is said about it. Never raises.
    """
    try:
        if error:
            return Outcome(False, _framework_error(_text(content)))
        if not isinstance(status, Mapping):
            return Outcome(True, "")
        ok = status.get("ok", True) is not False
        given = args if isinstance(args, Mapping) else {}
        if name == "marimo_execute" and status.get("error") in (None, "execution_failed", "cell_error"):
            ok, summary = _marimo_execute(given, _text(content), status)  # failed code is named by its exception
            return Outcome(ok, _clip(summary))
        if not ok:  # worded by its class, never its message
            return Outcome(False, _clip(_failure(name, status)))
        parser = _PARSERS.get(name)
        return Outcome(True, _clip(parser(given, _text(content), status)) if parser is not None else "")
    except Exception as exc:  # noqa: BLE001 - a summary is never worth a failed turn
        log.debug("no outcome for %s: %s", name, type(exc).__name__)
        return Outcome(not error and not (isinstance(status, Mapping) and status.get("ok") is False), "")


__all__ = ["DATA_SUFFIXES", "MAX_NAMES", "SUMMARY_LIMIT", "Outcome", "tool_outcome"]
