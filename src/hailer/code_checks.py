"""Checks for the notebook cells the agent writes: ruff (lint and format), ty (types) and Hailer's
own check for values typed into the text a reader sees.

The agent writes cells through marimo's code mode inside ``marimo_execute``. Around such a call
:class:`~hailer.tools.HailerTools` takes a :func:`snapshot_code` of the notebook's cells before and
after, and :func:`check` lints and type-checks the notebook with the cells that changed singled out;
:func:`report` is the text appended to the tool result, so the model can fix what it just wrote.

- **The notebook as one script.** The cells are joined in dependency order (marimo's topological
  order, the order ``marimo export script`` uses), so a name another cell defines is defined above
  its use and ty carries each variable's type from the cell that defines it (a Polars
  ``DataFrame`` stays one in the cells that use it). The order is worked out here, with marimo's
  compiler, from every cell's code: the kernel's own graph only holds the cells that have run. Findings are mapped back to a cell and a line
  in it; only those in the cells of interest are reported. A cell Python cannot parse is reported
  as a syntax error and left out of the script: an unclosed bracket would swallow the cells after
  it, and ruff lints nothing in a file that does not parse.
- **Rules that fit notebooks.** ruff runs isolated from any project configuration with a small,
  bug-oriented rule set (:data:`RUFF_SELECT`), minus rules that misfire on cells
  (:data:`RUFF_IGNORE`): an import may serve later cells, top-level ``await`` is allowed in a cell,
  a cell's last expression is its output. ty skips ``unresolved-import``: the packages the kernel
  has are not necessarily the ones Hailer runs with (a docker kernel has altair and plotly), and a
  missing package fails the cell anyway when it runs.
- **Formatting happens in the kernel.** marimo's code mode formats new and changed cells with
  ruff before applying them when the kernel's ``save.format_on_save`` setting is on. The snapshot
  switches that setting on in the kernel's memory (never in a configuration file), so a cell is
  formatted before it runs: formatting afterwards would change its code and make it run again.
- **Text follows the data.** People re-run a notebook on next month's data, so a number, date or
  month typed into ``mo.md`` text or a chart ``title=`` would go on describing the old data.
  :func:`hardcoded_values` reports them (rule ``hardcoded-value``); the values belong in variables
  that an f-string formats. It reads each cell's syntax tree here, with no tool to run.

Both tools come with Hailer (``ruff`` and ``ty`` are dependencies) and run as subprocesses on this
machine, whatever the kernel runtime. Nothing here raises into the agent loop: a tool that is
missing, fails or times out becomes a line in the report.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("hailer.code_checks")

#: Prefixes the JSON line :func:`snapshot_code` prints.
SNAPSHOT_MARKER = "HAILER_CELLS:"
#: ruff rules: pyflakes, pylint's errors and the bugbear rules that catch real mistakes in cells
#: (a mutable default argument, a closure over a loop variable) rather than style.
RUFF_SELECT: tuple[str, ...] = ("F", "PLE", "B006", "B023", "E722")
#: Rules that misfire on notebook cells: an import may be for a cell not written yet (F401), each
#: cell may define its own ``_private`` helper (F811), top-level ``await`` is allowed (F704,
#: PLE1142) and an f-string without placeholders is not worth running a cell again (F541).
RUFF_IGNORE: tuple[str, ...] = ("F401", "F541", "F704", "F811", "PLE1142")
#: ty rules left out (see the module docstring).
TY_IGNORE: tuple[str, ...] = ("unresolved-import",)
#: ty reports top-level ``await``/``async for``/``async with`` as syntax errors; a cell allows them.
_TOP_LEVEL_ASYNC = "outside of an asynchronous function"
#: ruff findings ty reports too, by the ty rule that reports the same thing.
_SAME_AS_TY = {"F821": "unresolved-reference", "invalid-syntax": "invalid-syntax"}
#: How long one ruff or ty run may take.
TIMEOUT_SEC = 30.0
#: The most findings :func:`report` lists.
MAX_REPORTED = 20
_SCRIPT_NAME = "hailer_notebook.py"
#: A value typed into text: 5, -3, 1,234, 5.6%, £41.1k, 2025, 2025-03, 25-03. Not a digit inside a
#: word or name (Q2, x2, COVID-19, 12px, 3rd), a version (v1.2), the second half of 1/2, or an
#: HTML entity or colour (&#163;, #1f77b4).
_TYPED_VALUE_RE = re.compile(r"(?<![\w.#/-])[-+]?[£$€]?\d[\d,]*(?:\.\d+)?(?:-\d{2}){0,2}(?:%|k|m|bn|pp)?(?!\w)")
#: Month names are a period typed into text too ("May" is left out: it is also a word).
_MONTH_RE = re.compile(r"\b(?:January|February|March|April|June|July|August|September|October|November|December)\b")
#: A numbered list item or heading ("1. Load", "## 2) Compare"): its number is layout, not data.
_NUMBERED_RE = re.compile(r"^\s*(?:#{1,6}\s+)?\d+[.)]\s")


def snapshot_code(*, enable_format: bool = False) -> str:
    """Scratchpad code that prints the notebook's cells (id, name, code) as one JSON line.

    It reads the notebook document rather than ``ctx.cells``: reading a cell's code there counts as
    the agent having read it, which would let a later ``edit_cell`` overwrite a change the user made
    in the browser without the agent seeing it. ``enable_format`` also switches on code mode's
    formatting in the kernel's memory.
    """
    lines = [
        "import json as _json",
        "import marimo._code_mode as _cm",
        "_ctx = _cm.get_context()",
    ]
    if enable_format:
        lines += [
            "try:",
            '    _ctx._kernel.user_config["save"]["format_on_save"] = True',
            "except Exception:",
            "    pass",
        ]
    lines += [
        '_cells = [{"id": str(_c.id), "name": _c.name or "", "code": _c.code or ""} for _c in _ctx._document.cells]',
        f'print({SNAPSHOT_MARKER!r} + _json.dumps({{"cells": _cells}}))',
    ]
    return "\n".join(lines)


@dataclass(frozen=True)
class Cell:
    id: str
    name: str
    code: str

    @property
    def label(self) -> str:
        """``Hbol`` or, for a named cell, ``revenue_by_region (Hbol)``."""
        return f"{self.name} ({self.id})" if self.name and self.name != "_" else self.id


@dataclass(frozen=True)
class Snapshot:
    """The notebook's cells, in notebook order."""

    cells: tuple[Cell, ...]

    def cell(self, cell_id: str) -> Cell | None:
        return next((c for c in self.cells if c.id == cell_id), None)


def parse_snapshot(stdout: str) -> Snapshot | None:
    """The snapshot in :func:`snapshot_code`'s output; ``None`` when it has none."""
    for line in reversed((stdout or "").splitlines()):
        if not line.startswith(SNAPSHOT_MARKER):
            continue
        try:
            data = json.loads(line[len(SNAPSHOT_MARKER):])
            cells = tuple(Cell(str(c["id"]), str(c.get("name") or ""), str(c.get("code") or "")) for c in data["cells"])
        except (ValueError, KeyError, TypeError, AttributeError):
            return None
        return Snapshot(cells)
    return None


def changed_cells(before: Snapshot, after: Snapshot) -> list[str]:
    """Ids of the cells in ``after`` that are new or whose code changed (notebook order); empty cells
    are left out."""
    old = {c.id: c.code for c in before.cells}
    return [c.id for c in after.cells if c.code.strip() and old.get(c.id) != c.code]


@dataclass(frozen=True)
class Finding:
    tool: str  # "ruff" or "ty"
    rule: str  # "F821", "unresolved-attribute", "invalid-syntax", ...
    message: str
    cell_id: str
    line: int  # 1-based, within the cell
    column: int  # 1-based


@dataclass
class CheckResult:
    findings: list[Finding] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)  # the tools that ran
    problems: list[str] = field(default_factory=list)  # tools that could not run, and why


def _lines(code: str) -> list[str]:
    return code.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def syntax_error(cell: Cell) -> Finding | None:
    """The cell's syntax error, if Python cannot parse it (top-level ``await`` is allowed, as in marimo)."""
    try:
        with warnings.catch_warnings():  # a SyntaxWarning ("\d" in a string) would land in the chat terminal
            warnings.simplefilter("ignore")
            compile(cell.code, f"<cell {cell.id}>", "exec", flags=ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT, dont_inherit=True)
    except SyntaxError as err:
        return Finding("python", "syntax-error", err.msg, cell.id, max(err.lineno or 1, 1), max(err.offset or 1, 1))
    except ValueError as err:  # a null byte in the source
        return Finding("python", "syntax-error", str(err), cell.id, 1, 1)
    return None


def dependency_order(cells: Sequence[Cell]) -> list[Cell]:
    """``cells`` in marimo's topological order: a cell after the cells whose names it uses, ties in
    notebook order. Cells marimo cannot place (a cycle, code it cannot compile) follow in notebook order."""
    from marimo._ast.compiler import compile_cell  # lazy: marimo is slow to import
    from marimo._runtime.dataflow import DirectedGraph, topological_sort
    from marimo._types.ids import CellId_t

    graph = DirectedGraph()
    for cell in cells:
        try:
            with warnings.catch_warnings():  # compiling warns like syntax_error does; keep it out of the terminal
                warnings.simplefilter("ignore")
                compiled = compile_cell(cell.code, cell_id=CellId_t(cell.id))
            graph.register_cell(CellId_t(cell.id), compiled)
        except Exception as err:  # noqa: BLE001 - such a cell is still checked, just not ordered
            log.debug("cell %s left unordered: %s: %s", cell.id, type(err).__name__, err)
    try:
        order = [str(cid) for cid in topological_sort(graph, list(graph.cells.keys()))]
    except Exception as err:  # noqa: BLE001 - notebook order is a usable fallback
        log.debug("no dependency order: %s: %s", type(err).__name__, err)
        order = []
    by_id = {c.id: c for c in cells}
    ordered = [by_id[cid] for cid in order if cid in by_id]
    placed = {c.id for c in ordered}
    return ordered + [c for c in cells if c.id not in placed]


def build_script(snapshot: Snapshot, *, skip: Collection[str] = ()) -> tuple[str, list[tuple[str, int] | None]]:
    """The notebook as one script, cells in :func:`dependency_order` except empty ones and those in
    ``skip``, and for each line of it (index = line - 1) the cell and line it came from."""
    lines: list[str] = []
    owners: list[tuple[str, int] | None] = []
    kept = [c for c in snapshot.cells if c.code.strip() and c.id not in skip]
    for cell in dependency_order(kept):
        if lines:
            lines += ["", ""]
            owners += [None, None]
        lines.append(f"# cell {cell.id}")
        owners.append(None)
        for number, text in enumerate(_lines(cell.code), start=1):
            lines.append(text)
            owners.append((cell.id, number))
    return "\n".join(lines) + "\n", owners


class _Unavailable(Exception):
    """A tool could not run; the message says why."""


def _binary(name: str) -> str:
    """The ruff or ty executable installed with Hailer, else one on PATH."""
    try:
        if name == "ruff":
            from ruff import find_ruff_bin

            return os.fsdecode(find_ruff_bin())
        from ty import find_ty_bin

        return os.fsdecode(find_ty_bin())
    except (ImportError, FileNotFoundError):
        found = shutil.which(name)
        if found:
            return found
    raise _Unavailable(f"{name} is not installed")


def _run(cmd: list[str], *, cwd: Path, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    tool = Path(cmd[0]).stem
    try:
        return subprocess.run(
            cmd, cwd=cwd, input=stdin, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=TIMEOUT_SEC, check=False,
        )
    except subprocess.TimeoutExpired:
        raise _Unavailable(f"{tool} did not finish within {TIMEOUT_SEC:.0f} s") from None
    except OSError as err:
        raise _Unavailable(f"{tool} could not start: {err}") from None


def _failure(tool: str, result: subprocess.CompletedProcess[str]) -> _Unavailable:
    detail = (result.stderr or result.stdout or "").strip().splitlines()
    return _Unavailable(f"{tool} failed (exit code {result.returncode})" + (f": {detail[-1]}" if detail else ""))


def _ruff(script: str, workdir: Path) -> list[tuple[str, str, int, int]]:
    """``(rule, message, line, column)`` for each ruff finding in ``script``."""
    target = f"py{sys.version_info[0]}{sys.version_info[1]}"
    cmd = [
        _binary("ruff"), "check", "--isolated", "--no-cache", "--output-format", "json", "--target-version", target,
        "--select", ",".join(RUFF_SELECT), "--ignore", ",".join(RUFF_IGNORE), "--stdin-filename", _SCRIPT_NAME, "-",
    ]
    result = _run(cmd, cwd=workdir, stdin=script)
    if result.returncode not in (0, 1):
        raise _failure("ruff", result)
    try:
        items = json.loads(result.stdout or "[]")
        return [
            (str(i.get("code") or "invalid-syntax"), str(i.get("message") or ""), int(i["location"]["row"]), int(i["location"]["column"]))
            for i in items
        ]
    except (ValueError, KeyError, TypeError) as err:
        raise _Unavailable(f"ruff's output could not be read ({type(err).__name__})") from None


def _ty(script: str, workdir: Path, *, search_paths: Sequence[Path]) -> list[tuple[str, str, int, int]]:
    """``(rule, message, line, column)`` for each ty finding in ``script``, type-checked against the
    packages of the Python that runs Hailer."""
    path = workdir / _SCRIPT_NAME
    path.write_text(script, encoding="utf-8")
    cmd = [
        _binary("ty"), "check", "--python", sys.executable, "--output-format", "gitlab", "--no-progress",
        "--color", "never", "--exit-zero",
    ]
    for rule in TY_IGNORE:
        cmd += ["--ignore", rule]
    for folder in search_paths:
        if Path(folder).is_dir():  # ty refuses a search path that is not a folder
            cmd += ["--extra-search-path", str(folder)]
    result = _run([*cmd, str(path)], cwd=workdir)
    if result.returncode != 0:
        raise _failure("ty", result)
    try:
        items = json.loads(result.stdout or "[]")
        out = []
        for item in items:
            rule = str(item.get("check_name") or "")
            message = str(item.get("description") or "")
            if message.startswith(f"{rule}: "):
                message = message[len(rule) + 2:]
            begin = item["location"]["positions"]["begin"]
            out.append((rule, message, int(begin["line"]), int(begin["column"])))
        return out
    except (ValueError, KeyError, TypeError) as err:
        raise _Unavailable(f"ty's output could not be read ({type(err).__name__})") from None


def _reader_text(tree: ast.AST) -> Iterator[ast.Constant]:
    """The strings typed into text a reader sees: the argument of ``mo.md(...)`` and any ``title=``
    (a chart's or an axis's). Of an f-string, only its literal parts; strings joined with ``+`` too."""

    def typed(node: ast.expr) -> Iterator[ast.Constant]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node
        elif isinstance(node, ast.JoinedStr):
            yield from (part for part in node.values if isinstance(part, ast.Constant))
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            yield from typed(node.left)
            yield from typed(node.right)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "md" and isinstance(func.value, ast.Name) and func.value.id in ("mo", "marimo") and node.args:
            yield from typed(node.args[0])
        for keyword in node.keywords:
            if keyword.arg == "title":
                yield from typed(keyword.value)


def hardcoded_values(cell: Cell) -> list[Finding]:
    """One ``hardcoded-value`` finding per line of text in ``cell`` that has numbers, dates or month
    names typed in (see :func:`_reader_text`); none for a cell Python cannot parse."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(cell.code)
    except (SyntaxError, ValueError):
        return []
    source = _lines(cell.code)
    values_by_line: dict[int, list[str]] = {}
    for text in _reader_text(tree):
        span = range(text.lineno, (text.end_lineno or text.lineno) + 1)
        for line in str(text.value).split("\n"):
            for value in _TYPED_VALUE_RE.findall(_NUMBERED_RE.sub("", line)) + _MONTH_RE.findall(line):
                # The cell line that shows the value: a string's text lines need not be its source
                # lines ("\n" written inside it, a string marimo laid out again).
                number = next((n for n in span if value in source[n - 1]), text.lineno)
                seen = values_by_line.setdefault(number, [])
                if value not in seen:
                    seen.append(value)
    message = " typed into the text; format values from variables so the text follows the data"
    return [
        Finding("hailer", "hardcoded-value", ", ".join(f'"{v}"' for v in values) + message, cell.id, line, 1)
        for line, values in sorted(values_by_line.items())
        if values
    ]


def check(
    snapshot: Snapshot,
    cell_ids: Collection[str] | None = None,
    *,
    lint: bool = True,
    typecheck: bool = True,
    hardcoded: bool = True,
    search_paths: Sequence[Path] = (),
) -> CheckResult:
    """Lint (ruff), type-check (ty) and look for values typed into text (:func:`hardcoded_values`);
    the findings in ``cell_ids`` (all cells when ``None``). ``search_paths`` are folders notebook
    code imports local modules from (the notebook's folder: marimo puts it on ``sys.path``)."""
    result = CheckResult()
    wanted = set(cell_ids) if cell_ids is not None else {c.id for c in snapshot.cells}
    if not (lint or typecheck or hardcoded) or not wanted:
        return result
    unparsable: set[str] = set()
    for cell in snapshot.cells:
        error = syntax_error(cell) if cell.code.strip() else None
        if error is not None:
            unparsable.add(cell.id)
            if cell.id in wanted:
                result.findings.append(error)
    script, owners = build_script(snapshot, skip=unparsable) if lint or typecheck else ("", [])
    raw: list[tuple[str, tuple[str, str, int, int]]] = []
    with tempfile.TemporaryDirectory(prefix="hailer-checks-") as tmp:
        workdir = Path(tmp)
        for tool, enabled in (("ty", typecheck), ("ruff", lint)):
            if not enabled:
                continue
            try:
                if tool == "ruff":
                    found = _ruff(script, workdir)
                else:
                    found = _ty(script, workdir, search_paths=search_paths)
            except _Unavailable as err:
                log.warning("code check skipped: %s", err)
                result.problems.append(str(err))
                continue
            result.tools.append(tool)
            raw += [(tool, item) for item in found]
    if hardcoded:
        result.tools.append("hailer")
        for cell in snapshot.cells:
            if cell.id in wanted and cell.id not in unparsable:
                result.findings += hardcoded_values(cell)
    seen: set[tuple[str, int, int, str]] = set()
    for tool, (rule, message, line, column) in raw:
        owner = owners[line - 1] if 0 < line <= len(owners) else None
        if owner is None or owner[0] not in wanted:
            continue
        if tool == "ty" and rule == "invalid-syntax" and _TOP_LEVEL_ASYNC in message:
            continue
        cell_id, cell_line = owner
        same = _SAME_AS_TY.get(rule, rule) if tool == "ruff" else rule
        # ty runs first, so ruff's copy of an undefined name or a syntax error is the one dropped (the
        # two tools may place a syntax error in different columns); with ty off, ruff's is kept.
        key = (cell_id, cell_line, 0 if same == "invalid-syntax" else column, same)
        if key in seen:
            continue
        seen.add(key)
        result.findings.append(Finding(tool, rule, message, cell_id, cell_line, column))
    order = {c.id: i for i, c in enumerate(snapshot.cells)}
    result.findings.sort(key=lambda f: (order.get(f.cell_id, len(order)), f.line, f.column, f.tool))
    return result


def report(result: CheckResult, snapshot: Snapshot, cell_ids: Sequence[str], *, changed: bool = True) -> str:
    """The text the model reads: a summary line and the findings, each with the line it is about.
    ``changed`` says the cells are the ones a call just changed rather than the whole notebook."""
    tools = ", ".join(sorted(result.tools))
    count = len(cell_ids)
    plural = "s" if count != 1 else ""
    cells = f"the {count} changed cell{plural}" if changed else f"the notebook's {count} cell{plural}"
    lines: list[str] = []
    if result.tools:
        if result.findings:
            n = len(result.findings)
            lines.append(f"Checks ({tools}) on {cells}: {n} finding{'s' if n != 1 else ''}.")
        else:
            lines.append(f"Checks ({tools}) on {cells}: no findings.")
    for finding in result.findings[:MAX_REPORTED]:
        cell = snapshot.cell(finding.cell_id)
        label = cell.label if cell is not None else finding.cell_id
        lines.append(f"- cell {label}, line {finding.line}: {finding.tool} {finding.rule}: {finding.message}")
        source = _lines(cell.code)[finding.line - 1] if cell is not None and finding.line <= len(_lines(cell.code)) else ""
        if source.strip():
            text = source.strip()
            lines.append(f"    {text[:117] + '...' if len(text) > 120 else text}")
    if len(result.findings) > MAX_REPORTED:
        lines.append(f"(and {len(result.findings) - MAX_REPORTED} more)")
    for problem in result.problems:
        lines.append(f"(Check skipped: {problem}.)")
    return "\n".join(lines)


__all__ = [
    "MAX_REPORTED",
    "RUFF_IGNORE",
    "RUFF_SELECT",
    "SNAPSHOT_MARKER",
    "TY_IGNORE",
    "Cell",
    "CheckResult",
    "Finding",
    "Snapshot",
    "build_script",
    "changed_cells",
    "check",
    "dependency_order",
    "hardcoded_values",
    "parse_snapshot",
    "report",
    "snapshot_code",
    "syntax_error",
]
