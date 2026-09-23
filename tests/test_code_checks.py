"""Tests for hailer.code_checks: the cell snapshot, the notebook as one script, and ruff and ty on it.

ruff and ty are Hailer dependencies, so these run the real tools (offline, a fraction of a second
each). The expectations name only stable rule codes and message fragments.
"""

from __future__ import annotations

import ast
import json
import subprocess
import warnings
from pathlib import Path

import pytest

from hailer import code_checks as cc
from hailer.code_checks import Cell, Snapshot


def snap(*cells: tuple[str, str] | tuple[str, str, str]) -> Snapshot:
    """A snapshot from ``(id, code)`` or ``(id, name, code)`` tuples, in notebook order."""
    out = []
    for cell in cells:
        cid, name, code = cell if len(cell) == 3 else (cell[0], "", cell[1])
        out.append(Cell(cid, name, code))
    return Snapshot(tuple(out))


IMPORTS = ("imp", "import marimo as mo\nimport polars as pl")


def rules(result: cc.CheckResult) -> list[tuple[str, str, int]]:
    return [(f.cell_id, f.rule, f.line) for f in result.findings]


# --------------------------------------------------------------------------- #
# The snapshot
# --------------------------------------------------------------------------- #


def test_snapshot_code_reads_the_document_so_the_agent_has_read_nothing():
    code = cc.snapshot_code()
    ast.parse(code)
    assert "_ctx._document.cells" in code
    assert "ctx.cells" not in code.replace("_ctx._document.cells", ""), "ctx.cells[...].code records a read"
    assert "format_on_save" not in code
    assert cc.SNAPSHOT_MARKER in code


def test_snapshot_code_can_switch_on_code_mode_formatting_in_the_kernel():
    code = cc.snapshot_code(enable_format=True)
    ast.parse(code)
    assert '_ctx._kernel.user_config["save"]["format_on_save"] = True' in code
    assert code.index("format_on_save") < code.index("print("), "set before the cells are read"


def test_parse_snapshot_takes_the_marker_line():
    payload = {"cells": [{"id": "Hbol", "name": "", "code": "import marimo as mo"}, {"id": "b", "name": "load", "code": None}]}
    out = f"noise\n{cc.SNAPSHOT_MARKER}{json.dumps(payload)}\n"
    assert cc.parse_snapshot(out) == snap(("Hbol", "import marimo as mo"), ("b", "load", ""))
    assert cc.parse_snapshot("no marker here") is None
    assert cc.parse_snapshot(f"{cc.SNAPSHOT_MARKER}{{not json") is None
    assert cc.parse_snapshot(f"{cc.SNAPSHOT_MARKER}{json.dumps({'cells': [{'name': 'x'}]})}") is None


def test_changed_cells_are_new_or_edited_and_not_empty():
    before = snap(("a", "x = 1"), ("b", "y = 2"), ("c", "z = 3"))
    after = snap(("a", "x = 1"), ("b", "y = 20"), ("d", "w = x"), ("e", "   "))
    assert cc.changed_cells(before, after) == ["b", "d"]
    assert cc.changed_cells(after, after) == []


def test_cell_labels_name_named_cells():
    assert Cell("Hbol", "", "x").label == "Hbol"
    assert Cell("Hbol", "_", "x").label == "Hbol"
    assert Cell("Hbol", "load", "x").label == "load (Hbol)"


# --------------------------------------------------------------------------- #
# The notebook as one script
# --------------------------------------------------------------------------- #


def test_dependency_order_puts_definitions_before_their_uses():
    cells = [Cell("use", "", "total = df.height"), Cell("load", "", "df = pl.DataFrame()"), Cell("imp", "", "import polars as pl")]
    assert [c.id for c in cc.dependency_order(cells)] == ["imp", "load", "use"]


def test_dependency_order_keeps_cells_it_cannot_place_in_notebook_order():
    cells = [Cell("a", "", "x = y"), Cell("b", "", "y = x"), Cell("c", "", "z = 1")]  # a cycle
    assert sorted(c.id for c in cc.dependency_order(cells)) == ["a", "b", "c"]
    assert [c.id for c in cc.dependency_order([Cell("a", "", "x = ("), Cell("b", "", "y = 1")])] == ["b", "a"]


def test_build_script_maps_every_line_back_to_its_cell():
    script, owners = cc.build_script(snap(("use", "print(x)\nprint(x + 1)"), ("empty", "  "), ("def", "x = 1"), ("skip", "q = 1")), skip={"skip"})
    lines = script.splitlines()
    assert len(lines) == len(owners)
    assert lines.index("x = 1") < lines.index("print(x)")
    assert owners[lines.index("print(x + 1)")] == ("use", 2)
    assert owners[lines.index("x = 1")] == ("def", 1)
    assert "q = 1" not in lines and not any(o and o[0] == "empty" for o in owners)
    assert all(owners[i] is None for i, line in enumerate(lines) if line.startswith("# cell ") or not line)


def test_syntax_error_allows_top_level_await_like_marimo():
    assert cc.syntax_error(Cell("a", "", "x = await thing()\nasync with lock:\n    pass")) is None
    error = cc.syntax_error(Cell("b", "", "y = 1\nx = ("))
    assert error is not None and (error.cell_id, error.line, error.rule, error.tool) == ("b", 2, "syntax-error", "python")


# --------------------------------------------------------------------------- #
# ruff and ty
# --------------------------------------------------------------------------- #


def test_ty_knows_the_types_other_cells_define():
    notebook = snap(
        IMPORTS,
        ("use", "revenue", "revenue = sales.group_by('region').agg(pl.col('revenue').sum())\nrevenue.sortt('revenue')"),
        ("load", "sales = pl.read_csv('sales.csv')"),
    )
    result = cc.check(notebook, ["use"])
    assert result.tools == ["ty", "ruff"] and not result.problems
    assert rules(result) == [("use", "unresolved-attribute", 2)]
    assert "sortt" in result.findings[0].message and result.findings[0].column == 1


def test_only_the_cells_asked_about_are_reported():
    notebook = snap(IMPORTS, ("old", "print(missing_one)"), ("new", "print(missing_two)"))
    assert rules(cc.check(notebook, ["new"])) == [("new", "unresolved-reference", 1)]
    assert [f.cell_id for f in cc.check(notebook).findings] == ["old", "new"]


def test_an_undefined_name_is_reported_once_with_both_tools_and_by_ruff_alone():
    notebook = snap(("a", "print(undefined_x)"))
    assert rules(cc.check(notebook)) == [("a", "unresolved-reference", 1)]
    only_ruff = cc.check(notebook, typecheck=False)
    assert only_ruff.tools == ["ruff"] and rules(only_ruff) == [("a", "F821", 1)]


def test_ruff_reports_likely_bugs():
    notebook = snap(("a", "def f(x=[]):\n    return x\n\nfns = [lambda: i for i in range(3)]\ntry:\n    f()\nexcept:\n    pass"))
    assert sorted(f.rule for f in cc.check(notebook, typecheck=False).findings) == ["B006", "B023", "E722"]


def test_what_notebook_cells_do_on_purpose_is_not_reported():
    notebook = snap(
        ("a", "import os\nimport altair as alt\nimport not_installed_here"),  # imports for later cells; packages the kernel has
        (
            "b",  # top-level await, async for and async with are allowed in a cell
            "import asyncio\n\nasync def _tick() -> int:\n    return 1\n\nasync def _numbers():\n    yield 1\n\n"
            "_x = await _tick()\nasync for _i in _numbers():\n    pass\nasync with asyncio.timeout(1):\n    pass",
        ),
        ("c", "df = [1]\ndf"),  # the last expression is the cell's output
        ("d", "def _helper():\n    return 1\n_helper()"),
        ("e", "def _helper():\n    return 2\n_helper()"),  # each cell's private helper
        ("f", "print(f'plain')"),
    )
    result = cc.check(notebook)
    assert result.tools == ["ty", "ruff"] and result.findings == []


def test_a_cell_that_does_not_parse_is_reported_and_the_rest_still_checked():
    notebook = snap(IMPORTS, ("bad", "x = ("), ("after", "print(undefined_after)"))
    result = cc.check(notebook)
    assert rules(result) == [("bad", "syntax-error", 1), ("after", "unresolved-reference", 1)]
    assert rules(cc.check(notebook, typecheck=False)) == [("bad", "syntax-error", 1), ("after", "F821", 1)]


def test_modules_next_to_the_notebook_are_found(tmp_path):
    (tmp_path / "helpers.py").write_text("def total(values: list[int]) -> int:\n    return sum(values)\n", encoding="utf-8")
    notebook = snap(("a", "from helpers import total\nresult = total('abc')"))
    assert rules(cc.check(notebook, typecheck=True, lint=False, search_paths=[tmp_path])) == [("a", "invalid-argument-type", 2)]
    missing = cc.check(notebook, lint=False, search_paths=[tmp_path / "gone", tmp_path])
    assert missing.tools == ["ty"] and rules(missing) == [("a", "invalid-argument-type", 2)], "a missing folder is skipped"


def test_checking_code_that_python_warns_about_prints_no_warning(capfd):
    notebook = snap(("a", 'import re\npattern = re.compile("\\d+")\nflag = 1\nprint(flag is 1)'))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cc.check(notebook)
    assert [w for w in caught if issubclass(w.category, SyntaxWarning)] == []
    assert "SyntaxWarning" not in capfd.readouterr().err, "a warning would land in the chat terminal"


def test_checks_switched_off_run_nothing():
    result = cc.check(snap(("a", "print(undefined_x)")), lint=False, typecheck=False)
    assert (result.findings, result.tools, result.problems) == ([], [], [])


def test_a_tool_that_cannot_run_becomes_a_problem_not_an_error(monkeypatch):
    def missing(name):
        if name == "ty":
            raise cc._Unavailable("ty is not installed")
        return original(name)

    original = cc._binary
    monkeypatch.setattr(cc, "_binary", missing)
    notebook = snap(("a", "print(undefined_x)"))
    result = cc.check(notebook)
    assert result.tools == ["ruff"] and result.problems == ["ty is not installed"]
    assert rules(result) == [("a", "F821", 1)], "ruff's copy of a finding stays when ty did not run"
    assert "(Check skipped: ty is not installed.)" in cc.report(result, notebook, ["a"])


def test_a_tool_that_fails_is_reported(monkeypatch):
    def failing_run(cmd, *, cwd, stdin=None):
        return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="error: something broke\n")

    monkeypatch.setattr(cc, "_run", failing_run)
    result = cc.check(snap(("a", "x = 1")))
    assert result.tools == [] and result.problems == ["ty failed (exit code 2): error: something broke", "ruff failed (exit code 2): error: something broke"]


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


def test_report_lists_findings_with_their_code():
    notebook = snap(IMPORTS, ("r", "revenue", "print(undefined_x)"))
    result = cc.check(notebook, ["r"])
    assert cc.report(result, notebook, ["r"]) == (
        "Checks (ruff, ty) on the 1 changed cell: 1 finding.\n"
        "- cell revenue (r), line 1: ty unresolved-reference: Name `undefined_x` used when not defined\n"
        "    print(undefined_x)"
    )
    clean = cc.check(notebook, ["imp"])
    assert cc.report(clean, notebook, ["imp", "r"], changed=False) == "Checks (ruff, ty) on the notebook's 2 cells: no findings."


def test_report_caps_the_findings():
    code = "\n".join(f"print(undefined_{i})" for i in range(cc.MAX_REPORTED + 5))
    notebook = snap(("a", code))
    text = cc.report(cc.check(notebook), notebook, ["a"])
    assert f"{cc.MAX_REPORTED + 5} findings" in text
    assert text.count("\n- cell a,") == cc.MAX_REPORTED
    assert text.endswith("(and 5 more)")


@pytest.mark.parametrize("name", ["ruff", "ty"])
def test_the_tools_come_with_hailer(name):
    assert Path(cc._binary(name)).is_file()
