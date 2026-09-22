"""Tests for hailer.tools: tool behaviour with a fake marimo client and the LangChain
registration the agent uses (offline)."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path

import pytest

from hailer import code_checks, notebooks
from hailer.errors import MarimoUnavailableError, NoSessionError
from hailer.models import CodeChecksConfig, ExecResult, HailerConfig, KernelConfig, MarimoServer, MarimoSession, WebConfig
from hailer.tools import TOOL_NAMES, HailerTools, hailer_tools

NOTEBOOK_SOURCE = 'import marimo\n\n__generated_with = "0.24.2"\napp = marimo.App()\n\n\n@app.cell\ndef _():\n    return\n\n\nif __name__ == "__main__":\n    app.run()\n'


def make_config(tmp_path: Path, **kw) -> HailerConfig:
    root = tmp_path / ".config" / "hailer"
    return HailerConfig(
        workspace=tmp_path,
        notebook=tmp_path / "notebooks" / "analysis.py",
        data_dir=tmp_path / "data",
        context_dir=root / "context",
        skills_dir=root / "skills",
        prompts_dir=root / "prompts",
        **kw,
    )


def write_notebook(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(NOTEBOOK_SOURCE, encoding="utf-8")
    return path.resolve()


def _key(path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


class FakeClient:
    """A stand-in for MarimoClient.

    ``open_paths`` (when given) is the set of notebooks that have a kernel session; sessions
    are then derived from it and ``resolve_session`` matches by path. Without it the legacy
    ``has_session`` flag applies to any notebook.
    """

    def __init__(
        self,
        *,
        has_session: bool = True,
        result: ExecResult | None = None,
        raise_on_execute: Exception | None = None,
        open_paths: set[Path] | None = None,
    ):
        self.has_session = has_session
        self.result = result or ExecResult(success=True, stdout="hello\n")
        self.raise_on_execute = raise_on_execute
        self.open_paths: set[str] | None = {_key(p) for p in open_paths} if open_paths is not None else None
        self.executed: list[str] = []
        self.executed_notebooks: list[Path | None] = []
        self.closed: list[str] = []

    def health(self) -> bool:
        return True

    def _session_for(self, key: str) -> MarimoSession:
        ordered = sorted(self.open_paths or [])
        return MarimoSession(f"s{ordered.index(key) + 1}", Path(key).name, key)

    def sessions(self) -> list[MarimoSession]:
        if self.open_paths is not None:
            return [self._session_for(k) for k in sorted(self.open_paths)]
        if not self.has_session:
            return []
        return [MarimoSession("s1", "analysis.py", "notebooks/analysis.py")]

    def resolve_session(self, notebook):
        if self.open_paths is not None:
            from hailer.marimo_client import match_session  # the real rule: exact path, no filename fallback

            session = match_session(self.sessions(), Path(notebook), None)
            if session is None:
                raise NoSessionError("no session", "open the notebook")
            return session
        if not self.has_session:
            raise NoSessionError("no session", "open the notebook")
        return MarimoSession("s1", "analysis.py", "notebooks/analysis.py")

    def execute(self, code, *, session_id=None, notebook=None, on_stdout=None, on_stderr=None, timeout=600.0):
        self.executed.append(code)
        self.executed_notebooks.append(Path(notebook).resolve() if notebook is not None else None)
        if self.raise_on_execute:
            raise self.raise_on_execute
        if self.open_paths is not None and notebook is not None:
            self.resolve_session(notebook)  # no session -> NoSessionError, like the real client
        return self.result

    def shutdown_session(self, session_id: str) -> None:
        self.closed.append(session_id)
        if self.open_paths is not None:
            for key in list(self.open_paths):
                if self._session_for(key).session_id == session_id:
                    self.open_paths.discard(key)
                    break

    def server_token(self) -> str:
        return "tok"


def factory_for(client: FakeClient, url: str = "http://127.0.0.1:2718", version: str = "0.24.2"):
    def factory():
        return client, MarimoServer(url=url, server_id="127.0.0.1:2718", version=version, source="config")

    return factory


def failing_factory():
    raise MarimoUnavailableError("marimo is not running", "Start it with:\n    uvx hailer notebook --foreground")


class Opener:
    """Records the URLs it was asked to open; optionally gives the notebook a session."""

    def __init__(self, client: FakeClient | None = None, *, succeed: bool = True, grant_session: bool = True):
        self.client = client
        self.succeed = succeed
        self.grant_session = grant_session
        self.urls: list[str] = []

    def __call__(self, url: str) -> bool:
        self.urls.append(url)
        if not self.succeed:
            return False
        if self.grant_session and self.client is not None and self.client.open_paths is not None:
            from urllib.parse import parse_qs, unquote, urlsplit

            key = parse_qs(urlsplit(url).query)["file"][0]
            self.client.open_paths.add(_key(unquote(key)))
        return True


@pytest.fixture(autouse=True)
def _no_notebook_override(monkeypatch):
    monkeypatch.delenv("HAILER_NOTEBOOK", raising=False)


@pytest.fixture
def ws(tmp_path):
    """A workspace with two notebooks; the configured one is analysis.py."""
    config = make_config(tmp_path, notebooks_dir=tmp_path / "notebooks")
    analysis = write_notebook(config.notebook)
    other = write_notebook(tmp_path / "notebooks" / "other.py")
    return config, analysis, other


# --------------------------------------------------------------------------- #
# HailerTools: kernel tools
# --------------------------------------------------------------------------- #


def test_marimo_execute_success_and_failure(tmp_path):
    client = FakeClient(result=ExecResult(success=True, stdout="42\n", output="42"))
    tools = HailerTools(make_config(tmp_path), factory_for(client))
    assert tools.marimo_execute("print(42)") == "42"
    assert client.executed == ["print(42)"]

    client = FakeClient(result=ExecResult(success=False, stderr="NameError: x"))
    tools = HailerTools(make_config(tmp_path), factory_for(client))
    text = tools.marimo_execute("x")
    assert text.startswith("Execution failed.")
    assert "NameError: x" in text


def test_marimo_execute_errors_become_text(tmp_path):
    tools = HailerTools(make_config(tmp_path), failing_factory)
    text = tools.marimo_execute("1")
    assert text.startswith("ERROR: marimo is not running")
    assert "uvx hailer notebook --foreground" in text

    client = FakeClient(raise_on_execute=NoSessionError("Notebook not open", "Open http://127.0.0.1:2718/?file=notebooks/analysis.py"))
    tools = HailerTools(make_config(tmp_path), factory_for(client))
    text = tools.marimo_execute("1")
    assert text.startswith("ERROR: Notebook not open") and "Open http://" in text

    client = FakeClient(raise_on_execute=RuntimeError("boom"))
    tools = HailerTools(make_config(tmp_path), factory_for(client))
    assert tools.marimo_execute("1").startswith("ERROR: unexpected RuntimeError: boom")


def test_marimo_execute_truncates_large_output(tmp_path):
    big = "".join(f"row {i}\n" for i in range(5000))
    client = FakeClient(result=ExecResult(success=True, stdout=big))
    tools = HailerTools(make_config(tmp_path, max_tool_output_chars=1000), factory_for(client))
    text = tools.marimo_execute("print(df)")
    assert len(text) <= 1020
    assert text.startswith("row 0\n")
    assert text.rstrip().endswith("row 4999")
    assert "truncated" in text


def test_kernel_tools_use_the_active_notebook_from_the_state_file(ws):
    config, analysis, other = ws
    client = FakeClient(open_paths={analysis, other}, result=ExecResult(success=True, stdout="ok\n"))
    tools = HailerTools(config, factory_for(client))

    assert tools.marimo_execute("1") == "ok"
    assert client.executed_notebooks[-1] == analysis  # no state file -> the configured notebook

    notebooks.save_active_notebook(config, other)  # switched by the CLI or an earlier tool call
    assert tools.marimo_execute("2") == "ok"
    assert client.executed_notebooks[-1] == other
    tools.notebook_cells()
    assert client.executed_notebooks[-1] == other

    other.unlink()  # deleted notebook -> fall back to the configured one on the next call
    tools.marimo_execute("3")
    assert client.executed_notebooks[-1] == analysis


def test_marimo_status_variants(tmp_path):
    tools = HailerTools(make_config(tmp_path), failing_factory)
    text = tools.marimo_status()
    assert text.startswith("marimo: not running") and "uvx hailer notebook --foreground" in text
    assert "active notebook: notebooks/analysis.py" in text

    tools = HailerTools(make_config(tmp_path), factory_for(FakeClient(has_session=True)))
    text = tools.marimo_status()
    assert "running at http://127.0.0.1:2718 (version 0.24.2)" in text
    assert "s1: notebooks/analysis.py (active notebook)" in text
    assert "active notebook: notebooks/analysis.py -> session s1 (ready)" in text

    tools = HailerTools(make_config(tmp_path), factory_for(FakeClient(has_session=False)))
    text = tools.marimo_status()
    assert "sessions: none" in text
    assert "NO session" in text and "http://127.0.0.1:2718" in text


def test_marimo_status_names_active_notebook_and_other_sessions(ws, tmp_path):
    config, analysis, other = ws
    outside = write_notebook(tmp_path / "elsewhere" / "x.py")
    notebooks.save_active_notebook(config, other)
    client = FakeClient(open_paths={analysis, other, outside})
    text = HailerTools(config, factory_for(client)).marimo_status()
    assert "active notebook: notebooks/other.py -> session" in text and "(ready)" in text
    assert "notebooks/analysis.py (another notebook in the notebooks folder" in text
    assert "elsewhere/x.py (outside the notebooks folder)" in text
    assert "notebooks/other.py (active notebook)" in text


def test_notebook_cells_filters(tmp_path):
    listing = "Hbol\timports\tidle\terrors=0\timport marimo as mo\nAbcd\tload\tidle\terrors=1\tdf = load_periods(files)\n"
    client = FakeClient(result=ExecResult(success=True, stdout=listing))
    tools = HailerTools(make_config(tmp_path), factory_for(client))
    assert tools.notebook_cells() == listing.rstrip()
    assert tools.notebook_cells("LOAD") == "Abcd\tload\tidle\terrors=1\tdf = load_periods(files)"
    assert tools.notebook_cells("zzz") == "No cells match 'zzz'."
    assert "marimo._code_mode" in client.executed[0]


# --------------------------------------------------------------------------- #
# HailerTools: code checks (hailer.code_checks runs the real ruff and ty)
# --------------------------------------------------------------------------- #

STARTER_CELLS = {"imp": "import marimo as mo\nimport polars as pl", "load": "sales = pl.read_csv('sales.csv')"}
WRITE_CELL = "import marimo._code_mode as cm\nasync with cm.get_context() as ctx:\n    ctx.create_cell('...')"


class NotebookClient(FakeClient):
    """A FakeClient with cells: it answers the cell snapshot from ``cells`` and applies ``writes``
    (code -> cells to set) when the agent's code runs."""

    def __init__(self, cells: dict[str, str], writes: dict[str, dict[str, str]] | None = None, **kw):
        super().__init__(**kw)
        self.cells = dict(cells)
        self.writes = writes or {}

    def execute(self, code, **kw):
        if code_checks.SNAPSHOT_MARKER in code:
            self.executed.append(code)
            payload = {"cells": [{"id": cid, "name": cid, "code": text} for cid, text in self.cells.items()]}
            return ExecResult(success=True, stdout=code_checks.SNAPSHOT_MARKER + json.dumps(payload) + "\n")
        result = super().execute(code, **kw)
        self.cells.update(self.writes.get(code, {}))
        return result


def test_marimo_execute_checks_the_cells_it_changed(tmp_path):
    new_cell = "revenue = sales.group_by('region').agg(pl.col('revenue').sum())\nrevenue.sortt('revenue')"
    client = NotebookClient(
        {**STARTER_CELLS, "old": "print(missing_before)"},
        {WRITE_CELL: {"revenue": new_cell}},
        result=ExecResult(success=True, stdout="created cell\n"),
    )
    text = HailerTools(make_config(tmp_path), factory_for(client)).marimo_execute(WRITE_CELL)
    assert text == (
        "created cell\n\n"
        "Checks (ruff, ty) on the 1 changed cell: 1 finding.\n"
        "- cell revenue (revenue), line 2: ty unresolved-attribute: Object of type `DataFrame` has no attribute `sortt`\n"
        "    revenue.sortt('revenue')"
    ), "the pre-existing problem in cell 'old' is not the agent's change"
    before, agent, after = client.executed
    assert "format_on_save" in before and agent == WRITE_CELL and "format_on_save" not in after


def test_marimo_execute_reports_a_clean_change_and_a_failed_run(tmp_path):
    client = NotebookClient(STARTER_CELLS, {WRITE_CELL: {"n": "rows = sales.height"}}, result=ExecResult(success=False, stderr="boom"))
    text = HailerTools(make_config(tmp_path), factory_for(client)).marimo_execute(WRITE_CELL)
    assert text.startswith("Execution failed.\n[stderr]\nboom")
    assert text.endswith("\n\nChecks (ruff, ty) on the 1 changed cell: no findings.")


def test_marimo_execute_without_cell_writes_runs_only_the_code(tmp_path):
    client = NotebookClient(STARTER_CELLS)
    tools = HailerTools(make_config(tmp_path), factory_for(client))
    assert tools.marimo_execute("print(sales.shape)") == "hello"
    assert client.executed == ["print(sales.shape)"]
    reading = "import marimo._code_mode as cm\nprint(cm.get_context().cells[0].code)"
    assert tools.marimo_execute(reading) == "hello" and client.executed[-1] == reading and len(client.executed) == 2


def test_marimo_execute_says_nothing_when_no_cell_changed(tmp_path):
    client = NotebookClient(STARTER_CELLS)
    assert HailerTools(make_config(tmp_path), factory_for(client)).marimo_execute(WRITE_CELL) == "hello"
    assert len(client.executed) == 3


def test_checks_config_switches_the_parts_off(tmp_path):
    writes = {WRITE_CELL: {"n": "print(undefined_x)"}}
    off = CodeChecksConfig(lint=False, typecheck=False, format=False)
    client = NotebookClient(STARTER_CELLS, writes)
    assert HailerTools(make_config(tmp_path, checks=off), factory_for(client)).marimo_execute(WRITE_CELL) == "hello"
    assert client.executed == [WRITE_CELL]

    format_only = CodeChecksConfig(lint=False, typecheck=False, format=True)
    client = NotebookClient(STARTER_CELLS, writes)
    assert HailerTools(make_config(tmp_path, checks=format_only), factory_for(client)).marimo_execute(WRITE_CELL) == "hello"
    assert len(client.executed) == 2 and "format_on_save" in client.executed[0], "formatting is switched on, nothing checked"

    lint_only = CodeChecksConfig(typecheck=False, format=False)
    client = NotebookClient(STARTER_CELLS, writes)
    text = HailerTools(make_config(tmp_path, checks=lint_only), factory_for(client)).marimo_execute(WRITE_CELL)
    assert "Checks (ruff) on the 1 changed cell: 1 finding." in text and "ruff F821" in text
    assert "format_on_save" not in client.executed[0]


def test_marimo_execute_runs_the_code_when_the_cells_cannot_be_read(tmp_path):
    class Unreadable(NotebookClient):
        def execute(self, code, **kw):
            if code_checks.SNAPSHOT_MARKER in code:
                self.executed.append(code)
                return ExecResult(success=False, stderr="AttributeError: no _document")
            return super().execute(code, **kw)

    client = Unreadable(STARTER_CELLS, {WRITE_CELL: {"n": "print(undefined_x)"}})
    assert HailerTools(make_config(tmp_path), factory_for(client)).marimo_execute(WRITE_CELL) == "hello"
    assert WRITE_CELL in client.executed


def test_the_findings_survive_the_output_cap(tmp_path):
    big = "".join(f"row {i}\n" for i in range(5000))
    client = NotebookClient(STARTER_CELLS, {WRITE_CELL: {"n": "print(undefined_x)"}}, result=ExecResult(success=True, stdout=big))
    text = HailerTools(make_config(tmp_path, max_tool_output_chars=1000), factory_for(client)).marimo_execute(WRITE_CELL)
    assert len(text) <= 1000 and text.startswith("row 0\n") and "truncated" in text
    assert text.endswith("- cell n (n), line 1: ty unresolved-reference: Name `undefined_x` used when not defined\n    print(undefined_x)")


def test_notebook_check_covers_the_whole_notebook(tmp_path):
    client = NotebookClient({**STARTER_CELLS, "empty": " ", "old": "print(missing_before)"})
    tools = HailerTools(make_config(tmp_path), factory_for(client))
    assert tools.notebook_check() == (
        "Checks (ruff, ty) on the notebook's 3 cells: 1 finding.\n"
        "- cell old (old), line 1: ty unresolved-reference: Name `missing_before` used when not defined\n"
        "    print(missing_before)"
    )
    assert "format_on_save" not in client.executed[0]

    assert HailerTools(make_config(tmp_path), factory_for(NotebookClient({"e": ""}))).notebook_check() == "The notebook has no code to check."
    off = make_config(tmp_path, checks=CodeChecksConfig(lint=False, typecheck=False))
    assert "switched off" in HailerTools(off, factory_for(client)).notebook_check()
    unreadable = FakeClient(result=ExecResult(success=False, stderr="NameError: x"))
    assert HailerTools(make_config(tmp_path), factory_for(unreadable)).notebook_check().startswith("Could not read the notebook's cells.")
    assert HailerTools(make_config(tmp_path), failing_factory).notebook_check().startswith("ERROR: marimo is not running")


# --------------------------------------------------------------------------- #
# HailerTools: notebook lifecycle tools
# --------------------------------------------------------------------------- #


def test_notebook_list_empty_folder(tmp_path):
    config = make_config(tmp_path, notebooks_dir=tmp_path / "notebooks")
    text = HailerTools(config, failing_factory).notebook_list()
    assert text.startswith("No notebooks in notebooks yet") and "notebook_create" in text


def test_notebook_list_marks_active_and_open(ws, tmp_path):
    config, analysis, other = ws
    (tmp_path / "notebooks" / "notes.txt").write_text("not a notebook", encoding="utf-8")
    write_notebook(tmp_path / "notebooks" / "__marimo__" / "cache.py")
    notebooks.save_active_notebook(config, other)
    client = FakeClient(open_paths={analysis})
    text = HailerTools(config, factory_for(client)).notebook_list()
    lines = text.splitlines()
    assert lines[0].startswith("Notebooks in notebooks (")
    analysis_line = next(ln for ln in lines if "notebooks/analysis.py" in ln)
    other_line = next(ln for ln in lines if "notebooks/other.py" in ln)
    assert "[open]" in analysis_line and "[active]" not in analysis_line
    assert "[active]" in other_line and "[open]" not in other_line
    assert "modified 20" in analysis_line and "B" in analysis_line
    assert "notes.txt" not in text and "__marimo__" not in text
    assert "not reachable" not in text


def test_notebook_list_when_marimo_is_down(ws):
    config, analysis, other = ws
    text = HailerTools(config, failing_factory).notebook_list()
    entries = [ln for ln in text.splitlines() if ln.startswith("  - ")]
    assert len(entries) == 2
    assert any("notebooks/analysis.py" in ln and "[active]" in ln for ln in entries)
    assert not any("[open]" in ln for ln in entries)
    assert "marimo is not reachable" in text


def test_notebook_create_ready(ws):
    config, analysis, other = ws
    client = FakeClient(open_paths={analysis})
    opener = Opener(client)
    tools = HailerTools(config, factory_for(client), open_url=opener, session_wait_sec=0)
    text = tools.notebook_create("Q2 Churn")
    created = config.notebooks_root / "q2_churn.py"
    assert created.is_file() and "import marimo" in created.read_text(encoding="utf-8")
    assert "# Q2 Churn" in created.read_text(encoding="utf-8")
    assert notebooks.load_active_notebook(config) == created.resolve()
    assert len(opener.urls) == 1
    assert opener.urls[0].startswith("http://127.0.0.1:2718/?file=") and opener.urls[0].endswith("&view-as=present")
    assert "q2_churn.py" in opener.urls[0]
    assert text.startswith("Created notebooks/q2_churn.py from the starter template; it is now the active notebook.")
    assert "kernel session ready (s" in text
    assert "starter template already defines mo, pl, duckdb" in text


def test_notebook_create_without_session_reports_url(ws):
    config, analysis, other = ws
    client = FakeClient(open_paths={analysis})
    opener = Opener(client, grant_session=False)
    tools = HailerTools(config, factory_for(client), open_url=opener, session_wait_sec=0)
    text = tools.notebook_create("draft", template="empty")
    assert (config.notebooks_root / "draft.py").is_file()
    assert "no kernel session appeared within 0 s" in text and "http://127.0.0.1:2718/?file=" in text
    assert "empty template defines nothing" in text
    assert "marimo_status" in text

    opener = Opener(client, succeed=False)
    tools = HailerTools(config, factory_for(client), open_url=opener, session_wait_sec=0)
    text = tools.notebook_create("second")
    assert "Could not open a browser" in text and "Ask the user to open http://" in text


def test_notebook_create_when_marimo_is_down_still_creates_and_switches(ws):
    config, analysis, other = ws
    opener = Opener()
    tools = HailerTools(config, failing_factory, open_url=opener, session_wait_sec=0)
    text = tools.notebook_create("offline one")
    assert (config.notebooks_root / "offline_one.py").is_file()
    assert notebooks.load_active_notebook(config).name == "offline_one.py"
    assert "marimo is not running" in text and opener.urls == []


def test_notebook_create_errors(ws):
    config, analysis, other = ws
    opener = Opener()
    tools = HailerTools(config, failing_factory, open_url=opener, session_wait_sec=0)
    text = tools.notebook_create("x", template="fancy")
    assert text.startswith("ERROR: unknown template 'fancy'") and "'starter'" in text
    text = tools.notebook_create("other")
    assert text.startswith("ERROR: A notebook named other.py already exists") and "open the existing notebook" in text
    text = tools.notebook_create("!!!")
    assert text.startswith("ERROR:") and "not a usable notebook name" in text
    assert opener.urls == []
    assert notebooks.load_active_notebook(config) == analysis


def test_notebook_open_with_session_skips_browser_and_lists_cells(ws):
    config, analysis, other = ws
    listing = "Hbol\timports\tidle\terrors=0\timport marimo as mo\n"
    client = FakeClient(open_paths={analysis, other}, result=ExecResult(success=True, stdout=listing))
    opener = Opener(client)
    tools = HailerTools(config, factory_for(client), open_url=opener, session_wait_sec=0)
    text = tools.notebook_open("other")
    assert text.startswith("notebooks/other.py is now the active notebook.")
    assert "already has a kernel session" in text and "browser was not opened" in text
    assert opener.urls == []
    assert "Cells (id, name, status, errors, first line):" in text and "Hbol\timports" in text
    assert client.executed_notebooks[-1] == other
    assert notebooks.load_active_notebook(config) == other


def test_notebook_open_opens_browser_when_not_open(ws):
    config, analysis, other = ws
    client = FakeClient(open_paths={analysis}, result=ExecResult(success=True, stdout=""))
    opener = Opener(client)
    tools = HailerTools(config, factory_for(client), open_url=opener, session_wait_sec=0)
    text = tools.notebook_open("notebooks/other.py")
    assert len(opener.urls) == 1 and "other.py" in opener.urls[0]
    assert "kernel session ready" in text and "(no cells)" in text

    # no session appears -> URL and instructions, no cell listing
    client = FakeClient(open_paths={analysis})
    opener = Opener(client, grant_session=False)
    tools = HailerTools(config, factory_for(client), open_url=opener, session_wait_sec=0)
    text = tools.notebook_open("other.py")
    assert "no kernel session appeared" in text and "Cells" not in text
    assert notebooks.load_active_notebook(config) == other


def test_notebook_open_errors(ws, tmp_path):
    config, analysis, other = ws
    opener = Opener()
    tools = HailerTools(config, factory_for(FakeClient()), open_url=opener, session_wait_sec=0)
    text = tools.notebook_open("missing")
    assert text.startswith("ERROR: No notebook named 'missing'") and "analysis" in text and "other" in text
    outside = write_notebook(tmp_path / "x.py")
    text = tools.notebook_open("../x.py")
    assert text.startswith("ERROR:") and "outside the notebooks folder" in text
    text = tools.notebook_open(str(outside))
    assert text.startswith("ERROR:") and "outside the notebooks folder" in text
    assert opener.urls == []
    assert notebooks.load_active_notebook(config) == analysis


def test_notebook_close_with_and_without_session(ws):
    config, analysis, other = ws
    client = FakeClient(open_paths={analysis, other})
    tools = HailerTools(config, factory_for(client), session_wait_sec=0)
    text = tools.notebook_close("other")
    assert text.startswith("Closed the kernel session (s2) of notebooks/other.py")
    assert "active notebook is still notebooks/analysis.py" in text
    assert client.closed == ["s2"]
    text = tools.notebook_close("other")
    assert text == "notebooks/other.py is not open (no kernel session), so there is nothing to close."

    text = tools.notebook_close()  # default: the active notebook
    assert text.startswith("Closed the kernel session (s1) of notebooks/analysis.py")
    assert "stays the active notebook" in text and "notebook_open" in text
    assert client.closed == ["s2", "s1"]
    assert tools.notebook_close("nope").startswith("ERROR: No notebook named 'nope'")


# --------------------------------------------------------------------------- #
# Data, skills, web
# --------------------------------------------------------------------------- #


def test_list_periods_missing_and_empty(tmp_path):
    config = make_config(tmp_path)
    tools = HailerTools(config, failing_factory)
    assert tools.list_periods().startswith("Data directory does not exist")
    config.data_dir.mkdir()
    text = tools.list_periods("sales")
    assert "No period files found" in text and "'sales'" in text and "25-01 sales.parquet" in text
    assert "Other data files" not in text


def test_list_periods_names_other_data_files(tmp_path):
    config = make_config(tmp_path)
    config.data_dir.mkdir()
    (config.data_dir / "customers.csv").write_text("id,name\n1,a\n", encoding="utf-8")
    (config.data_dir / "Survey 2024.json").write_text("[]", encoding="utf-8")
    (config.data_dir / "Sales.xlsx").touch()
    (config.data_dir / "Budget.xlsb").touch()
    (config.data_dir / "notes.txt").write_text("not data", encoding="utf-8")
    text = HailerTools(config, failing_factory).list_periods()
    assert text.startswith("No period files found")
    assert text.splitlines()[-1] == (
        "Other data files (load them directly with Polars or DuckDB): "
        "Budget.xlsb, customers.csv, Sales.xlsx, Survey 2024.json"
    )


def test_list_periods_names_the_data_folder_as_a_docker_kernel_sees_it(tmp_path):
    """list_periods runs on the host, but the model writes code for the container."""
    config = make_config(tmp_path, kernel=KernelConfig(runtime="docker"))
    tools = HailerTools(config, failing_factory)
    assert tools.list_periods() == "Data directory does not exist: /work/data"
    config.data_dir.mkdir()
    text = tools.list_periods()
    assert text.startswith("No period files found in /work/data.") and str(config.data_dir) not in text
    local = HailerTools(make_config(tmp_path), failing_factory).list_periods()
    assert local.startswith(f"No period files found in {config.data_dir}.")


def test_list_periods_with_files(tmp_path):
    pytest.importorskip("hailer.periods")
    pl = pytest.importorskip("polars")
    config = make_config(tmp_path)
    config.data_dir.mkdir()
    pl.DataFrame({"region": ["North"], "revenue": [1.0]}).write_parquet(config.data_dir / "25-01 sales.parquet")
    pl.DataFrame({"region": ["South"], "revenue": [2.0], "discount": [0.1]}).write_parquet(config.data_dir / "25-02 sales.parquet")
    (config.data_dir / "targets.csv").write_text("region,target\nNorth,5\n", encoding="utf-8")
    text = HailerTools(config, failing_factory).list_periods()
    assert "2025-01" in text and "2025-02" in text
    assert "Other data files (load them directly with Polars or DuckDB): targets.csv" in text
    assert "25-01 sales.parquet" not in text.splitlines()[-1], "period files are described, not listed again"


def test_skill_and_page_tools(tmp_path):
    config = make_config(tmp_path, web=WebConfig(allowed_domains=("docs.pola.rs",)))
    skill_dir = config.skills_dir / "alpha"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: alpha\ndescription: d\n---\nBody", encoding="utf-8")
    (skill_dir / "reference").mkdir()
    (skill_dir / "reference" / "r.md").write_text("ref", encoding="utf-8")
    tools = HailerTools(config, failing_factory)
    assert "Body" in tools.load_skill("alpha") and "- reference/r.md" in tools.load_skill("alpha")
    assert tools.read_skill_file("alpha", "reference/r.md") == "ref"
    assert tools.read_skill_file("alpha", "../../x").startswith("ERROR:")
    assert tools.load_skill("nope").startswith("ERROR: No skill named 'nope'")
    denied = tools.fetch_page("https://example.com/")
    assert denied.startswith("ERROR: Web access to 'example.com' is not allowed")
    assert "docs.pola.rs" in denied
    assert tools.fetch_page("ftp://docs.pola.rs/").startswith("ERROR:")


# --------------------------------------------------------------------------- #
# LangChain registration
# --------------------------------------------------------------------------- #


def test_hailer_tools_registers_twelve_tools_with_descriptions_and_schemas(tmp_path):
    tools = hailer_tools(make_config(tmp_path), client_factory=failing_factory)
    assert [t.name for t in tools] == list(TOOL_NAMES)
    assert len(tools) == 12
    assert all(t.description and len(t.description) > 40 for t in tools)
    by_name = {t.name: t for t in tools}
    schema = {name: t.tool_call_schema.model_json_schema() for name, t in by_name.items()}
    assert schema["marimo_execute"]["required"] == ["code"]
    assert schema["notebook_create"]["required"] == ["name"]
    assert schema["notebook_open"]["required"] == ["notebook"]
    assert schema["read_skill_file"]["required"] == ["name", "path"]
    assert not schema["notebook_close"].get("required")
    assert not schema["marimo_status"].get("required")
    assert not schema["notebook_check"].get("required")
    # the description is the method's docstring, the text the model reads
    assert "scratchpad" in by_name["marimo_execute"].description
    assert "ACTIVE" in by_name["marimo_status"].description
    assert "allowed-domains" in by_name["fetch_page"].description


def test_every_tool_name_is_a_documented_method():
    for name in TOOL_NAMES:
        assert (getattr(HailerTools, name).__doc__ or "").strip(), name


def test_tools_invoke_in_process(ws):
    config, analysis, other = ws
    client = FakeClient(open_paths={analysis}, result=ExecResult(success=True, stdout="ok"))
    opener = Opener(client)
    tools = {t.name: t for t in hailer_tools(config, client_factory=factory_for(client), open_url=opener, session_wait_sec=0)}

    assert tools["marimo_execute"].invoke({"code": "print('ok')"}) == "ok"
    text = tools["notebook_list"].invoke({})
    assert "notebooks/analysis.py" in text and "[active]" in text
    assert tools["notebook_create"].invoke({"name": "via tool"}).startswith("Created notebooks/via_tool.py")
    assert tools["notebook_open"].invoke({"notebook": "other"}).startswith("notebooks/other.py is now the active notebook.")
    assert tools["fetch_page"].invoke({"url": "https://example.com"}).startswith("ERROR:")

    assert len(opener.urls) == 2  # create + open (other had no session)
    assert notebooks.load_active_notebook(config) == other


def test_async_path_runs_the_tool_on_a_daemon_thread(ws):
    """The agent awaits tools; a blocking kernel call must never be joined at shutdown."""
    config, analysis, _other = ws
    seen: dict[str, object] = {}

    class RecordingClient(FakeClient):
        def execute(self, code, notebook=None, **kw):
            thread = threading.current_thread()
            seen.update(name=thread.name, daemon=thread.daemon, main=thread is threading.main_thread())
            return super().execute(code, notebook=notebook, **kw)

    client = RecordingClient(open_paths={analysis}, result=ExecResult(success=True, stdout="42"))
    tools = {t.name: t for t in hailer_tools(config, client_factory=factory_for(client))}

    async def go():
        assert await tools["marimo_execute"].ainvoke({"code": "print(42)"}) == "42"
        assert (await tools["fetch_page"].ainvoke({"url": "https://example.com"})).startswith("ERROR:")

    asyncio.run(go())
    assert seen == {"name": "hailer-tool-marimo_execute", "daemon": True, "main": False}


def test_cancelling_a_turn_does_not_wait_for_a_blocked_tool(ws):
    config, analysis, _other = ws
    release = threading.Event()
    finished = threading.Event()

    class BlockingClient(FakeClient):
        def execute(self, code, notebook=None, **kw):
            release.wait(10)
            finished.set()
            return super().execute(code, notebook=notebook, **kw)

    client = BlockingClient(open_paths={analysis}, result=ExecResult(success=True, stdout="late"))
    tools = {t.name: t for t in hailer_tools(config, client_factory=factory_for(client))}

    async def go():
        task = asyncio.ensure_future(tools["marimo_execute"].ainvoke({"code": "slow()"}))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(go())  # returns at once: closing the loop does not join the tool thread
        assert not finished.is_set()
    finally:
        release.set()
    assert finished.wait(5)  # the abandoned call still ends on its own; its result is dropped


# --------------------------------------------------------------------------- #
# Fix-wave additions: rich outputs, configured notebook/workspace on the client
# --------------------------------------------------------------------------- #


def test_rich_output_is_replaced_by_placeholder(tmp_path):
    html = "<table><tr><td>1</td></tr></table>" * 50
    client = FakeClient(result=ExecResult(success=True, stdout="shape: (5, 3)\n", output=html, mimetype="text/html"))
    tools = HailerTools(make_config(tmp_path), factory_for(client))
    text = tools.marimo_execute("df.head()")
    assert "shape: (5, 3)" in text
    assert "<table>" not in text
    assert "text/html output omitted" in text and "print()" in text
    # stderr survives alongside the placeholder
    client = FakeClient(result=ExecResult(success=True, stderr="warn\n", output="{}", mimetype="application/json"))
    text = HailerTools(make_config(tmp_path), factory_for(client)).marimo_execute("x")
    assert "application/json output omitted" in text and "[stderr]" in text
    # plain text output still passes through unchanged
    client = FakeClient(result=ExecResult(success=True, output="42", mimetype="text/plain"))
    assert HailerTools(make_config(tmp_path), factory_for(client)).marimo_execute("42") == "42"


def test_default_client_uses_active_notebook_folder_and_workspace(ws, monkeypatch):
    from hailer import marimo_client as mc

    config, analysis, other = ws
    captured: dict = {}

    class CapturingClient:
        def __init__(self, base_url, token=None, **kw):
            captured.update(base_url=base_url, token=token, **kw)

    monkeypatch.setattr(mc, "find_server", lambda config: MarimoServer(url="http://127.0.0.1:2718", source="config"))
    monkeypatch.setattr(mc, "MarimoClient", CapturingClient)
    cfg = make_config(config.workspace, notebooks_dir=config.notebooks_root, marimo_token="tok")
    client, server = HailerTools(cfg)._default_client()
    assert isinstance(client, CapturingClient) and server.url == "http://127.0.0.1:2718"
    assert captured["notebook"] == cfg.notebook and captured["workspace"] == cfg.workspace
    assert captured["token"] == "tok"

    notebooks.save_active_notebook(cfg, other)
    HailerTools(cfg)._default_client()
    assert captured["notebook"] == other
    assert captured["paths"] is None


def test_default_client_uses_the_token_and_paths_of_the_server_hailer_started(ws, monkeypatch):
    from hailer import marimo_client as mc
    from hailer.kernel import docker_paths

    config, analysis, other = ws
    captured: dict = {}

    class CapturingClient:
        def __init__(self, base_url, token=None, **kw):
            captured.update(base_url=base_url, token=token, **kw)

    paths = docker_paths(config)
    started = MarimoServer(url="http://127.0.0.1:2731", source="kernel", token="kernel-token", runtime="docker", paths=paths)
    monkeypatch.setattr(mc, "find_server", lambda config: started)
    monkeypatch.setattr(mc, "MarimoClient", CapturingClient)
    HailerTools(make_config(config.workspace, notebooks_dir=config.notebooks_root, marimo_token="user-token"))._default_client()
    assert captured["token"] == "kernel-token", "the server's own token wins over HAILER_MARIMO_TOKEN"
    assert captured["paths"] is paths
    assert captured.get("token_in_links", False) is False, "hints from this client reach the model"


# --------------------------------------------------------------------------- #
# The server token never reaches the model
# --------------------------------------------------------------------------- #

SECRET = "kernel-token-never-for-the-model"


def signed_factory(client: FakeClient, **server_kw):
    def factory():
        return client, MarimoServer(url="http://127.0.0.1:2718", source="kernel", token=SECRET, **server_kw)

    return factory


def test_tool_results_carry_token_free_urls_and_the_browser_gets_the_signed_one(ws):
    config, analysis, other = ws
    client = FakeClient(open_paths={analysis})
    opener = Opener(client, grant_session=False)
    tools = HailerTools(config, signed_factory(client), open_url=opener, session_wait_sec=0)
    texts = [tools.notebook_create("draft"), tools.notebook_open("other"), tools.marimo_status()]
    opener_fails = HailerTools(config, signed_factory(client), open_url=Opener(client, succeed=False), session_wait_sec=0)
    texts.append(opener_fails.notebook_open("other"))
    for text in texts:
        assert SECRET not in text and "access_token" not in text, text
    assert all("/notebook in the chat" in text for text in texts), "the user can get the signed-in link there"
    assert "http://127.0.0.1:2718/?file=" in texts[0] and "http://127.0.0.1:2718/?file=" in texts[3]
    assert opener.urls and all(url.endswith(f"&access_token={SECRET}") for url in opener.urls)


def test_no_session_errors_point_at_the_notebook_command(ws):
    config, analysis, other = ws
    client = FakeClient(raise_on_execute=NoSessionError("The notebook is not open in a browser.", "Open http://127.0.0.1:2718/?file=x in your browser."))
    text = HailerTools(config, signed_factory(client)).marimo_execute("1")
    assert text.startswith("ERROR: The notebook is not open") and "/notebook in the chat" in text and SECRET not in text


def test_tools_send_kernel_paths_for_a_docker_kernel(ws):
    from hailer.kernel import docker_paths

    config, analysis, other = ws
    client = FakeClient(open_paths={analysis})
    opener = Opener(client, grant_session=False)
    tools = HailerTools(config, signed_factory(client, runtime="docker", paths=docker_paths(config)), open_url=opener, session_wait_sec=0)
    text = tools.notebook_open("other")
    assert opener.urls == [f"http://127.0.0.1:2718/?file=/work/notebooks/other.py&view-as=present&access_token={SECRET}"]
    assert "?file=/work/notebooks/other.py&view-as=present" in text and SECRET not in text


# --------------------------------------------------------------------------- #
# marimo configured but down: the factory succeeds, the first request fails
# --------------------------------------------------------------------------- #


class DownClient:
    """What the real client looks like with ``marimo_url`` configured and nothing listening:
    ``find_server`` returns the server unchecked, so every request raises."""

    def __init__(self):
        self.closed: list[str] = []

    def _down(self):
        return MarimoUnavailableError(
            "Marimo is not running at http://127.0.0.1:2718 (connection refused).",
            hint="Start everything in one go:\n\n    uvx hailer notebook",
        )

    def health(self):
        return False

    def sessions(self):
        raise self._down()

    def resolve_session(self, notebook):
        raise self._down()

    def execute(self, code, **kw):
        raise self._down()

    def shutdown_session(self, session_id):
        self.closed.append(session_id)


def test_notebook_create_with_marimo_configured_but_down(ws):
    config, analysis, other = ws
    opener = Opener()
    tools = HailerTools(config, factory_for(DownClient()), open_url=opener, session_wait_sec=0)
    text = tools.notebook_create("q2 churn")
    created = config.notebooks_root / "q2_churn.py"
    assert created.is_file()
    assert notebooks.load_active_notebook(config) == created.resolve()
    assert text.startswith("Created notebooks/q2_churn.py from the starter template; it is now the active notebook.")
    assert "marimo is not running" in text and "connection refused" in text and "uvx hailer notebook" in text
    assert not text.startswith("ERROR"), "a create that succeeded must not read as a failure"
    assert opener.urls == []


def test_notebook_open_with_marimo_configured_but_down(ws):
    config, analysis, other = ws
    opener = Opener()
    tools = HailerTools(config, factory_for(DownClient()), open_url=opener, session_wait_sec=0)
    text = tools.notebook_open("other")
    assert text.startswith("notebooks/other.py is now the active notebook.")
    assert "marimo is not running" in text and "Cells" not in text
    assert notebooks.load_active_notebook(config) == other
    assert opener.urls == []


def test_marimo_status_with_marimo_configured_but_down(ws):
    config, analysis, other = ws
    notebooks.save_active_notebook(config, other)
    text = HailerTools(config, factory_for(DownClient())).marimo_status()
    assert text.startswith("marimo: not running")
    assert "active notebook: notebooks/other.py" in text
    assert "connection refused" in text and "uvx hailer notebook" in text


def test_notebook_close_and_list_with_marimo_configured_but_down(ws):
    config, analysis, other = ws
    client = DownClient()
    tools = HailerTools(config, factory_for(client))
    text = tools.notebook_close()
    assert text.startswith("marimo is not running, so notebooks/analysis.py has no kernel session to close.")
    assert "connection refused" in text and client.closed == []
    text = tools.notebook_list()
    entries = [ln for ln in text.splitlines() if ln.startswith("  - ")]
    assert any("notebooks/analysis.py" in ln for ln in entries) and any("notebooks/other.py" in ln for ln in entries)
    assert "marimo is not reachable" in text and not any("[open]" in ln for ln in entries)
    # The list also carries the error and the launch hint, like the other tools when marimo is down.
    assert "connection refused" in text and "uvx hailer notebook" in text


def test_session_wait_that_loses_marimo_is_reported_not_raised(ws):
    config, analysis, other = ws

    class FlakyClient(FakeClient):
        def __init__(self):
            super().__init__(open_paths={analysis})
            self.calls = 0

        def resolve_session(self, notebook):
            self.calls += 1
            if self.calls > 1:  # the first call finds no session; the wait then loses the server
                raise MarimoUnavailableError("Marimo is not running at http://127.0.0.1:2718 (timed out).", hint="restart it")
            return super().resolve_session(notebook)

    client = FlakyClient()
    opener = Opener(client, grant_session=False)
    tools = HailerTools(config, factory_for(client), open_url=opener, session_wait_sec=0)
    text = tools.notebook_open("other")
    assert text.startswith("notebooks/other.py is now the active notebook.")
    assert "timed out" in text and "restart it" in text
    assert len(opener.urls) == 1


# --------------------------------------------------------------------------- #
# The server the chat is pinned to; the kernel the model works in
# --------------------------------------------------------------------------- #


def test_a_pinned_server_is_used_without_discovering_one(ws, monkeypatch):
    """`hailer notebook` hands its chat the server it started: a kernel.json rewritten or removed
    underneath (another terminal) cannot take its token and paths away."""
    from hailer import marimo_client as mc
    from hailer.kernel import docker_paths

    config, analysis, other = ws
    captured: dict = {}

    class CapturingClient:
        def __init__(self, base_url, token=None, **kw):
            captured.update(base_url=base_url, token=token, **kw)

    def no_discovery(config):
        raise AssertionError("a pinned chat never discovers a server")

    pinned = MarimoServer(url="http://127.0.0.1:2731", source="kernel", token="pinned-token", runtime="docker", paths=docker_paths(config))
    monkeypatch.setattr(mc, "find_server", no_discovery)
    monkeypatch.setattr(mc, "MarimoClient", CapturingClient)
    _client, server = HailerTools(config, server=pinned)._default_client()
    assert server is pinned and captured["base_url"] == pinned.url and captured["token"] == "pinned-token"
    assert captured["paths"] is pinned.paths


def test_hailer_tools_pins_every_tool_to_the_server(ws):
    config, analysis, other = ws
    pinned = MarimoServer(url="http://127.0.0.1:2731", source="kernel", token="t")
    tools = hailer_tools(config, server=pinned)
    status = next(t for t in tools if t.name == "marimo_status")
    assert status.func.__self__.server is pinned


def test_marimo_status_names_the_kernel_in_use_and_its_paths(ws):
    """A chat whose kernel changed underneath (a docker kernel started in another terminal) still
    gets the truth: the runtime and where notebook code finds the folders."""
    from hailer.kernel import docker_paths

    config, analysis, other = ws
    client = FakeClient(open_paths={analysis})
    docker = MarimoServer(url="http://127.0.0.1:2731", source="kernel", token="t", runtime="docker", paths=docker_paths(config), network_access=True)
    text = HailerTools(config, lambda: (client, docker)).marimo_status()
    assert "kernel: docker (hailer-kernel " in text and "network on: the internet and this machine" in text
    assert "kernel paths: notebooks folder /work/notebooks (writable), data folder /work/data (read-only)" in text
    local = HailerTools(config, factory_for(client)).marimo_status()
    assert "kernel: local (runs as you; not isolated)" in local
    assert f"kernel paths: notebooks folder {config.notebooks_root}, data folder {config.data_dir}" in local


def test_notebook_open_and_close_accept_the_kernels_paths(ws):
    """The model sees /work/notebooks/... in a docker kernel and may pass such a path back."""
    from hailer.kernel import docker_paths

    config, analysis, other = ws
    client = FakeClient(open_paths={analysis, other})
    docker = MarimoServer(url="http://127.0.0.1:2731", source="kernel", token="t", runtime="docker", paths=docker_paths(config))
    tools = HailerTools(config, lambda: (client, docker), open_url=Opener(client), session_wait_sec=0)
    text = tools.notebook_open("/work/notebooks/other.py")
    assert text.startswith("notebooks/other.py is now the active notebook."), text
    text = tools.notebook_close("/work/notebooks/other.py")
    assert text.startswith("Closed the kernel session") and "notebooks/other.py" in text, text
    text = tools.notebook_open("/work/data/sales.py")
    assert text.startswith("ERROR:"), "a kernel path outside the notebooks folder is still refused"
    local = HailerTools(config, factory_for(FakeClient(open_paths={analysis, other})), open_url=Opener(client), session_wait_sec=0)
    assert local.notebook_open(str(other)).startswith("notebooks/other.py is now the active notebook."), "host paths as before"
