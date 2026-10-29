"""Tests for hailer.tools: tool behaviour against a sandbox (tests/fake_sandbox.py) with a fake
marimo client, the LangChain registration the agent uses, and an end-to-end pass over a real
sandbox and marimo's file API (tests/fake_marimo.py). Offline."""

from __future__ import annotations

import ast
import asyncio
import contextlib
import io
import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from urllib.parse import parse_qs, urlsplit

import pytest

from fake_sandbox import folder_sandbox
from hailer import code_checks, notebooks
from hailer.errors import MarimoUnavailableError, NoSessionError
from hailer.marimo_client import LIST_CELLS_CODE, name_in_folder
from hailer.models import CodeChecksConfig, ExecResult, HailerConfig, MarimoServer, MarimoSession, WebConfig
from hailer.tool_outcomes import SUMMARY_LIMIT, tool_outcome
from hailer.tools import TOOL_NAMES, HailerTools, ToolReply, hailer_tools

NOTEBOOK_SOURCE = 'import marimo\n\n__generated_with = "0.24.2"\napp = marimo.App()\n\n\n@app.cell\ndef _():\n    return\n\n\nif __name__ == "__main__":\n    app.run()\n'


def make_config(tmp_path: Path, **kw) -> HailerConfig:
    root = tmp_path / ".config" / "hailer"
    kw.setdefault("notebooks_dir", tmp_path / "notebooks")
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
    return path


class FakeClient:
    """A stand-in for MarimoClient that knows notebooks by name.

    ``open_names`` (when given) are the notebooks with a kernel session; sessions are derived from
    it and ``resolve_session`` uses the real rule (exact name). Without it the ``has_session`` flag
    applies to any notebook. ``outside`` adds sessions of files outside the notebooks folder.
    """

    def __init__(
        self,
        *,
        has_session: bool = True,
        result: ExecResult | None = None,
        raise_on_execute: Exception | None = None,
        open_names: set[str] | None = None,
        outside: tuple[str, ...] = (),
    ):
        self.has_session = has_session
        self.result = result or ExecResult(success=True, stdout="hello\n")
        self.raise_on_execute = raise_on_execute
        self.open_names = set(open_names) if open_names is not None else None
        self.outside = outside
        self.executed: list[str] = []
        self.executed_notebooks: list[str | None] = []
        self.closed: list[str] = []

    def health(self) -> bool:
        return True

    def sessions(self) -> list[MarimoSession]:
        if self.open_names is not None:
            ordered = sorted(self.open_names)
            named = [MarimoSession(f"s{i + 1}", name, f"/work/notebooks/{name}", name=name) for i, name in enumerate(ordered)]
            return named + [MarimoSession(f"x{i + 1}", Path(p).name, p) for i, p in enumerate(self.outside)]
        if not self.has_session:
            return []
        return [MarimoSession("s1", "analysis.py", "/work/notebooks/analysis.py", name="analysis.py")]

    def resolve_session(self, notebook):
        if self.open_names is not None:
            from hailer.marimo_client import match_session  # the real rule: exact name

            session = match_session(self.sessions(), notebook)
            if session is None:
                raise NoSessionError("no session", "open the notebook")
            return session
        if not self.has_session:
            raise NoSessionError("no session", "open the notebook")
        return MarimoSession("s1", "analysis.py", "/work/notebooks/analysis.py", name="analysis.py")

    def execute(self, code, *, session_id=None, notebook=None, on_stdout=None, on_stderr=None, timeout=600.0):
        self.executed.append(code)
        self.executed_notebooks.append(notebook)
        if self.raise_on_execute:
            raise self.raise_on_execute
        if self.open_names is not None and notebook is not None:
            self.resolve_session(notebook)  # no session -> NoSessionError, like the real client
        return self.result

    def shutdown_session(self, session_id: str) -> None:
        self.closed.append(session_id)
        if self.open_names is not None:
            for session in self.sessions():
                if session.session_id == session_id and session.name:
                    self.open_names.discard(session.name)


class DownClient:
    """What the real client looks like when the chat's kernel stopped (killed, or removed with
    ``uvx hailer kernel stop``): the sandbox is still held in memory, so every request raises."""

    def __init__(self):
        self.closed: list[str] = []

    def _down(self):
        return MarimoUnavailableError(
            "Marimo is not running at http://127.0.0.1:2718 (connection refused).",
            hint="End the chat (/exit) and run uvx hailer again.",
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


def sandbox(config: HailerConfig, client=None, *, docker: bool = False, token: str | None = None, network: bool = False):
    """The chat's sandbox on ``config``'s folders; its clients are ``client`` (default: a DownClient)."""
    client = client if client is not None else DownClient()
    server = MarimoServer(url="http://127.0.0.1:2718", token=token, runtime="docker" if docker else "unsafe-local", network_access=network)
    return folder_sandbox(config, server, docker=docker, client_factory=lambda **kw: client)


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
        if self.grant_session and self.client is not None and self.client.open_names is not None:
            key = parse_qs(urlsplit(url).query)["file"][0]
            name = name_in_folder("/work/notebooks", key, native=False) or key.rsplit("/notebooks/", 1)[-1]
            self.client.open_names.add(name)
        return True


@pytest.fixture
def ws(tmp_path):
    """A workspace with two notebooks; the configured one is analysis.py."""
    config = make_config(tmp_path)
    write_notebook(config.notebook)
    write_notebook(tmp_path / "notebooks" / "other.py")
    return config


# --------------------------------------------------------------------------- #
# HailerTools: kernel tools
# --------------------------------------------------------------------------- #


def test_marimo_execute_success_and_failure(ws):
    client = FakeClient(result=ExecResult(success=True, stdout="42\n", output="42"))
    tools = HailerTools(ws, sandbox(ws, client))
    assert tools.marimo_execute("print(42)") == "42"
    assert client.executed == ["print(42)"]

    client = FakeClient(result=ExecResult(success=False, stderr="NameError: x"))
    text = HailerTools(ws, sandbox(ws, client)).marimo_execute("x")
    assert text.startswith("Execution failed.")
    assert "NameError: x" in text


def test_marimo_execute_errors_become_text(ws):
    text = HailerTools(ws, sandbox(ws)).marimo_execute("1")
    assert text.startswith("ERROR: Marimo is not running") and "uvx hailer again" in text

    client = FakeClient(raise_on_execute=NoSessionError("Notebook not open", "Open http://127.0.0.1:2718/?file=/work/notebooks/analysis.py"))
    text = HailerTools(ws, sandbox(ws, client)).marimo_execute("1")
    assert text.startswith("ERROR: Notebook not open") and "Open http://" in text

    client = FakeClient(raise_on_execute=RuntimeError("boom"))
    assert HailerTools(ws, sandbox(ws, client)).marimo_execute("1").startswith("ERROR: unexpected RuntimeError: boom")


def test_marimo_execute_truncates_large_output(tmp_path):
    big = "".join(f"row {i}\n" for i in range(5000))
    config = make_config(tmp_path, max_tool_output_chars=1000)
    text = HailerTools(config, sandbox(config, FakeClient(result=ExecResult(success=True, stdout=big)))).marimo_execute("print(df)")
    assert len(text) <= 1020
    assert text.startswith("row 0\n")
    assert text.rstrip().endswith("row 4999")
    assert "truncated" in text


def test_kernel_tools_use_the_active_notebook_from_the_state_file(ws):
    client = FakeClient(open_names={"analysis.py", "other.py"}, result=ExecResult(success=True, stdout="ok\n"))
    tools = HailerTools(ws, sandbox(ws, client))

    assert tools.marimo_execute("1") == "ok"
    assert client.executed_notebooks[-1] == "analysis.py"  # no state file -> the configured notebook

    notebooks.save_active_notebook(ws, "other.py")  # switched by the CLI or an earlier tool call
    assert tools.marimo_execute("2") == "ok"
    assert client.executed_notebooks[-1] == "other.py"
    tools.notebook_cells()
    assert client.executed_notebooks[-1] == "other.py"


def test_the_client_is_made_by_the_sandbox_for_the_active_notebook_without_the_token_in_links(ws):
    made: list[dict] = []
    client = FakeClient()
    box = sandbox(ws, client, token="kernel-token")
    box.client_factory = lambda **kw: (made.append(kw) or client)
    HailerTools(ws, box).marimo_execute("1")
    notebooks.save_active_notebook(ws, "other.py")
    HailerTools(ws, box).marimo_execute("2")
    assert made == [
        {"notebook": "analysis.py", "token_in_links": False},
        {"notebook": "other.py", "token_in_links": False},
    ], "hints from these clients reach the model"


def test_without_a_kernel_every_kernel_tool_says_so(ws):
    """The tools never look for a server: without the chat's own kernel they report it."""
    tools = HailerTools(ws)
    for text in (tools.marimo_execute("1"), tools.notebook_list(), tools.notebook_open("other"), tools.list_periods()):
        assert text.startswith("ERROR: marimo is not running: this session has no kernel"), text
        assert "uvx hailer" in text
    assert tools.marimo_status().startswith("marimo: not running\nactive notebook: analysis.py")


def test_marimo_status_variants(ws):
    text = HailerTools(ws, sandbox(ws)).marimo_status()
    assert text.startswith("marimo: not running") and "uvx hailer again" in text
    assert "active notebook: analysis.py" in text

    text = HailerTools(ws, sandbox(ws, FakeClient(has_session=True))).marimo_status()
    assert "running at http://127.0.0.1:2718" in text
    assert "s1: analysis.py (active notebook)" in text
    assert "active notebook: analysis.py -> session s1 (ready)" in text

    text = HailerTools(ws, sandbox(ws, FakeClient(has_session=False))).marimo_status()
    assert "sessions: none" in text
    assert "NO session" in text and "http://127.0.0.1:2718" in text


def test_marimo_status_names_active_notebook_and_other_sessions(ws):
    notebooks.save_active_notebook(ws, "other.py")
    client = FakeClient(open_names={"analysis.py", "other.py"}, outside=("/tmp/elsewhere/x.py",))
    text = HailerTools(ws, sandbox(ws, client)).marimo_status()
    assert "active notebook: other.py -> session" in text and "(ready)" in text
    assert "analysis.py (another notebook in the notebooks folder" in text
    assert "/tmp/elsewhere/x.py (outside the notebooks folder)" in text
    assert "other.py (active notebook)" in text


def test_notebook_cells_filters(ws):
    listing = "Hbol\timports\tidle\terrors=0\timport marimo as mo\nAbcd\tload\tidle\terrors=1\tdf = load_periods(files)\n"
    client = FakeClient(result=ExecResult(success=True, stdout=listing))
    tools = HailerTools(ws, sandbox(ws, client))
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


def checked(config: HailerConfig, client) -> HailerTools:
    """The tools on ``config`` with a sandbox whose clients are ``client``."""
    return HailerTools(config, sandbox(config, client))


def test_marimo_execute_checks_the_cells_it_changed(tmp_path):
    new_cell = "revenue = sales.group_by('region').agg(pl.col('revenue').sum())\nrevenue.sortt('revenue')"
    client = NotebookClient(
        {**STARTER_CELLS, "old": "print(missing_before)"},
        {WRITE_CELL: {"revenue": new_cell}},
        result=ExecResult(success=True, stdout="created cell\n"),
    )
    text = checked(make_config(tmp_path), client).marimo_execute(WRITE_CELL)
    assert text == (
        "created cell\n\n"
        "Checks (hailer, ruff, ty) on the 1 changed cell: 1 finding.\n"
        "- cell revenue (revenue), line 2: ty unresolved-attribute: Object of type `DataFrame` has no attribute `sortt`\n"
        "    revenue.sortt('revenue')"
    ), "the pre-existing problem in cell 'old' is not the agent's change"
    before, agent, after = client.executed
    assert "format_on_save" in before and agent == WRITE_CELL and "format_on_save" not in after


def test_marimo_execute_reports_a_clean_change_and_a_failed_run(tmp_path):
    client = NotebookClient(STARTER_CELLS, {WRITE_CELL: {"n": "rows = sales.height"}}, result=ExecResult(success=False, stderr="boom"))
    text = checked(make_config(tmp_path), client).marimo_execute(WRITE_CELL)
    assert text.startswith("Execution failed.\n[stderr]\nboom")
    assert text.endswith("\n\nChecks (hailer, ruff, ty) on the 1 changed cell: no findings.")


def test_marimo_execute_without_cell_writes_runs_only_the_code(tmp_path):
    client = NotebookClient(STARTER_CELLS)
    tools = checked(make_config(tmp_path), client)
    assert tools.marimo_execute("print(sales.shape)") == "hello"
    assert client.executed == ["print(sales.shape)"]
    reading = "import marimo._code_mode as cm\nprint(cm.get_context().cells[0].code)"
    assert tools.marimo_execute(reading) == "hello" and client.executed[-1] == reading and len(client.executed) == 2


def test_marimo_execute_says_nothing_when_no_cell_changed(tmp_path):
    client = NotebookClient(STARTER_CELLS)
    assert checked(make_config(tmp_path), client).marimo_execute(WRITE_CELL) == "hello"
    assert len(client.executed) == 3


def test_checks_config_switches_the_parts_off(tmp_path):
    writes = {WRITE_CELL: {"n": "print(undefined_x)"}}
    off = CodeChecksConfig(lint=False, typecheck=False, hardcoded=False, format=False)
    client = NotebookClient(STARTER_CELLS, writes)
    assert checked(make_config(tmp_path, checks=off), client).marimo_execute(WRITE_CELL) == "hello"
    assert client.executed == [WRITE_CELL]

    format_only = CodeChecksConfig(lint=False, typecheck=False, hardcoded=False, format=True)
    client = NotebookClient(STARTER_CELLS, writes)
    assert checked(make_config(tmp_path, checks=format_only), client).marimo_execute(WRITE_CELL) == "hello"
    assert len(client.executed) == 2 and "format_on_save" in client.executed[0], "formatting is switched on, nothing checked"

    lint_only = CodeChecksConfig(typecheck=False, hardcoded=False, format=False)
    client = NotebookClient(STARTER_CELLS, writes)
    text = checked(make_config(tmp_path, checks=lint_only), client).marimo_execute(WRITE_CELL)
    assert "Checks (ruff) on the 1 changed cell: 1 finding." in text and "ruff F821" in text
    assert "format_on_save" not in client.executed[0]

    hardcoded_only = CodeChecksConfig(lint=False, typecheck=False, format=False)
    client = NotebookClient(STARTER_CELLS, {WRITE_CELL: {"summary": 'mo.md("Revenue rose 5.6% in March")'}})
    text = checked(make_config(tmp_path, checks=hardcoded_only), client).marimo_execute(WRITE_CELL)
    assert text == (
        "hello\n\n"
        "Checks (hailer) on the 1 changed cell: 1 finding.\n"
        '- cell summary (summary), line 1: hailer hardcoded-value: "5.6%", "March" typed into the text; '
        "format values from variables so the text follows the data\n"
        '    mo.md("Revenue rose 5.6% in March")'
    )
    assert "format_on_save" not in client.executed[0]


def test_marimo_execute_runs_the_code_when_the_cells_cannot_be_read(tmp_path):
    class Unreadable(NotebookClient):
        def execute(self, code, **kw):
            if code_checks.SNAPSHOT_MARKER in code:
                self.executed.append(code)
                return ExecResult(success=False, stderr="AttributeError: no _document")
            return super().execute(code, **kw)

    client = Unreadable(STARTER_CELLS, {WRITE_CELL: {"n": "print(undefined_x)"}})
    assert checked(make_config(tmp_path), client).marimo_execute(WRITE_CELL) == "hello"
    assert WRITE_CELL in client.executed


def test_the_findings_survive_the_output_cap(tmp_path):
    big = "".join(f"row {i}\n" for i in range(5000))
    client = NotebookClient(STARTER_CELLS, {WRITE_CELL: {"n": "print(undefined_x)"}}, result=ExecResult(success=True, stdout=big))
    text = checked(make_config(tmp_path, max_tool_output_chars=1000), client).marimo_execute(WRITE_CELL)
    assert len(text) <= 1000 and text.startswith("row 0\n") and "truncated" in text
    assert text.endswith("- cell n (n), line 1: ty unresolved-reference: Name `undefined_x` used when not defined\n    print(undefined_x)")


def test_notebook_check_covers_the_whole_notebook(tmp_path):
    client = NotebookClient({**STARTER_CELLS, "empty": " ", "old": "print(missing_before)"})
    tools = checked(make_config(tmp_path), client)
    assert tools.notebook_check() == (
        "Checks (hailer, ruff, ty) on the notebook's 3 cells: 1 finding.\n"
        "- cell old (old), line 1: ty unresolved-reference: Name `missing_before` used when not defined\n"
        "    print(missing_before)"
    )
    assert "format_on_save" not in client.executed[0]

    assert checked(make_config(tmp_path), NotebookClient({"e": ""})).notebook_check() == "The notebook has no code to check."
    off = make_config(tmp_path, checks=CodeChecksConfig(lint=False, typecheck=False, hardcoded=False))
    assert "switched off" in checked(off, client).notebook_check()
    typed = NotebookClient({**STARTER_CELLS, "summary": 'mo.md(f"Revenue {total:.0f} in 2025")'})
    only_hardcoded = make_config(tmp_path, checks=CodeChecksConfig(lint=False, typecheck=False))
    assert checked(only_hardcoded, typed).notebook_check().startswith(
        "Checks (hailer) on the notebook's 3 cells: 1 finding.\n"
        '- cell summary (summary), line 1: hailer hardcoded-value: "2025" typed into the text'
    )
    unreadable = FakeClient(result=ExecResult(success=False, stderr="NameError: x"))
    assert checked(make_config(tmp_path), unreadable).notebook_check().startswith("Could not read the notebook's cells.")
    assert HailerTools(make_config(tmp_path)).notebook_check().startswith("ERROR: marimo is not running")


# --------------------------------------------------------------------------- #
# HailerTools: notebook lifecycle tools
# --------------------------------------------------------------------------- #


def test_notebook_list_empty_folder(tmp_path):
    config = make_config(tmp_path)
    config.notebooks_root.mkdir()
    text = HailerTools(config, sandbox(config, docker=True)).notebook_list()
    assert text.startswith("No notebooks in /work/notebooks yet") and "notebook_create" in text


def test_notebook_list_marks_active_and_open(ws, tmp_path):
    (tmp_path / "notebooks" / "notes.txt").write_text("not a notebook", encoding="utf-8")
    write_notebook(tmp_path / "notebooks" / "__marimo__" / "cache.py")
    write_notebook(tmp_path / "notebooks" / "q3" / "review.py")
    notebooks.save_active_notebook(ws, "other.py")
    text = HailerTools(ws, sandbox(ws, FakeClient(open_names={"analysis.py"}), docker=True)).notebook_list()
    lines = text.splitlines()
    assert lines[0].startswith("Notebooks in /work/notebooks (")
    analysis_line = next(ln for ln in lines if "analysis.py" in ln)
    other_line = next(ln for ln in lines if "other.py" in ln)
    assert analysis_line.startswith("  - analysis.py  modified 20")
    assert "[open]" in analysis_line and "[active]" not in analysis_line
    assert "[active]" in other_line and "[open]" not in other_line
    assert "B" in analysis_line and any(ln.startswith("  - q3/review.py") for ln in lines)
    assert "notes.txt" not in text and "__marimo__" not in text
    assert "not reachable" not in text


def test_notebook_list_when_marimo_is_down(ws):
    text = HailerTools(ws, sandbox(ws)).notebook_list()
    entries = [ln for ln in text.splitlines() if ln.startswith("  - ")]
    assert len(entries) == 2
    assert any("analysis.py" in ln and "[active]" in ln for ln in entries)
    assert not any("[open]" in ln for ln in entries)
    assert "marimo is not reachable" in text
    # The list also carries the error and the launch hint, like the other tools when marimo is down.
    assert "connection refused" in text and "uvx hailer again" in text


def test_notebook_create_ready(ws):
    client = FakeClient(open_names={"analysis.py"})
    opener = Opener(client)
    box = sandbox(ws, client, docker=True)
    text = HailerTools(ws, box, open_url=opener, session_wait_sec=0).notebook_create("Q2 Churn")
    created = ws.notebooks_root / "q2_churn.py"
    assert box.writes == ["q2_churn.py"], "written through the sandbox"
    assert created.is_file() and "# Q2 Churn" in created.read_text(encoding="utf-8")
    assert notebooks.load_active_notebook(ws) == "q2_churn.py"
    assert opener.urls == ["http://127.0.0.1:2718/?file=/work/notebooks/q2_churn.py&view-as=present"]
    assert text.startswith("Created q2_churn.py from the starter template; it is now the active notebook.")
    assert "kernel session ready (s" in text
    assert "starter template already defines mo, pl, duckdb" in text


def test_notebook_create_without_session_reports_url(ws):
    client = FakeClient(open_names={"analysis.py"})
    opener = Opener(client, grant_session=False)
    text = HailerTools(ws, sandbox(ws, client), open_url=opener, session_wait_sec=0).notebook_create("draft", template="empty")
    assert (ws.notebooks_root / "draft.py").is_file()
    assert "no kernel session appeared within 0 s" in text and "http://127.0.0.1:2718/?file=" in text
    assert "empty template defines nothing" in text
    assert "marimo_status" in text

    opener = Opener(client, succeed=False)
    text = HailerTools(ws, sandbox(ws, client), open_url=opener, session_wait_sec=0).notebook_create("second")
    assert "Could not open a browser" in text and "Ask the user to open http://" in text


def test_notebook_create_when_marimo_is_down_still_creates_and_switches(ws):
    opener = Opener()
    text = HailerTools(ws, sandbox(ws), open_url=opener, session_wait_sec=0).notebook_create("offline one")
    assert (ws.notebooks_root / "offline_one.py").is_file()
    assert notebooks.load_active_notebook(ws) == "offline_one.py"
    assert text.startswith("Created offline_one.py from the starter template; it is now the active notebook.")
    assert "marimo is not running" in text and "connection refused" in text and "uvx hailer again" in text
    assert not text.startswith("ERROR"), "a create that succeeded must not read as a failure"
    assert opener.urls == []


def test_notebook_create_errors(ws):
    opener = Opener()
    tools = HailerTools(ws, sandbox(ws), open_url=opener, session_wait_sec=0)
    text = tools.notebook_create("x", template="fancy")
    assert text.startswith("ERROR: unknown template 'fancy'") and "'starter'" in text
    text = tools.notebook_create("other")
    assert text.startswith("ERROR: other.py already exists") and "open the existing notebook" in text
    text = tools.notebook_create("!!!")
    assert text.startswith("ERROR:") and "not a usable notebook name" in text
    assert opener.urls == []
    assert notebooks.load_active_notebook(ws) == "analysis.py"
    assert (ws.notebooks_root / "other.py").read_text(encoding="utf-8") == NOTEBOOK_SOURCE, "never overwritten"


def test_notebook_open_with_session_skips_browser_and_lists_cells(ws):
    listing = "Hbol\timports\tidle\terrors=0\timport marimo as mo\n"
    client = FakeClient(open_names={"analysis.py", "other.py"}, result=ExecResult(success=True, stdout=listing))
    opener = Opener(client)
    text = HailerTools(ws, sandbox(ws, client), open_url=opener, session_wait_sec=0).notebook_open("other")
    assert text.startswith("other.py is now the active notebook.")
    assert "already has a kernel session" in text and "browser was not opened" in text
    assert opener.urls == []
    assert "Cells (id, name, status, errors, first line):" in text and "Hbol\timports" in text
    assert client.executed_notebooks[-1] == "other.py"
    assert notebooks.load_active_notebook(ws) == "other.py"


def test_notebook_open_opens_browser_when_not_open(ws):
    client = FakeClient(open_names={"analysis.py"}, result=ExecResult(success=True, stdout=""))
    opener = Opener(client)
    text = HailerTools(ws, sandbox(ws, client, docker=True), open_url=opener, session_wait_sec=0).notebook_open("notebooks/other.py")
    assert opener.urls == ["http://127.0.0.1:2718/?file=/work/notebooks/other.py&view-as=present"]
    assert "kernel session ready" in text and "(no cells)" in text

    # no session appears -> URL and instructions, no cell listing
    client = FakeClient(open_names={"analysis.py"})
    opener = Opener(client, grant_session=False)
    text = HailerTools(ws, sandbox(ws, client), open_url=opener, session_wait_sec=0).notebook_open("other.py")
    assert "no kernel session appeared" in text and "Cells" not in text
    assert notebooks.load_active_notebook(ws) == "other.py"


def test_notebook_open_errors(ws, tmp_path):
    opener = Opener()
    tools = HailerTools(ws, sandbox(ws, FakeClient()), open_url=opener, session_wait_sec=0)
    text = tools.notebook_open("missing")
    assert text.startswith("ERROR: No notebook named 'missing'") and "analysis" in text and "other" in text
    outside = write_notebook(tmp_path / "x.py")
    text = tools.notebook_open("../x.py")
    assert text.startswith("ERROR:") and "outside the notebooks folder" in text
    text = tools.notebook_open(str(outside))
    assert text.startswith("ERROR:") and "outside the notebooks folder" in text
    assert opener.urls == []
    assert notebooks.load_active_notebook(ws) == "analysis.py"


def test_notebook_close_with_and_without_session(ws):
    client = FakeClient(open_names={"analysis.py", "other.py"})
    tools = HailerTools(ws, sandbox(ws, client), session_wait_sec=0)
    text = tools.notebook_close("other")
    assert text.startswith("Closed the kernel session (s2) of other.py")
    assert "active notebook is still analysis.py" in text
    assert client.closed == ["s2"]
    text = tools.notebook_close("other")
    assert text == "other.py is not open (no kernel session), so there is nothing to close."

    text = tools.notebook_close()  # default: the active notebook
    assert text.startswith("Closed the kernel session (s1) of analysis.py")
    assert "stays the active notebook" in text and "notebook_open" in text
    assert client.closed == ["s2", "s1"]
    assert tools.notebook_close("nope").startswith("ERROR: No notebook named 'nope'")


# --------------------------------------------------------------------------- #
# Data, skills, web
# --------------------------------------------------------------------------- #


def test_list_periods_missing_and_empty(ws):
    tools = HailerTools(ws, sandbox(ws))
    assert tools.list_periods().startswith(f"ERROR: Data directory does not exist: {ws.data_dir}")
    ws.data_dir.mkdir()
    text = tools.list_periods("sales")
    assert "No period files found" in text and "'sales'" in text and "25-01 sales.parquet" in text
    assert "Other data files" not in text


def test_list_periods_names_other_data_files(ws):
    ws.data_dir.mkdir()
    (ws.data_dir / "customers.csv").write_text("id,name\n1,a\n", encoding="utf-8")
    (ws.data_dir / "Survey 2024.json").write_text("[]", encoding="utf-8")
    (ws.data_dir / "Sales.xlsx").touch()
    (ws.data_dir / "Budget.xlsb").touch()
    (ws.data_dir / "notes.txt").write_text("not data", encoding="utf-8")
    (ws.data_dir / ".hidden.csv").write_text("x", encoding="utf-8")
    (ws.data_dir / "sub").mkdir()
    text = HailerTools(ws, sandbox(ws)).list_periods()
    assert text.startswith("No period files found")
    assert text.splitlines()[-1] == (
        "Other data files (load them directly with Polars or DuckDB): "
        "Budget.xlsb, customers.csv, Sales.xlsx, Survey 2024.json"
    )


def test_list_periods_names_the_data_folder_as_a_docker_kernel_sees_it(ws):
    """The data folder is listed on the host, but the model writes code for the container."""
    tools = HailerTools(ws, sandbox(ws, docker=True))
    assert tools.list_periods() == "ERROR: Data directory does not exist: /work/data"
    ws.data_dir.mkdir()
    text = tools.list_periods()
    assert text.startswith("No period files found in /work/data.") and str(ws.data_dir) not in text
    local = HailerTools(ws, sandbox(ws)).list_periods()
    assert local.startswith(f"No period files found in {ws.data_dir}.")


def test_list_periods_with_files(ws):
    pl = pytest.importorskip("polars")
    ws.data_dir.mkdir()
    pl.DataFrame({"region": ["North"], "revenue": [1.0]}).write_parquet(ws.data_dir / "25-01 sales.parquet")
    pl.DataFrame({"region": ["South"], "revenue": [2.0], "discount": [0.1]}).write_parquet(ws.data_dir / "25-02 sales.parquet")
    pl.DataFrame({"region": ["West"], "revenue": [3]}).write_parquet(ws.data_dir / "25-03 sales.parquet")
    pl.DataFrame({"n": [1]}).write_parquet(ws.data_dir / "25-01 costs.parquet")
    (ws.data_dir / "targets.csv").write_text("region,target\nNorth,5\n", encoding="utf-8")
    text = HailerTools(ws, sandbox(ws, docker=True)).list_periods("sales")
    assert text.splitlines()[:4] == [
        "3 period file(s) [sales]: 2025-01 -> 2025-03",
        "Periods: 2025-01, 2025-02, 2025-03",
        "Common columns (2): region String, revenue Float64",
        "Only in some periods: discount (1 file(s), 2025-02)",
    ]
    assert "Type differs across periods: revenue: Float64 / Int64" in text
    assert text.splitlines()[-1] == "Other data files (load them directly with Polars or DuckDB): 25-01 costs.parquet, targets.csv"
    assert "25-01 sales.parquet" not in text.splitlines()[-1], "period files are described, not listed again"


def test_skill_and_page_tools(tmp_path):
    config = make_config(tmp_path, web=WebConfig(allowed_domains=("docs.pola.rs",)))
    skill_dir = config.skills_dir / "alpha"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: alpha\ndescription: d\n---\nBody", encoding="utf-8")
    (skill_dir / "reference").mkdir()
    (skill_dir / "reference" / "r.md").write_text("ref", encoding="utf-8")
    tools = HailerTools(config)
    assert "Body" in tools.load_skill("alpha") and "- reference/r.md" in tools.load_skill("alpha")
    assert tools.read_skill_file("alpha", "reference/r.md") == "ref"
    assert tools.read_skill_file("alpha", "../../x").startswith("ERROR:")
    assert tools.load_skill("nope").startswith("ERROR: No skill named 'nope'")
    denied = tools.fetch_page("https://example.com/")
    assert denied.startswith("ERROR: Web access to 'example.com' is not allowed")
    assert "docs.pola.rs" in denied
    assert tools.fetch_page("ftp://docs.pola.rs/").startswith("ERROR:")


def test_the_tools_never_touch_the_notebooks_or_data_folders():
    """Regression: every file the tools know about comes from the sandbox. hailer.tools names
    neither folder and calls no file-system API (a cloud kernel's files are not on this machine)."""
    source = Path(__import__("hailer.tools", fromlist=["x"]).__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_attrs = {"notebooks_root", "notebooks_dir", "data_dir", "notebook"}
    file_calls = {"open", "read_text", "read_bytes", "write_text", "write_bytes", "iterdir", "glob", "rglob", "walk", "stat", "exists", "is_file", "is_dir", "mkdir", "unlink", "scandir", "listdir"}
    seen_attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    called = {
        (node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", ""))
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert not seen_attrs & forbidden_attrs, seen_attrs & forbidden_attrs
    assert not called & file_calls, called & file_calls
    assert "import os" not in source and "shutil" not in source


# --------------------------------------------------------------------------- #
# LangChain registration
# --------------------------------------------------------------------------- #


def test_hailer_tools_registers_twelve_tools_with_descriptions_and_schemas(tmp_path):
    tools = hailer_tools(make_config(tmp_path))
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
    client = FakeClient(open_names={"analysis.py"}, result=ExecResult(success=True, stdout="ok"))
    opener = Opener(client)
    tools = {t.name: t for t in hailer_tools(ws, sandbox=sandbox(ws, client), open_url=opener, session_wait_sec=0)}

    assert tools["marimo_execute"].invoke({"code": "print('ok')"}) == "ok"
    text = tools["notebook_list"].invoke({})
    assert "analysis.py" in text and "[active]" in text
    assert tools["notebook_create"].invoke({"name": "via tool"}).startswith("Created via_tool.py")
    assert tools["notebook_open"].invoke({"notebook": "other"}).startswith("other.py is now the active notebook.")
    assert tools["fetch_page"].invoke({"url": "https://example.com"}).startswith("ERROR:")

    assert len(opener.urls) == 2  # create + open (other had no session)
    assert notebooks.load_active_notebook(ws) == "other.py"


def test_hailer_tools_pins_every_tool_to_the_sandbox(ws):
    box = sandbox(ws)
    status = next(t for t in hailer_tools(ws, sandbox=box) if t.name == "marimo_status")
    assert status.func.__wrapped__.__self__.sandbox is box


def test_async_path_runs_the_tool_on_a_daemon_thread(ws):
    """The agent awaits tools; a blocking kernel call must never be joined at shutdown."""
    seen: dict[str, object] = {}

    class RecordingClient(FakeClient):
        def execute(self, code, notebook=None, **kw):
            thread = threading.current_thread()
            seen.update(name=thread.name, daemon=thread.daemon, main=thread is threading.main_thread())
            return super().execute(code, notebook=notebook, **kw)

    client = RecordingClient(open_names={"analysis.py"}, result=ExecResult(success=True, stdout="42"))
    tools = {t.name: t for t in hailer_tools(ws, sandbox=sandbox(ws, client))}

    async def go():
        assert await tools["marimo_execute"].ainvoke({"code": "print(42)"}) == "42"
        assert (await tools["fetch_page"].ainvoke({"url": "https://example.com"})).startswith("ERROR:")

    asyncio.run(go())
    assert seen == {"name": "hailer-tool-marimo_execute", "daemon": True, "main": False}


def test_cancelling_a_turn_does_not_wait_for_a_blocked_tool(ws):
    release = threading.Event()
    finished = threading.Event()

    class BlockingClient(FakeClient):
        def execute(self, code, notebook=None, **kw):
            release.wait(10)
            finished.set()
            return super().execute(code, notebook=notebook, **kw)

    client = BlockingClient(open_names={"analysis.py"}, result=ExecResult(success=True, stdout="late"))
    tools = {t.name: t for t in hailer_tools(ws, sandbox=sandbox(ws, client))}

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


def test_rich_output_is_replaced_by_placeholder(ws):
    html = "<table><tr><td>1</td></tr></table>" * 50
    client = FakeClient(result=ExecResult(success=True, stdout="shape: (5, 3)\n", output=html, mimetype="text/html"))
    text = HailerTools(ws, sandbox(ws, client)).marimo_execute("df.head()")
    assert "shape: (5, 3)" in text
    assert "<table>" not in text
    assert "text/html output omitted" in text and "print()" in text
    # stderr survives alongside the placeholder
    client = FakeClient(result=ExecResult(success=True, stderr="warn\n", output="{}", mimetype="application/json"))
    text = HailerTools(ws, sandbox(ws, client)).marimo_execute("x")
    assert "application/json output omitted" in text and "[stderr]" in text
    # plain text output still passes through unchanged
    client = FakeClient(result=ExecResult(success=True, output="42", mimetype="text/plain"))
    assert HailerTools(ws, sandbox(ws, client)).marimo_execute("42") == "42"


# --------------------------------------------------------------------------- #
# The server token never reaches the model
# --------------------------------------------------------------------------- #

SECRET = "kernel-token-never-for-the-model"


def test_tool_results_carry_token_free_urls_and_the_browser_gets_the_signed_one(ws):
    client = FakeClient(open_names={"analysis.py"})
    opener = Opener(client, grant_session=False)
    tools = HailerTools(ws, sandbox(ws, client, token=SECRET), open_url=opener, session_wait_sec=0)
    texts = [tools.notebook_create("draft"), tools.notebook_open("other"), tools.marimo_status()]
    opener_fails = HailerTools(ws, sandbox(ws, client, token=SECRET), open_url=Opener(client, succeed=False), session_wait_sec=0)
    texts.append(opener_fails.notebook_open("other"))
    for text in texts:
        assert SECRET not in text and "access_token" not in text, text
    assert all("/notebook in the chat" in text for text in texts), "the user can get the signed-in link there"
    assert "http://127.0.0.1:2718/?file=" in texts[0] and "http://127.0.0.1:2718/?file=" in texts[3]
    assert opener.urls and all(url.endswith(f"&access_token={SECRET}") for url in opener.urls)


def test_no_session_errors_point_at_the_notebook_command(ws):
    client = FakeClient(raise_on_execute=NoSessionError("The notebook is not open in a browser.", "Open http://127.0.0.1:2718/?file=x in your browser."))
    text = HailerTools(ws, sandbox(ws, client, token=SECRET)).marimo_execute("1")
    assert text.startswith("ERROR: The notebook is not open") and "/notebook in the chat" in text and SECRET not in text


def test_tools_send_kernel_paths_for_a_docker_kernel(ws):
    client = FakeClient(open_names={"analysis.py"})
    opener = Opener(client, grant_session=False)
    text = HailerTools(ws, sandbox(ws, client, docker=True, token=SECRET), open_url=opener, session_wait_sec=0).notebook_open("other")
    assert opener.urls == [f"http://127.0.0.1:2718/?file=/work/notebooks/other.py&view-as=present&access_token={SECRET}"]
    assert "?file=/work/notebooks/other.py&view-as=present" in text and SECRET not in text and str(ws.workspace) not in text


# --------------------------------------------------------------------------- #
# The kernel went away: the sandbox is held, the first request fails
# --------------------------------------------------------------------------- #


def test_notebook_open_when_the_kernel_went_away(ws):
    opener = Opener()
    text = HailerTools(ws, sandbox(ws), open_url=opener, session_wait_sec=0).notebook_open("other")
    assert text.startswith("other.py is now the active notebook.")
    assert "marimo is not running" in text and "Cells" not in text
    assert notebooks.load_active_notebook(ws) == "other.py"
    assert opener.urls == []


def test_marimo_status_when_the_kernel_went_away(ws):
    notebooks.save_active_notebook(ws, "other.py")
    text = HailerTools(ws, sandbox(ws)).marimo_status()
    assert text.startswith("marimo: not running")
    assert "active notebook: other.py" in text
    assert "connection refused" in text and "uvx hailer again" in text


def test_notebook_close_when_the_kernel_went_away(ws):
    client = DownClient()
    text = HailerTools(ws, sandbox(ws, client)).notebook_close()
    assert text.startswith("marimo is not running, so analysis.py has no kernel session to close.")
    assert "connection refused" in text and client.closed == []


def test_session_wait_that_loses_marimo_is_reported_not_raised(ws):
    class FlakyClient(FakeClient):
        def __init__(self):
            super().__init__(open_names={"analysis.py"})
            self.calls = 0

        def resolve_session(self, notebook):
            self.calls += 1
            if self.calls > 1:  # the first call finds no session; the wait then loses the server
                raise MarimoUnavailableError("Marimo is not running at http://127.0.0.1:2718 (timed out).", hint="restart it")
            return super().resolve_session(notebook)

    client = FlakyClient()
    opener = Opener(client, grant_session=False)
    text = HailerTools(ws, sandbox(ws, client), open_url=opener, session_wait_sec=0).notebook_open("other")
    assert text.startswith("other.py is now the active notebook.")
    assert "timed out" in text and "restart it" in text
    assert len(opener.urls) == 1


# --------------------------------------------------------------------------- #
# The kernel the model works in
# --------------------------------------------------------------------------- #


def test_marimo_status_names_the_kernel_in_use_and_its_paths(ws):
    client = FakeClient(open_names={"analysis.py"})
    text = HailerTools(ws, sandbox(ws, client, docker=True, network=True)).marimo_status()
    assert "kernel: docker (hailer-kernel " in text and "network on: the internet and this machine" in text
    assert "kernel paths: notebooks folder /work/notebooks, data folder /work/data; use these in code" in text
    local = HailerTools(ws, sandbox(ws, client)).marimo_status()
    assert "kernel: unsafe-local (runs as you; not isolated)" in local
    assert f"data folder {ws.data_dir}" in local and ws.notebooks_root.as_posix() in local


def test_notebook_open_and_close_accept_the_kernels_paths(ws):
    """The model sees /work/notebooks/... in a docker kernel and may pass such a path back."""
    client = FakeClient(open_names={"analysis.py", "other.py"})
    tools = HailerTools(ws, sandbox(ws, client, docker=True), open_url=Opener(client), session_wait_sec=0)
    assert tools.notebook_open("/work/notebooks/other.py").startswith("other.py is now the active notebook.")
    text = tools.notebook_close("/work/notebooks/other.py")
    assert text.startswith("Closed the kernel session") and "other.py" in text, text
    assert tools.notebook_open("/work/data/sales.py").startswith("ERROR:"), "a kernel path outside the notebooks folder is refused"
    local = HailerTools(ws, sandbox(ws, FakeClient(open_names={"analysis.py", "other.py"})), open_url=Opener(client), session_wait_sec=0)
    assert local.notebook_open(ws.notebooks_root.as_posix() + "/other.py").startswith("other.py is now the active notebook.")


# --------------------------------------------------------------------------- #
# End to end: a real sandbox over marimo's file API
# --------------------------------------------------------------------------- #


def test_the_tools_work_through_marimos_file_api(tmp_path):
    """No FolderSandbox: the tools on the shared MarimoSandbox, whose notebook files go through the
    (fake) marimo server's file endpoints, token-checked, for a kernel in a container."""
    from fake_marimo import serving
    from hailer.sandbox import MarimoSandbox

    config = make_config(tmp_path)
    write_notebook(config.notebook)
    with serving(token=SECRET) as srv:
        srv.serve_files(config.notebooks_root, "/work/notebooks")
        box = MarimoSandbox(
            MarimoServer(url=srv.url, token=SECRET, runtime="docker"),
            notebooks_path="/work/notebooks", data_path="/work/data", data_dir=config.data_dir,
        )  # fmt: skip
        opened: list[str] = []

        def open_tab(url: str) -> bool:
            opened.append(url)
            key = parse_qs(urlsplit(url).query)["file"][0]
            srv.sessions["s9"] = {"filename": key, "path": key}
            return True

        tools = HailerTools(config, box, open_url=open_tab, session_wait_sec=2)
        text = tools.notebook_create("Q3 review")
        assert text.startswith("Created q3_review.py"), text
        assert "kernel session ready (s9)" in text
        assert "# Q3 review" in (config.notebooks_root / "q3_review.py").read_text(encoding="utf-8")
        listing = tools.notebook_list()
        assert "  - q3_review.py" in listing and "[active] [open]" in listing and "  - analysis.py" in listing
        assert tools.notebook_open("/work/notebooks/analysis.py").startswith("analysis.py is now the active notebook.")
        assert tools.notebook_create("q3 review").startswith("ERROR: q3_review.py already exists")
    assert all(url.startswith(f"{srv.url}/?file=/work/notebooks/") for url in opened)
    files = [r for r in srv.requests if r["path"].startswith("/api/files/")]
    assert files and all(r["headers"]["Authorization"] == f"Bearer {SECRET}" for r in files)


# --------------------------------------------------------------------------- #
# One-line outcomes (hailer.tool_outcomes) of the replies the tools above write
# --------------------------------------------------------------------------- #


def outcome(name: str, args: dict, reply: str) -> tuple[bool, str]:
    """tool_outcome of a reply a tool wrote, with the status the tool put beside it, checked for what
    every summary must be: one short line of printable characters."""
    assert isinstance(reply, ToolReply), "every tool reply carries its status"
    ok, summary = tool_outcome(name, args, str(reply), reply.status)
    assert isinstance(ok, bool) and isinstance(summary, str)
    assert len(summary) <= SUMMARY_LIMIT and summary == " ".join(summary.split()), repr(summary)
    assert all(ch.isprintable() for ch in summary), repr(summary)
    return ok, summary


class CodeModeClient(FakeClient):
    """A FakeClient whose kernel runs Hailer's own read-only scratchpad code (the cell listing and the
    check snapshot) for real, in this process, with marimo's code-mode context replaced by one that
    holds ``cells`` (``id``, ``code`` and optionally ``name``, ``status``, ``errors``, and ``raises``:
    the exception its last run raised, in the kernel's graph). Any other code gets ``result`` and
    applies ``writes`` (code -> cells it adds or replaces)."""

    def __init__(self, cells: list[dict], writes: dict[str, list[dict]] | None = None, **kw):
        super().__init__(**kw)
        self.cells = [dict(c) for c in cells]
        self.writes = writes or {}

    def execute(self, code, **kw):
        if code == LIST_CELLS_CODE or code_checks.SNAPSHOT_MARKER in code:
            self.executed.append(code)
            return self._run_here(code)
        result = super().execute(code, **kw)
        for cell in self.writes.get(code, []):
            self.cells = [c for c in self.cells if c["id"] != cell["id"]] + [cell]
        return result

    def _run_here(self, code: str) -> ExecResult:
        import marimo._code_mode as cm
        from marimo._code_mode._context import CellStatusType

        views = [
            SimpleNamespace(
                id=c["id"], name=c.get("name", "_"), code=c["code"], status=CellStatusType(c.get("status", "idle")), errors=[object()] * c.get("errors", 0)
            )
            for c in self.cells
        ]
        graph = SimpleNamespace(cells={c["id"]: SimpleNamespace(exception=c.get("raises")) for c in self.cells})
        context = SimpleNamespace(cells=views, _document=SimpleNamespace(cells=views), _kernel=SimpleNamespace(user_config={"save": {}}), graph=graph)
        out = io.StringIO()
        with mock.patch.object(cm, "get_context", lambda: context), contextlib.redirect_stdout(out):
            exec(compile(code, "<scratchpad>", "exec"), {})
        return ExecResult(success=True, stdout=out.getvalue())


def code_mode_says(ops: list, *, names: dict[str, str], ran: set[str] = frozenset(), errored: set[str] = frozenset(), packages: tuple[str, ...] = ()) -> str:
    """What marimo's code mode prints once it has applied ``ops`` (its own ``_print_summary``, on a
    stand-in context: ``names`` are the cells' names, ``errored`` the ones whose run raised)."""
    from marimo._code_mode._context import AsyncCodeModeContext
    from marimo._code_mode._packages import _AddPackage

    documents = {cid: SimpleNamespace(name=name) for cid, name in names.items()}
    graph = SimpleNamespace(cells={cid: SimpleNamespace(exception=ZeroDivisionError() if cid in errored else None) for cid in names})
    context = SimpleNamespace(_document=SimpleNamespace(get=documents.get), graph=graph)
    context._cell_label = lambda cid: AsyncCodeModeContext._cell_label(context, cid)
    context._cell_errored = lambda cid: AsyncCodeModeContext._cell_errored(context, cid)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        AsyncCodeModeContext._print_summary(context, ops, [_AddPackage(p) for p in packages], [], set(ran))
    return out.getvalue()


def raised_by(fn) -> tuple[Exception, str]:
    """The exception ``fn`` raises and its traceback text."""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - the point is the exception
        return exc, traceback.format_exc()
    raise AssertionError("expected an exception")


def traceback_of(fn) -> str:
    return raised_by(fn)[1]


STARTER = [{"id": "imp", "code": "import marimo as mo\nimport polars as pl"}, {"id": "load", "name": "load", "code": "sales = pl.read_csv('sales.csv')"}]
CHECKS_OFF = CodeChecksConfig(lint=False, typecheck=False, hardcoded=False, format=False)


def test_tool_replies_carry_a_status_and_hand_it_to_langchain_as_the_artifact(ws):
    """The model reads the same text as before, byte for byte; the status rides beside it."""
    from langchain_core.messages import ToolMessage

    client = FakeClient(open_names={"analysis.py"}, result=ExecResult(success=False, stderr="boom\n"))
    direct = HailerTools(ws, sandbox(ws, client))
    tools = {t.name: t for t in hailer_tools(ws, sandbox=sandbox(ws, client))}
    for name, args in (("marimo_execute", {"code": "1/0"}), ("notebook_open", {"notebook": "missing"}), ("notebook_list", {})):
        expected = getattr(direct, name)(**args)
        message = tools[name].invoke({"type": "tool_call", "name": name, "args": args, "id": "c1"})
        assert isinstance(message, ToolMessage) and type(message.content) is str and message.content == str(expected)
        assert message.artifact == expected.status
        assert tools[name].invoke(args) == expected, "invoked without a tool call: the text only"
    assert direct.marimo_execute("1/0").status == {"ok": False, "error": "execution_failed"}
    assert direct.notebook_open("missing").status == {"ok": False, "error": "NotebookNotFoundError"}
    assert direct.notebook_list().status == {"ok": True}

    async def through_the_daemon_thread() -> ToolMessage:
        return await tools["marimo_execute"].ainvoke({"type": "tool_call", "name": "marimo_execute", "args": {"code": "1/0"}, "id": "c2"})

    assert asyncio.run(through_the_daemon_thread()).artifact == {"ok": False, "error": "execution_failed"}


def test_what_the_users_code_prints_never_decides_the_outcome(ws, tmp_path):
    def run(stdout: str, code: str = "print(x)") -> tuple[bool, str]:
        client = FakeClient(result=ExecResult(success=True, stdout=stdout))
        return outcome("marimo_execute", {"code": code}, HailerTools(ws, sandbox(ws, client)).marimo_execute(code))

    assert run("ERROR: customer ACME owes 1,234,567 EUR\n") == (True, "")
    assert run("Execution failed.\nZeroDivisionError: in the data\n") == (True, "")
    assert run("created and ran cell 'x' (ACME_owes_1234567) (error)\n") == (True, ""), "not code mode: its lines are not read"
    assert run("Checks (ruff, ty) on the 1 changed cell: 3 findings.\n") == (True, ""), "no checks ran"
    config = make_config(tmp_path)
    skill = config.skills_dir / "alpha"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: alpha\ndescription: d\n---\nBody", encoding="utf-8")
    (skill / "notes.md").write_text("ERROR: marimo is not running at http://x/?access_token=t\n", encoding="utf-8")
    assert outcome("read_skill_file", {"name": "alpha", "path": "notes.md"}, HailerTools(config).read_skill_file("alpha", "notes.md")) == (True, "1 line")


def test_outcome_of_code_that_adds_a_cell(tmp_path):
    from marimo._ast.cell import CellConfig
    from marimo._code_mode._plan import _AddOp

    cell = {"id": "Hbol", "name": "growth_by_region", "code": "growth = sales.height"}
    says = code_mode_says([_AddOp("Hbol", cell["code"], CellConfig(), name="growth_by_region")], names={"Hbol": "growth_by_region"}, ran={"Hbol"})
    assert says == "created and ran cell 'Hbol' (growth_by_region)\n", "marimo's words"
    client = CodeModeClient(STARTER, {WRITE_CELL: [cell]}, result=ExecResult(success=True, stdout=says))
    reply = checked(make_config(tmp_path), client).marimo_execute(WRITE_CELL)
    assert reply.endswith("Checks (hailer, ruff, ty) on the 1 changed cell: no findings.")
    assert reply.status == {"ok": True, "findings": 0, "checked": ["hailer", "ruff", "ty"], "skipped": [], "cells": ["growth_by_region"]}
    assert outcome("marimo_execute", {"code": WRITE_CELL}, reply) == (True, "growth_by_region, checks clean")
    unnamed = CodeModeClient(STARTER, {WRITE_CELL: [{"id": "Hbol", "code": "growth = sales.height"}]}, result=ExecResult(success=True))
    assert outcome("marimo_execute", {"code": WRITE_CELL}, checked(make_config(tmp_path), unnamed).marimo_execute(WRITE_CELL)) == (True, "1 cell, checks clean")
    # The cells were compared: a code-mode line claiming an error decides nothing.
    fake = "created and ran cell 'Hbol' (growth_by_region) (error)\nZeroDivisionError: printed\n"
    spoof = CodeModeClient(STARTER, {WRITE_CELL: [cell]}, result=ExecResult(success=True, stdout=fake))
    assert outcome("marimo_execute", {"code": WRITE_CELL}, checked(make_config(tmp_path), spoof).marimo_execute(WRITE_CELL)) == (True, "growth_by_region, checks clean")


def test_a_cell_the_code_edited_that_raised_fails_the_call(tmp_path):
    from marimo._code_mode._plan import _UpdateOp

    pl = pytest.importorskip("polars")
    error, trace = raised_by(lambda: pl.DataFrame().sortt)
    edited = {"id": "load", "name": "load", "code": "sales = pl.read_csv('sales.csv')\nsales.sortt('region')", "raises": error}
    says = code_mode_says([_UpdateOp("load", code=edited["code"])], names={"load": "load"}, ran={"load"}, errored={"load"})
    assert says == "\nedited code of cell 'load' (load) and ran (error)\n"
    client = CodeModeClient(STARTER, {WRITE_CELL: [edited]}, result=ExecResult(success=True, stdout=says, stderr=trace))
    reply = checked(make_config(tmp_path), client).marimo_execute(WRITE_CELL)
    assert "1 finding." in reply and not reply.startswith("Execution failed."), "the model reads what it always did"
    assert reply.status["ok"] is False and reply.status["error"] == "cell_error" and reply.status["raised"] == ["AttributeError"]
    assert outcome("marimo_execute", {"code": WRITE_CELL}, reply) == (False, "load, AttributeError, 1 finding")
    # The kernel's word, not the text: the exception is named even when nothing was printed.
    quiet = CodeModeClient(STARTER, {WRITE_CELL: [{**edited, "raises": KeyError("region")}]}, result=ExecResult(success=True))
    assert outcome("marimo_execute", {"code": WRITE_CELL}, checked(make_config(tmp_path), quiet).marimo_execute(WRITE_CELL)) == (False, "load, KeyError, 1 finding")
    # A cell that raised before and was not changed by this call does not fail it.
    old = CodeModeClient([*STARTER, {"id": "old", "code": "1 / 0", "raises": ZeroDivisionError()}], {WRITE_CELL: [{"id": "n", "name": "n", "code": "n = 1"}]})
    assert outcome("marimo_execute", {"code": WRITE_CELL}, checked(make_config(tmp_path), old).marimo_execute(WRITE_CELL)) == (True, "n, checks clean")


def test_outcome_of_code_mode_lines_without_checks(tmp_path):
    from marimo._ast.cell import CellConfig
    from marimo._code_mode._plan import _AddOp, _DeleteOp

    def run(says: str, stderr: str = "") -> tuple[bool, str]:
        client = CodeModeClient(STARTER, result=ExecResult(success=True, stdout=says, stderr=stderr))
        return outcome("marimo_execute", {"code": WRITE_CELL}, checked(make_config(tmp_path, checks=CHECKS_OFF), client).marimo_execute(WRITE_CELL))

    ops = [_AddOp("a1", "t = 1", CellConfig(), name="totals"), _AddOp("b2", "2", CellConfig()), _DeleteOp("c3")]
    says = code_mode_says(ops, names={"a1": "totals", "b2": "_", "c3": "scratch"}, packages=("altair",))
    assert says.splitlines() == ["installed altair", "created cell 'a1' (totals)", "created cell 'b2' (_)", "deleted cell 'c3' (scratch)"]
    assert run(says) == (True, "2 cells, deleted scratch, installed altair")
    named = code_mode_says([_AddOp(f"c{i}", "1", CellConfig(), name=f"n{i}") for i in range(2)], names={"c0": "n0", "c1": "n1"}, ran={"c0", "c1"})
    assert run(named) == (True, "n0 and n1")
    many = code_mode_says([_AddOp(f"c{i}", "1", CellConfig(), name=f"n{i}") for i in range(4)], names={f"c{i}": f"n{i}" for i in range(4)})
    assert run(many) == (True, "4 cells")
    # With [checks] all off there are no snapshots: code mode's own line is the only word on a cell that raised.
    unnamed_error = code_mode_says([_AddOp("u1", "1/0", CellConfig())], names={"u1": ""}, ran={"u1"}, errored={"u1"})
    assert unnamed_error.strip() == "created and ran cell 'u1' (error)"
    assert run(unnamed_error) == (False, "1 cell, cell error")
    assert run(unnamed_error, traceback_of(lambda: 1 / 0)) == (False, "1 cell, ZeroDivisionError")
    rerun = code_mode_says([], names={"r1": "report"}, ran={"r1"}, errored={"r1"})
    assert rerun.strip() == "re-ran cell 'r1' (report) (error)"
    assert run(rerun) == (True, "re-ran report"), "only a cell the code created or edited fails the call"


def test_outcome_when_a_checker_is_missing(tmp_path, monkeypatch):
    find = code_checks._binary

    def no_ty(name: str) -> str:
        if name == "ty":
            raise code_checks._Unavailable("ty is not installed")
        return find(name)

    monkeypatch.setattr(code_checks, "_binary", no_ty)
    cell = {"id": "Hbol", "name": "growth_by_region", "code": "growth = sales.height"}
    client = CodeModeClient(STARTER, {WRITE_CELL: [cell]}, result=ExecResult(success=True, stdout="created cell 'Hbol' (growth_by_region)\n"))
    reply = checked(make_config(tmp_path), client).marimo_execute(WRITE_CELL)
    assert "(Check skipped: ty is not installed.)" in reply
    assert outcome("marimo_execute", {"code": WRITE_CELL}, reply) == (True, "growth_by_region, checks clean, ty skipped")
    monkeypatch.setattr(code_checks, "_binary", lambda name: (_ for _ in ()).throw(code_checks._Unavailable(f"{name} is not installed")))
    client = CodeModeClient(STARTER, {WRITE_CELL: [cell]}, result=ExecResult(success=True, stdout="shape: (2, 1)\n"))
    no_hardcoded = make_config(tmp_path, checks=CodeChecksConfig(hardcoded=False))
    reply = checked(no_hardcoded, client).marimo_execute(WRITE_CELL)
    assert reply.startswith("shape: (2, 1)\n\n(Check skipped: ")
    assert outcome("marimo_execute", {"code": WRITE_CELL}, reply) == (True, "growth_by_region, 2 rows, checks skipped")
    client = CodeModeClient(STARTER, {WRITE_CELL: [cell]}, result=ExecResult(success=True, stdout="shape: (2, 1)\n"))
    reply = checked(make_config(tmp_path), client).marimo_execute(WRITE_CELL)  # Hailer's own check needs no tool
    assert reply.startswith("shape: (2, 1)\n\nChecks (hailer) ")
    assert outcome("marimo_execute", {"code": WRITE_CELL}, reply) == (True, "growth_by_region, 2 rows, checks clean, ruff and ty skipped")


def test_outcome_of_code_that_failed_or_printed_frames(ws):
    pl = pytest.importorskip("polars")

    def run(result: ExecResult, code: str = "x") -> tuple[bool, str]:
        return outcome("marimo_execute", {"code": code}, HailerTools(ws, sandbox(ws, FakeClient(result=result))).marimo_execute(code))

    assert run(ExecResult(success=False, stderr=traceback_of(lambda: 1 / 0))) == (False, "ZeroDivisionError")
    assert run(ExecResult(success=False, stderr="boom")) == (False, "failed")
    frame = pl.DataFrame({"region": ["N", "S", "E", "W"], "q1": [1, 2, 3, 4], "q2": [5, 6, 7, 8]})
    assert run(ExecResult(success=True, stdout=f"{frame}\n")) == (True, "4 rows")
    assert run(ExecResult(success=True, stdout=f"{pl.DataFrame({'a': range(1500)})}\n")) == (True, "1,500 rows")
    assert run(ExecResult(success=True, stdout=f"{pl.Series('s', [1, 2])}\n")) == (True, "2 values")
    pandas_footer = "     a\n0    0\n..  ..\n99  99\n\n[100 rows x 1 columns]\n"  # how pandas ends a long frame (not installed here)
    assert run(ExecResult(success=True, stdout=pandas_footer)) == (True, "100 rows")
    assert run(ExecResult(success=True, output="<table></table>", mimetype="text/html")) == (True, "")
    assert run(ExecResult(success=True)) == (True, "")  # "(no output)"
    assert run(ExecResult(success=True, stdout="42\n")) == (True, "")


def test_failures_are_worded_by_their_class_never_their_message(ws, tmp_path):
    from hailer.errors import MarimoExecutionError

    def failed(name: str, args: dict, reply: str) -> str:
        ok, summary = outcome(name, args, reply)
        assert not ok
        return summary

    assert failed("marimo_execute", {}, HailerTools(ws).marimo_execute("1")) == "marimo not running"
    assert failed("marimo_execute", {}, HailerTools(ws, sandbox(ws)).marimo_execute("1")) == "marimo not running"
    data = FakeClient(raise_on_execute=RuntimeError("customer 4711 owes 12,000"))
    assert failed("marimo_execute", {}, HailerTools(ws, sandbox(ws, data)).marimo_execute("1")) == "unexpected RuntimeError"
    lost = NoSessionError(f"Not open: http://127.0.0.1:2718/?file=x.py&access_token={SECRET}", f"Open http://127.0.0.1:2718/?access_token={SECRET}")
    assert failed("marimo_execute", {}, HailerTools(ws, sandbox(ws, FakeClient(raise_on_execute=lost))).marimo_execute("1")) == "notebook not open"
    refused = FakeClient(raise_on_execute=MarimoExecutionError("Marimo refused the execute request (HTTP 400): stale"))
    assert failed("notebook_cells", {}, HailerTools(ws, sandbox(ws, refused)).notebook_cells()) == "kernel request failed"
    web = HailerTools(make_config(tmp_path, web=WebConfig(allowed_domains=("docs.pola.rs",))))
    assert failed("fetch_page", {}, web.fetch_page(f"ftp://docs.pola.rs/x?access_token={SECRET}")) == "host not allowed"
    assert failed("fetch_page", {}, web.fetch_page("https://example.com/")) == "host not allowed"
    opener = Opener()
    tools = HailerTools(ws, sandbox(ws), open_url=opener, session_wait_sec=0)
    assert failed("notebook_create", {"name": "x"}, tools.notebook_create("x", template="fancy")) == "unknown template"
    assert failed("notebook_create", {"name": "other"}, tools.notebook_create("other")) == "notebook already exists"
    assert failed("notebook_create", {"name": "!!!"}, tools.notebook_create("!!!")) == "not a usable notebook name"
    assert failed("notebook_open", {}, HailerTools(ws, sandbox(ws, FakeClient())).notebook_open("missing")) == "notebook not found"
    assert failed("notebook_open", {}, HailerTools(ws, sandbox(ws, FakeClient())).notebook_open("../x.py")) == "not in the notebooks folder"
    assert failed("list_periods", {}, HailerTools(ws, sandbox(ws)).list_periods()) == "could not read the data folder"
    config = make_config(tmp_path)
    (config.skills_dir / "alpha").mkdir(parents=True)
    (config.skills_dir / "alpha" / "SKILL.md").write_text("---\nname: alpha\ndescription: d\n---\nBody", encoding="utf-8")
    assert failed("load_skill", {}, HailerTools(config).load_skill("nope")) == "skill not found"
    assert failed("read_skill_file", {}, HailerTools(config).read_skill_file("alpha", "../../x")) == "could not read the file"
    unreadable = FakeClient(result=ExecResult(success=False, stderr="AttributeError: no cells"))
    assert failed("notebook_cells", {}, HailerTools(ws, sandbox(ws, unreadable)).notebook_cells()) == "could not read the cells"
    assert failed("notebook_check", {}, checked(make_config(tmp_path), unreadable).notebook_check()) == "could not read the cells"


def test_outcome_of_calls_langchain_could_not_run():
    """status="error" tool messages, in LangGraph's own words: the arguments they quote never show."""
    from langgraph.prebuilt.tool_node import INVALID_TOOL_NAME_ERROR_TEMPLATE, TOOL_INVOCATION_ERROR_TEMPLATE

    invalid = TOOL_INVOCATION_ERROR_TEMPLATE.format(tool_name="notebook_create", tool_kwargs={"secret": SECRET}, error="name: Field required")
    assert tool_outcome("notebook_create", {}, invalid, error=True) == (False, "invalid arguments")
    unknown = INVALID_TOOL_NAME_ERROR_TEMPLATE.format(requested_tool="shell", available_tools="marimo_execute")
    assert tool_outcome("shell", {}, unknown, error=True) == (False, "unknown tool")
    assert tool_outcome("marimo_execute", {}, "Error: ValueError('x')\n Please fix your mistakes.", error=True) == (False, "ValueError")
    assert tool_outcome("marimo_execute", {}, "anything", error=True) == (False, "failed")


def test_outcome_of_marimo_status(ws):
    assert outcome("marimo_status", {}, HailerTools(ws).marimo_status()) == (True, "not running")
    assert outcome("marimo_status", {}, HailerTools(ws, sandbox(ws)).marimo_status()) == (True, "not running")
    assert outcome("marimo_status", {}, HailerTools(ws, sandbox(ws, FakeClient())).marimo_status()) == (True, "analysis.py ready")
    reply = HailerTools(ws, sandbox(ws, FakeClient(has_session=False))).marimo_status()
    assert outcome("marimo_status", {}, reply) == (True, "analysis.py not open")


def test_outcome_of_notebook_cells(ws):
    cells = [
        {"id": "Hbol", "name": "imports", "code": "import marimo as mo"},
        {"id": "Abcd", "name": "load", "code": "df = load()", "status": "exception", "errors": 2},
        {"id": "Efgh", "code": ""},  # the listing's last field is empty, and the reply is stripped
    ]
    tools = HailerTools(ws, sandbox(ws, CodeModeClient(cells)))
    listing = tools.notebook_cells()
    assert listing.splitlines()[1] == "Abcd\tload\texception\terrors=2\tdf = load()", "the real listing code ran"
    assert outcome("notebook_cells", {}, listing) == (True, "3 cells, 1 with errors")
    assert outcome("notebook_cells", {"pattern": "load"}, tools.notebook_cells("load")) == (True, "1 matching cell, 1 with errors")
    assert outcome("notebook_cells", {"pattern": " zzz "}, tools.notebook_cells(" zzz ")) == (True, "no matching cells")
    assert outcome("notebook_cells", {}, HailerTools(ws, sandbox(ws, CodeModeClient([]))).notebook_cells()) == (True, "no cells")
    big = [{"id": f"c{i:03}", "name": f"cell_{i}", "code": "x" * 80} for i in range(200)]
    truncated = HailerTools(make_config(ws.workspace, max_tool_output_chars=2000), sandbox(ws, CodeModeClient(big))).notebook_cells()
    assert truncated.status == {"ok": True, "truncated": True}
    ok, summary = outcome("notebook_cells", {}, truncated)
    assert ok and summary.endswith("+ cells")


def test_outcome_of_notebook_check(tmp_path):
    cells = [*STARTER, {"id": "old", "name": "old", "code": "print(missing_before)"}]
    assert outcome("notebook_check", {}, checked(make_config(tmp_path), CodeModeClient(cells)).notebook_check()) == (True, "1 finding")
    assert outcome("notebook_check", {}, checked(make_config(tmp_path), CodeModeClient(STARTER)).notebook_check()) == (True, "clean")
    assert outcome("notebook_check", {}, checked(make_config(tmp_path), CodeModeClient([{"id": "e", "code": " "}])).notebook_check()) == (True, "no code")
    off = make_config(tmp_path, checks=CodeChecksConfig(lint=False, typecheck=False, hardcoded=False))
    assert outcome("notebook_check", {}, checked(off, CodeModeClient(STARTER)).notebook_check()) == (True, "checks off")
    typed_values_only = make_config(tmp_path, checks=CodeChecksConfig(lint=False, typecheck=False))
    assert outcome("notebook_check", {}, checked(typed_values_only, CodeModeClient(STARTER)).notebook_check()) == (True, "clean")
    lint_only = make_config(tmp_path, checks=CodeChecksConfig(typecheck=False))
    assert outcome("notebook_check", {}, checked(lint_only, CodeModeClient(cells)).notebook_check()) == (True, "1 finding")


def test_outcome_of_notebook_list(ws, tmp_path):
    assert outcome("notebook_list", {}, HailerTools(ws, sandbox(ws, FakeClient(open_names={"analysis.py"}))).notebook_list()) == (True, "2 notebooks, 1 open")
    assert outcome("notebook_list", {}, HailerTools(ws, sandbox(ws)).notebook_list()) == (True, "2 notebooks"), "marimo down: none known open"
    empty = make_config(tmp_path / "other")
    empty.notebooks_root.mkdir(parents=True)
    assert outcome("notebook_list", {}, HailerTools(empty, sandbox(empty)).notebook_list()) == (True, "no notebooks")


def test_outcome_of_notebook_create_names_the_file_only_when_the_name_was_not_it(ws):
    client = FakeClient(open_names={"analysis.py"})

    def create(name: str, opener: Opener, box=None, **kw) -> tuple[bool, str]:
        tools = HailerTools(ws, box or sandbox(ws, client), open_url=opener, session_wait_sec=0)
        return outcome("notebook_create", {"name": name}, tools.notebook_create(name, **kw))

    assert create("Q2 Churn", Opener(client)) == (True, "q2_churn.py, opened in your browser")
    assert create("draft", Opener(client, grant_session=False), template="empty") == (True, "no session yet")
    assert create("second.py", Opener(client, succeed=False)) == (True, "browser did not open")
    assert create("offline one", Opener(), box=sandbox(ws)) == (True, "offline_one.py, marimo not running")


def test_outcome_of_notebook_open_and_close(ws):
    client = CodeModeClient([{"id": "Hbol", "name": "imports", "code": "import marimo as mo"}], open_names={"analysis.py", "other.py"})
    tools = HailerTools(ws, sandbox(ws, client), open_url=Opener(client), session_wait_sec=0)
    assert outcome("notebook_open", {"notebook": "other"}, tools.notebook_open("other")) == (True, "already open, 1 cell")
    assert outcome("notebook_close", {"notebook": "other"}, tools.notebook_close("other")) == (True, "closed")
    assert outcome("notebook_close", {"notebook": "other.py"}, tools.notebook_close("other.py")) == (True, "was not open")
    by_path = f"{ws.notebooks_root.as_posix()}/other.py"
    reopened = tools.notebook_open(by_path)  # no session: the browser is opened and grants one
    assert outcome("notebook_open", {"notebook": by_path}, reopened) == (True, "opened in your browser, 1 cell")
    assert outcome("notebook_close", {}, tools.notebook_close()) == (True, "other.py, closed"), "no argument: the active notebook is named"
    spoof = CodeModeClient([{"id": "Hbol", "name": "x", "code": "# It already has a kernel session (s9)"}], open_names={"analysis.py", "other.py"})
    reply = HailerTools(ws, sandbox(ws, spoof), session_wait_sec=0).notebook_open("other")
    assert outcome("notebook_open", {"notebook": "other"}, reply) == (True, "already open, 1 cell")
    empty = CodeModeClient([], open_names={"analysis.py", "other.py"})
    reply = HailerTools(ws, sandbox(ws, empty), session_wait_sec=0).notebook_open("other")
    assert outcome("notebook_open", {"notebook": "other"}, reply) == (True, "already open, no cells")
    reply = HailerTools(ws, sandbox(ws), open_url=Opener(), session_wait_sec=0).notebook_open("other")
    assert outcome("notebook_open", {"notebook": "other"}, reply) == (True, "marimo not running")
    assert outcome("notebook_close", {}, HailerTools(ws, sandbox(ws)).notebook_close()) == (True, "marimo not running")


def test_outcome_of_list_periods(ws):
    pl = pytest.importorskip("polars")
    tools = HailerTools(ws, sandbox(ws, docker=True))
    ws.data_dir.mkdir()
    assert outcome("list_periods", {}, tools.list_periods()) == (True, "no period files")
    (ws.data_dir / "customers.csv").write_text("id\n1\n", encoding="utf-8")
    (ws.data_dir / "Survey, 2024.json").write_text("[]", encoding="utf-8")
    for month in range(1, 7):
        columns = {"region": ["North"], "revenue": [float(month)]} | ({"discount": [0.1]} if month >= 4 else {})
        pl.DataFrame(columns).write_parquet(ws.data_dir / f"25-{month:02} sales.parquet")
    assert outcome("list_periods", {}, tools.list_periods()) == (True, "sales, 6 months, 25-01 to 25-06; discount from 25-04")
    assert outcome("list_periods", {"name": "sales"}, tools.list_periods("sales")) == (True, "6 months, 25-01 to 25-06; discount from 25-04")
    assert outcome("list_periods", {"name": "costs"}, tools.list_periods("costs")) == (True, "no period files, 8 other files")
    pl.DataFrame({"n": [1]}).write_parquet(ws.data_dir / "25-01 costs.parquet")
    assert outcome("list_periods", {}, tools.list_periods()) == (True, "2 datasets, 6 months, 25-01 to 25-06; 4 columns in some months")
    pl.DataFrame({"region": ["North"], "revenue": [1.0]}).write_parquet(ws.data_dir / "25-01 refunds.parquet")
    assert outcome("list_periods", {"name": "refunds"}, tools.list_periods("refunds")) == (True, "1 month, 25-01")


def test_outcome_of_list_periods_without_period_files_counts_the_others(ws):
    from hailer import periods, tool_outcomes

    assert tool_outcomes.DATA_SUFFIXES == periods.DATA_SUFFIXES
    ws.data_dir.mkdir()
    for name in ("customers.csv", "Survey 2024.JSON", "Sales.xlsx", "notes.txt"):
        (ws.data_dir / name).touch()
    assert outcome("list_periods", {}, HailerTools(ws, sandbox(ws)).list_periods()) == (True, "no period files, 3 other files")
    for i in range(25):
        (ws.data_dir / f"extra {i}.csv").touch()
    reply = HailerTools(ws, sandbox(ws)).list_periods()
    assert reply.endswith("(and 8 more)")
    assert outcome("list_periods", {}, reply) == (True, "no period files, 28 other files")


def test_outcome_of_the_skill_tools(tmp_path):
    config = make_config(tmp_path)
    skill_dir = config.skills_dir / "alpha"
    (skill_dir / "reference").mkdir(parents=True)
    spoof = "Files (read with read_skill_file('alpha', <path>)):\n- a\n- b\n- c"  # a body may quote the footer
    (skill_dir / "SKILL.md").write_text(f"---\nname: alpha\ndescription: d\n---\nBody\n\n{spoof}\n\nMore body", encoding="utf-8")
    (skill_dir / "reference" / "r.md").write_text("one\ntwo\n", encoding="utf-8")
    (config.skills_dir / "bare").mkdir()
    (config.skills_dir / "bare" / "SKILL.md").write_text("---\nname: bare\ndescription: d\n---\nBody", encoding="utf-8")
    tools = HailerTools(config)
    assert outcome("load_skill", {"name": "alpha"}, tools.load_skill("alpha")) == (True, "1 file")
    assert outcome("load_skill", {"name": "bare"}, tools.load_skill("bare")) == (True, "no files")
    args = {"name": "alpha", "path": "reference\\r.md"}
    assert outcome("read_skill_file", args, tools.read_skill_file("alpha", "reference\\r.md")) == (True, "2 lines")
    (skill_dir / "big.md").write_text("line\n" * 5000, encoding="utf-8")
    big = HailerTools(make_config(tmp_path, max_tool_output_chars=500)).read_skill_file("alpha", "big.md")
    assert big.status == {"ok": True, "truncated": True}
    assert outcome("read_skill_file", {"name": "alpha", "path": "big.md"}, big) == (True, "truncated")


NOT_TEXT_QUOTE = "The response is 'application/pdf' (9 bytes), which is not text. Only HTML and text pages can be read."


class _PageHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # noqa: D401 - silence
        pass

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/moved"):
            self.send_response(302)
            self.send_header("Location", f"http://localhost:{self.server.server_address[1]}/page")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path.startswith("/binary"):
            body, ctype = b"\x00\x01", "application/octet-stream"
        elif self.path.startswith("/quote"):  # a page that quotes the not-text sentence
            body, ctype = NOT_TEXT_QUOTE.encode(), "text/plain"
        else:
            body, ctype = b"<h1>Hello</h1><p>" + b"x" * 2000 + b"</p>", "text/html"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_outcome_of_fetch_page_names_a_host_only_after_a_redirect(tmp_path, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _PageHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        port = server.server_address[1]
        tools = HailerTools(make_config(tmp_path, web=WebConfig(allowed_domains=("127.0.0.1", "localhost"))))

        def fetch(url: str) -> tuple[bool, str]:
            return outcome("fetch_page", {"url": url}, tools.fetch_page(url))

        page = f"http://127.0.0.1:{port}/page?access_token={SECRET}"
        assert tools.fetch_page(page).startswith(f"Source: {page}")
        assert fetch(page) == (True, "2.0 KB")
        assert fetch(f"http://127.0.0.1:{port}/binary") == (True, "not text (application/octet-stream)")
        assert fetch(f"http://127.0.0.1:{port}/quote") == (True, f"{len(NOT_TEXT_QUOTE)} B"), "a page's own words are its text"
        assert fetch(f"http://127.0.0.1:{port}/moved") == (True, f"2.0 KB, redirected to localhost:{port}")
        ok, summary = fetch(f"http://user:pw@127.0.0.1:{port}/page")  # urllib cannot: an error that quotes the URL
        assert (ok, summary) == (False, "could not fetch the page")
        # A page cut to the output cap says so, and the model reads what fetch_page always gave it.
        from hailer.web import fetch_page

        capped_config = make_config(tmp_path, web=WebConfig(allowed_domains=("127.0.0.1",)), max_tool_output_chars=500)
        capped = HailerTools(capped_config).fetch_page(page)
        assert capped == fetch_page(page, capped_config) and "truncated" in capped and len(capped) <= 500
        assert capped.status == {"ok": True, "truncated": True}
        ok, summary = outcome("fetch_page", {"url": page}, capped)
        assert ok and summary.endswith(" B, truncated")
        small = make_config(tmp_path, web=WebConfig(allowed_domains=("127.0.0.1",), max_page_bytes=100))
        first_bytes = HailerTools(small).fetch_page(page)
        assert "(first 100 bytes only)" in first_bytes.splitlines()[0] and "truncated" not in first_bytes.status
        ok, summary = outcome("fetch_page", {"url": page}, first_bytes)
        assert ok and summary.endswith(" B, truncated"), "fetch_page's own header says the page was cut"
    finally:
        server.shutdown()
        server.server_close()


def test_outcome_is_empty_for_unknown_formats_and_never_raises():
    class Unprintable:
        def __str__(self) -> str:
            raise RuntimeError("no")

    assert tool_outcome("echo", {"text": "hi"}, "ERROR: x") == (True, ""), "no status: not a Hailer tool, nothing said"
    odd_status = [None, {"ok": "no"}, {"ok": True, "cells": 5, "findings": -1, "checked": "ruff", "truncated": "x"}, {"ok": False, "error": 3, "unexpected": True}]
    for name in TOOL_NAMES:
        for content in (None, "", "\x1b[2J\x1b[31mred\x00", ["block", {"text": "x"}, 3], Unprintable(), "x" * 100_000):
            for status in odd_status:
                ok, summary = tool_outcome(name, {"pattern": object(), "path": None, "name": 3, "code": 4}, content, status)
                assert isinstance(ok, bool) and len(summary) <= SUMMARY_LIMIT and all(ch.isprintable() for ch in summary)
    assert tool_outcome("marimo_execute", {}, "x", {"ok": False, "error": "x y", "unexpected": True}) == (False, "unexpected error")
    long = "a" * 200 + ".py is not open (no kernel session), so there is nothing to close."
    ok, summary = tool_outcome("notebook_close", {}, long, {"ok": True})
    assert ok and summary.endswith("...")
    bidi = "x‮\x1b[31my.py is not open (no kernel session), so there is nothing to close."
    assert tool_outcome("notebook_close", {}, bidi, {"ok": True}) == (True, "x[31my.py, was not open")
