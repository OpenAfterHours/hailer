"""Tests for hailer.mcp_server: tool behaviour with a fake marimo client, MCP registration,
and a real stdio round-trip in a subprocess (offline)."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from hailer import notebooks
from hailer.errors import MarimoUnavailableError, NoSessionError
from hailer.mcp_server import HailerTools, build_server
from hailer.models import ExecResult, HailerConfig, MarimoServer, MarimoSession, WebConfig

REPO_ROOT = Path(__file__).resolve().parents[1]

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
    raise MarimoUnavailableError("marimo is not running", "Start it with:\n    uv run marimo edit notebooks --no-token")


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
    assert "uv run marimo edit" in text

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
    assert text.startswith("marimo: not running") and "uv run marimo edit" in text
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
    text = tools.list_periods("pra101")
    assert "No period files found" in text and "'pra101'" in text and "25-01 pra101.parquet" in text


def test_list_periods_with_files(tmp_path):
    pytest.importorskip("hailer.periods")
    pl = pytest.importorskip("polars")
    config = make_config(tmp_path)
    config.data_dir.mkdir()
    pl.DataFrame({"exposure_class": ["Corporate"], "rwa": [1.0]}).write_parquet(config.data_dir / "25-01 pra101.parquet")
    pl.DataFrame({"exposure_class": ["Retail"], "rwa": [2.0], "risk_weight": [0.5]}).write_parquet(config.data_dir / "25-02 pra101.parquet")
    text = HailerTools(config, failing_factory).list_periods()
    assert "2025-01" in text and "2025-02" in text


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
# MCP registration (in-process)
# --------------------------------------------------------------------------- #


def test_build_server_registers_eleven_tools(tmp_path):
    server = build_server(make_config(tmp_path), client_factory=failing_factory)
    tools = asyncio.run(server.list_tools())
    assert [t.name for t in tools] == [
        "marimo_execute",
        "marimo_status",
        "notebook_cells",
        "notebook_list",
        "notebook_create",
        "notebook_open",
        "notebook_close",
        "list_periods",
        "load_skill",
        "read_skill_file",
        "fetch_page",
    ]
    assert all(t.description for t in tools)
    by_name = {t.name: t for t in tools}
    assert by_name["marimo_execute"].input_schema["required"] == ["code"]
    assert by_name["notebook_create"].input_schema["required"] == ["name"]
    assert by_name["notebook_open"].input_schema["required"] == ["notebook"]
    assert "required" not in by_name["notebook_close"].input_schema or by_name["notebook_close"].input_schema["required"] == []
    assert "ACTIVE notebook" in server.instructions or "active notebook" in server.instructions.lower()


def test_call_tool_in_process_via_client(ws):
    from mcp import Client

    config, analysis, other = ws
    client = FakeClient(open_paths={analysis}, result=ExecResult(success=True, stdout="ok"))
    opener = Opener(client)
    server = build_server(config, client_factory=factory_for(client), open_url=opener, session_wait_sec=0)

    async def go():
        async with Client(server) as c:
            r = await c.call_tool("marimo_execute", {"code": "print('ok')"})
            assert r.content[0].text == "ok"
            r = await c.call_tool("notebook_list", {})
            assert "notebooks/analysis.py" in r.content[0].text and "[active]" in r.content[0].text
            r = await c.call_tool("notebook_create", {"name": "via mcp"})
            assert r.content[0].text.startswith("Created notebooks/via_mcp.py")
            r = await c.call_tool("notebook_open", {"notebook": "other"})
            assert r.content[0].text.startswith("notebooks/other.py is now the active notebook.")
            r = await c.call_tool("fetch_page", {"url": "https://example.com"})
            assert r.content[0].text.startswith("ERROR:")

    asyncio.run(go())
    assert len(opener.urls) == 2  # create + open (other had no session)
    assert notebooks.load_active_notebook(config) == other


# --------------------------------------------------------------------------- #
# Real stdio server in a subprocess (offline)
# --------------------------------------------------------------------------- #


def test_stdio_server_roundtrip(tmp_path):
    pytest.importorskip("hailer.config")  # written by another owner; skip until it lands
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters

    (tmp_path / "hailer.toml").write_text(
        '[hailer]\nnotebook = "notebooks/analysis.py"\ndata_dir = "data"\n\n[web]\nallowed_domains = ["docs.pola.rs"]\n',
        encoding="utf-8",
    )
    (tmp_path / "data").mkdir()
    write_notebook(tmp_path / "notebooks" / "analysis.py")
    env = {
        "HAILER_WORKSPACE": str(tmp_path),
        "HAILER_CONFIG": str(tmp_path / "hailer.toml"),
        "HAILER_LOG_LEVEL": "WARNING",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONPATH": str(REPO_ROOT / "src"),
    }
    params = StdioServerParameters(command=sys.executable, args=["-m", "hailer.mcp_server"], env=env, cwd=str(tmp_path))

    async def go():
        async with Client(params) as c:
            tools = await asyncio.wait_for(c.list_tools(), 30)
            names = [t.name for t in tools.tools] if hasattr(tools, "tools") else [t.name for t in tools]
            assert "marimo_execute" in names and "fetch_page" in names
            assert {"notebook_list", "notebook_create", "notebook_open", "notebook_close"} <= set(names)
            r = await asyncio.wait_for(c.call_tool("fetch_page", {"url": "https://example.com/x"}), 30)
            assert r.content[0].text.startswith("ERROR: Web access to 'example.com' is not allowed")
            assert "docs.pola.rs" in r.content[0].text
            r = await asyncio.wait_for(c.call_tool("list_periods", {}), 30)
            assert "No period files found" in r.content[0].text
            # a pure file-system tool works over stdio without marimo (no browser is involved)
            r = await asyncio.wait_for(c.call_tool("notebook_list", {}), 30)
            assert "notebooks/analysis.py" in r.content[0].text and "[active]" in r.content[0].text

    asyncio.run(go())


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
    assert captured["notebooks_dir"] == cfg.notebooks_root
    assert captured["token"] == "tok"

    notebooks.save_active_notebook(cfg, other)
    HailerTools(cfg)._default_client()
    assert captured["notebook"] == other


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
            hint="Start everything in one go:\n\n    uv run hailer notebook",
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
    assert "marimo is not running" in text and "connection refused" in text and "uv run hailer notebook" in text
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
    assert "connection refused" in text and "uv run hailer notebook" in text


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
    assert "connection refused" in text and "uv run hailer notebook" in text


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
