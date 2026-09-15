"""Tests for hailer.mcp_server: tool behaviour with a fake marimo client, MCP registration,
and a real stdio round-trip in a subprocess (offline)."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from hailer.errors import MarimoUnavailableError, NoSessionError
from hailer.mcp_server import HailerTools, build_server
from hailer.models import ExecResult, HailerConfig, MarimoServer, MarimoSession, WebConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


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


class FakeClient:
    def __init__(self, *, has_session: bool = True, result: ExecResult | None = None, raise_on_execute: Exception | None = None):
        self.has_session = has_session
        self.result = result or ExecResult(success=True, stdout="hello\n")
        self.raise_on_execute = raise_on_execute
        self.executed: list[str] = []

    def health(self) -> bool:
        return True

    def sessions(self) -> list[MarimoSession]:
        if not self.has_session:
            return []
        return [MarimoSession("s1", "analysis.py", "notebooks/analysis.py")]

    def resolve_session(self, notebook):
        if not self.has_session:
            raise NoSessionError("no session", "open the notebook")
        return MarimoSession("s1", "analysis.py", "notebooks/analysis.py")

    def execute(self, code, *, session_id=None, notebook=None, on_stdout=None, on_stderr=None, timeout=600.0):
        self.executed.append(code)
        if self.raise_on_execute:
            raise self.raise_on_execute
        return self.result


def factory_for(client: FakeClient, url: str = "http://127.0.0.1:2718", version: str = "0.24.2"):
    def factory():
        return client, MarimoServer(url=url, server_id="127.0.0.1:2718", version=version, source="config")

    return factory


def failing_factory():
    raise MarimoUnavailableError("marimo is not running", "Start it with:\n    uv run marimo edit notebooks/analysis.py --no-token")


# --------------------------------------------------------------------------- #
# HailerTools
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


def test_marimo_status_variants(tmp_path):
    tools = HailerTools(make_config(tmp_path), failing_factory)
    text = tools.marimo_status()
    assert text.startswith("marimo: not running") and "uv run marimo edit" in text

    tools = HailerTools(make_config(tmp_path), factory_for(FakeClient(has_session=True)))
    text = tools.marimo_status()
    assert "running at http://127.0.0.1:2718 (version 0.24.2)" in text
    assert "s1: notebooks/analysis.py" in text
    assert "analysis.py -> session s1 (ready)" in text

    tools = HailerTools(make_config(tmp_path), factory_for(FakeClient(has_session=False)))
    text = tools.marimo_status()
    assert "sessions: none" in text
    assert "NO session" in text and "http://127.0.0.1:2718" in text


def test_notebook_cells_filters(tmp_path):
    listing = "Hbol\timports\tidle\terrors=0\timport marimo as mo\nAbcd\tload\tidle\terrors=1\tdf = load_periods(files)\n"
    client = FakeClient(result=ExecResult(success=True, stdout=listing))
    tools = HailerTools(make_config(tmp_path), factory_for(client))
    assert tools.notebook_cells() == listing.rstrip()
    assert tools.notebook_cells("LOAD") == "Abcd\tload\tidle\terrors=1\tdf = load_periods(files)"
    assert tools.notebook_cells("zzz") == "No cells match 'zzz'."
    assert "marimo._code_mode" in client.executed[0]


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


def test_build_server_registers_seven_tools(tmp_path):
    server = build_server(make_config(tmp_path), client_factory=failing_factory)
    tools = asyncio.run(server.list_tools())
    assert [t.name for t in tools] == [
        "marimo_execute", "marimo_status", "notebook_cells", "list_periods", "load_skill", "read_skill_file", "fetch_page",
    ]
    assert all(t.description for t in tools)
    assert tools[0].input_schema["required"] == ["code"]


def test_call_tool_in_process_via_client(tmp_path):
    from mcp import Client

    client = FakeClient(result=ExecResult(success=True, stdout="ok"))
    server = build_server(make_config(tmp_path), client_factory=factory_for(client))

    async def go():
        async with Client(server) as c:
            r = await c.call_tool("marimo_execute", {"code": "print('ok')"})
            assert r.content[0].text == "ok"
            r = await c.call_tool("fetch_page", {"url": "https://example.com"})
            assert r.content[0].text.startswith("ERROR:")

    asyncio.run(go())


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
            r = await asyncio.wait_for(c.call_tool("fetch_page", {"url": "https://example.com/x"}), 30)
            assert r.content[0].text.startswith("ERROR: Web access to 'example.com' is not allowed")
            assert "docs.pola.rs" in r.content[0].text
            r = await asyncio.wait_for(c.call_tool("list_periods", {}), 30)
            assert "No period files found" in r.content[0].text

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


def test_default_client_uses_configured_notebook_and_workspace(tmp_path, monkeypatch):
    from hailer import marimo_client as mc

    captured: dict = {}

    class CapturingClient:
        def __init__(self, base_url, token=None, **kw):
            captured.update(base_url=base_url, token=token, **kw)

    monkeypatch.setattr(mc, "find_server", lambda config: MarimoServer(url="http://127.0.0.1:2718", source="config"))
    monkeypatch.setattr(mc, "MarimoClient", CapturingClient)
    cfg = make_config(tmp_path, marimo_token="tok")
    client, server = HailerTools(cfg)._default_client()
    assert isinstance(client, CapturingClient) and server.url == "http://127.0.0.1:2718"
    assert captured["notebook"] == cfg.notebook and captured["workspace"] == cfg.workspace
    assert captured["token"] == "tok"
