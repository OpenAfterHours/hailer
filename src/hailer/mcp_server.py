"""Hailer's MCP server: the agent's only bridge to the live marimo kernel and the web.

Codex launches this as a stdio MCP server (``python -m hailer.mcp_server``) with
``HAILER_CONFIG`` / ``HAILER_WORKSPACE`` in its environment. stdout is the
transport, so this module never prints; diagnostics go to stderr via logging.

Every tool returns plain text. Hailer errors become ``ERROR: <message>\\n<hint>``
text so the model can act on them; nothing is raised across the protocol.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .errors import HailerError, MarimoUnavailableError, NoSessionError
from .models import HailerConfig, MarimoServer
from .web import truncate_text

log = logging.getLogger("hailer.mcp")

SERVER_NAME = "hailer"
SERVER_INSTRUCTIONS = (
    "Hailer tools. The live marimo notebook is the visual workspace: use marimo_execute to run "
    "Python in its kernel (scratchpad semantics; persist notebook changes through "
    "marimo._code_mode). Keep outputs compact: inspect schemas, samples and aggregates, never "
    "print whole datasets. fetch_page only reaches allow-listed domains."
)

# Fallback used only if hailer.marimo_client does not export LIST_CELLS_CODE.
_FALLBACK_LIST_CELLS_CODE = '''
import marimo._code_mode as _cm
_ctx = _cm.get_context()
for _c in _ctx.cells:
    _first = (_c.code.strip().splitlines() or [""])[0]
    print(f"{_c.id}\\t{_c.name}\\t{_c.status}\\terrors={len(_c.errors)}\\t{_first[:100]}")
'''.strip()

ClientFactory = Callable[[], tuple[Any, MarimoServer]]


def _error_text(err: HailerError) -> str:
    text = f"ERROR: {err}"
    if err.hint:
        text += f"\n{err.hint}"
    return text


def _config_from_env() -> HailerConfig:
    from .config import load_config  # lazy: keeps import-time coupling low

    workspace = os.environ.get("HAILER_WORKSPACE")
    config_path = os.environ.get("HAILER_CONFIG")
    return load_config(
        workspace=Path(workspace) if workspace else None,
        config_path=Path(config_path) if config_path else None,
    )


class HailerTools:
    """The tool implementations, independent of the MCP transport (unit-testable)."""

    def __init__(self, config: HailerConfig, client_factory: ClientFactory | None = None) -> None:
        self.config = config
        self._client_factory = client_factory or self._default_client

    # -- plumbing ----------------------------------------------------------- #

    def _default_client(self) -> tuple[Any, MarimoServer]:
        from . import marimo_client as mc  # lazy

        server = mc.find_server(self.config)
        if server is None:
            cmd = " ".join(mc.notebook_launch_command(self.config))
            raise MarimoUnavailableError(
                "marimo is not running (no server configured or discovered)",
                f"Start it with:\n    {cmd}\nThen open the notebook in your browser.",
            )
        return mc.MarimoClient(server.url, token=self.config.marimo_token), server

    def _truncate(self, text: str) -> str:
        return truncate_text(text, self.config.max_tool_output_chars)

    def _run(self, fn: Callable[[], str]) -> str:
        try:
            return self._truncate(fn())
        except HailerError as err:
            return self._truncate(_error_text(err))
        except Exception as exc:  # noqa: BLE001 - never let an exception cross the protocol
            log.exception("tool failed")
            return self._truncate(f"ERROR: unexpected {type(exc).__name__}: {exc}")

    def _open_url(self, server: MarimoServer) -> str:
        try:
            from . import marimo_client as mc

            return mc.open_notebook_url(server, self.config.notebook, self.config.workspace)
        except Exception:  # noqa: BLE001 - best effort
            return server.url

    # -- tools -------------------------------------------------------------- #

    def marimo_execute(self, code: str) -> str:
        def go() -> str:
            client, _ = self._client_factory()
            result = client.execute(code, notebook=self.config.notebook)
            text = result.as_text()
            return text if result.success else f"Execution failed.\n{text}"

        return self._run(go)

    def marimo_status(self) -> str:
        def go() -> str:
            try:
                client, server = self._client_factory()
            except MarimoUnavailableError as err:
                return f"marimo: not running\n{err}\n{err.hint}".rstrip()
            lines = [f"marimo: running at {server.url}" + (f" (version {server.version})" if server.version else "")]
            sessions = client.sessions()
            if sessions:
                lines.append("sessions:")
                for s in sessions:
                    lines.append(f"  - {s.session_id}: {s.path or s.filename or '(unsaved)'}")
            else:
                lines.append("sessions: none (the notebook is not open in a browser)")
            try:
                session = client.resolve_session(self.config.notebook)
                lines.append(f"configured notebook: {self.config.notebook.name} -> session {session.session_id} (ready)")
            except NoSessionError:
                lines.append(
                    f"configured notebook: {self.config.notebook.name} has NO session. "
                    f"Ask the user to open {self._open_url(server)} in a browser."
                )
            return "\n".join(lines)

        return self._run(go)

    def notebook_cells(self, pattern: str = "") -> str:
        def go() -> str:
            try:
                from .marimo_client import LIST_CELLS_CODE as code  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                code = _FALLBACK_LIST_CELLS_CODE
            client, _ = self._client_factory()
            result = client.execute(code, notebook=self.config.notebook)
            text = result.as_text()
            if not result.success:
                return f"Could not list cells.\n{text}"
            if pattern.strip():
                needle = pattern.strip().lower()
                lines = [ln for ln in text.splitlines() if needle in ln.lower()]
                return "\n".join(lines) if lines else f"No cells match '{pattern.strip()}'."
            return text or "(no cells)"

        return self._run(go)

    def list_periods(self, name: str = "") -> str:
        def go() -> str:
            from . import periods  # lazy

            data_dir = self.config.data_dir
            if not data_dir.is_dir():
                return f"Data directory does not exist: {data_dir}"
            files = periods.scan_period_files(data_dir, name.strip() or None)
            if not files:
                return (
                    f"No period files found in {data_dir}"
                    + (f" for '{name.strip()}'" if name.strip() else "")
                    + ". Files must be named like '25-01 pra101.parquet' (YY-MM then the dataset name)."
                )
            return periods.describe_periods(files)

        return self._run(go)

    def load_skill(self, name: str) -> str:
        from .context import read_skill

        return self._run(lambda: read_skill(self.config, name))

    def read_skill_file(self, name: str, path: str) -> str:
        from .context import read_skill_file

        return self._run(lambda: read_skill_file(self.config, name, path))

    def fetch_page(self, url: str) -> str:
        from .web import fetch_page

        return self._run(lambda: fetch_page(url, self.config))


def build_server(config: HailerConfig, *, client_factory: ClientFactory | None = None):
    """Create the MCP server with Hailer's tools registered. Returns an ``mcp.server.MCPServer``."""
    from mcp.server import MCPServer

    tools = HailerTools(config, client_factory)
    server = MCPServer(SERVER_NAME, instructions=SERVER_INSTRUCTIONS, log_level="WARNING")

    @server.tool(name="marimo_execute")
    def marimo_execute(code: str) -> str:
        """Run Python in the live marimo notebook kernel and return stdout, the result and stderr.

        The code runs in the kernel's scratchpad: notebook variables are readable by name, but
        new top-level assignments are discarded afterwards. To add or change notebook cells,
        use `import marimo._code_mode as cm` and `async with cm.get_context() as ctx:` with
        ctx.create_cell(...)/ctx.edit_cell(...)/ctx.run_cell(...) (top-level `async with` is
        allowed). Keep printed output small: schemas, head(), aggregates. Never print whole
        datasets.
        """
        return tools.marimo_execute(code)

    @server.tool(name="marimo_status")
    def marimo_status() -> str:
        """Report whether marimo is running, its sessions, and whether the configured notebook
        is open in a browser (a kernel session only exists while it is). If not, the reply
        contains the URL the user must open."""
        return tools.marimo_status()

    @server.tool(name="notebook_cells")
    def notebook_cells(pattern: str = "") -> str:
        """List the notebook's cells: id, name, status, error count and first code line.
        Optional case-insensitive substring filter. Use it before editing to find the cell
        that owns a variable and to avoid creating duplicate cells."""
        return tools.notebook_cells(pattern)

    @server.tool(name="list_periods")
    def list_periods(name: str = "") -> str:
        """Describe the monthly data files in the data directory (files named like
        '25-01 pra101.parquet'): periods available, common columns and columns that only
        appear in some months. Optional dataset name filter, e.g. 'pra101'."""
        return tools.list_periods(name)

    @server.tool(name="load_skill")
    def load_skill(name: str) -> str:
        """Load a project skill (its SKILL.md instructions and the list of bundled files) by
        name. Use when the task matches a skill listed in your instructions."""
        return tools.load_skill(name)

    @server.tool(name="read_skill_file")
    def read_skill_file(name: str, path: str) -> str:
        """Read a file bundled with a skill, by skill name and path relative to the skill
        folder (for example 'reference/checks.md')."""
        return tools.read_skill_file(name, path)

    @server.tool(name="fetch_page")
    def fetch_page(url: str) -> str:
        """Fetch a web page as readable text. Only hosts in the project's allowed-domains list
        can be fetched; a denied request says which domains are allowed. Fetched text is
        sent to the model like any other tool result."""
        return tools.fetch_page(url)

    server.hailer_tools = tools  # type: ignore[attr-defined]  # handy for tests
    return server


def _setup_logging() -> None:
    level_name = os.environ.get("HAILER_LOG_LEVEL", "WARNING").upper()
    level = getattr(logging, level_name, logging.WARNING)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


def main() -> None:
    """Console entry point (``hailer-mcp``). Speaks MCP over stdio."""
    _setup_logging()
    try:
        config = _config_from_env()
    except HailerError as err:
        sys.stderr.write(f"hailer-mcp: {err}\n{err.hint}\n")
        sys.exit(2)
    log.debug("hailer-mcp starting; workspace=%s notebook=%s", config.workspace, config.notebook)
    server = build_server(config)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
