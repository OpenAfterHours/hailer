"""Hailer's MCP server: the agent's only bridge to the live marimo kernel and the web.

Codex launches this as a stdio MCP server (``python -m hailer.mcp_server``) with
``HAILER_CONFIG`` / ``HAILER_WORKSPACE`` in its environment. stdout is the
transport, so this module never prints; diagnostics go to stderr via logging.

Every tool returns plain text. Hailer errors become ``ERROR: <message>\\n<hint>``
text so the model can act on them; nothing is raised across the protocol.

The *active notebook* (the one every kernel tool acts on) lives in
``<workspace>/.hailer/notebook.json`` and is re-read on every call: this process
outlives many turns, and both the model (``notebook_create`` / ``notebook_open``)
and the CLI (``/notebook`` commands) may switch it.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import notebooks
from .errors import HailerError, MarimoUnavailableError, NoSessionError
from .models import ExecResult, HailerConfig, MarimoServer, MarimoSession
from .web import truncate_text

log = logging.getLogger("hailer.mcp")

SERVER_NAME = "hailer"
SERVER_INSTRUCTIONS = (
    "Hailer tools. A live marimo notebook is the visual workspace: use marimo_execute to run "
    "Python in its kernel (scratchpad semantics; persist notebook changes through "
    "marimo._code_mode). Every kernel tool acts on the ACTIVE notebook, which marimo_status "
    "names; notebook_create and notebook_open switch it for all later calls. Keep outputs "
    "compact: inspect schemas, samples and aggregates, never print whole datasets. fetch_page "
    "only reaches allow-listed domains."
)

#: How long notebook_create / notebook_open wait for the browser tab to give the kernel a session.
DEFAULT_SESSION_WAIT_SEC = 30.0

# Fallback used only if hailer.marimo_client does not export LIST_CELLS_CODE.
_FALLBACK_LIST_CELLS_CODE = '''
import marimo._code_mode as _cm
_ctx = _cm.get_context()
for _c in _ctx.cells:
    _first = (_c.code.strip().splitlines() or [""])[0]
    print(f"{_c.id}\\t{_c.name}\\t{_c.status}\\terrors={len(_c.errors)}\\t{_first[:100]}")
'''.strip()

ClientFactory = Callable[[], tuple[Any, MarimoServer]]
UrlOpener = Callable[[str], bool]

# Mimetypes whose rendered value is useful to a language model as-is.
_TEXT_MIMETYPES = frozenset({"", "text/plain", "text/markdown", "text/csv"})


def _result_text(result: Any) -> str:
    """Model-friendly text for an ExecResult.

    Rich rendered values (``text/html`` tables, ``application/json`` widgets ...) are replaced
    by a short placeholder: they can be tens of kilobytes and carry nothing the model can
    read. stdout/stderr are kept intact, so ``print()`` remains the way to inspect values.
    """
    mimetype = (getattr(result, "mimetype", "") or "").split(";")[0].strip().lower()
    output = getattr(result, "output", "") or ""
    if output.strip() and mimetype not in _TEXT_MIMETYPES:
        placeholder = f"[{mimetype} output omitted - print() the values you need]"
        return ExecResult(
            success=result.success,
            stdout=result.stdout,
            stderr=result.stderr,
            output=placeholder,
            mimetype="text/plain",
        ).as_text()
    return result.as_text()


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


def _list_cells_code() -> str:
    try:
        from .marimo_client import LIST_CELLS_CODE  # type: ignore[attr-defined]

        return LIST_CELLS_CODE
    except Exception:  # noqa: BLE001
        return _FALLBACK_LIST_CELLS_CODE


def _default_open_url(url: str) -> bool:
    """Open ``url`` in the user's browser without touching stdout (the MCP transport).

    ``hailer.browser.open_url`` uses ``os.startfile`` on Windows (so a ``BROWSER`` variable
    pointing at a non-GUI command cannot turn it into a silent no-op) and spawns the POSIX
    launchers with their stdio discarded. It never raises.
    """
    from .browser import open_url  # lazy: keeps import-time coupling low

    return open_url(url)


def _same_file(a: Path | None, b: Path | None) -> bool:
    if a is None or b is None:
        return False
    try:
        return os.path.normcase(str(Path(a).resolve())) == os.path.normcase(str(Path(b).resolve()))
    except OSError:
        return False


def _absolute(config: HailerConfig, value: str | Path) -> Path:
    """A session/notebook path as an absolute path (relative values are taken from the workspace)."""
    path = Path(value)
    if not path.is_absolute():
        path = Path(config.workspace) / path
    try:
        return path.resolve()
    except OSError:
        return path


def _in_folder(config: HailerConfig, path: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(config.notebooks_root).resolve())
    except (ValueError, OSError):
        return False
    return True


def _session_label(config: HailerConfig, session: MarimoSession) -> str:
    """The session's notebook as shown to the model: workspace-relative when absolute, else as given."""
    raw = session.path or session.filename
    if not raw:
        return "(unsaved notebook)"
    if Path(raw).is_absolute():
        return notebooks.notebook_display_name(config, Path(raw))
    return raw


def _fmt_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _fmt_when(timestamp: float) -> str:
    try:
        return _dt.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
    except (OverflowError, OSError, ValueError):
        return "?"


@dataclasses.dataclass
class _SessionOutcome:
    """What happened while making sure a notebook has a kernel session."""

    client: Any = None
    session: MarimoSession | None = None
    url: str | None = None
    reused: bool = False  # a session already existed; no browser was opened
    opened: bool = False  # the browser was asked to open the URL
    marimo_error: HailerError | None = None  # marimo unreachable (or answered with an error)


class HailerTools:
    """The tool implementations, independent of the MCP transport (unit-testable)."""

    def __init__(
        self,
        config: HailerConfig,
        client_factory: ClientFactory | None = None,
        *,
        open_url: UrlOpener | None = None,
        session_wait_sec: float = DEFAULT_SESSION_WAIT_SEC,
    ) -> None:
        self.config = config
        self._client_factory = client_factory or self._default_client
        #: Opens a URL in the user's browser; injectable so tests never launch one.
        self.open_url: UrlOpener = open_url or _default_open_url
        #: How long to wait for a kernel session after opening the browser.
        self.session_wait_sec = session_wait_sec

    # -- plumbing ----------------------------------------------------------- #

    def _active_config(self) -> HailerConfig:
        """The configuration with ``notebook`` set to the active notebook (re-read every call)."""
        return dataclasses.replace(self.config, notebook=notebooks.load_active_notebook(self.config))

    def _default_client(self) -> tuple[Any, MarimoServer]:
        from . import marimo_client as mc  # lazy

        config = self._active_config()
        server = mc.find_server(config)
        if server is None:
            cmd = " ".join(mc.notebook_launch_command(config))
            raise MarimoUnavailableError(
                "marimo is not running (no server configured or discovered)",
                f"Start it with:\n    {cmd}\nThen open the notebook in your browser.",
            )
        client = mc.MarimoClient(
            server.url,
            token=config.marimo_token,
            notebook=config.notebook,
            workspace=config.workspace,
            notebooks_dir=config.notebooks_root,
        )
        return client, server

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

    def _open_url(self, server: MarimoServer, notebook: Path) -> str:
        try:
            from . import marimo_client as mc

            return mc.open_notebook_url(server, notebook)
        except Exception:  # noqa: BLE001 - best effort
            return server.url

    def _display(self, config: HailerConfig, path: Path) -> str:
        return notebooks.notebook_display_name(config, path)

    def _open_paths(self) -> set[str] | None:
        """Normalised paths of the notebooks that have a session; ``None`` when marimo is unreachable."""
        try:
            client, _ = self._client_factory()
            sessions = client.sessions()
        except HailerError:
            return None
        out: set[str] = set()
        for session in sessions:
            raw = session.path or session.filename
            if raw:
                out.add(os.path.normcase(str(_absolute(self.config, raw))))
        return out

    def _bring_up(self, notebook: Path) -> _SessionOutcome:
        """Make sure ``notebook`` has a kernel session: reuse one, else open the browser and wait."""
        from . import marimo_client as mc  # lazy

        outcome = _SessionOutcome()
        try:
            client, server = self._client_factory()
        except HailerError as err:
            outcome.marimo_error = err
            return outcome
        outcome.client = client
        outcome.url = self._open_url(server, notebook)
        # With a configured marimo_url the server is not health-checked up front, so "marimo is
        # down" first shows up here. The notebook was already created/switched by then; report it
        # as a session outcome rather than letting the error replace the whole reply.
        try:
            outcome.session = client.resolve_session(notebook)
            outcome.reused = True
            return outcome
        except NoSessionError:
            pass
        except HailerError as err:
            outcome.marimo_error = err
            return outcome
        try:
            outcome.opened = bool(self.open_url(outcome.url))
        except Exception as exc:  # noqa: BLE001 - the URL is reported instead
            log.debug("browser opener failed: %s: %s", type(exc).__name__, exc)
            outcome.opened = False
        if outcome.opened:
            try:
                outcome.session = mc.wait_for_session(client, notebook, self.session_wait_sec)
            except HailerError as err:
                outcome.marimo_error = err
        return outcome

    def _session_lines(self, outcome: _SessionOutcome) -> list[str]:
        if outcome.marimo_error is not None:
            err = outcome.marimo_error
            return [
                "marimo is not running (or not reachable), so the notebook has no kernel session yet; "
                "nothing can run in it until marimo is up. Call marimo_status before running code.",
                f"{err}",
                *([err.hint] if err.hint else []),
            ]
        if outcome.session is not None:
            if outcome.reused:
                return [f"It already has a kernel session ({outcome.session.session_id}); the browser was not opened."]
            return [f"Opened it in the browser; kernel session ready ({outcome.session.session_id})."]
        if outcome.opened:
            return [
                f"Opened {outcome.url} in the browser, but no kernel session appeared within {self.session_wait_sec:.0f} s. "
                "Ask the user to check that tab (or open the URL), then call marimo_status before running code."
            ]
        return [
            f"Could not open a browser from here. Ask the user to open {outcome.url}; "
            "the kernel gets a session once the tab loads. Call marimo_status before running code."
        ]

    # -- kernel tools -------------------------------------------------------- #

    def marimo_execute(self, code: str) -> str:
        def go() -> str:
            config = self._active_config()
            client, _ = self._client_factory()
            result = client.execute(code, notebook=config.notebook)
            text = _result_text(result)
            return text if result.success else f"Execution failed.\n{text}"

        return self._run(go)

    def marimo_status(self) -> str:
        def go() -> str:
            config = self._active_config()
            active_name = self._display(config, config.notebook)
            active_session: MarimoSession | None
            try:
                client, server = self._client_factory()
                sessions = client.sessions()
                try:
                    active_session = client.resolve_session(config.notebook)
                except NoSessionError:
                    active_session = None
            except MarimoUnavailableError as err:
                # A configured marimo_url is not health-checked up front, so a dead server can
                # surface here as well as from the factory; the active notebook is named either way.
                return f"marimo: not running\nactive notebook: {active_name}\n{err}\n{err.hint}".rstrip()
            lines = [f"marimo: running at {server.url}" + (f" (version {server.version})" if server.version else "")]
            if active_session is not None:
                lines.append(f"active notebook: {active_name} -> session {active_session.session_id} (ready)")
            else:
                lines.append(
                    f"active notebook: {active_name} has NO session. "
                    f"Ask the user to open {self._open_url(server, config.notebook)} in a browser."
                )
            if sessions:
                lines.append("sessions:")
                for s in sessions:
                    label = _session_label(config, s)
                    raw = s.path or s.filename
                    if active_session is not None and s.session_id == active_session.session_id:
                        marker = "(active notebook)"
                    elif raw and _in_folder(config, _absolute(config, raw)):
                        marker = "(another notebook in the notebooks folder; notebook_open switches to it)"
                    elif raw:
                        marker = "(outside the notebooks folder)"
                    else:
                        marker = "(unsaved)"
                    lines.append(f"  - {s.session_id}: {label} {marker}")
            else:
                lines.append("sessions: none (no notebook is open in a browser)")
            return "\n".join(lines)

        return self._run(go)

    def notebook_cells(self, pattern: str = "") -> str:
        def go() -> str:
            config = self._active_config()
            client, _ = self._client_factory()
            result = client.execute(_list_cells_code(), notebook=config.notebook)
            text = _result_text(result)
            if not result.success:
                return f"Could not list cells.\n{text}"
            if pattern.strip():
                needle = pattern.strip().lower()
                lines = [ln for ln in text.splitlines() if needle in ln.lower()]
                return "\n".join(lines) if lines else f"No cells match '{pattern.strip()}'."
            return text or "(no cells)"

        return self._run(go)

    # -- notebook lifecycle tools ------------------------------------------- #

    def notebook_list(self) -> str:
        def go() -> str:
            config = self._active_config()
            folder = self._display(config, config.notebooks_root)
            infos = notebooks.list_notebooks(config)
            if not infos:
                return f"No notebooks in {folder} yet. Create one with notebook_create(name)."
            open_paths = self._open_paths()
            lines = [f"Notebooks in {folder} ([active] = the one tools act on, [open] = has a kernel session):"]
            for info in infos:
                markers: list[str] = []
                if _same_file(info.path, config.notebook):
                    markers.append("[active]")
                if open_paths is not None and os.path.normcase(str(_absolute(config, info.path))) in open_paths:
                    markers.append("[open]")
                line = f"  - {self._display(config, info.path)}  modified {_fmt_when(info.modified)}  {_fmt_size(info.size)}"
                if markers:
                    line += "  " + " ".join(markers)
                lines.append(line)
            if open_paths is None:
                lines.append("(marimo is not reachable, so which notebooks are open is unknown.)")
            return "\n".join(lines)

        return self._run(go)

    def notebook_create(self, name: str, template: str = "starter") -> str:
        def go() -> str:
            kind = (template or "starter").strip().lower()
            if kind not in notebooks.TEMPLATE_KINDS:
                return f"ERROR: unknown template {template!r}.\nUse 'starter' (imports, data helpers, welcome cell) or 'empty'."
            config = self._active_config()
            path = notebooks.create_notebook(config, name, kind=kind)  # type: ignore[arg-type]
            notebooks.save_active_notebook(config, path)
            display = self._display(config, path)
            lines = [f"Created {display} from the {kind} template; it is now the active notebook."]
            lines.extend(self._session_lines(self._bring_up(path)))
            if kind == "starter":
                lines.append(
                    "The starter template already defines mo, pl, duckdb, Path, WORKSPACE, DATA_DIR, period_files "
                    "and the hailer.periods helpers; add analysis cells below the welcome cell."
                )
            else:
                lines.append(
                    "The empty template defines nothing: its first cell must import what you need "
                    "(import marimo as mo, import polars as pl, ...)."
                )
            return "\n".join(lines)

        return self._run(go)

    def notebook_open(self, notebook: str) -> str:
        def go() -> str:
            config = self._active_config()
            path = notebooks.resolve_notebook(config, notebook)
            notebooks.save_active_notebook(config, path)
            display = self._display(config, path)
            outcome = self._bring_up(path)
            lines = [f"{display} is now the active notebook."]
            lines.extend(self._session_lines(outcome))
            if outcome.session is not None and outcome.client is not None:
                result = outcome.client.execute(_list_cells_code(), notebook=path)
                text = _result_text(result)
                if result.success:
                    lines.append("Cells (id, name, status, errors, first line):")
                    lines.append("(no cells)" if text.strip() in ("", "(no output)") else text)
                else:
                    lines.append(f"Could not list its cells.\n{text}")
            return "\n".join(lines)

        return self._run(go)

    def notebook_close(self, notebook: str = "") -> str:
        def go() -> str:
            config = self._active_config()
            path = notebooks.resolve_notebook(config, notebook) if (notebook or "").strip() else config.notebook
            display = self._display(config, path)
            try:
                client, _ = self._client_factory()
                session = client.resolve_session(path)
            except NoSessionError:
                return f"{display} is not open (no kernel session), so there is nothing to close."
            except MarimoUnavailableError as err:
                return f"marimo is not running, so {display} has no kernel session to close.\n{err}\n{err.hint}".rstrip()
            client.shutdown_session(session.session_id)
            lines = [
                f"Closed the kernel session ({session.session_id}) of {display}; its memory is freed and its browser tab is disconnected."
            ]
            if _same_file(path, config.notebook):
                lines.append(
                    "It stays the active notebook, but nothing can run in it until it is opened again "
                    "(notebook_open reopens it, or notebook_open another notebook to switch)."
                )
            else:
                lines.append(f"The active notebook is still {self._display(config, config.notebook)}.")
            return "\n".join(lines)

        return self._run(go)

    # -- data, skills, web ------------------------------------------------- #

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


def build_server(
    config: HailerConfig,
    *,
    client_factory: ClientFactory | None = None,
    open_url: UrlOpener | None = None,
    session_wait_sec: float = DEFAULT_SESSION_WAIT_SEC,
):
    """Create the MCP server with Hailer's tools registered. Returns an ``mcp.server.MCPServer``."""
    from mcp.server import MCPServer

    tools = HailerTools(config, client_factory, open_url=open_url, session_wait_sec=session_wait_sec)
    server = MCPServer(SERVER_NAME, instructions=SERVER_INSTRUCTIONS, log_level="WARNING")

    @server.tool(name="marimo_execute")
    def marimo_execute(code: str) -> str:
        """Run Python in the active notebook's kernel and return stdout, the result and stderr.

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
        """Report whether marimo is running, which notebook is ACTIVE (the one every kernel tool
        acts on) and whether it has a kernel session (one exists only while the notebook is
        open in a browser), plus every open session. If the active notebook has no session,
        the reply contains the URL the user must open."""
        return tools.marimo_status()

    @server.tool(name="notebook_cells")
    def notebook_cells(pattern: str = "") -> str:
        """List the active notebook's cells: id, name, status, error count and first code line.
        Optional case-insensitive substring filter. Use it before editing to find the cell
        that owns a variable and to avoid creating duplicate cells."""
        return tools.notebook_cells(pattern)

    @server.tool(name="notebook_list")
    def notebook_list() -> str:
        """List the notebooks in the notebooks folder with modified time and size; [active]
        marks the one tools act on and [open] the ones with a kernel session. Call it before
        creating a notebook (to avoid duplicates) and when the user refers to an existing one."""
        return tools.notebook_list()

    @server.tool(name="notebook_create")
    def notebook_create(name: str, template: str = "starter") -> str:
        """Create a new notebook in the notebooks folder from a human name (saved as
        <slug>.py), make it the active notebook, open it in the browser and wait for its
        kernel session. template: 'starter' (imports, data helpers, welcome cell; the usual
        choice) or 'empty'. Only when the user asks for a new or separate notebook."""
        return tools.notebook_create(name, template)

    @server.tool(name="notebook_open")
    def notebook_open(notebook: str) -> str:
        """Open an existing notebook by name, filename or path and make it the active
        notebook for every later tool call. Opens the browser only when it has no kernel
        session yet. Returns its cells so you can continue where the notebook left off."""
        return tools.notebook_open(notebook)

    @server.tool(name="notebook_close")
    def notebook_close(notebook: str = "") -> str:
        """Close a notebook's kernel session (default: the active notebook) to free its
        memory. The notebook stays active until another one is opened; nothing can run in it
        until it is reopened with notebook_open."""
        return tools.notebook_close(notebook)

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
