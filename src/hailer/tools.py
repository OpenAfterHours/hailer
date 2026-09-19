"""Hailer's tools: the agent's only bridge to the live marimo kernel and the web.

The tools run inside the Hailer process. :class:`HailerTools` holds the implementations
(plain methods, unit-testable); :func:`hailer_tools` hands them to the agent as LangChain
tools. A method's docstring is the description the model sees.

Every tool returns plain text. Hailer errors become ``ERROR: <message>\\n<hint>``
text so the model can act on them; nothing is raised into the agent loop.

The *active notebook* (the one every kernel tool acts on) lives in
``<workspace>/.hailer/notebook.json`` and is re-read on every call: both the model
(``notebook_create`` / ``notebook_open``) and the CLI (``/notebook`` commands) may switch it.

The marimo server is the one the chat was pinned to (``hailer notebook`` hands over the server it
started or reused, kept in memory), else it is discovered anew on every call.

Tool results go to the model endpoint, so they never carry the marimo server's token: URLs in
them are token-free (the user gets a signed-in link from ``/notebook`` in the chat), while the
browser Hailer opens itself gets the signed-in URL.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as _dt
import inspect
import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import notebooks
from .errors import HailerError, MarimoUnavailableError, NoSessionError
from .models import KERNEL_RUNTIME_DOCKER, ExecResult, HailerConfig, MarimoServer, MarimoSession
from .web import truncate_text

log = logging.getLogger("hailer.tools")

#: The tools the agent gets, in the order they are offered to the model.
TOOL_NAMES = (
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
)

#: How long notebook_create / notebook_open wait for the browser tab to give the kernel a session.
DEFAULT_SESSION_WAIT_SEC = 30.0
#: Added wherever a tool asks the model to have the user open a URL: the URL in a tool result has
#: no token, so for a server Hailer started the browser would ask for one; /notebook prints the
#: signed-in link.
NOTEBOOK_LINK_TIP = "The user can also run /notebook in the chat to get the link."

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
    if isinstance(err, NoSessionError):  # its hint names a URL the user must open
        text += f"\n{NOTEBOOK_LINK_TIP}"
    return text


def _list_cells_code() -> str:
    try:
        from .marimo_client import LIST_CELLS_CODE  # type: ignore[attr-defined]

        return LIST_CELLS_CODE
    except Exception:  # noqa: BLE001
        return _FALLBACK_LIST_CELLS_CODE


def _default_open_url(url: str) -> bool:
    """Open ``url`` in the user's browser without touching the terminal the chat runs in.

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


def _kernel_data_dir(config: HailerConfig) -> str:
    """The data folder as notebook code reaches it: its mount point in a docker kernel (the host path
    does not exist there), the host path otherwise. ``list_periods`` runs on the host either way."""
    if config.kernel.runtime == KERNEL_RUNTIME_DOCKER:
        from .kernel import KERNEL_DATA_DIR  # lazy: keeps import-time coupling low

        return str(KERNEL_DATA_DIR)
    return str(config.data_dir)


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
    url: str | None = None  # token-free: this one is shown to the model
    reused: bool = False  # a session already existed; no browser was opened
    opened: bool = False  # the browser was asked to open the URL
    marimo_error: HailerError | None = None  # marimo unreachable (or answered with an error)


class HailerTools:
    """The tool implementations, independent of the agent framework (unit-testable).

    The docstring of each public tool method is the description the model is given.
    """

    def __init__(
        self,
        config: HailerConfig,
        client_factory: ClientFactory | None = None,
        *,
        server: MarimoServer | None = None,
        open_url: UrlOpener | None = None,
        session_wait_sec: float = DEFAULT_SESSION_WAIT_SEC,
    ) -> None:
        self.config = config
        #: The server this chat is pinned to (``hailer notebook``); ``None``: discover it per call.
        self.server = server
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
        server = self.server if self.server is not None else mc.find_server(config)
        if server is None:
            raise MarimoUnavailableError("marimo is not running for this workspace (no server configured or found)", mc.launch_hint())
        client = mc.MarimoClient(
            server.url,
            token=server.token or config.marimo_token,
            notebook=config.notebook,
            workspace=config.workspace,
            paths=server.paths,
        )
        return client, server

    def _truncate(self, text: str) -> str:
        return truncate_text(text, self.config.max_tool_output_chars)

    def _run(self, fn: Callable[[], str]) -> str:
        try:
            return self._truncate(fn())
        except HailerError as err:
            return self._truncate(_error_text(err))
        except Exception as exc:  # noqa: BLE001 - never let an exception reach the agent loop
            log.exception("tool failed")
            return self._truncate(f"ERROR: unexpected {type(exc).__name__}: {exc}")

    def _open_url(self, server: MarimoServer, notebook: Path, *, with_token: bool = False) -> str:
        """The notebook's URL: token-free for the model (the default), signed in for the browser."""
        try:
            from . import marimo_client as mc

            if with_token and not server.token and self.config.marimo_token:
                server = dataclasses.replace(server, token=self.config.marimo_token)
            return mc.open_notebook_url(server, notebook, with_token=with_token)
        except Exception:  # noqa: BLE001 - best effort
            return server.url

    def _host_notebook(self, config: HailerConfig, ref: str) -> str:
        """``ref`` as the host knows it: the model sees the kernel's paths (``/work/notebooks/...``
        in a container) and may pass one back; anything else is returned unchanged."""
        text = (ref or "").strip().strip("\"'").strip()
        if not text.startswith("/"):
            return ref
        try:
            _client, server = self._client_factory()
            paths = server.paths
        except HailerError:
            paths = None
        if paths is None and config.kernel.runtime == KERNEL_RUNTIME_DOCKER:
            from .kernel import docker_paths  # lazy: keeps import-time coupling low

            paths = docker_paths(config)
        if paths is None or paths.identity:
            return ref
        host = paths.to_host(text)
        return host if host is not None else ref

    def _kernel_lines(self, server: MarimoServer) -> list[str]:
        """Where notebook code runs and where it finds the folders, from the server in use (a chat
        whose kernel changed underneath still gets the truth)."""
        from .kernel import KERNEL_DATA_DIR, KERNEL_NOTEBOOKS_DIR, describe_runtime  # lazy

        config = self.config
        if server.runtime == KERNEL_RUNTIME_DOCKER:
            kernel = dataclasses.replace(config.kernel, runtime=KERNEL_RUNTIME_DOCKER, network=server.network_access)
            return [
                f"kernel: {describe_runtime(kernel)}",
                f"kernel paths: notebooks folder {KERNEL_NOTEBOOKS_DIR} (writable), data folder {KERNEL_DATA_DIR} "
                "(read-only); use these in code",
            ]
        kernel = dataclasses.replace(config.kernel, runtime="local")
        return [
            f"kernel: {describe_runtime(kernel)}",
            f"kernel paths: notebooks folder {config.notebooks_root}, data folder {config.data_dir}",
        ]

    def _display(self, config: HailerConfig, path: Path) -> str:
        return notebooks.notebook_display_name(config, path)

    def _open_paths(self) -> set[str] | None:
        """Normalised paths of the notebooks that have a session; ``None`` when marimo is unreachable."""
        return self._open_paths_or_error()[0]

    def _open_paths_or_error(self) -> tuple[set[str] | None, HailerError | None]:
        """Like ``_open_paths`` but also returns the error that made marimo unreachable."""
        try:
            client, _ = self._client_factory()
            sessions = client.sessions()
        except HailerError as err:
            return None, err
        out: set[str] = set()
        for session in sessions:
            raw = session.path or session.filename
            if raw:
                out.add(os.path.normcase(str(_absolute(self.config, raw))))
        return out, None

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
        browser_url = self._open_url(server, notebook, with_token=True)
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
            outcome.opened = bool(self.open_url(browser_url))
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
                f"Ask the user to check that tab (or open the URL), then call marimo_status before running code. {NOTEBOOK_LINK_TIP}"
            ]
        return [
            f"Could not open a browser from here. Ask the user to open {outcome.url}; "
            f"the kernel gets a session once the tab loads. Call marimo_status before running code. {NOTEBOOK_LINK_TIP}"
        ]

    # -- kernel tools -------------------------------------------------------- #

    def marimo_execute(self, code: str) -> str:
        """Run Python in the active notebook's kernel and return stdout, the result and stderr.

        The code runs in the kernel's scratchpad: notebook variables are readable by name, but
        new top-level assignments are discarded afterwards. To add or change notebook cells,
        use `import marimo._code_mode as cm` and `async with cm.get_context() as ctx:` with
        ctx.create_cell(...)/ctx.edit_cell(...)/ctx.run_cell(...) (top-level `async with` is
        allowed). Keep printed output small: schemas, head(), aggregates. Never print whole
        datasets.
        """

        def go() -> str:
            config = self._active_config()
            client, _ = self._client_factory()
            result = client.execute(code, notebook=config.notebook)
            text = _result_text(result)
            return text if result.success else f"Execution failed.\n{text}"

        return self._run(go)

    def marimo_status(self) -> str:
        """Report whether marimo is running, which notebook is ACTIVE (the one every kernel tool
        acts on) and whether it has a kernel session (one exists only while the notebook is
        open in a browser), plus every open session. If the active notebook has no session,
        the reply contains the URL the user must open."""

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
            lines += self._kernel_lines(server)
            if active_session is not None:
                lines.append(f"active notebook: {active_name} -> session {active_session.session_id} (ready)")
            else:
                lines.append(
                    f"active notebook: {active_name} has NO session. "
                    f"Ask the user to open {self._open_url(server, config.notebook)} in a browser. {NOTEBOOK_LINK_TIP}"
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
        """List the active notebook's cells: id, name, status, error count and first code line.
        Optional case-insensitive substring filter. Use it before editing to find the cell
        that owns a variable and to avoid creating duplicate cells."""

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
        """List the notebooks in the notebooks folder with modified time and size; [active]
        marks the one tools act on and [open] the ones with a kernel session. Call it before
        creating a notebook (to avoid duplicates) and when the user refers to an existing one."""

        def go() -> str:
            config = self._active_config()
            folder = self._display(config, config.notebooks_root)
            infos = notebooks.list_notebooks(config)
            if not infos:
                return f"No notebooks in {folder} yet. Create one with notebook_create(name)."
            open_paths, marimo_error = self._open_paths_or_error()
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
                if marimo_error is not None:
                    lines.append(str(marimo_error))
                    if marimo_error.hint:
                        lines.append(marimo_error.hint)
            return "\n".join(lines)

        return self._run(go)

    def notebook_create(self, name: str, template: str = "starter") -> str:
        """Create a new notebook in the notebooks folder from a human name (saved as
        <slug>.py), make it the active notebook, open it in the browser and wait for its
        kernel session. template: 'starter' (imports, data helpers, welcome cell; the usual
        choice) or 'empty'. Only when the user asks for a new or separate notebook."""

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
                    "The starter template already defines mo, pl, duckdb, Path, WORKSPACE, DATA_DIR, data_files, "
                    "period_files and the hailer.periods helpers; add analysis cells below the welcome cell."
                )
            else:
                lines.append(
                    "The empty template defines nothing: its first cell must import what you need "
                    "(import marimo as mo, import polars as pl, ...)."
                )
            return "\n".join(lines)

        return self._run(go)

    def notebook_open(self, notebook: str) -> str:
        """Open an existing notebook by name, filename or path and make it the active
        notebook for every later tool call. Opens the browser only when it has no kernel
        session yet. Returns its cells so you can continue where the notebook left off."""

        def go() -> str:
            config = self._active_config()
            path = notebooks.resolve_notebook(config, self._host_notebook(config, notebook))
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
        """Close a notebook's kernel session (default: the active notebook) to free its
        memory. The notebook stays active until another one is opened; nothing can run in it
        until it is reopened with notebook_open."""

        def go() -> str:
            config = self._active_config()
            ref = self._host_notebook(config, notebook) if (notebook or "").strip() else ""
            path = notebooks.resolve_notebook(config, ref) if ref else config.notebook
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
        """Describe the data directory: monthly files named like '25-01 sales.parquet'
        (YY-MM then the dataset name) with the periods available, common columns and columns
        that only appear in some months, plus the names of any other data files (CSV, JSON,
        ...). Optional dataset name filter for the monthly files, e.g. 'sales'."""

        def go() -> str:
            from . import periods  # lazy

            data_dir = self.config.data_dir
            shown = _kernel_data_dir(self.config)
            if not data_dir.is_dir():
                return f"Data directory does not exist: {shown}"
            files = periods.scan_period_files(data_dir, name.strip() or None)
            if files:
                text = periods.describe_periods(files)
            else:
                text = (
                    f"No period files found in {shown}"
                    + (f" for '{name.strip()}'" if name.strip() else "")
                    + ". Monthly files are named like '25-01 sales.parquet' (YY-MM then the dataset name)."
                )
            listed = {pf.path.name for pf in files}
            others = [p.name for p in periods.list_data_files(data_dir) if p.name not in listed]
            if others:
                shown = ", ".join(others[:20]) + (f" (and {len(others) - 20} more)" if len(others) > 20 else "")
                text += f"\nOther data files (load them directly with Polars or DuckDB): {shown}"
            return text

        return self._run(go)

    def load_skill(self, name: str) -> str:
        """Load a project skill (its SKILL.md instructions and the list of bundled files) by
        name. Use when the task matches a skill listed in your instructions."""
        from .context import read_skill

        return self._run(lambda: read_skill(self.config, name))

    def read_skill_file(self, name: str, path: str) -> str:
        """Read a file bundled with a skill, by skill name and path relative to the skill
        folder (for example 'reference/checks.md')."""
        from .context import read_skill_file

        return self._run(lambda: read_skill_file(self.config, name, path))

    def fetch_page(self, url: str) -> str:
        """Fetch a web page as readable text. Only hosts in the project's allowed-domains list
        can be fetched; a denied request says which domains are allowed. Fetched text is
        sent to the model like any other tool result."""
        from .web import fetch_page

        return self._run(lambda: fetch_page(url, self.config))


# --------------------------------------------------------------------------- #
# LangChain registration
# --------------------------------------------------------------------------- #


def _in_daemon_thread(fn: Callable[..., str]) -> Callable[..., Any]:
    """An async twin of a blocking tool that runs it on a daemon thread.

    Kernel calls can block for minutes. LangChain would otherwise run them on the event loop's
    default executor, whose threads are joined when the loop closes and again at interpreter
    exit, so quitting after Ctrl+C waited for the abandoned call (measured: 7 s for a call with
    7 s left). A daemon thread lets the cancelled turn and the process end at once; the call
    itself finishes (or times out) in the background and its result is dropped.
    """

    async def run(**kwargs: Any) -> str:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()

        def deliver(result: str | None, error: BaseException | None) -> None:
            if future.done():  # the turn was cancelled meanwhile
                return
            if error is not None:
                future.set_exception(error)
            else:
                future.set_result(result or "")

        def work() -> None:
            try:
                outcome: tuple[str | None, BaseException | None] = (fn(**kwargs), None)
            except BaseException as exc:  # noqa: BLE001 - handed to the awaiting turn
                outcome = (None, exc)
            try:
                loop.call_soon_threadsafe(deliver, *outcome)
            except RuntimeError:  # the loop is closed: Hailer is shutting down
                pass

        threading.Thread(target=work, name=f"hailer-tool-{fn.__name__}", daemon=True).start()
        return await future

    return run


def hailer_tools(
    config: HailerConfig,
    *,
    server: MarimoServer | None = None,
    client_factory: ClientFactory | None = None,
    open_url: UrlOpener | None = None,
    session_wait_sec: float = DEFAULT_SESSION_WAIT_SEC,
) -> list[Any]:
    """Hailer's tools as LangChain tools (``TOOL_NAMES`` order), bound to ``config`` (and pinned to
    ``server`` when given)."""
    from langchain_core.tools import StructuredTool  # lazy: keeps `hailer --help` and friends fast

    impl = HailerTools(config, client_factory, server=server, open_url=open_url, session_wait_sec=session_wait_sec)
    out: list[Any] = []
    for name in TOOL_NAMES:
        method = getattr(impl, name)
        out.append(
            StructuredTool.from_function(
                func=method,
                coroutine=_in_daemon_thread(method),
                name=name,
                description=inspect.getdoc(method) or name,
            )
        )
    return out

