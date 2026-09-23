"""Hailer's tools: the agent's only bridge to the live marimo kernel and the web.

The tools run inside the Hailer process. :class:`HailerTools` holds the implementations
(plain methods, unit-testable); :func:`hailer_tools` hands them to the agent as LangChain
tools. A method's docstring is the description the model sees.

Every tool returns plain text. Hailer errors become ``ERROR: <message>\\n<hint>``
text so the model can act on them; nothing is raised into the agent loop.

The kernel is the :class:`~hailer.sandbox.MarimoSandbox` this Hailer process started, handed over by
the chat and kept in memory; the tools never look for another. Everything they know about files
comes from it: notebooks are names relative to its notebooks folder (``"sales.py"``,
``"q3/review.py"``) and data files are names in its data folder. The tools never read or write
either folder themselves.

The *active notebook* (the one every kernel tool acts on) is a name in
``<workspace>/.hailer/notebook.json`` and is re-read on every call: both the model
(``notebook_create`` / ``notebook_open``) and the CLI (``/notebook`` commands) may switch it.

Tool results go to the model endpoint, so they never carry the marimo server's token: URLs in
them are token-free (the user gets a signed-in link from ``/notebook`` in the chat), while the
browser Hailer opens itself gets the signed-in URL.

A ``marimo_execute`` call that creates or edits cells is bracketed by two snapshots of the
notebook's cells (:mod:`hailer.code_checks`): the first also switches on code mode's ruff
formatting in the kernel, and the cells that differ afterwards are checked with ruff and ty, the
findings appended to the result (``[checks]`` in ``hailer.toml`` turns each part off).
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as _dt
import inspect
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import code_checks, notebooks
from .errors import HailerError, MarimoUnavailableError, NoSessionError
from .models import ExecResult, HailerConfig, MarimoSession
from .web import TRUNCATED_MARKER, truncate_text

if TYPE_CHECKING:  # pragma: no cover - annotations only; the sandbox module imports marimo_client
    from .sandbox import MarimoSandbox

log = logging.getLogger("hailer.tools")

#: The tools the agent gets, in the order they are offered to the model.
TOOL_NAMES = (
    "marimo_execute",
    "marimo_status",
    "notebook_cells",
    "notebook_check",
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


def _writes_cells(code: str) -> bool:
    """Whether scratchpad ``code`` creates or edits notebook cells through code mode."""
    return "create_cell" in code or "edit_cell" in code


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


def _session_label(session: MarimoSession) -> str:
    """The session's notebook as shown to the model: its name, else the path marimo reported."""
    return session.name or session.path or session.filename or "(unsaved notebook)"


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
        sandbox: MarimoSandbox | None = None,
        *,
        open_url: UrlOpener | None = None,
        session_wait_sec: float = DEFAULT_SESSION_WAIT_SEC,
    ) -> None:
        self.config = config
        #: The sandbox this process started (:class:`~hailer.sandbox.MarimoSandbox`); ``None``: no kernel
        #: (every kernel tool says so).
        self.sandbox = sandbox
        #: Opens a URL in the user's browser; injectable so tests never launch one.
        self.open_url: UrlOpener = open_url or _default_open_url
        #: How long to wait for a kernel session after opening the browser.
        self.session_wait_sec = session_wait_sec

    # -- plumbing ----------------------------------------------------------- #

    def _active(self) -> str:
        """The active notebook's name (re-read from the state file every call)."""
        return notebooks.load_active_notebook(self.config)

    def _kernel(self) -> Any:
        """The sandbox, or ``MarimoUnavailableError`` when this session has none."""
        if self.sandbox is None:
            from .marimo_client import launch_hint  # lazy

            raise MarimoUnavailableError("marimo is not running: this session has no kernel", launch_hint())
        return self.sandbox

    def _client(self, notebook: str | None = None) -> Any:
        return self._kernel().client(notebook=notebook)

    def _truncate(self, text: str) -> str:
        return truncate_text(text, self.config.max_tool_output_chars)

    def _with_footer(self, text: str, footer: str) -> str:
        """``text`` then ``footer``, ``text`` shortened so the footer survives the output cap."""
        if not footer:
            return text
        limit = self.config.max_tool_output_chars
        if limit > 0:
            room = limit - len(footer) - 2
            text = truncate_text(text, max(room, len(TRUNCATED_MARKER) + 20))
        return f"{text}\n\n{footer}"

    # -- code checks -------------------------------------------------------- #

    def _snapshot(self, client: Any, notebook: str, *, enable_format: bool = False) -> code_checks.Snapshot | None:
        """The notebook's cells, read from the kernel; ``None`` when they could not be read."""
        try:
            result = client.execute(code_checks.snapshot_code(enable_format=enable_format), notebook=notebook)
        except HailerError as err:
            log.debug("cell snapshot failed: %s", err)
            return None
        snapshot = code_checks.parse_snapshot(result.stdout) if result.success else None
        if snapshot is None:  # the reason only: the output may hold the notebook's code
            log.debug("cell snapshot unreadable: %s", ((result.stderr or "").strip().splitlines() or ["no snapshot line"])[-1])
        return snapshot

    def _check_report(self, snapshot: code_checks.Snapshot, cell_ids: list[str], *, changed: bool) -> str:
        # The cells come from the kernel; no search path is passed, because the notebooks folder the
        # kernel imports local modules from is the sandbox's, not a folder on this machine.
        checks = self.config.checks
        result = code_checks.check(snapshot, cell_ids, lint=checks.lint, typecheck=checks.typecheck)
        return code_checks.report(result, snapshot, cell_ids, changed=changed)

    def _run(self, fn: Callable[[], str]) -> str:
        try:
            return self._truncate(fn())
        except HailerError as err:
            return self._truncate(_error_text(err))
        except Exception as exc:  # noqa: BLE001 - never let an exception reach the agent loop
            log.exception("tool failed")
            return self._truncate(f"ERROR: unexpected {type(exc).__name__}: {exc}")

    def _url(self, notebook: str, *, with_token: bool = False) -> str:
        """The notebook's URL: token-free for the model (the default), signed in for the browser."""
        sandbox = self._kernel()
        try:
            return sandbox.notebook_url(notebook, with_token=with_token)
        except Exception:  # noqa: BLE001 - best effort
            return sandbox.server.url

    def _resolve(self, ref: str) -> str:
        """``ref`` as the name of an existing notebook: the model may pass a name, a file name or the
        kernel's path (``/work/notebooks/...`` in a container)."""
        sandbox = self._kernel()
        names = [info.name for info in sandbox.list_notebooks()]
        return notebooks.resolve_notebook(ref, names, folders=notebooks.reference_folders(self.config, sandbox.notebooks_path))

    def _kernel_lines(self) -> list[str]:
        """Where notebook code runs and where it finds the folders, from the sandbox in use."""
        sandbox = self._kernel()
        return [
            f"kernel: {sandbox.describe()}",
            f"kernel paths: notebooks folder {sandbox.notebooks_path}, data folder {sandbox.data_path}; use these in code",
        ]

    def _open_names_or_error(self) -> tuple[set[str] | None, HailerError | None]:
        """The names of the notebooks that have a session (``None`` when marimo is unreachable)
        and the error that made it unreachable."""
        try:
            sessions = self._client().sessions()
        except HailerError as err:
            return None, err
        return {s.name for s in sessions if s.name}, None

    def _bring_up(self, notebook: str) -> _SessionOutcome:
        """Make sure ``notebook`` has a kernel session: reuse one, else open the browser and wait."""
        from . import marimo_client as mc  # lazy

        outcome = _SessionOutcome()
        try:
            client = self._client(notebook)
        except HailerError as err:
            outcome.marimo_error = err
            return outcome
        outcome.client = client
        outcome.url = self._url(notebook)
        browser_url = self._url(notebook, with_token=True)
        # A kernel that stopped (or went away) first shows up here. The notebook was already
        # created/switched by then; report it as a session outcome rather than letting the error
        # replace the whole reply.
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
        allowed). New and edited cells are formatted with ruff, and when the call changed cells
        the result ends with ruff and ty findings for them. Keep printed output small: schemas,
        head(), aggregates. Never print whole datasets.
        """

        def go() -> str:
            active = self._active()
            client = self._client(active)
            checks = self.config.checks
            before = None
            if (checks.lint or checks.typecheck or checks.format) and _writes_cells(code):
                before = self._snapshot(client, active, enable_format=checks.format)
            result = client.execute(code, notebook=active)
            text = _result_text(result)
            text = text if result.success else f"Execution failed.\n{text}"
            if before is None or not (checks.lint or checks.typecheck):
                return text
            after = self._snapshot(client, active)
            changed = code_checks.changed_cells(before, after) if after is not None else []
            if after is None or not changed:
                return text
            return self._with_footer(text, self._check_report(after, changed, changed=True))

        return self._run(go)

    def marimo_status(self) -> str:
        """Report whether marimo is running, which notebook is ACTIVE (the one every kernel tool
        acts on) and whether it has a kernel session (one exists only while the notebook is
        open in a browser), plus every open session. If the active notebook has no session,
        the reply contains the URL the user must open."""

        def go() -> str:
            active = self._active()
            active_session: MarimoSession | None
            try:
                sandbox = self._kernel()
                client = self._client(active)
                sessions = client.sessions()
                try:
                    active_session = client.resolve_session(active)
                except NoSessionError:
                    active_session = None
            except MarimoUnavailableError as err:
                # A kernel that stopped surfaces here as well as without a sandbox; the active
                # notebook is named either way.
                return f"marimo: not running\nactive notebook: {active}\n{err}\n{err.hint}".rstrip()
            lines = [f"marimo: running at {sandbox.server.url}"]
            lines += self._kernel_lines()
            if active_session is not None:
                lines.append(f"active notebook: {active} -> session {active_session.session_id} (ready)")
            else:
                lines.append(
                    f"active notebook: {active} has NO session. "
                    f"Ask the user to open {self._url(active)} in a browser. {NOTEBOOK_LINK_TIP}"
                )
            if sessions:
                lines.append("sessions:")
                for s in sessions:
                    if active_session is not None and s.session_id == active_session.session_id:
                        marker = "(active notebook)"
                    elif s.name:
                        marker = "(another notebook in the notebooks folder; notebook_open switches to it)"
                    elif s.path or s.filename:
                        marker = "(outside the notebooks folder)"
                    else:
                        marker = "(unsaved)"
                    lines.append(f"  - {s.session_id}: {_session_label(s)} {marker}")
            else:
                lines.append("sessions: none (no notebook is open in a browser)")
            return "\n".join(lines)

        return self._run(go)

    def notebook_cells(self, pattern: str = "") -> str:
        """List the active notebook's cells: id, name, status, error count and first code line.
        Optional case-insensitive substring filter. Use it before editing to find the cell
        that owns a variable and to avoid creating duplicate cells."""

        def go() -> str:
            active = self._active()
            result = self._client(active).execute(_list_cells_code(), notebook=active)
            text = _result_text(result)
            if not result.success:
                return f"Could not list cells.\n{text}"
            if pattern.strip():
                needle = pattern.strip().lower()
                lines = [ln for ln in text.splitlines() if needle in ln.lower()]
                return "\n".join(lines) if lines else f"No cells match '{pattern.strip()}'."
            return text or "(no cells)"

        return self._run(go)

    def notebook_check(self) -> str:
        """Check every cell of the active notebook with ruff (lint) and ty (types) and list the
        findings by cell and line. Cells you create or edit are checked automatically, so use
        this for the notebook as a whole: when the user asks for a review, or before building
        on a notebook written elsewhere."""

        def go() -> str:
            if not (self.config.checks.lint or self.config.checks.typecheck):
                return "Code checks are switched off ([checks] lint and typecheck are false in hailer.toml)."
            active = self._active()
            result = self._client(active).execute(code_checks.snapshot_code(), notebook=active)
            snapshot = code_checks.parse_snapshot(result.stdout) if result.success else None
            if snapshot is None:
                return f"Could not read the notebook's cells.\n{_result_text(result)}"
            cell_ids = [c.id for c in snapshot.cells if c.code.strip()]
            if not cell_ids:
                return "The notebook has no code to check."
            return self._check_report(snapshot, cell_ids, changed=False)

        return self._run(go)

    # -- notebook lifecycle tools ------------------------------------------- #

    def notebook_list(self) -> str:
        """List the notebooks in the notebooks folder with modified time and size; [active]
        marks the one tools act on and [open] the ones with a kernel session. Call it before
        creating a notebook (to avoid duplicates) and when the user refers to an existing one."""

        def go() -> str:
            active = self._active()
            sandbox = self._kernel()
            folder = sandbox.notebooks_path
            infos = sandbox.list_notebooks()
            if not infos:
                return f"No notebooks in {folder} yet. Create one with notebook_create(name)."
            open_names, marimo_error = self._open_names_or_error()
            lines = [f"Notebooks in {folder} ([active] = the one tools act on, [open] = has a kernel session):"]
            for info in infos:
                markers: list[str] = []
                if notebooks.same_name(info.name, active):
                    markers.append("[active]")
                if open_names is not None and info.name in open_names:
                    markers.append("[open]")
                line = f"  - {info.name}  modified {_fmt_when(info.modified)}  {_fmt_size(info.size)}"
                if markers:
                    line += "  " + " ".join(markers)
                lines.append(line)
            if open_names is None:
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
            filename, source = notebooks.new_notebook(name, kind=kind)  # type: ignore[arg-type]
            created = self._kernel().write_notebook(filename, source, replace=False).name
            notebooks.save_active_notebook(self.config, created)
            lines = [f"Created {created} from the {kind} template; it is now the active notebook."]
            lines.extend(self._session_lines(self._bring_up(created)))
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
            name = self._resolve(notebook)
            notebooks.save_active_notebook(self.config, name)
            outcome = self._bring_up(name)
            lines = [f"{name} is now the active notebook."]
            lines.extend(self._session_lines(outcome))
            if outcome.session is not None and outcome.client is not None:
                result = outcome.client.execute(_list_cells_code(), notebook=name)
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
            active = self._active()
            try:
                name = self._resolve(notebook) if (notebook or "").strip() else active
                client = self._client(name)
                session = client.resolve_session(name)
            except NoSessionError:
                return f"{name} is not open (no kernel session), so there is nothing to close."
            except MarimoUnavailableError as err:
                shown = (notebook or "").strip() or active
                return f"marimo is not running, so {shown} has no kernel session to close.\n{err}\n{err.hint}".rstrip()
            client.shutdown_session(session.session_id)
            lines = [
                f"Closed the kernel session ({session.session_id}) of {name}; its memory is freed and its browser tab is disconnected."
            ]
            if notebooks.same_name(name, active):
                lines.append(
                    "It stays the active notebook, but nothing can run in it until it is opened again "
                    "(notebook_open reopens it, or notebook_open another notebook to switch)."
                )
            else:
                lines.append(f"The active notebook is still {active}.")
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

            sandbox = self._kernel()
            shown = sandbox.data_path
            files = sandbox.list_data()
            wanted = name.strip().lower() or None
            found = []
            for info in files:
                pf = periods.parse_period_file(Path(info.name)) if info.name.lower().endswith(".parquet") else None
                if pf is not None and (wanted is None or pf.stem == wanted):
                    found.append(pf)
            if found:
                text = periods.describe_periods(found, schema=lambda pf: sandbox.data_schema(pf.path.name))
            else:
                text = (
                    f"No period files found in {shown}"
                    + (f" for '{name.strip()}'" if name.strip() else "")
                    + ". Monthly files are named like '25-01 sales.parquet' (YY-MM then the dataset name)."
                )
            listed = {pf.path.name for pf in found}
            others = [f.name for f in files if f.name.lower().endswith(periods.DATA_SUFFIXES) and f.name not in listed]
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
    sandbox: MarimoSandbox | None = None,
    open_url: UrlOpener | None = None,
    session_wait_sec: float = DEFAULT_SESSION_WAIT_SEC,
) -> list[Any]:
    """Hailer's tools as LangChain tools (``TOOL_NAMES`` order), bound to ``config`` and to
    ``sandbox``, the kernel this process started."""
    from langchain_core.tools import StructuredTool  # lazy: keeps `hailer --help` and friends fast

    impl = HailerTools(config, sandbox, open_url=open_url, session_wait_sec=session_wait_sec)
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

