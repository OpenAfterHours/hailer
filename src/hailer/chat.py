"""Shared session operations for the inline composer and the plain chat loop.

CLI collaborators are injected so both presentations use the same notebook, context
and credential operations. Only the synchronous facade drives an asyncio.Runner.
"""

from __future__ import annotations

import asyncio
import textwrap
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console, RenderableType
from rich.table import Table
from rich.text import Text

from hailer import notebooks
from hailer.errors import HailerError, MarimoUnavailableError, NoSessionError
from hailer.marimo_client import request_deadline
from hailer.models import CompletionData, ContextBundle, HailerConfig, MarimoSession, ProviderConfig, SessionState, SkillInfo, TurnSummary
from hailer.session import EXIT_COMMANDS, help_text, load_session, parse_command, save_session
from hailer.statedir import input_history_path
from hailer.steps import plural, single_line
from hailer.transcript import FooterData, NotebookRow, console_glyphs, detail_card, help_card, notebook_table, status_card
from hailer.transcript import session_note, session_row

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from hailer.sandbox import MarimoSandbox

#: The command menu's background notebook listing: all its marimo requests together.
NAMES_DEADLINE_SEC = 10.0
#: How long the end of a chat waits for that listing before leaving it behind.
NAMES_SETTLE_SEC = 1.0


class _CommandStopped(Exception):
    """A blocking notebook operation reached a cooperative cancellation point."""


def _when(timestamp: float) -> str:
    """A file time in the composer's notebook list: "today 14:02", "yesterday", "12 Oct", "12 Oct 2025"."""
    moment, today = datetime.fromtimestamp(timestamp), datetime.now()
    days = (today.date() - moment.date()).days
    if days == 0:
        return f"today {moment:%H:%M}"
    if days == 1:
        return "yesterday"
    if moment.year == today.year:
        return f"{moment.day} {moment:%b}"
    return f"{moment.day} {moment:%b %Y}"


@dataclass(frozen=True)
class _Opened:
    """What :meth:`ChatController._ensure_session` came to: the session and client, the notebook's
    state in words for its row, whether that is fine, a note and the signed-in link to open by hand."""

    session: MarimoSession | None = None
    client: Any = None
    state: str = "not open in a browser yet"
    ok: bool = False
    note: str = ""
    link: str = ""


class ChatController:
    def __init__(self, console: Console, config: HailerConfig, opts: Any, *, services: Any, sandbox: MarimoSandbox | None = None) -> None:
        self._cli = services
        self.console = console
        self.config = config
        self.opts = opts
        #: The kernel this chat started (:class:`~hailer.sandbox.MarimoSandbox`); ``None``: no kernel.
        #: Its warnings (notebook copies, from any thread) go to this chat's current console.
        self.sandbox = sandbox
        if sandbox is not None:
            sandbox.notice = lambda text: self.console.print(text, style="yellow", markup=False)
        #: The active notebook's name (the state file is the truth; see _sync_active_notebook).
        self.notebook: str = notebooks.load_active_notebook(config)
        #: Whether it has a kernel session (a browser tab), as last learned (startup, a switch,
        #: /status, /notebook, /exec); None: not known. The composer's footer shows it.
        self.notebook_open: bool | None = None
        #: True while the inline composer runs (cards instead of plain lines).
        self.interactive = False
        #: False with --no-browser: a switch prints the notebook's signed-in link instead of opening it.
        self.open_browser = True
        self._link_cache: tuple[str | None, str] = (None, "")
        #: The notebook before a switch a turn's tools made, until _sync_active_notebook reports it.
        self._switched_from: str | None = None
        self.state: SessionState = load_session(config.workspace)
        #: This session's turns and tokens (the state's counters span sessions and restart at /new):
        #: the CLI's exit summary.
        self.session_turns = 0
        self.session_tokens = 0
        self.bundle: ContextBundle = ContextBundle()
        self.agent: Any = None
        self.startup_failed = False
        self.reader = (
            self._cli._LineReader(console, interactive=False)
            if opts.plain
            else self._cli._make_line_reader(console)
        )
        self.reader.history_file = self._input_history_file()
        #: The sandbox's notebooks as last listed, never on the UI loop (a /notebook command's own
        #: listing, or _refresh_names_soon after startup and turns): the command menu's names.
        self._notebook_names: tuple[str, ...] = ()
        self._names_task: asyncio.Task[None] | None = None
        self._names_again = False
        self._names_closed = False
        # One-line notice sent with the next user message after a /notebook switch (the model
        # learns about switches it made itself from its own tool results).
        self._pending_preamble: str | None = None
        self._stop_event: threading.Event | None = None

    # -- lifecycle --------------------------------------------------------- #

    def start(self) -> None:
        began = time.monotonic()
        self.bundle = self._cli._load_context(self.config)
        self._print_context_warnings()
        self.agent = self._cli._make_agent(self.config, self.bundle, self.sandbox)
        resume = None if self.opts.new_thread else self.state.thread_id
        forget = self.state.thread_id if self.opts.new_thread else None  # --new: drop the stored conversation
        thread_id = self.agent.start(resume_thread_id=resume, forget_thread_id=forget)
        self._started(thread_id, resume, time.monotonic() - began)

    def _started(self, thread_id: str, resume: str | None, took: float | None = None) -> None:
        """Record the thread the agent started on, then the startup rows: ``✓ Agent  model ·
        provider · key source`` and whether the conversation was resumed."""
        resumed: tuple[str, str, str] | None = None
        if resume and thread_id != resume:
            resumed = ("warn", "Thread", "previous conversation could not be resumed; started a new one")
            self.state = SessionState()
        elif resume:
            resumed = ("resume", "Resumed", self._glyphs().join(plural(self.state.turns, "earlier turn"), "/new starts fresh"))
        elif self.opts.new_thread:
            self.state = SessionState()
        self.state.thread_id = thread_id
        # The agent starts with configured settings, even when the stored thread's last
        # session used /model. Report the endpoint actually used by this process.
        model = getattr(self.agent, "model", None)
        provider = getattr(self.agent, "provider_id", None)
        self.state.model = model if isinstance(model, str) else self.config.model.name
        self.state.provider = provider if isinstance(provider, str) else self.config.model.provider
        save_session(self.config.workspace, self.state)
        self.console.print(self._agent_row(took))
        if resumed is not None:
            self.console.print(session_row(self._glyphs(), *resumed))

    def _agent_row(self, took: float | None) -> RenderableType:
        """``✓ Agent  gpt-5.5 · openai · OPENAI_API_KEY from keyring``. The key's source is the one
        the agent resolved: nothing is looked up here (this runs on the composer's loop)."""
        provider = self._active_provider()
        source = getattr(self.agent, "key_source", None)
        key = ""
        if provider is not None and isinstance(source, str) and source:
            key = f"{provider.env_key or 'API key'} " + ("missing" if source == "missing" else f"from {source}")
        g = self._glyphs()
        status = "warn" if source == "missing" else "ok"
        return session_row(g, status, "Agent", g.join(single_line(self.state.model or ""), single_line(self.state.provider or ""), key), took)

    @contextmanager
    def progress(self, label: str, detail: str = "") -> Iterator[None]:
        """Show ``label`` and ``detail`` on the composer's live row while the block runs, from any
        thread (``Notebook  opening analysis.py in your browser``); plain mode, which has no live
        row, prints them once as a ``○`` row when the block starts."""
        ui = getattr(self, "ui", None) if self.interactive else None
        if ui is None:
            self.console.print(session_row(self._glyphs(), "wait", label, detail))
            yield
            return
        key = object()
        ui.begin_task(key, label, detail)
        try:
            yield
        finally:
            ui.end_task(key)

    def close(self) -> None:
        if self.agent is not None:
            try:
                self.agent.close()
            except Exception:  # pragma: no cover - best effort
                pass

    def _print_context_warnings(self) -> None:
        for warning in self.bundle.warnings:
            self.console.print(session_row(self._glyphs(), "warn", "Context", warning))

    # -- persistent interactive session ----------------------------------- #

    async def _blocking(self, function: Any, *args: Any, **kwargs: Any) -> Any:
        """Keep blocking notebook work off the UI loop, but serialize its lifetime.

        Cancelling to_thread does not stop its worker. Keep the command busy until
        that worker settles so a second command cannot race its state changes.
        """
        stop = threading.Event()
        self._stop_event = stop
        task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            stop.set()
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not task.cancelled():
                task.exception()  # retrieve a worker failure while preserving cancellation
            raise
        finally:
            self._stop_event = None

    def _check_stopped(self) -> None:
        if self._stop_event is not None and self._stop_event.is_set():
            raise _CommandStopped()

    async def astart(self) -> None:
        began = time.monotonic()
        self.bundle = await self._blocking(self._cli._load_context, self.config)
        self._print_context_warnings()
        self.agent = self._cli._make_agent(self.config, self.bundle, self.sandbox)
        resume = None if self.opts.new_thread else self.state.thread_id
        forget = self.state.thread_id if self.opts.new_thread else None
        thread_id = await self.agent.astart(resume_thread_id=resume, forget_thread_id=forget)
        self._started(thread_id, resume, time.monotonic() - began)

    def footer(self) -> FooterData:
        """The composer's footer, from state this controller already holds: it is read on every
        redraw, so it never asks marimo, the agent or the disk anything."""
        return FooterData(
            notebook=self.notebook,
            notebook_open=self.notebook_open,
            kernel=self.config.kernel.runtime,
            model=self.state.model or self.config.model.name,
            turns=self.state.turns,
            tokens=self.state.input_tokens + self.state.output_tokens,
            notebook_url=self._notebook_link(),
        )

    def completions(self) -> CompletionData:
        """The command menu's arguments, from state this controller already holds (like ``footer``:
        it is read while a slash command is typed)."""
        return CompletionData(
            notebooks=self._notebook_names,
            active_notebook=self.notebook,
            skills=tuple((skill.name, skill.description) for skill in self.bundle.skills),
            prompts=tuple(sorted(self.bundle.prompts)),
            model=self.state.model or self.config.model.name,
            providers=tuple(sorted({"openai", *self.config.providers})),
        )

    def _refresh_names_soon(self) -> None:
        """List the notebooks for the command menu in the background, nothing waiting for it (call on
        the UI loop). One listing at a time: a request during one makes it list once more after."""
        if self._names_closed or self.sandbox is None:
            return
        if self._names_task is not None and not self._names_task.done():
            self._names_again = True
            return
        self._names_task = asyncio.ensure_future(self._list_names())

    async def _list_names(self) -> None:
        while True:
            self._names_again = False
            names = await self._listing()
            if self._names_closed:
                return  # the session is over: its kernel may be stopping
            if names is not None:
                self._notebook_names = names
            if not self._names_again:
                return

    def _listing(self) -> asyncio.Future[tuple[str, ...] | None]:
        """The sandbox's notebook names (None if they could not be listed), from a daemon thread
        whose marimo requests all end within NAMES_DEADLINE_SEC (a kernel can trickle a reply).
        Not the loop's default executor, which asyncio.run waits for at exit (LEARNINGS §4)."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[str, ...] | None] = loop.create_future()

        def deliver(names: tuple[str, ...] | None) -> None:
            if not future.done():  # cancelled meanwhile: the chat is ending
                future.set_result(names)

        def work() -> None:
            names: tuple[str, ...] | None = None
            try:
                with request_deadline(NAMES_DEADLINE_SEC):
                    names = tuple(info.name for info in self.sandbox.list_notebooks()) if self.sandbox is not None else ()
            except Exception:  # noqa: BLE001 - the menu keeps its last list
                names = None
            try:
                loop.call_soon_threadsafe(deliver, names)
            except RuntimeError:  # the loop is closed: Hailer is shutting down
                pass

        threading.Thread(target=work, name="hailer-notebook-names", daemon=True).start()
        return future

    async def _settle_names(self) -> None:
        """At shutdown, before the kernel stops: no listing starts, and one still running gets
        NAMES_SETTLE_SEC to end before it is left behind (its result ignored, its thread a daemon)."""
        self._names_closed = True
        task = self._names_task
        if task is None or task.done():
            return
        try:
            await asyncio.wait({task}, timeout=NAMES_SETTLE_SEC)
        except asyncio.CancelledError:  # a second interrupt: finish shutting down
            pass
        finally:
            task.cancel()

    def _input_history_file(self) -> Path | None:
        """``.hailer/input_history`` while ``[hailer].input_history`` is on (Up recalls earlier
        sessions); ``None`` keeps this session's history in memory."""
        return input_history_path(self.config.workspace) if self.config.input_history else None

    def _notebook_link(self) -> str:
        """The active notebook's URL without its token (it is shown, not opened), made once per name."""
        name, url = self._link_cache
        if name != self.notebook:
            url = ""
            if self.sandbox is not None:
                try:
                    url = self.sandbox.notebook_url(self.notebook)
                except Exception:  # noqa: BLE001 - an odd name only loses the link
                    url = ""
            self._link_cache = (self.notebook, url)
        return url

    def _note_session(self, name: str, session: MarimoSession | None, *, known: bool = True) -> None:
        """Remember whether ``name`` has a kernel session when it is the active notebook."""
        if notebooks.same_name(name, self.notebook):
            self.notebook_open = (session is not None) if known else None

    async def run_interactive(
        self,
        ui_factory: Any = None,
        *,
        prepare_notebook: Callable[[], Awaitable[bool | None]] | None = None,
        timings: Any = None,
    ) -> None:
        if ui_factory is None:
            from hailer.chat_ui import ChatUI

            ui_factory = partial(ChatUI, completions=self.completions, history_file=self._input_history_file())
        original_console = self.console
        self.startup_failed = False
        self.ui = ui_factory(original_console, self.ahandle, self.footer)
        self.console = self.ui.console
        self.interactive = True

        async def prepare() -> bool:
            async def agent_ready() -> None:
                with self.progress("Agent", f"starting {single_line(self.config.model.name)}"):
                    await self.astart()
                if timings is not None:
                    timings.mark("agent_ready")

            async def notebook_ready() -> None:
                if prepare_notebook is not None:
                    opened = await prepare_notebook()
                    if isinstance(opened, bool):  # the CLI's wait reports whether a session came
                        self.notebook_open = opened
                if timings is not None:
                    timings.mark("notebook_wait_complete")
                self._refresh_names_soon()  # the command menu's names, in the background

            tasks = [asyncio.create_task(agent_ready()), asyncio.create_task(notebook_ready())]
            try:
                await asyncio.gather(*tasks)
                if timings is not None:
                    timings.mark("ready_to_answer")
                return True
            except HailerError as err:
                self.startup_failed = True
                self._cli._print_error(self.console, err, verbose=self.opts.verbose)
                return False
            except Exception as exc:
                self.startup_failed = True
                self._cli._print_unexpected(self.console, exc, verbose=self.opts.verbose)
                return False
            finally:
                # gather does not cancel siblings after a failure. Settle both
                # resource owners before a failed startup or exit closes agent.
                for task in tasks:
                    if not task.done():
                        task.cancel()
                cleanup = asyncio.gather(*tasks, return_exceptions=True)
                interrupted = False
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        interrupted = True
                cleanup.result()
                if interrupted:
                    raise asyncio.CancelledError

        self.ui.configure_startup(
            prepare,
            on_input_ready=(lambda: timings.mark("input_ready")) if timings is not None else None,
        )
        try:
            await self.ui.run()
        finally:
            try:
                await self._settle_names()
                if self.agent is not None:
                    await self.agent.aclose()
            finally:
                try:
                    await self.ui.flush()
                finally:
                    self.console = original_console
                    self.interactive = False

    async def ahandle(self, text: str) -> None:
        """Execute one accepted submission. The UI owns busy gating and user echo."""
        command = parse_command(text)
        try:
            if command is None:
                await self._aturn(text)
            elif command.name in EXIT_COMMANDS:
                self.ui.exit()  # the CLI's exit summary follows
            elif command.name == "new":
                self._reset_thread(await self.agent.anew_thread(), "New conversation started.")
            elif command.name == "model":
                reason = self._select_model(command.args)
                if reason is not None:
                    self._reset_thread(await self.agent.anew_thread(), reason)
            elif command.name == "skill":
                request = self._skill_request(command.args)
                if request is not None:
                    message, skill = request
                    await self._aturn(message, skill=skill)
            elif command.name == "prompt":
                message = await self._blocking(self._prompt_request, command.args)
                if message is not None:
                    await self._aturn(message)
            elif command.name == "reload":
                bundle = await self._blocking(self._cli._load_context, self.config)
                self._apply_bundle(bundle)
            else:
                try:
                    await self._blocking(self._command, command)
                finally:
                    if command.name == "notebook" and command.args.lower().split()[:1] == ["new"]:
                        self._refresh_names_soon()  # a notebook was (perhaps) created
        except HailerError as err:
            self._cli._print_error(self.console, err, verbose=self.opts.verbose)
        except Exception as exc:  # keep the composer usable after a failed operation
            self._cli._print_unexpected(self.console, exc, verbose=self.opts.verbose)

    async def _aturn(self, text: str, *, skill: SkillInfo | None = None) -> None:
        self.ui.set_activity("Thinking...")
        try:
            try:
                summary = await self.agent.arun_turn(
                    text, on_event=self.ui.on_event, skill=skill, preamble=self._pending_preamble
                )
            except asyncio.CancelledError:
                self._pending_preamble = None
                raise
            except Exception:
                self.ui.settle_steps()  # the rows of steps left open come before the error ahandle prints
                raise
            self._pending_preamble = None
            self._peek_active_notebook()
            self.ui.finish(summary)
            self._record_turn(summary)
        finally:
            # A tool may have switched notebooks before cancellation or failure.
            await self._blocking(self._sync_active_notebook)
            self._copy_notebooks_back()
            self._refresh_names_soon()  # a tool may have created one: the menu's names, in the background

    # -- REPL -------------------------------------------------------------- #

    def run(self) -> None:
        """The plain chat loop, until /exit, EOF or Ctrl+C at the prompt; the CLI's exit summary follows."""
        while True:
            try:
                line = self.reader.read()
            except (EOFError, KeyboardInterrupt):
                if not self.reader.interactive:
                    self.console.print()  # the plain prompt leaves the cursor after "You > "
                return
            text = line.strip()
            if not text:
                continue
            command = parse_command(text)
            try:
                if command is None:
                    self._turn(text)
                    continue
                if command.name in EXIT_COMMANDS:
                    return
                self._command(command)
            except HailerError as err:
                self._cli._print_error(self.console, err, verbose=self.opts.verbose)
            except KeyboardInterrupt:
                self.console.print("\nInterrupted.")
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                self._cli._print_unexpected(self.console, exc, verbose=self.opts.verbose)

    # -- turns ------------------------------------------------------------- #

    def _turn(self, text: str, *, skill: SkillInfo | None = None) -> None:
        display = self._cli._TurnDisplay(self.console)
        display.start()
        preamble = self._pending_preamble
        try:
            try:
                summary: TurnSummary = self.agent.run_turn(text, on_event=display, skill=skill, preamble=preamble)
            except KeyboardInterrupt:  # the agent has already cancelled the turn
                self._pending_preamble = None  # the notice went out with the interrupted turn
                display.cancel()
                self.console.print("\nInterrupted.")
                return
            except BaseException:
                display.stop()
                raise
            self._pending_preamble = None
            display.finish(summary)
            self._record_turn(summary)
        finally:
            # Also after Ctrl+C or a failed turn: a notebook_create/notebook_open tool call may have
            # completed (and switched the state file) before the turn was cut short.
            self._sync_active_notebook()
            self._copy_notebooks_back()

    def _copy_notebooks_back(self) -> None:
        """After a turn and on a notebook switch: the sandbox copies changed notebooks back to the
        workspace in the background (a no-op when its notebooks are the workspace's)."""
        if self.sandbox is not None:
            self.sandbox.sync_soon()

    def _record_turn(self, summary: TurnSummary) -> None:
        self.state.turns += 1
        self.state.input_tokens += summary.input_tokens or 0
        self.state.output_tokens += summary.output_tokens or 0
        self.session_turns += 1
        self.session_tokens += (summary.input_tokens or 0) + (summary.output_tokens or 0)
        if summary.thread_id:
            self.state.thread_id = summary.thread_id
        save_session(self.config.workspace, self.state)

    def _peek_active_notebook(self) -> None:
        """Before a turn's answer prints: take the notebook a tool switched to from the state file
        (a local read), so the footer and the answer's link name it. Nothing asks marimo or opens
        a browser here; :meth:`_sync_active_notebook` still reports the switch after the answer."""
        try:
            active = notebooks.load_active_notebook(self.config)
        except Exception:  # noqa: BLE001 - the sync after the answer handles a bad state file
            return
        if active != self.notebook:
            self._switched_from = self._switched_from or self.notebook
            self.notebook, self.notebook_open = active, None

    def _sync_active_notebook(self, *, open_browser: bool = True) -> bool:
        """Pick up a switch the model made during a turn (its tools write the same state file).

        Returns True when the active notebook changed (since before a :meth:`_peek_active_notebook`).
        With ``open_browser`` the notebook's URL is opened once when it has no kernel session yet
        (with --no-browser its signed-in link is printed instead).
        """
        try:
            active = notebooks.load_active_notebook(self.config)
        except Exception:  # noqa: BLE001 - a bad state file must never spoil a finished turn
            return False
        before, self._switched_from = self._switched_from or self.notebook, None
        if active == before:
            return False
        self.notebook = active
        self.notebook_open = None
        state, link = "", ""
        if open_browser:
            sandbox, session, _err = self._cli._marimo_state(self.sandbox, active)
            self._note_session(active, session, known=sandbox is not None)
            if sandbox is not None and session is None and self.open_browser:
                self._cli._open_browser(sandbox.notebook_url(active, with_token=True))  # opened, never printed
                state = "opened in your browser"
            elif sandbox is not None and session is None:
                state, link = "not open in a browser", sandbox.notebook_url(active, with_token=True)
            elif session is not None:
                state = "open in your browser"
        g = self._glyphs()
        self.console.print(session_row(g, "wait" if link else "ok", "Notebook", g.join(single_line(active), "now active", state)))
        if link:  # --no-browser: the user opens it
            self.console.print(session_note("Open it in your browser with this link (it signs you in):"))
            self.console.print(session_note(link, style=""), soft_wrap=True)
        return True

    # -- slash commands ---------------------------------------------------- #

    def _command(self, command: Any) -> None:
        name, args = command.name, command.args
        if name == "help":
            if self.interactive:
                self.console.print(help_card(self._glyphs()))
            else:
                self.console.print(help_text(), markup=False)
        elif name == "status":
            self._status()
        elif name == "new":
            self._new_thread(reason="New conversation started.")
        elif name == "model":
            self._model(args)
        elif name == "notebook":
            self._notebook(args)
        elif name == "exec":
            self._exec(args)
        elif name == "clear":
            self.console.clear()
        elif name == "context":
            self._context()
        elif name == "skill":
            self._skill(args)
        elif name == "prompt":
            self._prompt(args)
        elif name == "reload":
            self._reload()
        elif name == "":
            self.console.print("Type /help for commands.", markup=False)
        else:
            self.console.print(f"Unknown command /{name}; type /help for commands.", markup=False)

    def _new_thread(self, *, reason: str) -> None:
        """Start a new thread and reset the counters."""
        self._reset_thread(self.agent.new_thread(), reason)

    def _reset_thread(self, thread_id: str, reason: str) -> None:
        self.state = SessionState(
            thread_id=thread_id,
            model=self.state.model,
            provider=self.state.provider,
        )
        save_session(self.config.workspace, self.state)
        self.console.print(reason, markup=False)

    def _status(self) -> None:
        self._sync_active_notebook(open_browser=False)  # a late tool call may have switched notebooks
        sandbox, session, _err = self._cli._marimo_state(self.sandbox, self.notebook)
        self._note_session(self.notebook, session, known=sandbox is not None)
        if self.interactive:
            self._status_card(sandbox, session)
            return
        if sandbox is None:
            marimo = "not running"
        elif session is None:
            marimo = f"{sandbox.server.url} (notebook not open in a browser)"
        else:
            marimo = f"{sandbox.server.url} (session {session.session_id})"
        table = Table.grid(padding=(0, 2))
        rows = [
            ("Model", f"{self.state.model or self.config.model.name}"),
            ("Provider", self._active_provider_line()),
            ("Credentials", self._credentials_line()),
            ("Thread", self.state.thread_id or "-"),
            ("Turns", str(self.state.turns)),
            ("Tokens", f"{self.state.input_tokens} in / {self.state.output_tokens} out"),
            ("Marimo", marimo),
            ("Kernel", self._cli._kernel_line(self.config)),
            *([("Kernel log", log)] if (log := self._log_hint()) else []),
            ("Notebook", self.notebook),
            ("Notebooks", self._cli._relative(self.config.notebooks_root, self.config.workspace)),
            ("Web access", self._cli._web_line(self.config)),
        ]
        for label, value in rows:
            style = self._cli._kernel_style(self.config) if label == "Kernel" else ""  # unsafe-local: a warning
            table.add_row(Text(label), Text(value, style=style))
        self.console.print(table)

    def _status_card(self, sandbox: Any, session: MarimoSession | None) -> None:
        """/status in the composer: the plain table's information, grouped, with a health mark on
        what can be wrong (credentials, an unisolated kernel, marimo, the notebook's tab)."""
        g = self._glyphs()
        credentials, have_key = self._credentials()
        model = g.join(self.state.model or self.config.model.name, self._active_provider_line(), credentials)
        unsafe = bool(self._cli._kernel_style(self.config))  # unsafe-local: a warning
        marimo: tuple[str | None, str] = ("warn", "not running")
        notebook: tuple[str | None, str] = (None, self.notebook)
        if sandbox is not None:
            marimo = ("ok", sandbox.server.url)
            notebook = (
                ("ok", g.join(self.notebook, f"session {session.session_id}")) if session is not None
                else ("off", g.join(self.notebook, "not open in a browser"))
            )
        tokens = f"{self.state.input_tokens} in / {self.state.output_tokens} out tokens"
        rows: list[tuple[str | None, str, str]] = [
            ("ok" if have_key else "warn", "Model", model),
            ("warn" if unsafe else "ok", "Kernel", self._cli._kernel_line(self.config)),
            *([(None, "Kernel log", log)] if (log := self._log_hint()) else []),
            (marimo[0], "Marimo", marimo[1]),
            (notebook[0], "Notebook", notebook[1]),
            (None, "Notebooks", self._cli._relative(self.config.notebooks_root, self.config.workspace)),
            (None, "Thread", g.join(self.state.thread_id or "-", plural(self.state.turns, "turn"), tokens)),
            (None, "Web access", self._cli._web_line(self.config)),
        ]
        self.console.print(status_card(g, rows))

    def _glyphs(self) -> Any:
        return console_glyphs(self.console)

    def _log_hint(self) -> str:
        """Where this chat's kernel logs (``docker logs hailer-kernel-...``, a log file): /status."""
        return single_line(str(getattr(self.sandbox, "log_hint", "") or "")) if self.sandbox is not None else ""

    def _active_provider(self) -> ProviderConfig | None:
        provider_id = self.state.provider or self.config.model.provider
        if provider_id in self.config.providers:
            return self.config.providers[provider_id]
        if provider_id == "openai":
            return ProviderConfig(id="openai", env_key="OPENAI_API_KEY")
        return None

    def _active_provider_line(self) -> str:
        provider_id = self.state.provider or self.config.model.provider
        if provider_id == self.config.model.provider:
            return self._cli._provider_line(self.config)
        provider = self._active_provider()
        if provider is None:
            return f"{provider_id} (not declared in hailer.toml)"
        return self._cli._describe_provider(provider)

    def _credentials_line(self) -> str:
        return self._credentials()[0]

    def _credentials(self) -> tuple[str, bool]:
        """The ``Credentials`` line, and whether the active provider has a key."""
        provider = self._active_provider()
        if provider is None:
            return "unknown provider", False
        source = getattr(self.agent, "key_source", None)
        if not isinstance(source, str) or not source:
            try:
                _value, source = self._cli._resolve_key(provider)
            except HailerError:
                source = "missing"
        if source == "missing":
            return f"{provider.env_key or 'API key'} missing (run: uvx hailer login {provider.id})", False
        return f"{provider.env_key or 'API key'} from {source}", True

    def _model(self, args: str) -> None:
        reason = self._select_model(args)
        if reason is not None:
            self._new_thread(reason=reason)

    def _select_model(self, args: str) -> str | None:
        if not args:
            self.console.print(
                f"Model: {self.state.model or self.config.model.name}  Provider: {self.state.provider or self.config.model.provider}",
                markup=False,  # a model name is the user's text
            )
            self.console.print("Usage: /model <name>  or  /model <provider>:<name>")
            return
        provider: str | None = None
        name = args.strip()
        if ":" in name:
            provider, _, name = name.partition(":")
            provider = provider.strip()
            name = name.strip()
        if not name:
            self.console.print("Usage: /model <name>  or  /model <provider>:<name>")
            return
        if provider is not None and provider != "openai" and provider not in self.config.providers:
            known = ", ".join(sorted({"openai", *self.config.providers}))
            self.console.print(f"Unknown provider {provider!r}. Declared providers: {known}.", markup=False)
            return
        self.agent.set_model(name, provider)
        self.state.model = name
        if provider is not None:
            self.state.provider = provider
        return f"Model set to {name}" + (f" ({provider})" if provider else "") + "; started a new thread."

    # -- /exec ------------------------------------------------------------- #

    def _exec(self, code: str) -> None:
        """Run ``code`` in the scratchpad of the active notebook's kernel session (this chat's own
        kernel) and print what it printed and returned. Nothing goes to the model."""
        if not code.strip():
            self.console.print("Usage: /exec <python code>  (runs in the active notebook's kernel; the model does not see it)", markup=False)
            return
        sandbox, session, err = self._cli._marimo_state(self.sandbox, self.notebook)
        self._note_session(self.notebook, session, known=sandbox is not None)
        if sandbox is None:
            self.console.print("Marimo is not running.", markup=False)
            return
        if session is None:
            self._cli._print_error(self.console, err or NoSessionError("The notebook is not open in a browser."), verbose=False)
            return
        self._check_stopped()
        client = sandbox.client(notebook=self.notebook, token_in_links=True)
        result = client.execute(textwrap.dedent(code), session_id=session.session_id)
        if result.stdout.strip():
            self.console.print(result.stdout.rstrip(), markup=False)
        if result.output.strip() and result.output.strip() != result.stdout.strip():
            self.console.print(result.output.rstrip(), markup=False)
        if result.stderr.strip():
            self.console.print(result.stderr.rstrip(), style="red", markup=False)
        if not result.success:
            self.console.print("(the code failed)", style="red", markup=False)
        elif not (result.stdout.strip() or result.output.strip() or result.stderr.strip()):
            self.console.print("(no output)", style="dim", markup=False)

    # -- /notebook --------------------------------------------------------- #

    def _notebook(self, args: str) -> None:
        self._sync_active_notebook(open_browser=False)  # act on the current state, not a stale copy
        sub, _, rest = args.strip().partition(" ")
        sub, rest = sub.lower(), rest.strip()
        if not sub:
            self._notebook_show()
        elif sub == "list":
            self._notebook_list()
        elif sub == "new":
            self._notebook_new(rest)
        elif sub == "open":
            self._notebook_open(rest)
        elif sub == "close":
            self._notebook_close(rest)
        else:
            self.console.print(f"Unknown /notebook subcommand {sub!r}. {self._cli.NOTEBOOK_USAGE}", markup=False)

    def _names(self) -> list[str]:
        """The notebooks in the sandbox; empty without a kernel. The command menu keeps the list."""
        names = [info.name for info in self.sandbox.list_notebooks()] if self.sandbox is not None else []
        self._notebook_names = tuple(names)
        return names

    def _resolve(self, ref: str) -> str:
        """``ref`` as the name of an existing notebook (a name, a file name or a path in the folder)."""
        if self.sandbox is None:
            raise MarimoUnavailableError("Marimo is not running, so the notebooks cannot be listed.")
        folders = notebooks.reference_folders(self.config, self.sandbox.notebooks_path)
        return notebooks.resolve_notebook(ref, self._names(), folders=folders)

    def _notebook_show(self) -> None:
        cfg, out = self.config, self.console
        out.print(f"Notebook:  {self.notebook} (active)", markup=False)
        folder = self._cli._relative(cfg.notebooks_root, cfg.workspace)
        names = self._names()
        if self.sandbox is not None:
            out.print(f"Notebooks: {folder} ({plural(len(names), 'notebook')}; /notebook list)", markup=False)
        recent = [name for name in notebooks.load_recent(cfg) if name != self.notebook and name in names]
        if recent:
            out.print("Recent:    " + ", ".join(recent[:5]), markup=False)
        sandbox, session, _err = self._cli._marimo_state(self.sandbox, self.notebook)
        self._note_session(self.notebook, session, known=sandbox is not None)
        if sandbox is None:
            out.print("Marimo:    not running", markup=False)
        else:
            state = f"session {session.session_id}" if session is not None else "not open in a browser"
            out.print(f"Marimo:    {sandbox.server.url} ({state})", markup=False)
            out.print(f"URL:       {sandbox.notebook_url(self.notebook, with_token=True)}", markup=False)
            out.print("View:      app view (results only); Ctrl+. in the notebook toggles the code editor", markup=False)
        out.print(self._cli.NOTEBOOK_USAGE, markup=False)

    def _server_sessions(self) -> tuple[Any, Any, list[MarimoSession]]:
        """(sandbox, client, open sessions) for this chat's kernel; sandbox is None when it does not answer."""
        sandbox = self.sandbox
        if sandbox is None:
            return None, None, []
        client = sandbox.client(notebook=self.notebook, token_in_links=True)
        try:
            if not client.health():
                return None, None, []
            return sandbox, client, client.sessions()
        except HailerError:
            return sandbox, client, []

    def _notebook_list(self) -> None:
        folder = self._cli._relative(self.config.notebooks_root, self.config.workspace)
        if self.sandbox is None:
            self.console.print("Marimo is not running, so the notebooks cannot be listed.", markup=False)
            return
        infos = self.sandbox.list_notebooks()
        self._notebook_names = tuple(info.name for info in infos)  # the command menu's list, too
        if not infos:
            self.console.print(f"No notebooks in {folder} yet. Create one with /notebook new <name>.", markup=False)
            return
        sandbox, _client, sessions = self._server_sessions()
        if sandbox is not None:
            self._note_session(self.notebook, self._cli._session_for(sessions, self.notebook))
        note = f"{plural(len(infos), 'notebook')} in {folder}. /notebook open <name> switches."
        if self.interactive:
            rows = [
                NotebookRow(
                    info.name, _when(info.modified), f"{info.size / 1024:.1f} KB",
                    active=info.name == self.notebook, open=self._cli._session_for(sessions, info.name) is not None,
                )
                for info in infos
            ]
            self.console.print(notebook_table(self._glyphs(), rows, note))
            return
        table = Table.grid(padding=(0, 2))
        for info in infos:
            markers = []
            if info.name == self.notebook:
                markers.append("active")
            if self._cli._session_for(sessions, info.name) is not None:
                markers.append("open")
            modified = datetime.fromtimestamp(info.modified).strftime("%Y-%m-%d %H:%M")
            table.add_row(
                Text(info.name),
                Text(modified),
                Text(f"{info.size / 1024:.1f} KB"),
                Text(", ".join(markers)),
            )
        self.console.print(table)
        self.console.print(note, markup=False)

    def _notebook_new(self, rest: str) -> None:
        tokens = rest.split()
        kind = "empty" if "--empty" in tokens else "starter"
        name = " ".join(t for t in tokens if t != "--empty")
        if not name:
            self.console.print("Usage: /notebook new <name> [--empty]", markup=False)
            return
        if self.sandbox is None:
            self.console.print("Marimo is not running, so no notebook can be created.", markup=False)
            return
        filename, source = notebooks.new_notebook(name, kind=kind)
        created = self.sandbox.write_notebook(filename, source, replace=False).name
        self._switch_notebook(created, how=f"created from the {kind} template")

    def _notebook_open(self, rest: str) -> None:
        if not rest:
            self.console.print("Usage: /notebook open <name>", markup=False)
            return
        name = self._resolve(rest)
        if notebooks.same_name(name, self.notebook):
            began = time.monotonic()
            opened = self._ensure_session()
            self._notebook_row([name, "already the active notebook"], opened, time.monotonic() - began)
            return
        self._switch_notebook(name, how="reopened")

    def _notebook_close(self, rest: str) -> None:
        name = self._resolve(rest) if rest else self.notebook
        sandbox, client, _sessions = self._server_sessions()
        if sandbox is None:
            self.console.print("Marimo is not running.", markup=False)
            return
        try:
            session = client.resolve_session(name)
        except NoSessionError:
            self.console.print(f"{name} is not open (no kernel session).", markup=False)
            return
        client.shutdown_session(session.session_id)
        self._note_session(name, None)
        g = self._glyphs()
        closed = g.join(single_line(name), "closed", f"session {session.session_id}", "its browser tab is disconnected")
        self.console.print(session_row(g, "ok", "Notebook", closed))
        if notebooks.same_name(name, self.notebook):
            self.console.print(session_note(f"It stays the active notebook; /notebook open {name.removesuffix('.py')} reopens it."))

    def _switch_notebook(self, name: str, *, how: str) -> None:
        """Make ``name`` the active notebook for this chat, the agent's tools and the next session,
        then one row: ``✓ Notebook  q2.py · active · open in your browser  0.6s``."""
        began = time.monotonic()
        notebooks.save_active_notebook(self.config, name)
        self.notebook = name
        self.notebook_open = None
        self._copy_notebooks_back()
        # Queue the notice first: if the wait for the browser tab is interrupted (Ctrl+C) or marimo
        # fails, the switch has still happened and the model must hear about it.
        self._pending_preamble = self._switch_notice(name, f"{how}, not open in a browser yet")
        opened = _Opened()
        try:
            opened = self._ensure_session()
            self._check_stopped()
            detail = how
            if opened.session is not None:
                count = self._cell_count(opened.client)
                if count is not None:
                    detail += f", {plural(count, 'cell')}"
            else:
                detail += ", not open in a browser yet"
            self._pending_preamble = self._switch_notice(name, detail)
        finally:
            self._notebook_row([name, "" if how == "reopened" else how, "active"], opened, time.monotonic() - began)

    @staticmethod
    def _switch_notice(name: str, detail: str) -> str:
        return f"[Hailer] The active notebook is now {name} ({detail}). Call notebook_cells before editing."

    def _ensure_session(self) -> _Opened:
        """Open the active notebook in the browser when it has no kernel session and wait for one
        (the live row says so); what came of it. Nothing is printed here: the caller's row is."""
        name = self.notebook
        self._check_stopped()
        sandbox, session, _err = self._cli._marimo_state(self.sandbox, name)
        self._note_session(name, session, known=sandbox is not None)
        self._check_stopped()
        if sandbox is None:
            return _Opened(state="marimo is not running", note="Marimo is not running; the notebook opens once it is (uvx hailer notebook).")
        client = sandbox.client(notebook=name, token_in_links=True)
        if session is not None:
            return _Opened(session, client, "already open", ok=True)
        url = sandbox.notebook_url(name, with_token=True)  # opened, and printed only when it must be opened by hand
        if not self.open_browser:  # --no-browser: the link at once, then the wait
            g = self._glyphs()
            self.console.print(session_row(g, "wait", "Notebook", f"open {single_line(name)} in your browser with this link (it signs you in):"))
            self.console.print(session_note(url, style=""), soft_wrap=True)
        doing = f"opening {single_line(name)} in your browser" if self.open_browser else f"waiting for {single_line(name)} to open"
        with self.progress("Notebook", doing):
            if self.open_browser:
                self._cli._open_browser(url)
            if self._stop_event is None:
                session = self._cli._wait_for_session(client, name, self._cli.SWITCH_SESSION_TIMEOUT_SEC)
            else:
                session = self._cli._wait_for_session(
                    client, name, self._cli.SWITCH_SESSION_TIMEOUT_SEC,
                    should_stop=self._stop_event.is_set,
                )
        self._note_session(name, session)
        self._check_stopped()
        if session is None and not self.open_browser:
            return _Opened(None, client, "no kernel session yet", note="Open it with the link above.")
        if session is None:
            return _Opened(None, client, "no kernel session yet", note="Open it in your browser with this link:", link=url)
        return _Opened(session, client, "open in your browser", ok=True)

    def _notebook_row(self, parts: list[str], opened: _Opened, took: float | None) -> None:
        """``✓ Notebook  q2.py · active · open in your browser  0.6s``; a notebook that did not
        open is a warning, with the signed-in link under it when it must be opened by hand."""
        g = self._glyphs()
        detail = g.join(*(single_line(part) for part in parts), opened.state)
        self.console.print(session_row(g, "ok" if opened.ok else "warn", "Notebook", detail, took))
        if opened.note:
            self.console.print(session_note(opened.note))
        if opened.link:
            self.console.print(session_note(opened.link, style=""), soft_wrap=True)  # whole: it copies as one

    def _cell_count(self, client: Any) -> int | None:
        """Number of cells in the active notebook, or None when it cannot be determined."""
        if client is None:
            return None
        self._check_stopped()
        try:
            result = client.execute(self._cli._list_cells_code(), notebook=self.notebook, timeout=5.0)
        except Exception:  # noqa: BLE001 - informational only; never fail a switch over it
            return None
        if not result.success:
            return None
        return sum(1 for line in (result.stdout or "").splitlines() if line.strip())

    def _context(self) -> None:
        if self.interactive:
            self._context_card()
            return
        bundle = self.bundle
        out = self.console
        if bundle.context_files:
            out.print("Context files (sent with every session):", markup=False)
            for path in bundle.context_files:
                try:
                    size = path.stat().st_size
                except OSError:
                    size = 0
                out.print(f"  {self._cli._relative(path, self.config.workspace)}  ({size} bytes)", markup=False)
        else:
            out.print(f"Context files: none ({self._cli._relative(self.config.context_dir, self.config.workspace)})", markup=False)
        if bundle.skills:
            out.print("Skills:", markup=False)
            for skill in bundle.skills:
                out.print(f"  {skill.name}: {skill.description}", markup=False)
        else:
            out.print(f"Skills: none ({self._cli._relative(self.config.skills_dir, self.config.workspace)})", markup=False)
        if bundle.prompts:
            out.print("Prompts: " + ", ".join(sorted(bundle.prompts)), markup=False)
        else:
            out.print(f"Prompts: none ({self._cli._relative(self.config.prompts_dir, self.config.workspace)})", markup=False)
        out.print(f"Web access: {self._cli._web_line(self.config)}", markup=False)
        for warning in bundle.warnings:
            out.print(self._cli._labelled("warn", "yellow", f"  {warning}"))

    def _context_card(self) -> None:
        """/context in the composer: the plain lines' lists, one item per row."""
        bundle, workspace = self.bundle, self.config.workspace

        def none(folder: Any) -> list[RenderableType | str]:
            return [Text(f"none ({self._cli._relative(folder, workspace)})", style="dim")]

        files: list[RenderableType | str] = []
        for path in bundle.context_files:
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            files.append(Text.assemble(self._cli._relative(path, workspace), (f"  ({size} bytes)", "dim")))
        skills: list[RenderableType | str] = [Text.assemble(skill.name, (f"  {skill.description}", "dim")) for skill in bundle.skills]
        sections: list[tuple[str, list[RenderableType | str]]] = [
            ("Context files", files or none(self.config.context_dir)),
            ("Skills", skills or none(self.config.skills_dir)),
            ("Prompts", [", ".join(sorted(bundle.prompts))] if bundle.prompts else none(self.config.prompts_dir)),
            ("Web access", [self._cli._web_line(self.config)]),
        ]
        if bundle.context_files:
            sections.insert(1, ("", [Text("sent with every session", style="dim")]))
        self.console.print(detail_card(self._glyphs(), sections, bundle.warnings))

    def _skill(self, args: str) -> None:
        request = self._skill_request(args)
        if request is not None:
            message, skill = request
            self._turn(message, skill=skill)

    def _skill_request(self, args: str) -> tuple[str, SkillInfo] | None:
        if not args:
            names = ", ".join(s.name for s in self.bundle.skills) or "none"
            self.console.print(f"Usage: /skill <name> [message]. Available: {names}", markup=False)
            return
        name, _, message = args.partition(" ")
        skill = next((s for s in self.bundle.skills if s.name == name), None)
        if skill is None:
            names = ", ".join(s.name for s in self.bundle.skills) or "none"
            self.console.print(f"Unknown skill {name!r}. Available: {names}", markup=False)
            return
        message = message.strip() or f"Summarise the '{skill.name}' skill and how you would apply it to this workspace."
        return message, skill

    def _prompt(self, args: str) -> None:
        text = self._prompt_request(args)
        if text is not None:
            self._turn(text)

    def _prompt_request(self, args: str) -> str | None:
        if not args:
            names = ", ".join(sorted(self.bundle.prompts)) or "none"
            self.console.print(f"Usage: /prompt <name> [args]. Available: {names}", markup=False)
            return
        name, _, rest = args.partition(" ")
        return self._cli._render_prompt(self.config, name, rest.strip())

    def _reload(self) -> None:
        self._apply_bundle(self._cli._load_context(self.config))

    def _apply_bundle(self, bundle: ContextBundle) -> None:
        self.bundle = bundle
        self._print_context_warnings()
        if hasattr(self.agent, "bundle"):
            self.agent.bundle = self.bundle
        self.console.print(
            f"Reloaded {len(self.bundle.context_files)} context file(s), {len(self.bundle.skills)} skill(s), "
            f"{len(self.bundle.prompts)} prompt(s). Applies from your next message.",
            markup=False,
        )
