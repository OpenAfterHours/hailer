"""Shared session operations for the inline composer and the plain chat loop.

CLI collaborators are injected so both presentations use the same notebook, context
and credential operations. Only the synchronous facade drives an asyncio.Runner.
"""

from __future__ import annotations

import asyncio
import textwrap
import threading
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text

from hailer import notebooks
from hailer.errors import HailerError, NoSessionError
from hailer.models import ContextBundle, HailerConfig, MarimoServer, MarimoSession, ProviderConfig, SessionState, SkillInfo, TurnSummary
from hailer.session import EXIT_COMMANDS, help_text, load_session, parse_command, save_session


class _CommandStopped(Exception):
    """A blocking notebook operation reached a cooperative cancellation point."""


class ChatController:
    def __init__(self, console: Console, config: HailerConfig, opts: Any, *, services: Any, server: MarimoServer | None = None) -> None:
        self._cli = services
        self.console = console
        self.config = config
        self.opts = opts
        self.server = server
        self.state: SessionState = load_session(config.workspace)
        self.bundle: ContextBundle = ContextBundle()
        self.agent: Any = None
        self.startup_failed = False
        self.reader = (
            self._cli._LineReader(console, interactive=False)
            if opts.plain
            else self._cli._make_line_reader(console)
        )
        # One-line notice sent with the next user message after a /notebook switch (the model
        # learns about switches it made itself from its own tool results).
        self._pending_preamble: str | None = None
        self._stop_event: threading.Event | None = None

    # -- lifecycle --------------------------------------------------------- #

    def start(self) -> None:
        self.bundle = self._cli._load_context(self.config)
        self._print_context_warnings()
        self.agent = self._cli._make_agent(self.config, self.bundle, self.server)
        resume = None if self.opts.new_thread else self.state.thread_id
        forget = self.state.thread_id if self.opts.new_thread else None  # --new: drop the stored conversation
        thread_id = self.agent.start(resume_thread_id=resume, forget_thread_id=forget)
        self._started(thread_id, resume)

    def _started(self, thread_id: str, resume: str | None) -> None:
        if resume and thread_id != resume:
            self.console.print(Text("Previous conversation could not be resumed; started a new one.", style="dim"))
            self.state = SessionState()
        elif resume:
            self.console.print(Text(f"Resumed conversation ({self.state.turns} turns so far). /new starts fresh.", style="dim"))
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

    def close(self) -> None:
        if self.agent is not None:
            try:
                self.agent.close()
            except Exception:  # pragma: no cover - best effort
                pass

    def _print_context_warnings(self) -> None:
        for warning in self.bundle.warnings:
            self.console.print(self._cli._labelled("warn", "yellow", f"  {warning}"))

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
        self.bundle = await self._blocking(self._cli._load_context, self.config)
        self._print_context_warnings()
        self.agent = self._cli._make_agent(self.config, self.bundle, self.server)
        resume = None if self.opts.new_thread else self.state.thread_id
        forget = self.state.thread_id if self.opts.new_thread else None
        thread_id = await self.agent.astart(resume_thread_id=resume, forget_thread_id=forget)
        self._started(thread_id, resume)

    def context_line(self) -> str:
        notebook = self._display_name(self.config.notebook)
        model = self.state.model or self.config.model.name
        return f"Notebook: {notebook} | Model: {model} | Context: {len(self.bundle.context_files)} files"

    async def run_interactive(
        self,
        ui_factory: Any = None,
        *,
        prepare_notebook: Callable[[], Awaitable[None]] | None = None,
        timings: Any = None,
    ) -> None:
        if ui_factory is None:
            from hailer.chat_ui import ChatUI

            ui_factory = ChatUI
        original_console = self.console
        self.startup_failed = False
        self.ui = ui_factory(original_console, self.ahandle, self.context_line)
        self.console = self.ui.console

        async def prepare() -> bool:
            async def agent_ready() -> None:
                await self.astart()
                if timings is not None:
                    timings.mark("agent_ready")

            async def notebook_ready() -> None:
                if prepare_notebook is not None:
                    await prepare_notebook()
                if timings is not None:
                    timings.mark("notebook_wait_complete")

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
                if self.agent is not None:
                    await self.agent.aclose()
            finally:
                try:
                    await self.ui.flush()
                finally:
                    self.console = original_console

    async def ahandle(self, text: str) -> None:
        """Execute one accepted submission. The UI owns busy gating and user echo."""
        command = parse_command(text)
        try:
            if command is None:
                await self._aturn(text)
            elif command.name in EXIT_COMMANDS:
                self.console.print("Bye.")
                self.ui.exit()
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
                await self._blocking(self._command, command)
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
            self._pending_preamble = None
            self.ui.finish(summary.final_response)
            self._record_turn(summary)
        finally:
            # A tool may have switched notebooks before cancellation or failure.
            await self._blocking(self._sync_active_notebook)

    # -- REPL -------------------------------------------------------------- #

    def run(self) -> None:
        # The plain prompt leaves the cursor after "You > "; prompt_toolkit has ended the line already.
        bye = "Bye." if self.reader.interactive else "\nBye."
        while True:
            try:
                line = self.reader.read()
            except (EOFError, KeyboardInterrupt):
                self.console.print(bye)
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
                    self.console.print("Bye.")
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
                display.stop()
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

    def _record_turn(self, summary: TurnSummary) -> None:
        self.state.turns += 1
        self.state.input_tokens += summary.input_tokens or 0
        self.state.output_tokens += summary.output_tokens or 0
        if summary.thread_id:
            self.state.thread_id = summary.thread_id
        save_session(self.config.workspace, self.state)

    def _sync_active_notebook(self, *, open_browser: bool = True) -> bool:
        """Pick up a switch the model made during a turn (its tools write the same state file).

        Returns True when the active notebook changed. With ``open_browser`` the notebook's URL is
        opened once when it has no kernel session yet.
        """
        try:
            active = notebooks.load_active_notebook(self.config)
        except Exception:  # noqa: BLE001 - a bad state file must never spoil a finished turn
            return False
        if self._cli._same_file(active, self.config.notebook):
            return False
        self.config = replace(self.config, notebook=active)
        name = notebooks.notebook_display_name(self.config, active)
        self.console.print(Text(f"Active notebook is now {name}.", style="dim"))
        if open_browser:
            server, session, _err = self._cli._marimo_state(self.config, self.server)
            if server is not None and session is None:
                url = self._cli._notebook_link(self.console, server, self.config)
                self.console.print(f"Opening {url} in your browser...", markup=False)
                self._cli._open_browser(url)
        return True

    # -- slash commands ---------------------------------------------------- #

    def _command(self, command: Any) -> None:
        name, args = command.name, command.args
        if name == "help":
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
        server, session, _err = self._cli._marimo_state(self.config, self.server)
        if server is None:
            marimo = "not running"
        elif session is None:
            marimo = f"{server.url} (notebook not open in a browser)"
        else:
            marimo = f"{server.url} (session {session.session_id})"
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
            ("Notebook", self._cli._relative(self.config.notebook, self.config.workspace)),
            ("Notebooks", self._cli._relative(self.config.notebooks_root, self.config.workspace)),
            ("Web access", self._cli._web_line(self.config)),
        ]
        for label, value in rows:
            table.add_row(Text(label), Text(value))
        self.console.print(table)

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
        provider = self._active_provider()
        if provider is None:
            return "unknown provider"
        source = getattr(self.agent, "key_source", None)
        if not isinstance(source, str) or not source:
            try:
                _value, source = self._cli._resolve_key(provider)
            except HailerError:
                source = "missing"
        if source == "missing":
            return f"{provider.env_key or 'API key'} missing (run: uvx hailer login {provider.id})"
        return f"{provider.env_key or 'API key'} from {source}"

    def _model(self, args: str) -> None:
        reason = self._select_model(args)
        if reason is not None:
            self._new_thread(reason=reason)

    def _select_model(self, args: str) -> str | None:
        if not args:
            self.console.print(f"Model: {self.state.model or self.config.model.name}  Provider: {self.state.provider or self.config.model.provider}")
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
        server, session, err = self._cli._marimo_state(self.config, self.server)
        if server is None:
            self.console.print("Marimo is not running.", markup=False)
            return
        if session is None:
            self._cli._print_error(self.console, err or NoSessionError("The notebook is not open in a browser."), verbose=False)
            return
        self._check_stopped()
        client = self._cli._make_client(server, self.config)
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

    def _display_name(self, path: Path) -> str:
        return notebooks.notebook_display_name(self.config, path)

    def _notebook_show(self) -> None:
        cfg, out = self.config, self.console
        out.print(f"Notebook:  {self._display_name(cfg.notebook)} (active)", markup=False)
        count = len(notebooks.list_notebooks(cfg))
        out.print(f"Notebooks: {self._display_name(cfg.notebooks_root)} ({self._cli._plural(count, 'notebook')}; /notebook list)", markup=False)
        recent = [p for p in notebooks.load_recent(cfg) if not self._cli._same_file(p, cfg.notebook)]
        if recent:
            out.print("Recent:    " + ", ".join(self._display_name(p) for p in recent[:5]), markup=False)
        server, session, _err = self._cli._marimo_state(cfg, self.server)
        if server is None:
            out.print("Marimo:    not running", markup=False)
        else:
            state = f"session {session.session_id}" if session is not None else "not open in a browser"
            out.print(f"Marimo:    {server.url} ({state})", markup=False)
            out.print(f"URL:       {self._cli._notebook_link(out, server, cfg)}", markup=False)
            out.print("View:      app view (results only); Ctrl+. in the notebook toggles the code editor", markup=False)
        out.print(self._cli.NOTEBOOK_USAGE, markup=False)

    def _server_sessions(self) -> tuple[MarimoServer | None, Any, list[MarimoSession]]:
        """(server, client, open sessions) for this chat's kernel; server is None when it does not answer."""
        server = self.server
        if server is None:
            return None, None, []
        client = self._cli._make_client(server, self.config)
        try:
            if not client.health():
                return None, None, []
            return server, client, client.sessions()
        except HailerError:
            return server, client, []

    def _notebook_list(self) -> None:
        infos = notebooks.list_notebooks(self.config)
        folder = self._display_name(self.config.notebooks_root)
        if not infos:
            self.console.print(f"No notebooks in {folder} yet. Create one with /notebook new <name>.", markup=False)
            return
        _server, _client, sessions = self._server_sessions()
        table = Table.grid(padding=(0, 2))
        for info in infos:
            markers = []
            if self._cli._same_file(info.path, self.config.notebook):
                markers.append("active")
            if self._cli._session_for(sessions, info.path, self.config.workspace) is not None:
                markers.append("open")
            modified = datetime.fromtimestamp(info.modified).strftime("%Y-%m-%d %H:%M")
            table.add_row(
                Text(self._display_name(info.path)),
                Text(modified),
                Text(f"{info.size / 1024:.1f} KB"),
                Text(", ".join(markers)),
            )
        self.console.print(table)
        self.console.print(f"{self._cli._plural(len(infos), 'notebook')} in {folder}. /notebook open <name> switches.", markup=False)

    def _notebook_new(self, rest: str) -> None:
        tokens = rest.split()
        kind = "empty" if "--empty" in tokens else "starter"
        name = " ".join(t for t in tokens if t != "--empty")
        if not name:
            self.console.print("Usage: /notebook new <name> [--empty]", markup=False)
            return
        path = notebooks.create_notebook(self.config, name, kind=kind)
        self.console.print(f"Created {self._display_name(path)} from the {kind} template.", markup=False)
        self._switch_notebook(path, how=f"created from the {kind} template")

    def _notebook_open(self, rest: str) -> None:
        if not rest:
            self.console.print("Usage: /notebook open <name>", markup=False)
            return
        path = notebooks.resolve_notebook(self.config, rest)
        if self._cli._same_file(path, self.config.notebook):
            self.console.print(f"{self._display_name(path)} is already the active notebook.", markup=False)
            self._ensure_session()
            return
        self._switch_notebook(path, how="reopened")

    def _notebook_close(self, rest: str) -> None:
        path = notebooks.resolve_notebook(self.config, rest) if rest else self.config.notebook
        name = self._display_name(path)
        server, client, _sessions = self._server_sessions()
        if server is None:
            self.console.print("Marimo is not running.", markup=False)
            return
        try:
            session = client.resolve_session(path)
        except NoSessionError:
            self.console.print(f"{name} is not open (no kernel session).", markup=False)
            return
        client.shutdown_session(session.session_id)
        self.console.print(f"Closed {name} (session {session.session_id}); its browser tab is disconnected.", markup=False)
        if self._cli._same_file(path, self.config.notebook):
            self.console.print(f"It stays the active notebook; /notebook open {Path(path).stem} reopens it.", markup=False)

    def _switch_notebook(self, path: Path, *, how: str) -> None:
        """Make ``path`` the active notebook for this chat, the tool server and the next session."""
        notebooks.save_active_notebook(self.config, path)
        self.config = replace(self.config, notebook=path)
        name = self._display_name(path)
        # Queue the notice first: if the wait for the browser tab is interrupted (Ctrl+C) or marimo
        # fails, the switch has still happened and the model must hear about it.
        self._pending_preamble = self._switch_notice(name, f"{how}, not open in a browser yet")
        try:
            session, client = self._ensure_session()
            self._check_stopped()
            detail = how
            if session is not None:
                count = self._cell_count(client)
                if count is not None:
                    detail += f", {self._cli._plural(count, 'cell')}"
            else:
                detail += ", not open in a browser yet"
            self._pending_preamble = self._switch_notice(name, detail)
        finally:
            self.console.print(f"Active notebook: {name}.", markup=False)

    @staticmethod
    def _switch_notice(name: str, detail: str) -> str:
        return f"[Hailer] The active notebook is now {name} ({detail}). Call notebook_cells before editing."

    def _ensure_session(self) -> tuple[MarimoSession | None, Any]:
        """Open the active notebook in the browser when it has no kernel session; (session, client)."""
        cfg = self.config
        self._check_stopped()
        server, session, _err = self._cli._marimo_state(cfg, self.server)
        self._check_stopped()
        if server is None:
            self.console.print("Marimo is not running; the notebook opens once it is (uvx hailer notebook).", markup=False)
            return None, None
        client = self._cli._make_client(server, cfg)
        if session is not None:
            self.console.print(f"Notebook is open (session {session.session_id}).", markup=False)
            return session, client
        url = self._cli._notebook_link(self.console, server, cfg)
        self.console.print(f"Opening {url} in your browser...", markup=False)
        self._cli._open_browser(url)
        with self.console.status("Waiting for the notebook to open..."):
            if self._stop_event is None:
                session = self._cli._wait_for_session(client, cfg.notebook, self._cli.SWITCH_SESSION_TIMEOUT_SEC)
            else:
                session = self._cli._wait_for_session(
                    client, cfg.notebook, self._cli.SWITCH_SESSION_TIMEOUT_SEC,
                    should_stop=self._stop_event.is_set,
                )
        self._check_stopped()
        if session is None:
            self.console.print(f"No kernel session yet. Open {url} in your browser.", style="yellow", markup=False)
        else:
            self.console.print(f"Notebook is open (session {session.session_id}).", markup=False)
        return session, client

    def _cell_count(self, client: Any) -> int | None:
        """Number of cells in the active notebook, or None when it cannot be determined."""
        if client is None:
            return None
        self._check_stopped()
        try:
            result = client.execute(self._cli._list_cells_code(), notebook=self.config.notebook, timeout=5.0)
        except Exception:  # noqa: BLE001 - informational only; never fail a switch over it
            return None
        if not result.success:
            return None
        return sum(1 for line in (result.stdout or "").splitlines() if line.strip())

    def _context(self) -> None:
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
