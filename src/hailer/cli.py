"""Hailer command line: the conversational control plane.

``uvx hailer``            chat with the agent (default; marimo must already be running)
``uvx hailer notebook``   one-command session: start (or reuse) marimo, open the notebook, chat,
                          stop marimo on exit
``uvx hailer exec``       run code in the live kernel (debugging / scripting)
``uvx hailer status``     show configuration, credentials source and marimo state
``uvx hailer doctor``     run the preflight checks and print a table
``uvx hailer login``      store an API key in the OS credential store
``uvx hailer logout``     remove it
``uvx hailer init``       set up a workspace: hailer.toml, .config/hailer, the starter notebook, data/
``uvx hailer kernel``     the Docker kernel: ``pull`` or ``build`` its image, ``stop`` this workspace's kernel

Collaborators (config, marimo client, agent, secrets, context) are reached through
small module-level factory functions so tests can replace them.
"""

from __future__ import annotations

import os
import shutil
import sys
import traceback
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from hailer import __version__, notebooks
from hailer.errors import (
    ConfigError,
    CredentialsError,
    HailerError,
    MarimoUnavailableError,
    NoSessionError,
)
from hailer.models import (
    KERNEL_RUNTIME_DOCKER,
    VALID_KERNEL_RUNTIMES,
    WIRE_API_CHAT,
    AgentEvent,
    Check,
    ContextBundle,
    HailerConfig,
    MarimoServer,
    MarimoSession,
    ProviderConfig,
    SessionState,
    SkillInfo,
    TurnSummary,
)
from hailer.session import (
    COMMANDS,
    EXIT_COMMANDS,
    help_text,
    load_session,
    parse_command,
    save_session,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    invoke_without_command=True,
    help="Hailer: chat with your data. Explore, analyse and chart local data files in a live marimo notebook.",
)

PROMPT = "You > "
ANSWER_HEADER = "Hailer >"


# --------------------------------------------------------------------------- #
# Options and injectable collaborators
# --------------------------------------------------------------------------- #


@dataclass
class CliOptions:
    verbose: bool = False
    config_path: Path | None = None
    workspace: Path | None = None
    new_thread: bool = False


def console_factory() -> Console:
    return Console(highlight=False, soft_wrap=True)


def _reconfigure_streams() -> None:
    """Never crash on characters the console encoding cannot represent (cp1252 pipes on Windows)."""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="replace")
        except (ValueError, OSError):  # pragma: no cover - closed or exotic streams
            pass


def _load_config(opts: CliOptions) -> HailerConfig:
    """The resolved configuration with ``notebook`` set to the *active* notebook.

    Every subcommand goes through here, so from this point on ``config.notebook`` means the
    notebook the chat is working in (the state file wins over ``[hailer].notebook``; see
    :func:`hailer.notebooks.load_active_notebook`). ``HAILER_NOTEBOOK`` is an explicit choice:
    it is written to the state file so the tool server (a separate process that reads only the
    file) and later sessions agree with this one.
    """
    from hailer.config import load_config

    config = load_config(workspace=opts.workspace, config_path=opts.config_path)
    if os.environ.get("HAILER_NOTEBOOK"):
        try:
            notebooks.save_active_notebook(config, config.notebook)
        except OSError:  # pragma: no cover - a read-only workspace must not stop the chat
            pass
        return config
    return replace(config, notebook=notebooks.load_active_notebook(config))


def _validate_config(config: HailerConfig) -> list[str]:
    from hailer.config import validate

    return validate(config)


def _setup_logging(config: HailerConfig, opts: CliOptions) -> None:
    from hailer.log import setup_logging

    setup_logging(config.log_level, verbose=opts.verbose)


def _find_server(config: HailerConfig) -> MarimoServer | None:
    from hailer.marimo_client import find_server

    return find_server(config)


def _make_client(server: MarimoServer, config: HailerConfig) -> Any:
    """A client for ``server``: its token (else ``HAILER_MARIMO_TOKEN``) and its path map.

    Links in its error hints carry the token: the CLI prints them for the user and never sends
    them to the model (the agent's tools build their own clients, without it).
    """
    from hailer.marimo_client import MarimoClient

    return MarimoClient(
        server.url,
        server.token or config.marimo_token,
        notebook=config.notebook,
        workspace=config.workspace,
        paths=server.paths,
        token_in_links=True,
    )


def _signed_in(server: MarimoServer, config: HailerConfig) -> MarimoServer:
    """``server`` with the token a browser needs: its own, else ``HAILER_MARIMO_TOKEN``."""
    if server.token or not config.marimo_token:
        return server
    return replace(server, token=config.marimo_token)


def _notebook_url(server: MarimoServer, config: HailerConfig) -> str:
    """The active notebook's URL for the user (printed or opened here, never sent to the model);
    marimo's home page when the kernel cannot see the notebook (:func:`_outside_kernel_note` says why)."""
    from hailer.errors import NotebookPathError
    from hailer.marimo_client import open_notebook_url

    try:
        return open_notebook_url(_signed_in(server, config), config.notebook, config.workspace, with_token=True)
    except NotebookPathError:
        return _home_url(server, config)


def _outside_kernel_note(server: MarimoServer, config: HailerConfig) -> str | None:
    """Why the notebook's link is marimo's home page: the running (docker) kernel does not mount
    the folder the notebook is in. ``None`` when the kernel sees it."""
    paths = server.paths
    if paths is None or paths.identity:
        return None
    try:
        paths.to_kernel(config.notebook)
    except ValueError:
        return (
            f"The running kernel cannot see {config.notebook}: it mounts other folders. The link opens marimo's "
            "home page instead. Keep notebooks in the notebooks folder, or stop the kernel (uvx hailer kernel stop) "
            "so the next start mounts the folders in hailer.toml."
        )
    return None


def _notebook_link(console: Console, server: MarimoServer, config: HailerConfig) -> str:
    """:func:`_notebook_url`, with a note printed when it had to fall back to the home page."""
    note = _outside_kernel_note(server, config)
    if note is not None:
        console.print(note, style="yellow", markup=False)
    return _notebook_url(server, config)


def _home_url(server: MarimoServer, config: HailerConfig) -> str:
    from hailer.marimo_client import home_url

    return home_url(_signed_in(server, config), with_token=True)


def _runtime_for(config: HailerConfig) -> Any:
    """The kernel runtime ``[kernel] runtime`` asks for (``ConfigError`` for an unknown one). Nothing
    is checked until its ``check``, ``prepare`` or ``start`` runs."""
    from hailer.kernel import runtime_for

    return runtime_for(config)


def _docker_runner() -> Any:
    """What ``hailer kernel ...`` runs ``docker`` with (the CLI on PATH)."""
    from hailer.kernel_docker import SubprocessDockerRunner

    return SubprocessDockerRunner()


def _kernel_mismatch(config: HailerConfig, server: MarimoServer | None) -> list[str]:
    """How the docker kernel ``server`` (found through ``.hailer/kernel.json``) was started
    differently from what ``config`` asks for now; empty for any other server."""
    if server is None or server.source != "kernel" or server.runtime != KERNEL_RUNTIME_DOCKER:
        return []
    from hailer.kernel import read_kernel_state
    from hailer.kernel_docker import settings_mismatch

    state = read_kernel_state(config.workspace)
    if state is None or state.token != server.token:
        return []
    return settings_mismatch(state, config)


def _docker_on_path() -> bool:
    return shutil.which("docker") is not None


def _periods_probe_code() -> str:
    return "import hailer.periods\nprint('hailer.periods imported')"


def _attach_runtime(config: HailerConfig, server: MarimoServer | None) -> HailerConfig:
    from hailer.kernel import attach_runtime

    return attach_runtime(config, server)


def _kernel_line(config: HailerConfig) -> str:
    from hailer.kernel import describe_runtime

    return describe_runtime(config.kernel)


def _launch_command() -> list[str]:
    from hailer.marimo_client import launch_command

    return launch_command()


def _cm_help_code() -> str:
    from hailer.marimo_client import CM_HELP_CODE

    return CM_HELP_CODE


def _list_cells_code() -> str:
    from hailer.marimo_client import LIST_CELLS_CODE

    return LIST_CELLS_CODE


def _resolve_key(provider: ProviderConfig) -> tuple[str | None, str]:
    from hailer.secrets import resolve_provider_key

    return resolve_provider_key(provider)


def _store_key(provider: ProviderConfig, value: str) -> None:
    from hailer.secrets import store_provider_key

    store_provider_key(provider, value)


def _delete_key(provider: ProviderConfig) -> bool:
    from hailer.secrets import delete_provider_key

    return delete_provider_key(provider)


def _load_context(config: HailerConfig) -> ContextBundle:
    from hailer.context import load_context

    return load_context(config)


def _render_prompt(config: HailerConfig, name: str, args: str) -> str:
    from hailer.context import render_prompt

    return render_prompt(config, name, args)


def _make_agent(config: HailerConfig, bundle: ContextBundle, server: MarimoServer | None = None) -> Any:
    """The agent; ``server`` pins its tools to the server ``hailer notebook`` started or reused."""
    from hailer.agent import HailerAgent

    return HailerAgent(config, bundle, server=server)


def _open_browser(url: str) -> None:
    from hailer.browser import open_url  # os.startfile first on Windows (ignores BROWSER), never raises

    open_url(url)


def _find_free_port(preferred: int) -> int:
    from hailer.marimo_client import find_free_port

    return find_free_port(preferred)


def _wait_for_session(client: Any, notebook: Path, timeout: float) -> MarimoSession | None:
    from hailer.marimo_client import wait_for_session

    return wait_for_session(client, notebook, timeout)


def _write_default_config(path: Path) -> None:
    from hailer.config import write_default_config

    write_default_config(path)


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #


def _print_error(console: Console, err: HailerError, *, verbose: bool = False) -> None:
    # Error text can contain "[web]" / "[model_providers.x]" which Rich markup would swallow.
    console.print(str(err), style="red", markup=False)
    if err.hint:
        for line in str(err.hint).splitlines():
            console.print(f"    {line}", markup=False)
    if verbose:
        console.print(traceback.format_exc(), markup=False)


def _print_unexpected(console: Console, exc: BaseException, *, verbose: bool) -> None:
    console.print(f"Unexpected error: {type(exc).__name__}: {exc}", style="red", markup=False)
    if verbose:
        console.print(traceback.format_exc(), markup=False)
    else:
        console.print("    Run with --verbose for details.", markup=False)


def _labelled(label: str, style: str, text: str) -> Text:
    """A coloured label followed by verbatim (non-markup) text."""
    return Text.assemble((label, style), text)


def _provider_line(config: HailerConfig) -> str:
    try:
        provider = config.provider
    except KeyError:
        return f"{config.model.provider} (not declared in hailer.toml)"
    return _describe_provider(provider)


def _describe_provider(provider: ProviderConfig) -> str:
    """``id (base_url[, chat completions][, no streaming])`` for a custom endpoint, ``id`` for the built-in one."""
    if not provider.base_url:
        return provider.id
    notes = [provider.base_url]
    if provider.wire_api == WIRE_API_CHAT:
        notes.append("chat completions")
    if not provider.stream:
        notes.append("no streaming")
    return f"{provider.id} ({', '.join(notes)})"


def _web_line(config: HailerConfig) -> str:
    domains = list(config.web.allowed_domains)
    if not domains:
        return "none"
    shown = ", ".join(domains[:4])
    if len(domains) > 4:
        shown += f" (+{len(domains) - 4} more)"
    return shown


def _relative(path: Path, workspace: Path) -> str:
    try:
        return str(Path(path).relative_to(workspace))
    except ValueError:
        return str(path)


def _same_file(a: Path, b: Path) -> bool:
    """Path equality that tolerates case and separator differences (Windows)."""
    try:
        return os.path.normcase(str(Path(a).resolve())) == os.path.normcase(str(Path(b).resolve()))
    except OSError:
        return False


def _inside(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except (ValueError, OSError):
        return False
    return True


def _session_for(sessions: list[MarimoSession], path: Path, workspace: Path | None = None) -> MarimoSession | None:
    """The kernel session for ``path`` among ``sessions``: exact path match only (no filename fallback,
    so two notebooks called ``report.py`` in different folders never share a session)."""
    from hailer.marimo_client import match_session

    return match_session(sessions, path, workspace)


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" + ("" if count == 1 else "s")


def _startup_panel(console: Console, config: HailerConfig) -> None:
    body = "\n".join(
        [
            f"Model:      {config.model.name}",
            f"Provider:   {_provider_line(config)}",
            f"Notebook:   {_relative(config.notebook, config.workspace)}",
            f"Notebooks:  {_relative(config.notebooks_root, config.workspace)}",
            f"Workspace:  {config.workspace}",
            f"Kernel:     {_kernel_line(config)}",
            f"Web access: {_web_line(config)}",
        ]
    )
    console.print(Panel(Text(body), title="Hailer", expand=False))
    console.print("Type /help for commands.\n", markup=False)


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #


def _marimo_state(
    config: HailerConfig, pinned: MarimoServer | None = None
) -> tuple[MarimoServer | None, MarimoSession | None, HailerError | None]:
    """(server, session, error) without raising. ``pinned``: the server this chat was started
    with (``hailer notebook``), used instead of discovering one."""
    if pinned is not None:
        server: MarimoServer | None = pinned
    else:
        try:
            server = _find_server(config)
        except HailerError as err:
            return None, None, err
    if server is None:
        return None, None, None
    try:
        client = _make_client(server, config)
        if not client.health():
            return None, None, None
        session = client.resolve_session(config.notebook)
        return server, session, None
    except NoSessionError as err:
        return server, None, err
    except HailerError as err:
        return server, None, err


def _discovered_server(config: HailerConfig) -> MarimoServer | None:
    try:
        return _find_server(config)
    except HailerError:
        return None


def _launch_hint() -> str:
    from hailer.marimo_client import launch_hint

    return launch_hint()


def kernel_checks(config: HailerConfig, *, starting: bool = False) -> list[Check]:
    """The kernel runtime's rows (``hailer doctor``, ``hailer notebook``); a runtime that cannot be
    used is one failed ``kernel`` row. ``starting``: a start follows, which downloads a missing
    image itself, so that warning is left out."""
    try:
        checks = list(_runtime_for(config).check())
    except HailerError as err:
        return [Check("kernel", False, str(err), hint=err.hint)]
    if starting:
        checks = [c for c in checks if not (c.name == "image" and not c.ok and not c.fatal)]
    return checks


def preflight(config: HailerConfig) -> list[Check]:
    """Startup checks. Fatal failures stop the chat; the rest are printed as warnings."""
    checks = local_checks(config)
    checks.extend(marimo_checks(config))
    return checks


def marimo_checks(config: HailerConfig) -> list[Check]:
    """The marimo server and kernel-session checks (the part ``hailer notebook`` handles itself)."""
    checks: list[Check] = []
    server, session, err = _marimo_state(config)
    if server is None:
        summary = "Marimo is not running." if err is None else str(err)
        hint = err.hint if err is not None and err.hint else _launch_hint()
        checks.append(Check("marimo", False, summary, hint=hint))
    else:
        checks.append(Check("marimo", True, server.url, fatal=False))
        if session is not None:
            checks.append(Check("session", True, f"notebook is open (session {session.session_id})", fatal=False))
        else:
            url = _notebook_url(server, config)
            hint = err.hint if err is not None and err.hint else f"Open {url} in your browser."
            note = _outside_kernel_note(server, config)
            checks.append(Check("session", False, "the notebook is not open in a browser", hint=f"{note}\n{hint}" if note else hint, fatal=False))
    return checks


CONFIG_FIX_HINT = "Fix hailer.toml (uvx hailer init writes a starter file and creates a missing notebook)."


def _config_check(problems: list[str]) -> Check:
    """The ``config`` row, each problem on a line of its own. Warnings make it a warning row, so
    they show when a session starts too (a setting in the wrong table must not go unnoticed)."""
    fatal = [p for p in problems if not p.lower().startswith("warning")]
    warnings = [p for p in problems if p.lower().startswith("warning")]
    if fatal:
        summary = fatal[0] if len(fatal) == 1 else f"{len(fatal)} problems in the configuration:"
        listed = [] if len(fatal) == 1 else [f"- {p}" for p in fatal]
        return Check("config", False, summary, hint="\n".join([*listed, *warnings, CONFIG_FIX_HINT]), fatal=True)
    if not warnings:
        return Check("config", True, "ok", fatal=False)
    if len(warnings) == 1:
        return Check("config", False, warnings[0], fatal=False)
    return Check("config", False, f"{len(warnings)} warnings:", hint="\n".join(f"- {w}" for w in warnings), fatal=False)


def local_checks(config: HailerConfig) -> list[Check]:
    """Config, notebook and credential checks (everything except the marimo server)."""
    checks: list[Check] = [_config_check(_validate_config(config))]

    if config.notebook.exists():
        checks.append(Check("notebook", True, f"{_relative(config.notebook, config.workspace)} (active)"))
    else:
        checks.append(
            Check(
                "notebook",
                False,
                f"not found: {config.notebook}",
                hint=(
                    "Run uvx hailer init to create it from the starter template, or fix [hailer].notebook in "
                    "hailer.toml. Notebooks created from the chat (/notebook new <name>) live in "
                    f"{_relative(config.notebooks_root, config.workspace)}."
                ),
            )
        )

    try:
        provider = config.provider
    except KeyError:
        checks.append(
            Check(
                "credentials",
                False,
                f"provider {config.model.provider!r} is not declared",
                hint="Add a [model_providers.<id>] table to hailer.toml or set [model].provider.",
            )
        )
    else:
        value, source = _resolve_key(provider)
        del value
        if source != "missing":
            checks.append(Check("credentials", True, f"{provider.env_key or 'API key'} from {source}", fatal=False))
        else:
            checks.append(
                Check(
                    "credentials",
                    False,
                    f"{provider.env_key} is not set (required by provider {provider.id!r})",
                    hint=f"Store it once with:\n\n    uvx hailer login {provider.id}\n\nor set the {provider.env_key} environment variable in this terminal.",
                )
            )
    return checks


def _print_checks(console: Console, checks: list[Check], *, only_failures: bool = False) -> None:
    for check in checks:
        if check.ok and only_failures:
            continue
        if check.ok:
            console.print(_labelled("ok", "green", f"    {check.name}: {check.summary}"))
            continue
        if check.fatal:
            console.print(_labelled("FAIL", "red", f"  {check.name}: {check.summary}"))
        else:
            console.print(_labelled("warn", "yellow", f"  {check.name}: {check.summary}"))
        if check.hint:
            for line in check.hint.splitlines():
                console.print(f"        {line}", markup=False)


# --------------------------------------------------------------------------- #
# Chat loop
# --------------------------------------------------------------------------- #


@dataclass
class _TurnDisplay:
    """Drives the progress line for one turn and prints the final answer exactly once.

    Message deltas are not printed as they arrive: a model can write interim commentary next
    to its tool calls and then the final answer, and streaming both printed the answer twice.
    The spinner shows the latest tool instead, and ``finish`` prints ``final_response``.
    """

    console: Console
    status: Any = None
    last_activity: str = ""
    deltas: int = 0

    def start(self) -> None:
        self.status = self.console.status("Thinking...", spinner="dots")
        self.status.start()

    def stop(self) -> None:
        if self.status is not None:
            self.status.stop()
            self.status = None

    def _update(self, text: str) -> None:
        if self.status is not None:
            self.status.update(Text(text))

    def __call__(self, event: AgentEvent) -> None:
        if event.kind == "message_delta":
            self.deltas += 1
            if self.deltas == 1:
                self._update("Writing reply...")
            return
        if event.kind == "tool_call":
            width = max(20, self.console.width - 12)
            text = event.text.replace("\n", " ")
            self.last_activity = text[: width - 1] + "…" if len(text) > width else text
            self._update(f"Using: {self.last_activity}")
            return
        if event.kind == "status" and event.text:
            self._update(event.text)

    def finish(self, summary: TurnSummary) -> None:
        self.stop()
        final = (summary.final_response or "").strip()
        self.console.print(Text(ANSWER_HEADER, style="dim"))
        if final:
            self.console.print(final, markup=False)
        else:
            self.console.print(Text("(no reply)", style="dim"))
        self.console.print()


class ChatLoop:
    """The REPL. ``server`` pins the chat to the server ``hailer notebook`` started or reused: the
    chat and the agent's tools keep its URL, token and paths in memory, so a ``kernel.json``
    rewritten or removed underneath (another terminal) cannot take them away. Without it every
    call discovers the server anew (chat-only ``uvx hailer`` follows a restarted kernel)."""

    def __init__(self, console: Console, config: HailerConfig, opts: CliOptions, server: MarimoServer | None = None) -> None:
        self.console = console
        self.config = config
        self.opts = opts
        self.server = server
        self.state: SessionState = load_session(config.workspace)
        self.bundle: ContextBundle = ContextBundle()
        self.agent: Any = None
        # One-line notice sent with the next user message after a /notebook switch (the model
        # learns about switches it made itself from its own tool results).
        self._pending_preamble: str | None = None

    # -- lifecycle --------------------------------------------------------- #

    def start(self) -> None:
        self.bundle = _load_context(self.config)
        self._print_context_warnings()
        self.agent = _make_agent(self.config, self.bundle, self.server)
        resume = None if self.opts.new_thread else self.state.thread_id
        forget = self.state.thread_id if self.opts.new_thread else None  # --new: drop the stored conversation
        thread_id = self.agent.start(resume_thread_id=resume, forget_thread_id=forget)
        if resume and thread_id != resume:
            self.console.print(Text("Previous conversation could not be resumed; started a new one.", style="dim"))
            self.state = SessionState()
        elif resume:
            self.console.print(Text(f"Resumed conversation ({self.state.turns} turns so far). /new starts fresh.", style="dim"))
        elif self.opts.new_thread:
            self.state = SessionState()
        self.state.thread_id = thread_id
        self.state.model = self.state.model or self.config.model.name
        self.state.provider = self.state.provider or self.config.model.provider
        save_session(self.config.workspace, self.state)

    def close(self) -> None:
        if self.agent is not None:
            try:
                self.agent.close()
            except Exception:  # pragma: no cover - best effort
                pass

    def _print_context_warnings(self) -> None:
        for warning in self.bundle.warnings:
            self.console.print(_labelled("warn", "yellow", f"  {warning}"))

    # -- REPL -------------------------------------------------------------- #

    def run(self) -> None:
        while True:
            try:
                line = self.console.input(f"[bold cyan]{PROMPT}[/bold cyan]")
            except EOFError:
                self.console.print("\nBye.")
                return
            except KeyboardInterrupt:
                self.console.print("\nBye.")
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
                _print_error(self.console, err, verbose=self.opts.verbose)
            except KeyboardInterrupt:
                self.console.print("\nInterrupted.")
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                _print_unexpected(self.console, exc, verbose=self.opts.verbose)

    # -- turns ------------------------------------------------------------- #

    def _turn(self, text: str, *, skill: SkillInfo | None = None) -> None:
        display = _TurnDisplay(self.console)
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
            self.state.turns += 1
            self.state.input_tokens += summary.input_tokens or 0
            self.state.output_tokens += summary.output_tokens or 0
            if summary.thread_id:
                self.state.thread_id = summary.thread_id
            save_session(self.config.workspace, self.state)
        finally:
            # Also after Ctrl+C or a failed turn: a notebook_create/notebook_open tool call may have
            # completed (and switched the state file) before the turn was cut short.
            self._sync_active_notebook()

    def _sync_active_notebook(self, *, open_browser: bool = True) -> bool:
        """Pick up a switch the model made during a turn (its tools write the same state file).

        Returns True when the active notebook changed. With ``open_browser`` the notebook's URL is
        opened once when it has no kernel session yet.
        """
        try:
            active = notebooks.load_active_notebook(self.config)
        except Exception:  # noqa: BLE001 - a bad state file must never spoil a finished turn
            return False
        if _same_file(active, self.config.notebook):
            return False
        self.config = replace(self.config, notebook=active)
        name = notebooks.notebook_display_name(self.config, active)
        self.console.print(Text(f"Active notebook is now {name}.", style="dim"))
        if open_browser:
            server, session, _err = _marimo_state(self.config, self.server)
            if server is not None and session is None:
                url = _notebook_link(self.console, server, self.config)
                self.console.print(f"Opening {url} in your browser...", markup=False)
                _open_browser(url)
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
        self.state = SessionState(
            thread_id=self.agent.new_thread(),
            model=self.state.model,
            provider=self.state.provider,
        )
        save_session(self.config.workspace, self.state)
        self.console.print(reason, markup=False)

    def _status(self) -> None:
        self._sync_active_notebook(open_browser=False)  # a late tool call may have switched notebooks
        server, session, _err = _marimo_state(self.config, self.server)
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
            ("Kernel", _kernel_line(self.config)),
            ("Notebook", _relative(self.config.notebook, self.config.workspace)),
            ("Notebooks", _relative(self.config.notebooks_root, self.config.workspace)),
            ("Web access", _web_line(self.config)),
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
            return _provider_line(self.config)
        provider = self._active_provider()
        if provider is None:
            return f"{provider_id} (not declared in hailer.toml)"
        return _describe_provider(provider)

    def _credentials_line(self) -> str:
        provider = self._active_provider()
        if provider is None:
            return "unknown provider"
        source = getattr(self.agent, "key_source", None)
        if not isinstance(source, str) or not source:
            try:
                _value, source = _resolve_key(provider)
            except HailerError:
                source = "missing"
        if source == "missing":
            return f"{provider.env_key or 'API key'} missing (run: uvx hailer login {provider.id})"
        return f"{provider.env_key or 'API key'} from {source}"

    def _model(self, args: str) -> None:
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
        self._new_thread(reason=f"Model set to {name}" + (f" ({provider})" if provider else "") + "; started a new thread.")

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
            self.console.print(f"Unknown /notebook subcommand {sub!r}. {NOTEBOOK_USAGE}", markup=False)

    def _display_name(self, path: Path) -> str:
        return notebooks.notebook_display_name(self.config, path)

    def _notebook_show(self) -> None:
        cfg, out = self.config, self.console
        out.print(f"Notebook:  {self._display_name(cfg.notebook)} (active)", markup=False)
        count = len(notebooks.list_notebooks(cfg))
        out.print(f"Notebooks: {self._display_name(cfg.notebooks_root)} ({_plural(count, 'notebook')}; /notebook list)", markup=False)
        recent = [p for p in notebooks.load_recent(cfg) if not _same_file(p, cfg.notebook)]
        if recent:
            out.print("Recent:    " + ", ".join(self._display_name(p) for p in recent[:5]), markup=False)
        server, session, _err = _marimo_state(cfg, self.server)
        if server is None:
            out.print("Marimo:    not running", markup=False)
        else:
            state = f"session {session.session_id}" if session is not None else "not open in a browser"
            out.print(f"Marimo:    {server.url} ({state})", markup=False)
            out.print(f"URL:       {_notebook_link(out, server, cfg)}", markup=False)
            out.print("View:      app view (results only); Ctrl+. in the notebook toggles the code editor", markup=False)
        out.print("Start everything in one go:  uvx hailer notebook", markup=False)
        out.print(f"Launch:    {' '.join(_launch_command())}", markup=False)
        out.print(NOTEBOOK_USAGE, markup=False)

    def _server_sessions(self) -> tuple[MarimoServer | None, Any, list[MarimoSession]]:
        """(server, client, open sessions); server is None when marimo is not running."""
        try:
            server = self.server if self.server is not None else _find_server(self.config)
        except HailerError:
            return None, None, []
        if server is None:
            return None, None, []
        client = _make_client(server, self.config)
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
            if _same_file(info.path, self.config.notebook):
                markers.append("active")
            if _session_for(sessions, info.path, self.config.workspace) is not None:
                markers.append("open")
            modified = datetime.fromtimestamp(info.modified).strftime("%Y-%m-%d %H:%M")
            table.add_row(
                Text(self._display_name(info.path)),
                Text(modified),
                Text(f"{info.size / 1024:.1f} KB"),
                Text(", ".join(markers)),
            )
        self.console.print(table)
        self.console.print(f"{_plural(len(infos), 'notebook')} in {folder}. /notebook open <name> switches.", markup=False)

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
        if _same_file(path, self.config.notebook):
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
        if _same_file(path, self.config.notebook):
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
            detail = how
            if session is not None:
                count = self._cell_count(client)
                if count is not None:
                    detail += f", {_plural(count, 'cell')}"
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
        server, session, _err = _marimo_state(cfg, self.server)
        if server is None:
            self.console.print("Marimo is not running; the notebook opens once it is (uvx hailer notebook).", markup=False)
            return None, None
        client = _make_client(server, cfg)
        if session is not None:
            self.console.print(f"Notebook is open (session {session.session_id}).", markup=False)
            return session, client
        url = _notebook_link(self.console, server, cfg)
        self.console.print(f"Opening {url} in your browser...", markup=False)
        _open_browser(url)
        with self.console.status("Waiting for the notebook to open..."):
            session = _wait_for_session(client, cfg.notebook, SWITCH_SESSION_TIMEOUT_SEC)
        if session is None:
            self.console.print(f"No kernel session yet. Open {url} in your browser.", style="yellow", markup=False)
        else:
            self.console.print(f"Notebook is open (session {session.session_id}).", markup=False)
        return session, client

    def _cell_count(self, client: Any) -> int | None:
        """Number of cells in the active notebook, or None when it cannot be determined."""
        if client is None:
            return None
        try:
            result = client.execute(_list_cells_code(), notebook=self.config.notebook)
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
                out.print(f"  {_relative(path, self.config.workspace)}  ({size} bytes)", markup=False)
        else:
            out.print(f"Context files: none ({_relative(self.config.context_dir, self.config.workspace)})", markup=False)
        if bundle.skills:
            out.print("Skills:", markup=False)
            for skill in bundle.skills:
                out.print(f"  {skill.name}: {skill.description}", markup=False)
        else:
            out.print(f"Skills: none ({_relative(self.config.skills_dir, self.config.workspace)})", markup=False)
        if bundle.prompts:
            out.print("Prompts: " + ", ".join(sorted(bundle.prompts)), markup=False)
        else:
            out.print(f"Prompts: none ({_relative(self.config.prompts_dir, self.config.workspace)})", markup=False)
        out.print(f"Web access: {_web_line(self.config)}", markup=False)
        for warning in bundle.warnings:
            out.print(_labelled("warn", "yellow", f"  {warning}"))

    def _skill(self, args: str) -> None:
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
        self._turn(message, skill=skill)

    def _prompt(self, args: str) -> None:
        if not args:
            names = ", ".join(sorted(self.bundle.prompts)) or "none"
            self.console.print(f"Usage: /prompt <name> [args]. Available: {names}", markup=False)
            return
        name, _, rest = args.partition(" ")
        text = _render_prompt(self.config, name, rest.strip())
        self._turn(text)

    def _reload(self) -> None:
        self.bundle = _load_context(self.config)
        self._print_context_warnings()
        if hasattr(self.agent, "bundle"):
            self.agent.bundle = self.bundle
        self.console.print(
            f"Reloaded {len(self.bundle.context_files)} context file(s), {len(self.bundle.skills)} skill(s), "
            f"{len(self.bundle.prompts)} prompt(s). Applies from your next message.",
            markup=False,
        )


def run_chat(opts: CliOptions) -> None:
    console = console_factory()
    try:
        config = _load_config(opts)
    except ConfigError as err:
        _print_error(console, err, verbose=opts.verbose)
        raise typer.Exit(code=1)
    _setup_logging(config, opts)
    requested = config
    # A docker kernel Hailer started for this workspace makes this session docker (see attach_runtime).
    discovered = _discovered_server(config)
    config = _attach_runtime(config, discovered)
    _startup_panel(console, config)

    checks = preflight(requested)  # the settings as written; the attached kernel is checked by marimo_checks
    diffs = _kernel_mismatch(requested, discovered)
    if diffs:
        checks.append(_mismatch_check(diffs))
    _print_checks(console, checks, only_failures=True)
    if any(not c.ok and c.fatal for c in checks):
        raise typer.Exit(code=1)
    session_check = next((c for c in checks if c.name == "session"), None)
    if session_check is not None and not session_check.ok:
        server, _s, _e = _marimo_state(config)
        if server is not None:
            url = _notebook_link(console, server, config)
            console.print(f"Opening {url} in your browser...", markup=False)
            _open_browser(url)
            console.print(APP_VIEW_HINT, markup=False)
    if any(not c.ok for c in checks):
        console.print()

    _run_chat_loop(console, config, opts)


def _mismatch_check(diffs: list[str]) -> Check:
    """A warning row: the kernel this chat attaches to was started with other settings."""
    return Check(
        "kernel",
        False,
        f"the running kernel was started with other settings ({'; '.join(diffs)}); the chat uses it as it is",
        hint="uvx hailer kernel stop, then uvx hailer notebook, starts one with the settings in hailer.toml.",
        fatal=False,
    )


def _run_chat_loop(console: Console, config: HailerConfig, opts: CliOptions, server: MarimoServer | None = None) -> None:
    """Start the agent and run the REPL (pinned to ``server`` when given); exit code 1 when the
    agent cannot start."""
    loop = ChatLoop(console, config, opts, server)
    try:
        loop.start()
    except HailerError as err:
        _print_error(console, err, verbose=opts.verbose)
        loop.close()
        raise typer.Exit(code=1)
    except Exception as exc:  # noqa: BLE001
        _print_unexpected(console, exc, verbose=opts.verbose)
        loop.close()
        raise typer.Exit(code=1)
    try:
        loop.run()
    finally:
        loop.close()


# --------------------------------------------------------------------------- #
# Typer wiring
# --------------------------------------------------------------------------- #


def _opts(ctx: typer.Context) -> CliOptions:
    return ctx.obj if isinstance(ctx.obj, CliOptions) else CliOptions()


def _config_or_exit(console: Console, opts: CliOptions) -> HailerConfig:
    try:
        config = _load_config(opts)
    except ConfigError as err:
        _print_error(console, err, verbose=opts.verbose)
        raise typer.Exit(code=1)
    _setup_logging(config, opts)
    return config


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"hailer {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging and full tracebacks."),
    config: Path | None = typer.Option(None, "--config", help="Path to hailer.toml.", show_default=False),
    workspace: Path | None = typer.Option(None, "--workspace", help="Project directory (default: auto-detect).", show_default=False),
    new: bool = typer.Option(False, "--new", help="Start a new conversation instead of resuming."),
    version: bool = typer.Option(False, "--version", callback=_version_callback, is_eager=True, help="Show the version and exit."),
) -> None:
    del version
    ctx.obj = CliOptions(verbose=verbose, config_path=config, workspace=workspace, new_thread=new)
    if ctx.invoked_subcommand is None:
        run_chat(ctx.obj)


SESSION_TIMEOUT_SEC = 90.0
SWITCH_SESSION_TIMEOUT_SEC = 30.0  # how long /notebook new|open waits for the browser tab
NOTEBOOK_USAGE = "Usage: /notebook [list | new <name> [--empty] | open <name> | close [name]]"
APP_VIEW_HINT = (
    "The notebook opens in app view (results only). "
    "Press Ctrl+. in it, or use Toggle app view, to see and edit the code."
)


def _has_session_under(client: Any, root: Path) -> bool:
    """True when the server hosts a kernel session for any notebook inside ``root``."""
    try:
        sessions = client.sessions()
    except HailerError:
        return False
    return any(_inside(Path(s.path or s.filename or ""), root) for s in sessions if s.path or s.filename)


def _reusable_server(config: HailerConfig) -> tuple[MarimoServer, MarimoSession | None] | None:
    """A running server Hailer may attach to. In order: the one Hailer started for this workspace
    (``.hailer/kernel.json``, verified with its token; any runtime ``find_server`` allows), the
    configured URL, or (local runtime) a live registry server that already has a kernel session for
    the active notebook or for another notebook in the notebooks folder (one server hosts them all).
    Anything else gets a fresh server on its own port.

    A docker kernel started with other settings (folders, network, image, memory, cpus) than
    ``config`` asks for is refused with ``KernelRuntimeError``: attaching would silently ignore
    hailer.toml."""
    server = _discovered_server(config)
    if server is None:
        return None
    diffs = _kernel_mismatch(config, server)
    if diffs:
        from hailer.kernel_docker import mismatch_error

        raise mismatch_error(diffs)
    client = _make_client(server, config)
    try:
        if not client.health():
            return None
        session = client.resolve_session(config.notebook)
        return server, session
    except NoSessionError:
        if server.source == "kernel" or config.marimo_url or _has_session_under(client, config.notebooks_root):
            return server, None
        return None
    except HailerError:
        return None


def _kernel_choice(console: Console, value: str | None) -> str | None:
    """A ``--kernel`` value ("local" or "docker"); exit 2 for anything else."""
    if value is None:
        return None
    choice = value.strip().lower()
    if choice not in VALID_KERNEL_RUNTIMES:
        console.print(f'--kernel must be "local" or "docker", not "{value}".', style="red", markup=False)
        raise typer.Exit(code=2)
    return choice


def _free_port_url(console: Console, port: int) -> tuple[int, str]:
    chosen = _find_free_port(port)
    if chosen != port:
        console.print(f"Port {port} is busy; using {chosen}.", markup=False)
    return chosen, f"http://127.0.0.1:{chosen}"


def _prepare_runtime(console: Console, runtime: Any) -> None:
    """The runtime's slow steps with their own progress (docker: downloading the kernel image), before
    any spinner: a spinner line would fight with docker's progress bars."""
    runtime.prepare(say=lambda text: console.print(text, markup=False))


def _where(runtime: Any) -> str:
    return " in Docker" if getattr(runtime, "name", "") == KERNEL_RUNTIME_DOCKER else ""


def _start_kernel(console: Console, runtime: Any, port: int, *, verbose: bool) -> Any:
    """Start marimo through ``runtime`` and wait until ``/health`` answers; exit 1 when it does not."""
    try:
        _prepare_runtime(console, runtime)
        chosen, url = _free_port_url(console, port)
        with console.status(f"Starting marimo{_where(runtime)} on {url} ..."):
            running = runtime.start(chosen)
    except HailerError as err:  # the runtime has cleaned up; the hint carries the log tail
        _print_error(console, err, verbose=verbose)
        raise typer.Exit(code=1)
    console.print(f"Marimo is running at {running.server.url}  (log: {running.log_hint})", markup=False)
    return running


def _run_foreground(console: Console, config: HailerConfig, runtime: Any, port: int, *, open_browser: bool, verbose: bool) -> int:
    """``hailer notebook --foreground``: marimo in this terminal, no chat, until it exits or Ctrl+C.

    It is recorded in ``.hailer/kernel.json`` like any server Hailer starts, so ``uvx hailer`` in
    another terminal attaches to it. In docker mode the kernel's log is followed here, and Ctrl+C
    stops and removes the containers.
    """
    try:
        _prepare_runtime(console, runtime)
    except HailerError as err:
        _print_error(console, err, verbose=verbose)
        return 1
    chosen, url = _free_port_url(console, port)
    console.print(f"Starting marimo{_where(runtime)} on {url} (Ctrl+C stops it) ...", markup=False)
    try:
        running = runtime.start(chosen, foreground=True)
    except HailerError as err:
        _print_error(console, err, verbose=verbose)
        return 1
    try:
        console.print(f"Kernel:     {runtime.describe()}", markup=False)
        console.print(f"Marimo is running at {running.server.url}. Chat with it from another terminal: uvx hailer", markup=False)
        home = _home_url(running.server, config)
        if open_browser:
            console.print(f"Opening {home} in your browser...", markup=False)
            _open_browser(home)
        else:
            console.print(f"Open {home} in your browser.", markup=False)
        code = running.wait()
        if running.ended:
            console.print(running.ended, style="yellow", markup=False)
        return code
    finally:
        running.stop()
        _warn_about_planted_files(console, running)


def _warn_about_planted_files(console: Console, running: Any) -> None:
    """After a docker kernel stopped: files at the top of its notebooks folder that git or an
    editor would run code from (the kernel could write them there). Loud, and never removed."""
    planted = getattr(running, "planted", None)
    names = planted() if callable(planted) else []
    if names:
        from hailer.kernel_docker import planted_warning

        console.print(planted_warning(running.notebooks_folder, names), style="bold red", markup=False)


def _wait_for_notebook(console: Console, config: HailerConfig, server: MarimoServer, *, open_browser: bool) -> None:
    client = _make_client(server, config)
    url = _notebook_link(console, server, config)
    try:
        session = client.resolve_session(config.notebook)
    except HailerError:
        session = None
    if session is not None:
        console.print(f"Notebook is open (session {session.session_id}).", markup=False)
        return
    if open_browser:
        console.print(f"Opening {url} in your browser...", markup=False)
        _open_browser(url)
    else:
        console.print(f"Open {url} in your browser.", markup=False)
    console.print(APP_VIEW_HINT, markup=False)
    with console.status("Waiting for the notebook to open..."):
        session = _wait_for_session(client, config.notebook, SESSION_TIMEOUT_SEC)
    if session is None:
        console.print(
            f"No kernel session yet. Open {url} in your browser; the agent reports the notebook state when asked.",
            style="yellow",
            markup=False,
        )
    else:
        console.print(f"Notebook is open (session {session.session_id}).", markup=False)


@app.command()
def notebook(
    ctx: typer.Context,
    port: int = typer.Option(2718, "--port", "-p", help="Port for the marimo server (a free one is chosen if busy)."),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not open the notebook in a browser."),
    keep_marimo: bool = typer.Option(False, "--keep-marimo", help="Leave marimo running when the chat ends."),
    foreground: bool = typer.Option(False, "--foreground", help="Run marimo attached to this terminal, without the chat."),
    new: bool = typer.Option(False, "--new", help="Start a new conversation instead of resuming."),
    kernel: str | None = typer.Option(
        None,
        "--kernel",
        help='Where notebook code runs for this run: "local" (as you) or "docker" (an isolated container). Default: \\[kernel] runtime.',
        show_default=False,
    ),
) -> None:
    """Start marimo (or reuse a running one), open the notebook, and chat here; stop marimo on exit."""
    opts = _opts(ctx)
    if new:
        opts.new_thread = True
    console = console_factory()
    choice = _kernel_choice(console, kernel)
    config = _config_or_exit(console, opts)
    if choice is not None:  # the flag beats HAILER_KERNEL and the file
        config = replace(config, kernel=replace(config.kernel, runtime=choice))

    if foreground:
        config.notebooks_root.mkdir(parents=True, exist_ok=True)  # marimo edit refuses a missing folder
        if not config.notebook.exists():
            console.print(
                _labelled(
                    "Notebook not found:",
                    "yellow",
                    f" {config.notebook}. Marimo runs on the notebooks folder; run uvx hailer init to create "
                    "it, create another in the chat with /notebook new <name>, or fix [hailer].notebook in hailer.toml.",
                )
            )
        checks = kernel_checks(config, starting=True)
        _print_checks(console, checks, only_failures=True)
        if any(not c.ok and c.fatal for c in checks):
            raise typer.Exit(code=1)
        runtime = _runtime_for(config)
        raise typer.Exit(code=_run_foreground(console, config, runtime, port, open_browser=not no_browser, verbose=opts.verbose))

    checks = local_checks(config) + kernel_checks(config, starting=True)
    _print_checks(console, checks, only_failures=True)
    if any(not c.ok and c.fatal for c in checks):
        raise typer.Exit(code=1)
    config.notebooks_root.mkdir(parents=True, exist_ok=True)  # marimo edit refuses a missing folder

    running: Any = None
    try:
        reusable = _reusable_server(config)
    except HailerError as err:  # a kernel started with other settings
        _print_error(console, err, verbose=opts.verbose)
        raise typer.Exit(code=1)
    if reusable is not None:
        server, _session = reusable
        console.print(f"Using the running marimo at {server.url}.", markup=False)
        config = _attach_runtime(config, server)
    else:
        running = _start_kernel(console, _runtime_for(config), port, verbose=opts.verbose)
        server = running.server
    # The chat and the agent's tools keep this server (URL, token, paths) in memory: a kernel.json
    # rewritten or removed underneath cannot take it away. marimo_url names it in the prompt.
    config = replace(config, marimo_url=server.url)

    try:
        _wait_for_notebook(console, config, server, open_browser=not no_browser)
        console.print()
        _startup_panel(console, config)
        _run_chat_loop(console, config, opts, server)
    finally:
        if running is not None:
            if keep_marimo:
                console.print(
                    f"Marimo is still running at {server.url} ({running.stop_hint}; log: {running.log_hint}). "
                    "uvx hailer in this workspace attaches to it.",
                    markup=False,
                )
            else:
                running.stop()
                console.print("Stopped marimo.", markup=False)
                _warn_about_planted_files(console, running)


@app.command("exec")
def exec_(
    ctx: typer.Context,
    source: str | None = typer.Argument(None, help="Python file to run, or '-' for stdin.", show_default=False),
    code: str | None = typer.Option(None, "-c", "--code", help="Inline Python to run in the kernel scratchpad."),
) -> None:
    """Run Python in the live notebook kernel (scratchpad) and print the result."""
    opts = _opts(ctx)
    console = console_factory()
    config = _config_or_exit(console, opts)
    if code is None:
        if source is None:
            console.print("Give code with -c 'CODE', a file path, or '-' for stdin.", markup=False)
            raise typer.Exit(code=2)
        if source == "-":
            code = sys.stdin.read()
        else:
            path = Path(source)
            if not path.exists():
                console.print(f"File not found: {path}", style="red", markup=False)
                raise typer.Exit(code=1)
            code = path.read_text(encoding="utf-8")
    try:
        server = _find_server(config)
        if server is None:
            raise MarimoUnavailableError("Marimo is not running.", hint=_launch_hint())
        client = _make_client(server, config)
        result = client.execute(
            code,
            notebook=config.notebook,
            on_stdout=lambda s: console.print(s, end="", markup=False),
            on_stderr=lambda s: console.print(s, end="", style="red", markup=False) if s else None,
        )
    except HailerError as err:
        _print_error(console, err, verbose=opts.verbose)
        raise typer.Exit(code=1)
    if result.output and result.output.strip() and result.output.strip() != result.stdout.strip():
        console.print(result.output, markup=False)
    if not result.success:
        raise typer.Exit(code=1)


@app.command()
def status(ctx: typer.Context) -> None:
    """Show configuration, credential source, marimo state and loaded context."""
    opts = _opts(ctx)
    console = console_factory()
    config = _config_or_exit(console, opts)
    # Like doctor: the kernel in use, found without the health-gated session check (a busy server
    # must not flip the Kernel line back to hailer.toml's settings).
    config = _attach_runtime(config, _discovered_server(config))
    server, session, err = _marimo_state(config)
    _startup_panel(console, config)
    try:
        provider = config.provider
        _value, source = _resolve_key(provider)
        if source == "missing":
            console.print(f"Credentials: {provider.env_key or 'API key'} missing (run: uvx hailer login {provider.id})", markup=False)
        else:
            console.print(f"Credentials: {provider.env_key or 'API key'} from {source}", markup=False)
    except KeyError:
        console.print(f"Credentials: provider {config.model.provider!r} is not declared", markup=False)
    if server is None:
        console.print("Marimo:      not running", markup=False)
    elif session is None:
        console.print(f"Marimo:      {server.url} (notebook not open in a browser)", markup=False)
        if err is not None and err.hint:
            console.print(f"             {err.hint}", markup=False)
    else:
        console.print(f"Marimo:      {server.url} (session {session.session_id})", markup=False)
    try:
        bundle = _load_context(config)
    except HailerError as context_err:
        _print_error(console, context_err, verbose=opts.verbose)
        return
    console.print(
        f"Context:     {len(bundle.context_files)} file(s), {len(bundle.skills)} skill(s), {len(bundle.prompts)} prompt(s)",
        markup=False,
    )
    for warning in bundle.warnings:
        console.print(_labelled("warn", "yellow", f"  {warning}"))
    if config.config_path:
        console.print(f"Config:      {config.config_path}", markup=False)
    else:
        console.print("Config:      defaults (no hailer.toml found; run: uvx hailer init)", markup=False)


@app.command()
def doctor(ctx: typer.Context) -> None:
    """Run the startup checks and print a table with fixes."""
    opts = _opts(ctx)
    console = console_factory()
    config = _config_or_exit(console, opts)
    # Like status: a docker kernel Hailer started for this workspace is the kernel in use.
    discovered = _discovered_server(config)
    attached = _attach_runtime(config, discovered)
    checks = local_checks(config) + kernel_checks(attached)
    diffs = _kernel_mismatch(config, discovered)
    if diffs:
        checks.append(_mismatch_check(diffs))
    checks += marimo_checks(attached)
    config = attached
    table = Table(title="Hailer doctor", show_lines=False)
    table.add_column("Check")
    table.add_column("Result")
    table.add_column("Details")
    for check in checks:
        result = "OK" if check.ok else ("FAIL" if check.fatal else "WARN")
        details = check.summary
        if not check.ok and check.hint:
            details += "\n" + check.hint
        table.add_row(Text(check.name), Text(result), Text(details))
    console.print(table)

    server, session, _err = _marimo_state(config)
    if server is not None and session is not None:
        try:
            client = _make_client(server, config)
            result = client.execute(_cm_help_code(), notebook=config.notebook)
            if result.success and "get_context" in (result.stdout + result.output):
                console.print("ok    code mode: marimo._code_mode is available in the kernel", markup=False)
            else:
                console.print("warn  code mode: marimo._code_mode did not respond as expected; check the marimo version (0.24.x expected)", markup=False)
            result = client.execute(_periods_probe_code(), notebook=config.notebook)
            if result.success:
                console.print("ok    notebook helpers: hailer.periods imports in the kernel", markup=False)
            else:
                lines = (result.stderr or result.output or "").strip().splitlines()
                detail = f" ({lines[-1].strip()})" if lines else ""
                console.print(
                    f"warn  notebook helpers: hailer.periods did not import in the kernel{detail}; the starter "
                    "notebook needs it. Docker: uvx hailer kernel pull (or build) gets the right image.",
                    markup=False,
                )
        except HailerError as err:
            console.print(f"warn  code mode: {err}", markup=False)
    if any(not c.ok and c.fatal for c in checks):
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------- #
# hailer kernel: the Docker kernel's image and containers
# --------------------------------------------------------------------------- #

kernel_app = typer.Typer(
    no_args_is_help=True,
    help="The Docker kernel: download or build its image, stop this workspace's kernel.",
)
app.add_typer(kernel_app, name="kernel")


def _docker_runtime(config: HailerConfig) -> Any:
    from hailer.kernel_docker import DockerRuntime

    return DockerRuntime(config, runner=_docker_runner())


@kernel_app.command("pull")
def kernel_pull(ctx: typer.Context) -> None:
    """Download the kernel image for this Hailer version (or \\[kernel].image) ahead of the first start."""
    opts = _opts(ctx)
    console = console_factory()
    config = _config_or_exit(console, opts)
    runtime = _docker_runtime(config)
    try:
        version = runtime.pull(say=lambda text: console.print(text, markup=False))
    except HailerError as err:
        _print_error(console, err, verbose=opts.verbose)
        raise typer.Exit(code=1)
    console.print(f"{runtime.image} is ready (Hailer {version}).", markup=False)


@kernel_app.command("build")
def kernel_build(
    ctx: typer.Context,
    tag: str | None = typer.Option(
        None,
        "--tag",
        help="Name and tag for the image (default: \\[kernel].image, else ghcr.io/openafterhours/hailer-kernel:<version>).",
        show_default=False,
    ),
) -> None:
    """Build the kernel image on this machine, for machines that cannot download it."""
    from hailer import kernel_image
    from hailer.errors import KernelRuntimeError

    opts = _opts(ctx)
    console = console_factory()
    config = _config_or_exit(console, opts)
    runtime = _docker_runtime(config)
    image = tag or config.kernel.effective_image
    try:
        runtime.engine_version()
        args = kernel_image.build_args()
        console.print(
            f"Building {image} for Hailer {args['HAILER_VERSION']} (marimo {args['MARIMO_VERSION']}, "
            f"Polars {args['POLARS_VERSION']}, DuckDB {args['DUCKDB_VERSION']}) ...",
            markup=False,
        )
        code = kernel_image.build(image, runtime.runner)
        if code != 0:
            raise KernelRuntimeError(
                f"docker build failed (exit code {code}).",
                hint="docker's output is above. The build downloads the base image from Docker Hub and the Python packages from PyPI.",
            )
    except HailerError as err:
        _print_error(console, err, verbose=opts.verbose)
        raise typer.Exit(code=1)
    console.print(f"Built {image}.", markup=False)
    if image != config.kernel.effective_image:
        console.print(f'Hailer uses {config.kernel.effective_image}; set image = "{image}" under [kernel] in hailer.toml to use this one.', markup=False)


@kernel_app.command("stop")
def kernel_stop(ctx: typer.Context) -> None:
    """Stop this workspace's kernel (local or docker) and remove its Docker containers and network."""
    from hailer.kernel_docker import stop_workspace_kernels

    opts = _opts(ctx)
    console = console_factory()
    config = _config_or_exit(console, opts)
    try:
        report = stop_workspace_kernels(config, runner=_docker_runner())
    except HailerError as err:
        _print_error(console, err, verbose=opts.verbose)
        raise typer.Exit(code=1)
    for line in report.done or ([] if report.failed else ["No kernel is running for this workspace."]):
        console.print(line, markup=False)
    for line in report.failed:
        console.print(line, style="red", markup=False)
    for line in report.warnings:
        console.print(line, style="bold red", markup=False)
    if report.failed:
        raise typer.Exit(code=1)


def _provider_for(config: HailerConfig, provider_id: str) -> ProviderConfig:
    if provider_id in config.providers:
        return config.providers[provider_id]
    if provider_id == "openai":
        return ProviderConfig(id="openai", env_key="OPENAI_API_KEY")
    known = ", ".join(sorted({"openai", *config.providers}))
    raise CredentialsError(
        f"Unknown provider {provider_id!r}.",
        hint=f"Declared providers: {known}. Add a [model_providers.{provider_id}] table to hailer.toml first.",
    )


@app.command()
def login(
    ctx: typer.Context,
    provider: str = typer.Argument(..., help="Provider id from hailer.toml (or 'openai')."),
) -> None:
    """Store the provider's API key in the OS credential store (never in a file)."""
    opts = _opts(ctx)
    console = console_factory()
    config = _config_or_exit(console, opts)
    try:
        prov = _provider_for(config, provider)
        if not prov.env_key:
            raise CredentialsError(
                f"Provider {provider!r} has no env_key.",
                hint=f"Set model_providers.{provider}.env_key in hailer.toml to the environment variable that names the key.",
            )
        value = typer.prompt(f"API key for {provider} ({prov.env_key})", hide_input=True)
        if not value.strip():
            console.print("Nothing stored (empty key).", markup=False)
            raise typer.Exit(code=1)
        _store_key(prov, value.strip())
    except HailerError as err:
        _print_error(console, err, verbose=opts.verbose)
        raise typer.Exit(code=1)
    console.print(f"Stored {prov.env_key} for provider {provider!r} in the OS credential store.", markup=False)


@app.command()
def logout(
    ctx: typer.Context,
    provider: str = typer.Argument(..., help="Provider id from hailer.toml (or 'openai')."),
) -> None:
    """Remove the provider's API key from the OS credential store."""
    opts = _opts(ctx)
    console = console_factory()
    config = _config_or_exit(console, opts)
    try:
        prov = _provider_for(config, provider)
        removed = _delete_key(prov)
    except HailerError as err:
        _print_error(console, err, verbose=opts.verbose)
        raise typer.Exit(code=1)
    console.print("Removed." if removed else "No stored key found.", markup=False)


def _example_config_dir() -> Path | None:
    candidate = Path(__file__).resolve().parents[2] / ".config" / "hailer"
    return candidate if candidate.is_dir() else None


_PLACEHOLDERS: dict[str, str] = {
    "context/README.md": (
        "# Project context\n\nMarkdown files in this folder are sent to the model endpoint with every "
        "Hailer session. Put glossaries, column meanings and house rules here. Never put secrets here.\n"
    ),
    "skills/README.md": (
        "# Skills\n\nEach sub-folder holds a SKILL.md with `name:` and `description:` frontmatter. "
        "Hailer lists them to the agent and loads one on demand or via /skill <name>.\n"
    ),
    "prompts/README.md": "# Prompts\n\nEach `<name>.md` can be sent with /prompt <name> [args]; `{{args}}` is replaced.\n",
}
INIT_NOTEBOOK_TITLE = "Hailer workspace"


def _init_notebook_and_data(console: Console, workspace: Path, config_path: Path, *, verbose: bool) -> HailerConfig | None:
    """Create the configured notebook (starter template) and the notebooks and data folders when missing.

    Never overwrites a file. Returns the loaded configuration, or ``None`` when hailer.toml does not load.
    """
    from hailer.config import load_config

    try:
        config = load_config(workspace=workspace, config_path=config_path)
    except HailerError as err:
        _print_error(console, err, verbose=verbose)
        console.print("Skipped the notebook and data folder; fix hailer.toml and run uvx hailer init again.", markup=False)
        return None
    shown = notebooks.notebook_display_name(config, config.notebook)
    created: list[str] = []
    try:
        if notebooks.ensure_notebook(config.notebook, title=INIT_NOTEBOOK_TITLE):
            created.append(f"{shown} (starter notebook)")
        for folder in (config.notebooks_root, config.data_dir):
            if not folder.is_dir():
                folder.mkdir(parents=True)
                created.append(notebooks.notebook_display_name(config, folder) + "/")
    except OSError as err:
        console.print(f"Could not create the notebook or data folder: {err}", style="red", markup=False)
        return config
    console.print("Created " + ", ".join(created) if created else f"{shown} already exists.", markup=False)
    return config


@app.command()
def init(
    ctx: typer.Context,
    force: bool = typer.Option(False, "--force", help="Overwrite an existing hailer.toml."),
    kernel: str | None = typer.Option(
        None,
        "--kernel",
        help='Write \\[kernel] runtime = "local" or "docker" into hailer.toml (docker: notebook code runs in an isolated container).',
        show_default=False,
    ),
) -> None:
    """Set up a workspace: hailer.toml, .config/hailer, the starter notebook and the data folder.

    Only hailer.toml is ever overwritten (with --force); anything else that exists is kept, so running
    it again fills in whatever is missing.
    """
    opts = _opts(ctx)
    console = console_factory()
    choice = _kernel_choice(console, kernel)
    workspace = opts.workspace or Path.cwd()
    workspace = workspace.resolve()
    config_path = workspace / "hailer.toml"
    if config_path.exists() and not force:
        console.print(f"{config_path} already exists (use --force to overwrite).", markup=False)
    else:
        from hailer.config import write_default_config

        write_default_config(config_path, overwrite=force, kernel=choice)
        console.print(f"Wrote {config_path}" + (f' ([kernel] runtime = "{choice}")' if choice else ""), markup=False)

    target = workspace / ".config" / "hailer"
    example = _example_config_dir()
    created: list[str] = []
    for sub in ("context", "skills", "prompts"):
        (target / sub).mkdir(parents=True, exist_ok=True)
    if example is not None and example != target:
        for src in example.rglob("*"):
            if src.is_dir():
                continue
            rel = src.relative_to(example)
            dst = target / rel
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            created.append(str(rel))
    else:
        for rel, text in _PLACEHOLDERS.items():
            dst = target / rel
            if not dst.exists():
                dst.write_text(text, encoding="utf-8")
                created.append(rel)
    if created:
        console.print(f"Created {target} with: " + ", ".join(created), markup=False)
    else:
        console.print(f"{target} already set up.", markup=False)

    config = _init_notebook_and_data(console, workspace, config_path, verbose=opts.verbose)
    runtime = config.kernel.runtime if config is not None else (choice or "local")
    kept = choice is not None and config is not None and runtime != choice
    if kept:
        console.print(
            f'hailer.toml was kept, so [kernel] runtime is still "{runtime}". To change it, set '
            f'runtime = "{choice}" under [kernel] in hailer.toml (uncomment the [kernel] line too), or run '
            f"uvx hailer init --kernel {choice} --force to rewrite the file.",
            markup=False,
        )
    provider = config.model.provider if config is not None else "<provider>"
    data = notebooks.notebook_display_name(config, config.data_dir) if config is not None else "data"
    console.print(
        "\nNext steps:\n"
        "  1. Edit hailer.toml (model, provider, notebook).\n"
        f"  2. Store the API key once:   uvx hailer login {provider}\n"
        f"  3. Put the files you want to analyse in {data}/ (CSV, Parquet or JSON, any name).\n"
        "  4. Start marimo and chat:     uvx hailer notebook\n"
        '     then ask, for example: "what is in my data?" or "chart revenue by month"',
        markup=False,
    )
    for line in _kernel_next_steps(runtime, kept=kept):
        console.print(line, markup=False)


def _kernel_next_steps(runtime: str, *, kept: bool = False) -> list[str]:
    """What ``init`` adds to the next steps about where notebook code runs (nothing more after the
    "hailer.toml was kept" advice, which already says how to switch)."""
    docker_found = _docker_on_path()
    if runtime == KERNEL_RUNTIME_DOCKER:
        lines = [
            "\nNotebook code runs in a Docker container (no network; the data folder is read-only).",
            "The first uvx hailer notebook downloads the kernel image (uvx hailer kernel pull does it now); where "
            "it cannot be downloaded, uvx hailer kernel build builds it on this machine.",
        ]
        if not docker_found:
            lines.append("Docker was not found on this machine: install Docker Desktop (or Docker Engine) first.")
        return lines
    if docker_found and not kept:
        return [
            "\nOptional: to isolate notebook code in a container (Docker was found on this machine), uncomment the "
            '[kernel] and runtime lines in hailer.toml and set runtime = "docker", or try it once with '
            "uvx hailer notebook --kernel docker."
        ]
    return []


def main() -> None:
    _reconfigure_streams()
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
