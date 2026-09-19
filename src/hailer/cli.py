"""Hailer command line: the conversational control plane.

``uvx hailer``            chat with the agent: reuse this workspace's marimo or start one, open the
                          notebook, chat, stop the marimo it started on exit
``uvx hailer notebook``   the same, with options (--port, --no-browser, --keep-marimo, --foreground)
``uvx hailer exec``       run code in the live kernel (debugging / scripting)
``uvx hailer status``     show configuration, credentials source and marimo state
``uvx hailer doctor``     run the preflight checks and print a table
``uvx hailer login``      store an API key in the OS credential store
``uvx hailer logout``     remove it
``uvx hailer init``       set up a workspace: hailer.toml, .config/hailer, the starter notebook, data/

Collaborators (config, marimo client, agent, secrets, context) are reached through
small module-level factory functions so tests can replace them.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from hailer import __version__, notebooks
from hailer.chat import ChatController
from hailer.errors import (
    ConfigError,
    CredentialsError,
    HailerError,
    MarimoUnavailableError,
    NoSessionError,
)
from hailer.models import (
    WIRE_API_CHAT,
    AgentEvent,
    Check,
    ContextBundle,
    HailerConfig,
    MarimoServer,
    MarimoSession,
    ProviderConfig,
    TurnSummary,
)
from hailer.session import (
    COMMANDS,
)
from hailer.statedir import ensure_state_dir

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    invoke_without_command=True,
    help="Hailer: chat with your data. Explore, analyse and chart local data files in a live marimo notebook.",
)

PROMPT = "You > "
NO_WORKSPACE_SERVER = "No marimo server is running for this workspace."
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
    plain: bool = False


def console_factory() -> Console:
    return Console(highlight=False, soft_wrap=True)


def _chat_console(opts: CliOptions) -> Console:
    console = console_factory()
    if opts.plain or os.environ.get("TERM", "").lower() in {"dumb", "unknown"}:
        # Rich live status also emits cursor controls on a TTY. Plain mode must
        # disable those throughout startup, notebook waits and model turns.
        return Console(file=console.file, width=console.width, force_terminal=False,
                       color_system=None, highlight=False, soft_wrap=True)
    return console


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
    from hailer.marimo_client import MarimoClient

    return MarimoClient(
        server.url,
        config.marimo_token,
        notebook=config.notebook,
        workspace=config.workspace,
    )


def _notebook_url(server: MarimoServer, config: HailerConfig) -> str:
    from hailer.marimo_client import open_notebook_url

    return open_notebook_url(server, config.notebook, config.workspace)


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


def _make_agent(config: HailerConfig, bundle: ContextBundle) -> Any:
    from hailer.agent import HailerAgent

    return HailerAgent(config, bundle)


def _open_browser(url: str) -> None:
    from hailer.browser import open_url  # os.startfile first on Windows (ignores BROWSER), never raises

    open_url(url)


def _marimo_server_command(config: HailerConfig, port: int, *, headless: bool = True) -> list[str]:
    from hailer.marimo_client import marimo_server_command

    return marimo_server_command(config.notebooks_root, config.workspace, port, headless=headless)


def _find_free_port(preferred: int) -> int:
    from hailer.marimo_client import find_free_port

    return find_free_port(preferred)


def _wait_for_health(url: str, timeout: float, should_stop: Callable[[], bool] | None = None) -> bool:
    from hailer.marimo_client import wait_for_health

    return wait_for_health(url, timeout, should_stop=should_stop)


def _wait_for_session(
    client: Any, notebook: Path, timeout: float, *, should_stop: Callable[[], bool] | None = None
) -> MarimoSession | None:
    from hailer.marimo_client import wait_for_session

    return wait_for_session(client, notebook, timeout, should_stop=should_stop)


def _registry_remove(url: str) -> bool:
    from hailer.marimo_client import remove_registry_entry

    return remove_registry_entry(url)


def _spawn_marimo(cmd: list[str], cwd: Path, log_path: Path) -> subprocess.Popen:
    """Start marimo as a background child of this process, logging to ``log_path`` (its folder must exist).

    No new console window: the chat runs in this terminal and marimo is stopped when it
    ends. On Windows the child gets its own process group so Ctrl+C in the chat is not
    delivered to marimo.
    """
    log = open(log_path, "ab")  # noqa: SIM115 - handed to the child; closed with it
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    try:
        return subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )
    finally:
        log.close()


def _run_foreground(cmd: list[str], cwd: Path) -> int:
    """Run marimo attached to this terminal (no chat); returns its exit code."""
    return subprocess.run(cmd, cwd=str(cwd), check=False).returncode


def _kill_tree(pid: int) -> None:
    """Windows: kill a process and everything it spawned (marimo runs kernels as children)."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def _stop_process(proc: Any, timeout: float = 5.0) -> None:
    """terminate → wait up to ``timeout`` → kill; never raises."""
    try:
        if proc.poll() is not None:
            return
        if os.name == "nt":
            _kill_tree(proc.pid)
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except Exception:  # noqa: BLE001 - subprocess.TimeoutExpired or a fake
            proc.kill()
            try:
                proc.wait(timeout=timeout)
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001 - best effort on the way out
        pass


def _log_tail(path: Path, lines: int = 15) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return text.splitlines()[-lines:]


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
            f"Web access: {_web_line(config)}",
        ]
    )
    console.print(Panel(Text(body), title="Hailer", expand=False))
    console.print("Type /help for commands.\n", markup=False)


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #


def _marimo_state(config: HailerConfig) -> tuple[MarimoServer | None, MarimoSession | None, HailerError | None]:
    """(server, session, error) without raising."""
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


def _launch_hint() -> str:
    from hailer.marimo_client import launch_hint

    return launch_hint()


def preflight(config: HailerConfig) -> list[Check]:
    """What ``hailer doctor`` checks: :func:`local_checks` plus the marimo server and session."""
    checks = local_checks(config)
    checks.extend(marimo_checks(config))
    return checks


def marimo_checks(config: HailerConfig) -> list[Check]:
    """The marimo server and kernel-session checks (the part ``hailer`` and ``hailer notebook`` handle themselves).

    With several servers running, the one serving this workspace is checked. None is a warning, not a
    failure (starting one is the chat's job); a pinned URL that does not answer is a failure.
    """
    checks: list[Check] = []
    server, session, err = _marimo_state(config)
    if server is None and config.marimo_url:
        checks.append(Check("marimo", False, f"not running at {config.marimo_url}", hint=PINNED_URL_HINT))
    elif server is None:
        # Not a failure: `hailer` and `hailer notebook` start a server for this workspace.
        summary = NO_WORKSPACE_SERVER if err is None else str(err)
        hint = err.hint if err is not None and err.hint else _launch_hint()
        checks.append(Check("marimo", False, summary, hint=hint, fatal=False))
    else:
        checks.append(Check("marimo", True, server.url, fatal=False))
        if session is not None:
            checks.append(Check("session", True, f"notebook is open (session {session.session_id})", fatal=False))
        else:
            url = _notebook_url(server, config)
            hint = err.hint if err is not None and err.hint else f"Open {url} in your browser."
            checks.append(Check("session", False, "the notebook is not open in a browser", hint=hint, fatal=False))
    return checks


def local_checks(config: HailerConfig) -> list[Check]:
    """Config, notebook and credential checks (everything except the marimo server)."""
    checks: list[Check] = []

    problems = _validate_config(config)
    fatal = [p for p in problems if not p.lower().startswith("warning")]
    warnings = [p for p in problems if p.lower().startswith("warning")]
    if fatal:
        checks.append(Check("config", False, "; ".join(fatal), hint="Fix hailer.toml (uvx hailer init writes a starter file and creates a missing notebook).", fatal=True))
    else:
        summary = "ok" if not warnings else "; ".join(warnings)
        checks.append(Check("config", True, summary, fatal=False))

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


CTRL_Z = "\x1a"  # Ctrl+Z then Enter: end of input on Windows, as in Python's own REPL


def _stdio_is_terminal() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):  # replaced or closed streams
        return False


class _LineReader:
    """Reads one message at the ``You > `` prompt.

    In a terminal it is a prompt_toolkit prompt. While it waits, bracketed paste is on
    (``ESC[?2004h``), so a pasted block arrives as one message with its newlines, and it is off
    again (``ESC[?2004l``) before the prompt returns, so before the turn runs. Up/Down recall this
    session's messages (in memory only). No mouse capture and no alternate screen, so the
    terminal's own scrollback and selection keep working. Without a terminal (a pipe, the tests)
    it is Rich's plain ``Console.input``, as before.

    Ctrl+C at the prompt raises ``KeyboardInterrupt``; Ctrl+D on an empty line, or Ctrl+Z then
    Enter, raises ``EOFError``. ``pt_input`` / ``pt_output`` replace the terminal (tests).
    """

    def __init__(self, console: Console, *, interactive: bool | None = None, pt_input: Any = None, pt_output: Any = None) -> None:
        self.console = console
        self.interactive = _stdio_is_terminal() if interactive is None else interactive
        self._pt_input = pt_input
        self._pt_output = pt_output
        self._session: Any = None

    def read(self) -> str:
        if self.interactive:
            line = self._prompt_session().prompt()
        else:
            line = self.console.input(f"[bold cyan]{PROMPT}[/bold cyan]")
        if line.lstrip().startswith(CTRL_Z):
            raise EOFError
        return line

    def _prompt_session(self) -> Any:
        if self._session is None:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.formatted_text import FormattedText
            from prompt_toolkit.history import InMemoryHistory
            from prompt_toolkit.output import create_output
            from prompt_toolkit.output.vt100 import Vt100_Output

            output = self._pt_output or create_output()
            if isinstance(output, Vt100_Output):
                # No completion menu needs the cursor row, and a terminal that never answers the
                # ESC[6n request would get a "does not support CPR" warning printed into it.
                output.enable_cpr = False
            self._session = PromptSession(
                FormattedText([("bold ansicyan", PROMPT)]),
                history=InMemoryHistory(),
                mouse_support=False,
                input=self._pt_input,
                output=output,
            )
        return self._session


def _make_line_reader(console: Console) -> _LineReader:
    interactive = _stdio_is_terminal() and os.environ.get("TERM", "").lower() not in {"dumb", "unknown"}
    return _LineReader(console, interactive=interactive)


def _use_composer(opts: CliOptions) -> bool:
    """Pipes and minimal terminal emulators retain the linear interface."""
    return not opts.plain and _stdio_is_terminal() and os.environ.get("TERM", "").lower() not in {"dumb", "unknown"}


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


class ChatLoop(ChatController):
    """Bind the shared controller to the CLI's injectable collaborators."""

    def __init__(self, console: Console, config: HailerConfig, opts: CliOptions) -> None:
        super().__init__(console, config, opts, services=sys.modules[__name__])


def run_chat(opts: CliOptions) -> None:
    """Bare ``hailer``: the same session as ``hailer notebook`` with its default options."""
    console = _chat_console(opts)
    config = _config_or_exit(console, opts)
    _run_session(console, config, opts, port=DEFAULT_MARIMO_PORT, open_browser=True, keep_marimo=False)


def _run_chat_loop(console: Console, config: HailerConfig, opts: CliOptions) -> None:
    """Start the agent and run the REPL; exit code 1 when the agent cannot start."""
    loop = ChatLoop(console, config, opts)
    if _use_composer(opts):
        try:
            asyncio.run(loop.run_interactive())
        except HailerError as err:
            _print_error(console, err, verbose=opts.verbose)
            raise typer.Exit(code=1)
        except KeyboardInterrupt:
            console.print("Bye.")
        except Exception as exc:  # startup/terminal failures still restore the terminal
            _print_unexpected(console, exc, verbose=opts.verbose)
            raise typer.Exit(code=1)
        return
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
    plain: bool = typer.Option(False, "--plain", help="Use line-oriented chat without the persistent composer."),
    version: bool = typer.Option(False, "--version", callback=_version_callback, is_eager=True, help="Show the version and exit."),
) -> None:
    del version
    ctx.obj = CliOptions(verbose=verbose, config_path=config, workspace=workspace, new_thread=new, plain=plain)
    if ctx.invoked_subcommand is None:
        run_chat(ctx.obj)


DEFAULT_MARIMO_PORT = 2718
MARIMO_LOG_NAME = "marimo.log"
HEALTH_TIMEOUT_SEC = 60.0
SESSION_TIMEOUT_SEC = 90.0
SWITCH_SESSION_TIMEOUT_SEC = 30.0  # how long /notebook new|open waits for the browser tab
NOTEBOOK_USAGE = "Usage: /notebook [list | new <name> [--empty] | open <name> | close [name]]"
PINNED_URL_HINT = (
    "[hailer].marimo_url (or HAILER_MARIMO_URL) pins Hailer to that server, so Hailer does not start "
    "one in its place. Start marimo there, or remove the setting and Hailer starts its own for this workspace."
)
APP_VIEW_HINT = (
    "The notebook opens in app view (results only). "
    "Press Ctrl+. in it, or use Toggle app view, to see and edit the code."
)


def _serves_workspace(client: Any, config: HailerConfig) -> bool:
    """True when the server was started on the notebooks folder or hosts a notebook inside it."""
    from hailer.marimo_client import AFFINITY_NONE, workspace_affinity

    return workspace_affinity(client, config) > AFFINITY_NONE


def _reusable_server(config: HailerConfig) -> MarimoServer | None:
    """The running marimo server this workspace may attach to; ``None`` means start one.

    A pinned URL (``[hailer].marimo_url`` / ``HAILER_MARIMO_URL``) is returned whether or not it
    answers: the caller reports a pinned server that is down and never starts one in its place.
    Otherwise it is the server found for this workspace, if it answers and serves the notebooks
    folder (one server hosts every notebook there). A server for another folder, such as another
    git worktree of the same repository, is never attached to.
    """
    try:
        server = _find_server(config)
    except HailerError:
        return None
    if server is None or config.marimo_url:
        return server
    client = _make_client(server, config)
    if not client.health() or not _serves_workspace(client, config):
        return None
    return server


def _start_marimo(console: Console, config: HailerConfig, port: int) -> tuple[MarimoServer, Any, Path]:
    """Start marimo in the background and wait until ``/health`` answers."""
    chosen = _find_free_port(port)
    if chosen != port:
        console.print(f"Port {port} is busy; using {chosen}.", markup=False)
    url = f"http://127.0.0.1:{chosen}"
    cmd = _marimo_server_command(config, chosen)
    try:
        log_path = ensure_state_dir(config.workspace) / MARIMO_LOG_NAME
        proc = _spawn_marimo(cmd, config.workspace, log_path)
    except OSError as err:
        console.print(f"Could not start marimo: {err}", style="red", markup=False)
        raise typer.Exit(code=1)
    with console.status(f"Starting marimo on {url} ..."):
        healthy = _wait_for_health(url, HEALTH_TIMEOUT_SEC, lambda: proc.poll() is not None)
    if not healthy:
        if proc.poll() is not None:
            console.print(f"Marimo exited early (code {proc.returncode}).", style="red", markup=False)
        else:
            console.print(f"Marimo did not answer on {url} within {int(HEALTH_TIMEOUT_SEC)} s.", style="red", markup=False)
            _stop_process(proc)
        console.print(f"Log: {log_path}", markup=False)
        for line in _log_tail(log_path):
            console.print(f"    {line}", markup=False)
        raise typer.Exit(code=1)
    console.print(f"Marimo is running at {url}  (log: {log_path})", markup=False)
    return MarimoServer(url=url, server_id=f"127.0.0.1:{chosen}", pid=getattr(proc, "pid", None), source="config"), proc, log_path


def _wait_for_notebook(console: Console, config: HailerConfig, server: MarimoServer, *, open_browser: bool) -> None:
    client = _make_client(server, config)
    url = _notebook_url(server, config)
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


def _run_session(console: Console, config: HailerConfig, opts: CliOptions, *, port: int, open_browser: bool, keep_marimo: bool) -> None:
    """Attach to this workspace's marimo or start one, open the notebook, chat, then stop what was started.

    Bare ``hailer`` and ``hailer notebook`` both run this. A pinned URL is used as is; when it does
    not answer the session ends with an error instead of starting a server in its place.
    """
    checks = local_checks(config)
    _print_checks(console, checks, only_failures=True)
    if any(not c.ok and c.fatal for c in checks):
        raise typer.Exit(code=1)
    config.notebooks_root.mkdir(parents=True, exist_ok=True)  # marimo edit refuses a missing folder

    proc: Any = None
    log_path: Path | None = None
    server = _reusable_server(config)
    if config.marimo_url and (server is None or not _make_client(server, config).health()):
        console.print(f"Marimo is not running at {config.marimo_url}.", style="red", markup=False)
        console.print(f"    {PINNED_URL_HINT}", markup=False)
        raise typer.Exit(code=1)
    if server is not None:
        console.print(f"Using the running marimo at {server.url}.", markup=False)
    else:
        server, proc, log_path = _start_marimo(console, config, port)
    # Pin the chat to this server so registry discovery cannot pick another one.
    config = replace(config, marimo_url=server.url)

    try:
        _wait_for_notebook(console, config, server, open_browser=open_browser)
        console.print()
        _startup_panel(console, config)
        _run_chat_loop(console, config, opts)
    finally:
        if proc is not None:
            if keep_marimo:
                console.print(
                    f"Marimo is still running at {server.url} (stop it by closing its process; log: {log_path}).",
                    markup=False,
                )
            else:
                _stop_process(proc)
                _registry_remove(server.url)
                console.print("Stopped marimo.", markup=False)


@app.command()
def notebook(
    ctx: typer.Context,
    port: int = typer.Option(DEFAULT_MARIMO_PORT, "--port", "-p", help="Port for the marimo server (a free one is chosen if busy)."),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not open the notebook in a browser."),
    keep_marimo: bool = typer.Option(False, "--keep-marimo", help="Leave marimo running when the chat ends."),
    foreground: bool = typer.Option(False, "--foreground", help="Run marimo attached to this terminal, without the chat."),
    new: bool = typer.Option(False, "--new", help="Start a new conversation instead of resuming."),
    plain: bool = typer.Option(False, "--plain", help="Use line-oriented chat without the persistent composer."),
) -> None:
    """Start marimo (or reuse this workspace's), open the notebook, and chat here; stop marimo on exit."""
    opts = _opts(ctx)
    if new:
        opts.new_thread = True
    if plain:
        opts.plain = True
    console = _chat_console(opts)
    config = _config_or_exit(console, opts)

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
        # Hailer's own interpreter, like the background server: `uv run marimo` would need a project .venv.
        cmd = _marimo_server_command(config, port, headless=no_browser)
        console.print("Launching: " + " ".join(cmd), markup=False)
        raise typer.Exit(code=_run_foreground(cmd, config.workspace))

    _run_session(console, config, opts, port=port, open_browser=not no_browser, keep_marimo=keep_marimo)


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
            raise MarimoUnavailableError(NO_WORKSPACE_SERVER, hint=_launch_hint())
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
    server, session, err = _marimo_state(config)
    if server is None and config.marimo_url:
        console.print(f"Marimo:      not running at {config.marimo_url} (pinned by marimo_url)", markup=False)
    elif server is None:
        console.print("Marimo:      none for this workspace (uvx hailer starts one)", markup=False)
    elif session is None:
        console.print(f"Marimo:      {server.url} (notebook not open in a browser)", markup=False)
        if err is not None and err.hint:
            console.print(f"             {err.hint}", markup=False)
    else:
        console.print(f"Marimo:      {server.url} (session {session.session_id})", markup=False)
    try:
        bundle = _load_context(config)
    except HailerError as err:
        _print_error(console, err, verbose=opts.verbose)
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
    checks = preflight(config)
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
        except HailerError as err:
            console.print(f"warn  code mode: {err}", markup=False)
    if any(not c.ok and c.fatal for c in checks):
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
) -> None:
    """Set up a workspace: hailer.toml, .config/hailer, the starter notebook and the data folder.

    Only hailer.toml is ever overwritten (with --force); anything else that exists is kept, so running
    it again fills in whatever is missing.
    """
    opts = _opts(ctx)
    console = console_factory()
    workspace = opts.workspace or Path.cwd()
    workspace = workspace.resolve()
    config_path = workspace / "hailer.toml"
    if config_path.exists() and not force:
        console.print(f"{config_path} already exists (use --force to overwrite).", markup=False)
    else:
        from hailer.config import write_default_config

        write_default_config(config_path, overwrite=force)
        console.print(f"Wrote {config_path}", markup=False)

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


def main() -> None:
    _reconfigure_streams()
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
