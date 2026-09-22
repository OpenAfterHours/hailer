"""Hailer command line: the conversational control plane.

``uvx hailer``            chat with the agent: start this session's own marimo kernel, open the
                          notebook, chat, stop the kernel on exit
``uvx hailer notebook``   the same, with options (--port, --no-browser, --foreground, --kernel)
``uvx hailer status``     show configuration, credentials source and this workspace's running kernels
``uvx hailer doctor``     check the configuration, credentials, Docker, the kernel image and folders
``uvx hailer login``      store an API key in the OS credential store
``uvx hailer logout``     remove it
``uvx hailer init``       set up a workspace: hailer.toml, .config/hailer, the starter notebook, data/
``uvx hailer kernel``     the Docker kernel: ``pull`` or ``build`` its image, ``stop`` this workspace's kernels

Collaborators (config, marimo client, agent, secrets, context) are reached through
small module-level factory functions so tests can replace them.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import threading
import traceback
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from hailer import __version__, notebooks
from hailer.chat import ChatController
from hailer.errors import ConfigError, CredentialsError, HailerError, NoSessionError
from hailer.models import (
    KERNEL_RUNTIME_DOCKER,
    VALID_KERNEL_RUNTIMES,
    WIRE_API_CHAT,
    AgentEvent,
    Check,
    ContextBundle,
    HailerConfig,
    MarimoSession,
    ProviderConfig,
    TurnSummary,
)
from hailer.startup import StartupTimings

if TYPE_CHECKING:  # pragma: no cover - annotations only; the runtimes import it when a kernel starts
    from hailer.sandbox import MarimoSandbox

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    invoke_without_command=True,
    help="Hailer: chat with your data. Explore, analyse and chart local data files in a live marimo notebook.",
)

DEFAULT_MARIMO_PORT = 2718
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
    """The resolved configuration. ``config.notebook`` stays the *configured* notebook; the active
    one is a name in the state file (:func:`hailer.notebooks.load_active_notebook`).
    ``HAILER_NOTEBOOK`` is an explicit choice: it is written to the state file so the agent's tools
    (which read only the file) and later sessions agree with this one.
    """
    from hailer.config import load_config

    config = load_config(workspace=opts.workspace, config_path=opts.config_path)
    if os.environ.get("HAILER_NOTEBOOK"):
        name = notebooks.notebook_name(config, config.notebook)
        if name is not None:
            try:
                notebooks.save_active_notebook(config, name)
            except OSError:  # pragma: no cover - a read-only workspace must not stop the chat
                pass
    return config


def _validate_config(config: HailerConfig) -> list[str]:
    from hailer.config import validate

    return validate(config)


def _setup_logging(config: HailerConfig, opts: CliOptions) -> None:
    from hailer.log import setup_logging

    setup_logging(config.log_level, verbose=opts.verbose)


def _active_notebook(config: HailerConfig, sandbox: MarimoSandbox) -> str:
    """The active notebook for a session that just started ``sandbox``: the state file's, else the
    configured one when the sandbox does not have it (it was deleted or renamed; one existence
    check, not a listing). Saved, so an old state file is converted and the agent's tools read the
    same name."""
    active = notebooks.load_active_notebook(config)
    try:
        exists = sandbox.has_notebook(active)
    except HailerError:
        exists = True  # marimo does not answer: keep the choice, the chat reports the kernel
    if not exists:
        active = notebooks.default_notebook(config)
    try:
        notebooks.save_active_notebook(config, active)
    except (OSError, HailerError):  # pragma: no cover - a read-only workspace must not stop the chat
        pass
    return active


def _runtime_for(config: HailerConfig) -> Any:
    """The kernel runtime ``[kernel] runtime`` asks for (``ConfigError`` for an unknown one). Nothing
    is checked until its ``check``, ``prepare`` or ``start`` runs."""
    from hailer.kernel import runtime_for

    return runtime_for(config)


def _docker_runner() -> Any:
    """What ``hailer kernel ...`` runs ``docker`` with (the CLI on PATH)."""
    from hailer.kernel_docker import SubprocessDockerRunner

    return SubprocessDockerRunner()


def _workspace_kernels(config: HailerConfig) -> list[str]:
    """This workspace's running kernels, one line each (``hailer status``)."""
    from hailer.kernel_docker import workspace_kernels

    return workspace_kernels(config, runner=_docker_runner())


def _docker_on_path() -> bool:
    return shutil.which("docker") is not None


def _kernel_line(config: HailerConfig) -> str:
    from hailer.kernel import describe_runtime

    return describe_runtime(config.kernel)


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


def _make_agent(config: HailerConfig, bundle: ContextBundle, sandbox: MarimoSandbox | None = None) -> Any:
    """The agent; its tools use ``sandbox``, the kernel this process started."""
    from hailer.agent import HailerAgent

    return HailerAgent(config, bundle, sandbox=sandbox)


def _start_dependency_warmup() -> None:
    """Start dependency imports without creating provider, HTTP or database resources."""
    from hailer.agent import start_dependency_warmup

    start_dependency_warmup()


def _open_browser(url: str) -> None:
    from hailer.browser import open_url  # os.startfile first on Windows (ignores BROWSER), never raises

    open_url(url)


def _find_free_port(preferred: int) -> int:
    from hailer.marimo_client import find_free_port

    return find_free_port(preferred)


def _wait_for_session(
    client: Any, notebook: str, timeout: float, *, should_stop: Callable[[], bool] | None = None
) -> MarimoSession | None:
    from hailer.marimo_client import wait_for_session

    return wait_for_session(client, notebook, timeout, should_stop=should_stop)


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


def _session_for(sessions: list[MarimoSession], notebook: str) -> MarimoSession | None:
    """The kernel session for the notebook ``notebook`` (a name) among ``sessions``: exact name match
    only (no file-name fallback, so two notebooks called ``report.py`` in different folders never
    share a session)."""
    from hailer.marimo_client import match_session

    return match_session(sessions, notebook)


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" + ("" if count == 1 else "s")


def _startup_panel(console: Console, config: HailerConfig, notebook: str | None = None) -> None:
    """The panel with the settings; ``notebook``: the active notebook (default: the state file's)."""
    body = "\n".join(
        [
            f"Model:      {config.model.name}",
            f"Provider:   {_provider_line(config)}",
            f"Notebook:   {notebook or notebooks.load_active_notebook(config)}",
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


def _marimo_state(sandbox: MarimoSandbox | None, notebook: str) -> tuple[MarimoSandbox | None, MarimoSession | None, HailerError | None]:
    """(sandbox, session of ``notebook``, error) for the kernel this chat started, without raising.
    ``sandbox`` is ``None`` (and so is the result) when there is none or it does not answer."""
    if sandbox is None:
        return None, None, None
    try:
        client = sandbox.client(notebook=notebook, token_in_links=True)
        if not client.health():
            return None, None, None
        session = client.resolve_session(notebook)
        return sandbox, session, None
    except NoSessionError as err:
        return sandbox, None, err
    except HailerError as err:
        return sandbox, None, err


def kernel_checks(config: HailerConfig, *, starting: bool = False) -> list[Check]:
    """The kernel runtime's rows (``hailer doctor``, ``hailer notebook``): Docker, the image and the
    folders for the docker runtime; no kernel is started or probed. A runtime that cannot be
    used is one failed ``kernel`` row. ``starting``: a start follows, which downloads a missing
    image itself, so that warning is left out."""
    try:
        checks = list(_runtime_for(config).check())
    except HailerError as err:
        return [Check("kernel", False, str(err), hint=err.hint)]
    if starting:
        checks = [c for c in checks if not (c.name == "image" and not c.ok and not c.fatal)]
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
    """Config, notebook and credential checks (everything except the kernel runtime)."""
    checks: list[Check] = [_config_check(_validate_config(config))]

    # Workspace setup is host-side: the active notebook (else the configured one) must be in the folder.
    active = config.notebooks_root / notebooks.load_active_notebook(config)
    if active.is_file():
        checks.append(Check("notebook", True, f"{_relative(active, config.workspace)} (active)"))
    elif config.notebook.exists():
        checks.append(Check("notebook", True, f"{_relative(config.notebook, config.workspace)} (configured)"))
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

    def __init__(self, console: Console, config: HailerConfig, opts: CliOptions, sandbox: MarimoSandbox | None = None) -> None:
        super().__init__(console, config, opts, services=sys.modules[__name__], sandbox=sandbox)


def run_chat(opts: CliOptions) -> None:
    """Bare ``hailer``: the same session as ``hailer notebook`` with its default options."""
    console = _chat_console(opts)
    config = _config_or_exit(console, opts)
    _run_session(console, config, opts, port=DEFAULT_MARIMO_PORT, open_browser=True)


def _run_chat_loop(
    console: Console,
    config: HailerConfig,
    opts: CliOptions,
    sandbox: MarimoSandbox | None = None,
    *,
    prepare_notebook: Callable[[Console], Awaitable[None]] | None = None,
    timings: StartupTimings | None = None,
) -> None:
    """Start the agent and run the REPL; exit code 1 when the agent cannot start."""
    loop = ChatLoop(console, config, opts, sandbox)
    if _use_composer(opts):
        async def prepare() -> None:
            if prepare_notebook is not None:
                await prepare_notebook(loop.console)

        try:
            asyncio.run(loop.run_interactive(prepare_notebook=prepare, timings=timings))
        except HailerError as err:
            _print_error(console, err, verbose=opts.verbose)
            raise typer.Exit(code=1)
        except KeyboardInterrupt:
            console.print("Bye.")
        except Exception as exc:  # startup/terminal failures still restore the terminal
            _print_unexpected(console, exc, verbose=opts.verbose)
            raise typer.Exit(code=1)
        if loop.startup_failed:
            # The composer keeps the draft available until the user exits, but
            # a preparation error remains a failed CLI invocation.
            raise typer.Exit(code=1)
        return
    try:
        loop.start()
        if timings is not None:
            timings.mark("agent_ready")
            timings.mark("input_ready")
            timings.mark("ready_to_answer")
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


SESSION_TIMEOUT_SEC = 90.0
SWITCH_SESSION_TIMEOUT_SEC = 30.0  # how long /notebook new|open waits for the browser tab
NOTEBOOK_USAGE = "Usage: /notebook [list | new <name> [--empty] | open <name> | close [name]]"
APP_VIEW_HINT = (
    "The notebook opens in app view (results only). "
    "Press Ctrl+. in it, or use Toggle app view, to see and edit the code."
)


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
    """Start marimo through ``runtime`` and wait until it answers with its token; exit 1 when it
    does not. Ctrl+C after the runtime started it still stops it (the caller owns it only once
    this returns)."""
    running: Any = None
    try:
        _prepare_runtime(console, runtime)
        chosen, url = _free_port_url(console, port)
        with console.status(f"Starting marimo{_where(runtime)} on {url} ..."):
            running = runtime.start(chosen)
        console.print(f"Marimo is running at {running.server.url}  (log: {running.log_hint})", markup=False)
    except HailerError as err:  # the runtime has cleaned up; the hint carries the log tail
        _print_error(console, err, verbose=verbose)
        raise typer.Exit(code=1)
    except BaseException:
        if running is not None:
            running.stop()
        raise
    return running


def _run_foreground(console: Console, config: HailerConfig, runtime: Any, port: int, *, open_browser: bool, verbose: bool) -> int:
    """``hailer notebook --foreground``: marimo in this terminal, no chat, until it exits or Ctrl+C.

    Only the browser uses it (nothing else attaches to a kernel it did not start). In docker mode
    the kernel's log is followed here, and Ctrl+C stops and removes the containers.
    """
    try:
        _prepare_runtime(console, runtime)
    except HailerError as err:
        _print_error(console, err, verbose=verbose)
        return 1
    chosen, url = _free_port_url(console, port)
    console.print(f"Starting marimo{_where(runtime)} on {url} (Ctrl+C stops it) ...", markup=False)
    running: Any = None
    try:
        try:
            running = runtime.start(chosen, foreground=True)
        except HailerError as err:
            _print_error(console, err, verbose=verbose)
            return 1
        console.print(f"Kernel:     {runtime.describe()}", markup=False)
        console.print(f"Marimo is running at {running.server.url}. Ctrl+C stops it.", markup=False)
        home = running.home_url(with_token=True)
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
        if running is not None:
            running.stop()
            _warn_about_planted_files(console, running)
            if getattr(running, "stop_error", ""):
                console.print(running.stop_error, style="red", markup=False)
                raise typer.Exit(code=1)


def _warn_about_planted_files(console: Console, running: Any) -> None:
    """After a docker kernel stopped: files at the top of its notebooks folder that git or an
    editor would run code from (the kernel could write them there). Loud, and never removed."""
    planted = getattr(running, "planted", None)
    names = planted() if callable(planted) else []
    if names:
        from hailer.kernel_docker import planted_warning

        console.print(planted_warning(running.notebooks_folder, names), style="bold red", markup=False)


def _wait_for_notebook(
    console: Console,
    sandbox: MarimoSandbox,
    notebook: str,
    *,
    open_browser: bool,
    should_stop: Callable[[], bool] | None = None,
    show_status: bool = True,
) -> None:
    """Open ``notebook`` and poll its session, stopping between blocking operations.

    The live composer owns its activity display, so its background preparation
    suppresses the Rich spinner. Cancellation never leaves a later browser open
    or session message racing the next session or kernel shutdown.
    """
    stopped = should_stop or (lambda: False)
    if stopped():
        return
    client = sandbox.client(notebook=notebook, token_in_links=True)
    url = sandbox.notebook_url(notebook, with_token=True)
    if stopped():
        return
    try:
        session = client.resolve_session(notebook)
    except HailerError:
        session = None
    if stopped():
        return
    if session is not None:
        console.print(f"Notebook is open (session {session.session_id}).", markup=False)
        return
    if open_browser:
        console.print(f"Opening {url} in your browser...", markup=False)
        _open_browser(url)
    else:
        console.print(f"Open {url} in your browser.", markup=False)
    if stopped():
        return
    console.print(APP_VIEW_HINT, markup=False)
    status = console.status("Waiting for the notebook to open...") if show_status else nullcontext()
    with status:
        session = _wait_for_session(client, notebook, SESSION_TIMEOUT_SEC, should_stop=should_stop)
    if stopped():
        return
    if session is None:
        console.print(
            f"No kernel session yet. Open {url} in your browser; the agent reports the notebook state when asked.",
            style="yellow",
            markup=False,
        )
    else:
        console.print(f"Notebook is open (session {session.session_id}).", markup=False)


async def _prepare_notebook(console: Console, sandbox: MarimoSandbox, notebook: str, *, open_browser: bool) -> None:
    """Wait off the UI loop and settle the worker before the kernel may be stopped.

    Cancelling ``to_thread`` alone abandons its work. The stop flag ends polling
    promptly; an in-flight request or browser launcher must still finish before
    cleanup proceeds.
    """
    stop = threading.Event()
    task = asyncio.create_task(asyncio.to_thread(
        _wait_for_notebook, console, sandbox, notebook,
        open_browser=open_browser, should_stop=stop.is_set, show_status=False,
    ))
    try:
        await asyncio.shield(task)
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
            task.exception()  # consume a worker error while preserving cancellation
        raise


def _run_session(console: Console, config: HailerConfig, opts: CliOptions, *, port: int, open_browser: bool) -> None:
    """Start this session's own kernel, chat with it, and stop it when the chat ends."""
    timings = StartupTimings()
    checks = local_checks(config) + kernel_checks(config, starting=True)
    _print_checks(console, checks, only_failures=True)
    if any(not c.ok and c.fatal for c in checks):
        raise typer.Exit(code=1)
    timings.mark("checks_complete")
    _start_dependency_warmup()
    config.notebooks_root.mkdir(parents=True, exist_ok=True)  # marimo edit refuses a missing folder

    running: Any = None
    try:
        running = _start_kernel(console, _runtime_for(config), port, verbose=opts.verbose)
        # The chat and the agent's tools keep this sandbox (endpoint, token, files) in memory;
        # nothing else finds it, and it is stopped below whatever happens.
        timings.mark("kernel_ready")
        notebook = _active_notebook(config, running)
        interactive = _use_composer(opts)
        if not interactive:
            _wait_for_notebook(console, running, notebook, open_browser=open_browser)
            timings.mark("notebook_wait_complete")
        console.print()
        _startup_panel(console, config, notebook)
        timings.mark("panel_shown")

        async def prepare_notebook(chat_console: Console) -> None:
            await _prepare_notebook(chat_console, running, notebook, open_browser=open_browser)

        _run_chat_loop(
            console, config, opts, running,
            prepare_notebook=prepare_notebook if interactive else None, timings=timings,
        )
    finally:
        if running is not None:
            running.stop()
            if not getattr(running, "stop_error", ""):
                console.print("Stopped marimo.", markup=False)
            _warn_about_planted_files(console, running)
            if getattr(running, "stop_error", ""):
                console.print(running.stop_error, style="red", markup=False)
                raise typer.Exit(code=1)


@app.command()
def notebook(
    ctx: typer.Context,
    port: int = typer.Option(DEFAULT_MARIMO_PORT, "--port", "-p", help="Port for the marimo server (a free one is chosen if busy)."),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not open the notebook in a browser."),
    foreground: bool = typer.Option(False, "--foreground", help="Run marimo attached to this terminal, without the chat."),
    new: bool = typer.Option(False, "--new", help="Start a new conversation instead of resuming."),
    kernel: str | None = typer.Option(
        None,
        "--kernel",
        help='Where notebook code runs for this run: "local" (as you) or "docker" (an isolated container). Default: \\[kernel] runtime.',
        show_default=False,
    ),
    plain: bool = typer.Option(False, "--plain", help="Use line-oriented chat without the persistent composer."),
) -> None:
    """Start this session's own marimo kernel, open the notebook, and chat here; stop it on exit."""
    opts = _opts(ctx)
    if new:
        opts.new_thread = True
    if plain:
        opts.plain = True
    console = _chat_console(opts)
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

    _run_session(console, config, opts, port=port, open_browser=not no_browser)


@app.command()
def status(ctx: typer.Context) -> None:
    """Show configuration, credential source, this workspace's running kernels and loaded context."""
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
    kernels = _workspace_kernels(config)
    if not kernels:
        console.print("Kernels:     none running for this workspace (each uvx hailer session starts its own)", markup=False)
    for index, line in enumerate(kernels):
        console.print(("Kernels:     " if index == 0 else "             ") + line, markup=False)
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
    """Check the configuration, credentials and the kernel runtime (Docker, the image, the folders)
    and print a table with fixes. No kernel is started or probed."""
    opts = _opts(ctx)
    console = console_factory()
    config = _config_or_exit(console, opts)
    checks = local_checks(config) + kernel_checks(config)
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
    if any(not c.ok and c.fatal for c in checks):
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------- #
# hailer kernel: the Docker kernel's image and containers
# --------------------------------------------------------------------------- #

kernel_app = typer.Typer(
    no_args_is_help=True,
    help="The Docker kernel: download or build its image; stop this workspace's kernels.",
)
app.add_typer(kernel_app, name="kernel")


def _docker_runtime(config: HailerConfig) -> Any:
    from hailer.kernel_docker import DockerRuntime

    return DockerRuntime(config, runner=_docker_runner())


@kernel_app.command("pull")
def kernel_pull(ctx: typer.Context) -> None:
    """Download the kernel image for this Hailer's kernel contract (or \\[kernel].image) ahead of the first start."""
    opts = _opts(ctx)
    console = console_factory()
    config = _config_or_exit(console, opts)
    runtime = _docker_runtime(config)
    try:
        contract = runtime.pull(say=lambda text: console.print(text, markup=False))
    except HailerError as err:
        _print_error(console, err, verbose=opts.verbose)
        raise typer.Exit(code=1)
    console.print(f"{runtime.image} is ready (kernel contract {contract}).", markup=False)


@kernel_app.command("build")
def kernel_build(
    ctx: typer.Context,
    tag: str | None = typer.Option(
        None,
        "--tag",
        help="Name and tag for the image (default: \\[kernel].image, else ghcr.io/openafterhours/hailer-kernel:<kernel contract>).",
        show_default=False,
    ),
    base_image: str | None = typer.Option(
        None, "--base-image", help="Python base image, e.g. registry.company/python:3.13 (Python 3.12+ with venv).",
    ),
    pip_config: Path | None = typer.Option(
        None, "--pip-config", help="Pip configuration build secret; overrides automatic host pip/uv discovery.",
        exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True,
    ),
    pip_cert: Path | None = typer.Option(
        None, "--pip-cert", help="PEM CA bundle for pip during the build; mounted as a build secret.",
        exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True,
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Reinstall packages without cached build layers (use after changing mirror settings).",
    ),
    no_host_config: bool = typer.Option(
        False, "--no-host-config", help="Disable automatic discovery of host pip/uv package settings.",
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
            f"Building {image} for kernel contract {args['KERNEL_CONTRACT']} (marimo {args['MARIMO_VERSION']}, "
            f"Polars {args['POLARS_VERSION']}, DuckDB {args['DUCKDB_VERSION']}) ...",
            markup=False,
        )
        code = kernel_image.build(
            image, runtime.runner, base_image=base_image, pip_config=pip_config, pip_cert=pip_cert, no_cache=no_cache,
            no_host_config=no_host_config, say=lambda message: console.print(message, markup=False),
        )
        if code != 0:
            raise KernelRuntimeError(
                f"docker build failed (exit code {code}).",
                hint=(
                    "Docker's output is above. Check access to the base image and package index. "
                    "Corporate mirrors: use --base-image and --pip-config; for a private CA, add --pip-cert. "
                    "The build needs BuildKit and a Linux base with Python 3.12+, venv and ensurepip."
                ),
            )
    except HailerError as err:
        _print_error(console, err, verbose=opts.verbose)
        raise typer.Exit(code=1)
    console.print(f"Built {image}.", markup=False)
    if image != config.kernel.effective_image:
        console.print(f'Hailer uses {config.kernel.effective_image}; set image = "{image}" under [kernel] in hailer.toml to use this one.', markup=False)


@kernel_app.command("stop")
def kernel_stop(ctx: typer.Context) -> None:
    """Stop every kernel of this workspace (local or docker, whichever session started it) and remove its Docker containers and networks."""
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
        f"  3. Put the files you want to analyse in {data}/ (CSV, Parquet, JSON or Excel, any name).\n"
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
