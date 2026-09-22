"""``uvx hailer`` and ``uvx hailer notebook``: start this session's own kernel, open the notebook,
chat, and stop the kernel on every way out (``--foreground``: marimo without the chat)."""

from __future__ import annotations

import asyncio
import os
import threading
from contextlib import nullcontext
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import typer
from rich.console import Console

from hailer import notebooks
from hailer.chat import ChatController
from hailer.cli import common
from hailer.cli.common import (
    CliOptions,
    _config_or_exit,
    _kernel_choice,
    _kernel_style,
    _labelled,
    _opts,
    _print_checks,
    _print_error,
    _print_unexpected,
    _startup_panel,
    kernel_checks,
    local_checks,
)
from hailer.errors import HailerError
from hailer.models import KERNEL_RUNTIME_DOCKER, HailerConfig
from hailer.startup import StartupTimings

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from hailer.sandbox import MarimoSandbox

DEFAULT_MARIMO_PORT = 2718
SESSION_TIMEOUT_SEC = 90.0
APP_VIEW_HINT = (
    "The notebook opens in app view (results only). "
    "Press Ctrl+. in it, or use Toggle app view, to see and edit the code."
)


def _chat_console(opts: CliOptions) -> Console:
    console = common.console_factory()
    if opts.plain or os.environ.get("TERM", "").lower() in {"dumb", "unknown"}:
        # Rich live status also emits cursor controls on a TTY. Plain mode must
        # disable those throughout startup, notebook waits and model turns.
        return Console(file=console.file, width=console.width, force_terminal=False,
                       color_system=None, highlight=False, soft_wrap=True)
    return console


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


def _use_composer(opts: CliOptions) -> bool:
    """Pipes and minimal terminal emulators retain the linear interface."""
    return not opts.plain and common._stdio_is_terminal() and os.environ.get("TERM", "").lower() not in {"dumb", "unknown"}


class ChatLoop(ChatController):
    """Bind the shared controller to the CLI's injectable collaborators."""

    def __init__(self, console: Console, config: HailerConfig, opts: CliOptions, sandbox: MarimoSandbox | None = None) -> None:
        super().__init__(console, config, opts, services=common, sandbox=sandbox)


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


def _free_port_url(console: Console, port: int) -> tuple[int, str]:
    chosen = common._find_free_port(port)
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
        console.print(f"Kernel:     {runtime.describe()}", style=_kernel_style(config), markup=False)
        console.print(f"Marimo is running at {running.server.url}. Ctrl+C stops it.", markup=False)
        home = running.home_url(with_token=True)
        if open_browser:
            console.print(f"Opening {home} in your browser...", markup=False)
            common._open_browser(home)
        else:
            console.print(f"Open {home} in your browser.", markup=False)
        code = running.wait()
        if running.ended:
            console.print(running.ended, style="yellow", markup=False)
        return code
    finally:
        if running is not None:
            running.stop()  # a docker kernel copies its notebooks back first
            if getattr(running, "stop_error", ""):
                console.print(running.stop_error, style="red", markup=False)
                raise typer.Exit(code=1)


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
        common._open_browser(url)
    else:
        console.print(f"Open {url} in your browser.", markup=False)
    if stopped():
        return
    console.print(APP_VIEW_HINT, markup=False)
    status = console.status("Waiting for the notebook to open...") if show_status else nullcontext()
    with status:
        session = common._wait_for_session(client, notebook, SESSION_TIMEOUT_SEC, should_stop=should_stop)
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
    common._start_dependency_warmup()
    config.notebooks_root.mkdir(parents=True, exist_ok=True)  # marimo edit refuses a missing folder

    running: Any = None
    try:
        running = _start_kernel(console, common._runtime_for(config), port, verbose=opts.verbose)
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
            running.stop()  # a docker kernel copies its notebooks back first, on every way out
            if not getattr(running, "stop_error", ""):
                console.print("Stopped marimo.", markup=False)
            if getattr(running, "stop_error", ""):
                console.print(running.stop_error, style="red", markup=False)
                raise typer.Exit(code=1)


def notebook(
    ctx: typer.Context,
    port: int = typer.Option(DEFAULT_MARIMO_PORT, "--port", "-p", help="Port for the marimo server (a free one is chosen if busy)."),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not open the notebook in a browser."),
    foreground: bool = typer.Option(False, "--foreground", help="Run marimo attached to this terminal, without the chat."),
    new: bool = typer.Option(False, "--new", help="Start a new conversation instead of resuming."),
    kernel: str | None = typer.Option(
        None,
        "--kernel",
        help='Where notebook code runs for this run: "docker" (an isolated container) or "unsafe-local" (as you, with your files and network). Default: \\[kernel] runtime.',
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
        runtime = common._runtime_for(config)
        raise typer.Exit(code=_run_foreground(console, config, runtime, port, open_browser=not no_browser, verbose=opts.verbose))

    _run_session(console, config, opts, port=port, open_browser=not no_browser)
