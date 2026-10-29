"""``uvx hailer`` and ``uvx hailer notebook``: start this session's own kernel, open the notebook,
chat, and stop the kernel on every way out (``--foreground``: marimo without the chat).

A session reads as a checklist: a header line, a row per thing that got ready (``✓ Kernel``,
``✓ Notebook``, ``✓ Agent``; the last two inside the composer as they finish), and an exit summary.
In the chat the notebook's signed-in URL (it carries the access token) is printed only when the
user must open it by hand: ``--no-browser``, or no session came in time. ``--foreground`` has no
chat: its signed-in home URL is the way in, so it prints it.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import typer
from rich.console import Console
from rich.text import Text

from hailer import __version__, notebooks
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
    _relative,
    kernel_checks,
    local_checks,
)
from hailer.errors import HailerError
from hailer.models import (
    KERNEL_RUNTIME_DOCKER,
    KERNEL_RUNTIME_UNSAFE_LOCAL,
    Check,
    HailerConfig,
)
from hailer.startup import StartupTimings
from hailer.steps import single_line
from hailer.transcript import (
    MUTED,
    callout,
    console_glyphs,
    done_items,
    session_ended,
    session_header,
    session_note,
    session_row,
)

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from hailer.sandbox import MarimoSandbox

DEFAULT_MARIMO_PORT = 2718
SESSION_TIMEOUT_SEC = 90.0
#: The muted line under ``✓ Notebook`` when this start had it opened in a browser.
APP_VIEW_TIP = "Opens in app view (results only); Ctrl+. in the notebook shows the code"
#: The startup checklist's label for a preflight check that did not pass (``check.name``).
CHECK_LABELS = {"config": "Config", "notebook": "Notebook", "credentials": "API key", "kernel": "Kernel", "docker": "Docker", "image": "Image", "data": "Data"}
#: ``progress(label, detail)``: what the composer's live row shows while a block runs.
Progress = Callable[[str, str], AbstractContextManager[Any]]


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
    check, not a listing). Saved, so the agent's tools read the same name."""
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

    def __init__(
        self, console: Console, config: HailerConfig, opts: CliOptions, sandbox: MarimoSandbox | None = None, *, open_browser: bool = True,
    ) -> None:
        super().__init__(console, config, opts, services=common, sandbox=sandbox)
        self.open_browser = open_browser  # --no-browser: switches print the link instead


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
    prepare_notebook: Callable[[Console, Progress], Awaitable[bool | None]] | None = None,
    timings: StartupTimings | None = None,
    open_browser: bool = True,
) -> None:
    """Start the agent and run the REPL, then say the session ended; exit code 1 when the agent
    cannot start."""
    loop = ChatLoop(console, config, opts, sandbox, open_browser=open_browser)
    started = timings.started_at if timings is not None else time.perf_counter()
    if _use_composer(opts):
        async def prepare() -> bool | None:
            if prepare_notebook is not None:
                return await prepare_notebook(loop.console, loop.progress)
            return None

        try:
            asyncio.run(loop.run_interactive(prepare_notebook=prepare, timings=timings))
        except HailerError as err:
            _print_error(console, err, verbose=opts.verbose)
            raise typer.Exit(code=1)
        except KeyboardInterrupt:
            pass
        except Exception as exc:  # startup/terminal failures still restore the terminal
            _print_unexpected(console, exc, verbose=opts.verbose)
            raise typer.Exit(code=1)
        _session_ended(console, loop, started)
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
    console.print()
    console.print(Text("Type /help for commands.", style=MUTED))
    try:
        loop.run()
    finally:
        loop.close()
    console.print()
    _session_ended(console, loop, started)


def _note(console: Console, renderable: Any) -> None:
    """A note or callout on the CLI's own console (before and after the composer): on a terminal
    Rich wraps it, so its lines stay under the detail column; pipes and plain mode get each line
    whole (the console is soft-wrapped)."""
    console.print(renderable, soft_wrap=not console.is_terminal, crop=False)


def _session_ended(console: Console, loop: ChatController, started: float) -> None:
    """``Session ended · 4 turns · 4m 12s · 18.2k tokens``: this session's turns, time (since the
    command started) and tokens. The kernel's line follows once it has stopped."""
    console.print(session_ended(
        console_glyphs(console), turns=loop.session_turns, elapsed=time.perf_counter() - started, tokens_used=loop.session_tokens,
    ))


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
    """Start marimo through ``runtime`` and wait until it answers with its token, then its row
    (:func:`_kernel_row`, a busy port and, with ``--verbose``, the URL and log under it); exit 1 when
    it does not start. Ctrl+C after the runtime started it still stops it (the caller owns it only
    once this returns)."""
    running: Any = None
    try:
        _prepare_runtime(console, runtime)
        chosen = common._find_free_port(port)
        began = time.monotonic()
        with console.status(f"Starting marimo{_where(runtime)} on http://127.0.0.1:{chosen} ..."):
            running = runtime.start(chosen)
        _kernel_row(console, running, time.monotonic() - began)
        if chosen != port:
            _note(console, session_note(f"Port {port} is busy; using {chosen}."))
        if verbose:
            console.print(session_note(f"Marimo is running at {running.server.url}  (log: {running.log_hint})"), soft_wrap=True)
    except HailerError as err:  # the runtime has cleaned up; the hint carries the log tail
        _print_error(console, err, verbose=verbose)
        raise typer.Exit(code=1)
    except BaseException:
        if running is not None:
            running.stop()
        raise
    return running


def _kernel_row(console: Console, sandbox: MarimoSandbox, took: float | None) -> None:
    """``✓ Kernel  docker · isolated · no network · data read-only  1.9s``; the unsafe-local kernel
    is a warning row (it runs as the user). The image and the log are in ``/status``."""
    g = console_glyphs(console)
    server = sandbox.server
    if server.runtime == KERNEL_RUNTIME_UNSAFE_LOCAL:
        console.print(session_row(g, "warn", "Kernel", g.join("unsafe-local", "runs as you, not isolated"), took))
        return
    if server.runtime == KERNEL_RUNTIME_DOCKER:
        network = ("network on",) if server.network_access else ("isolated", "no network")
        console.print(session_row(g, "ok", "Kernel", g.join("docker", *network, "data read-only"), took))
        return
    console.print(session_row(g, "ok", "Kernel", single_line(server.runtime), took))


def _stop_kernel(console: Console, config: HailerConfig, running: MarimoSandbox) -> None:
    """Stop the session's kernel (a docker kernel copies its notebooks back first, which can take
    a while: a spinner says so on a terminal) and say how it went on one line:
    ``✓ Notebooks copied back to notebooks/   ✓ Kernel stopped``.

    The stop runs whatever the terminal does: a spinner or line that cannot be written must never
    leave the kernel running (or its last copy not made). The first item is the last copy's own
    record (``copied_back``), not its warnings, which say a repeated problem only once; they are
    printed as they come. A second Ctrl+C while the kernel is removed says it may still run; a stop
    error stays red; both exit 1."""
    g = console_glyphs(console)
    docker = running.server.runtime == KERNEL_RUNTIME_DOCKER
    running.notice = lambda text: _note(console, callout(g, "warn", text))
    began = False
    try:
        with console.status("Copying the notebooks back and stopping the kernel ..." if docker else "Stopping the kernel ..."):
            began = True
            running.stop()  # a docker kernel copies its notebooks back first, on every way out
    except KeyboardInterrupt:
        if not began:
            raise  # the stop below still runs
        how = "Run uvx hailer kernel stop to remove it." if docker else "End its python -m marimo process (Task Manager or kill)."
        _note(console, callout(g, "warn", f"Interrupted while the kernel stopped: it may still be running. {how}"))
        raise typer.Exit(code=1)
    finally:
        if not began:
            running.stop()
    error = getattr(running, "stop_error", "")
    items: list[tuple[str, str]] = []
    copied = getattr(running, "copied_back", None)
    if docker and copied is not None:
        folder = f"{_relative(config.notebooks_root, config.workspace)}{os.sep}"
        items.append(("ok", f"Notebooks copied back to {folder}") if copied else ("warn", f"Notebooks not all copied back to {folder}"))
    if not error:
        items.append(("ok", "Kernel stopped"))
    console.print(done_items(g, items))
    if error:
        _note(console, callout(g, "fail", error))
        raise typer.Exit(code=1)


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
    progress: Progress | None = None,
) -> bool | None:
    """Open ``notebook`` and poll its session, stopping between blocking operations, then its row:
    ``✓ Notebook  analysis.py · open in your browser  2.2s`` and the app-view tip, ``already open``,
    or a warning with the signed-in link when no session came. Returns whether it has a session
    (the composer's footer shows it); None when stopped.

    The signed-in URL is printed only when the user must open it by hand (``--no-browser``, or no
    session in time), never as a hyperlink. ``progress`` shows the wait on the composer's live row;
    without it (plain mode) a ``○ Notebook`` row says what is awaited, and a Rich spinner may run
    (``show_status``). Cancellation never leaves a later browser open or session message racing
    the next session or kernel shutdown.
    """
    stopped = should_stop or (lambda: False)
    if stopped():
        return None
    g = console_glyphs(console)
    name = single_line(notebook)
    began = time.monotonic()
    client = sandbox.client(notebook=notebook, token_in_links=True)
    url = sandbox.notebook_url(notebook, with_token=True)
    if stopped():
        return None
    try:
        session = client.resolve_session(notebook)
    except HailerError:
        session = None
    if stopped():
        return None
    if session is not None:
        console.print(session_row(g, "ok", "Notebook", g.join(name, "already open"), time.monotonic() - began))
        return True
    doing = f"opening {name} in your browser" if open_browser else f"waiting for {name} to open"
    if not open_browser:
        console.print(session_row(g, "wait", "Notebook", f"open {name} in your browser with this link (it signs you in):"))
        console.print(session_note(url, style=""), soft_wrap=True)
    elif progress is None:
        console.print(session_row(g, "wait", "Notebook", doing))  # plain mode: no live row says it
    if progress is not None:
        waiting = progress("Notebook", doing)
    else:
        waiting = console.status("Waiting for the notebook to open...") if show_status else nullcontext()
    with waiting:
        if open_browser:
            common._open_browser(url)
        if stopped():
            return None
        session = common._wait_for_session(client, notebook, SESSION_TIMEOUT_SEC, should_stop=should_stop)
    if stopped():
        return None
    took = time.monotonic() - began
    if session is None:
        console.print(session_row(g, "warn", "Notebook", g.join(name, "no kernel session yet"), took))
        if open_browser:
            console.print(session_note("Open it in your browser with this link; the agent reports the notebook state when asked:"))
            console.print(session_note(url, style=""), soft_wrap=True)
        else:
            console.print(session_note("Open it with the link above; the agent reports the notebook state when asked."))
        return False
    console.print(session_row(g, "ok", "Notebook", g.join(name, "open in your browser"), took))
    console.print(session_note(APP_VIEW_TIP))
    return True


async def _prepare_notebook(
    console: Console, sandbox: MarimoSandbox, notebook: str, *, open_browser: bool, progress: Progress | None = None,
) -> bool | None:
    """Wait off the UI loop and settle the worker before the kernel may be stopped.

    Cancelling ``to_thread`` alone abandons its work. The stop flag ends polling
    promptly; an in-flight request or browser launcher must still finish before
    cleanup proceeds.
    """
    stop = threading.Event()
    task = asyncio.create_task(asyncio.to_thread(
        _wait_for_notebook, console, sandbox, notebook,
        open_browser=open_browser, should_stop=stop.is_set, show_status=False, progress=progress,
    ))
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
            task.exception()  # consume a worker error while preserving cancellation
        raise


def _home_relative(path: Path) -> str:
    """``path`` with ``~`` for the home folder (the header line)."""
    try:
        inside = Path(path).relative_to(Path.home())
    except (ValueError, RuntimeError):  # elsewhere, or no home folder
        return str(path)
    return "~" if inside == Path() else str(Path("~") / inside)


def _check_rows(console: Console, checks: list[Check]) -> None:
    """The preflight checks that did not pass, as rows of the startup checklist: a warning
    (yellow) or a failure (red), each with its hint lines under it, their text unchanged."""
    g = console_glyphs(console)
    for check in checks:
        if check.ok:
            continue
        console.print(session_row(g, "fail" if check.fatal else "warn", CHECK_LABELS.get(check.name, check.name.capitalize()), check.summary))
        for line in (check.hint or "").splitlines():
            _note(console, session_note(line, style=""))


def _run_session(console: Console, config: HailerConfig, opts: CliOptions, *, port: int, open_browser: bool) -> None:
    """Start this session's own kernel, chat with it, and stop it when the chat ends: the header,
    the checklist's rows, the chat, the exit summary."""
    timings = StartupTimings()
    console.print(session_header(console_glyphs(console), __version__, _home_relative(config.workspace)))
    console.print()
    checks = local_checks(config) + kernel_checks(config, starting=True)
    _check_rows(console, checks)
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
        timings.mark("checklist_shown")

        async def prepare_notebook(chat_console: Console, progress: Progress) -> bool | None:
            return await _prepare_notebook(chat_console, running, notebook, open_browser=open_browser, progress=progress)

        _run_chat_loop(
            console, config, opts, running,
            prepare_notebook=prepare_notebook if interactive else None, timings=timings, open_browser=open_browser,
        )
    finally:
        if running is not None:
            _stop_kernel(console, config, running)


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
