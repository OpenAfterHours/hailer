"""What the Hailer commands and the chat controller share: options, the injectable collaborators,
output helpers, the preflight checks and the terminal pieces of the chat.

Collaborators (config, runtimes, marimo client, agent, secrets, context, browser) are small
module-level factory functions here. The command modules call them as ``common.<name>`` and
:class:`hailer.chat.ChatController` receives this module as its ``services``, so a test replaces one
with ``monkeypatch.setattr(common, "<name>", ...)`` and every caller sees it.
"""

from __future__ import annotations

import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from hailer import notebooks
from hailer.errors import ConfigError, HailerError, NoSessionError
from hailer.models import (
    KERNEL_RUNTIME_UNSAFE_LOCAL,
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

if TYPE_CHECKING:  # pragma: no cover - annotations only; the runtimes import it when a kernel starts
    from hailer.sandbox import MarimoSandbox

PROMPT = "You > "
ANSWER_HEADER = "Hailer >"
SWITCH_SESSION_TIMEOUT_SEC = 30.0  # how long /notebook new|open waits for the browser tab
NOTEBOOK_USAGE = "Usage: /notebook [list | new <name> [--empty] | open <name> | close [name]]"


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
    """
    from hailer.config import load_config

    return load_config(workspace=opts.workspace, config_path=opts.config_path)


def _validate_config(config: HailerConfig) -> list[str]:
    from hailer.config import validate

    return validate(config)


def _setup_logging(config: HailerConfig, opts: CliOptions) -> None:
    from hailer.log import setup_logging

    setup_logging(config.log_level, verbose=opts.verbose)


def _runtime_for(config: HailerConfig) -> Any:
    """The kernel runtime ``[kernel] runtime`` asks for (``ConfigError`` for an unknown one). Nothing
    is checked until its ``check``, ``prepare`` or ``start`` runs."""
    from hailer.kernel import runtime_for

    return runtime_for(config)


def _docker_runner() -> Any:
    """What ``hailer kernel ...`` runs ``docker`` with (the CLI on PATH)."""
    from hailer.kernel_docker import SubprocessDockerRunner

    return SubprocessDockerRunner()


def _kernel_line(config: HailerConfig) -> str:
    from hailer.kernel import describe_runtime

    return describe_runtime(config.kernel)


def _kernel_style(config: HailerConfig) -> str:
    """A warning style (yellow) for the ``Kernel:`` line of the unsafe-local runtime, whose notebook
    code is not isolated (the startup panel, ``hailer status``, ``/status``, ``--foreground``)."""
    return "yellow" if config.kernel.runtime == KERNEL_RUNTIME_UNSAFE_LOCAL else ""


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
    from hailer.browser import (
        open_url,  # os.startfile first on Windows (ignores BROWSER), never raises
    )

    open_url(url)


def _find_free_port(preferred: int) -> int:
    from hailer.marimo_client import find_free_port

    return find_free_port(preferred)


def _wait_for_session(
    client: Any, notebook: str, timeout: float, *, should_stop: Callable[[], bool] | None = None
) -> MarimoSession | None:
    from hailer.marimo_client import wait_for_session

    return wait_for_session(client, notebook, timeout, should_stop=should_stop)


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
    """The panel with the settings; ``notebook``: the active notebook (default: the state file's).
    An unsafe-local ``Kernel:`` line is shown in a warning style (:func:`_kernel_style`)."""
    body = Text("\n").join(
        [
            Text(f"Model:      {config.model.name}"),
            Text(f"Provider:   {_provider_line(config)}"),
            Text(f"Notebook:   {notebook or notebooks.load_active_notebook(config)}"),
            Text(f"Notebooks:  {_relative(config.notebooks_root, config.workspace)}"),
            Text(f"Workspace:  {config.workspace}"),
            Text(f"Kernel:     {_kernel_line(config)}", style=_kernel_style(config)),
            Text(f"Web access: {_web_line(config)}"),
        ]
    )
    console.print(Panel(body, title="Hailer", expand=False))
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
    image itself and shows the ``Kernel:`` line (unsafe-local: in a warning style), so those two
    warnings are left out. A runtime that is not one has no rows: the ``config`` row reports it."""
    if config.kernel.runtime not in VALID_KERNEL_RUNTIMES:
        return []
    try:
        checks = list(_runtime_for(config).check())
    except HailerError as err:
        return [Check("kernel", False, str(err), hint=err.hint)]
    if starting:
        checks = [c for c in checks if not (c.name in ("image", "kernel") and not c.ok and not c.fatal)]
    return checks


CONFIG_FIX_HINT = "Fix hailer.toml (uvx hailer init writes a starter file and creates a missing notebook)."


def _config_fix_hint(fatal: list[str]) -> str:
    """How to fix the fatal ``config`` problems: in hailer.toml, or in ``HAILER_KERNEL`` when that
    variable set the runtime a problem names."""
    if os.environ.get("HAILER_KERNEL") and any(p.startswith("Invalid [kernel].runtime") for p in fatal):
        return "HAILER_KERNEL in the environment overrides [kernel] runtime in hailer.toml: change or unset it."
    return CONFIG_FIX_HINT


def _config_check(problems: list[str]) -> Check:
    """The ``config`` row, each problem on a line of its own. Warnings make it a warning row, so
    they show when a session starts too (a setting in the wrong table must not go unnoticed)."""
    fatal = [p for p in problems if not p.lower().startswith("warning")]
    warnings = [p for p in problems if p.lower().startswith("warning")]
    if fatal:
        summary = fatal[0] if len(fatal) == 1 else f"{len(fatal)} problems in the configuration:"
        listed = [] if len(fatal) == 1 else [f"- {p}" for p in fatal]
        return Check("config", False, summary, hint="\n".join([*listed, *warnings, _config_fix_hint(fatal)]), fatal=True)
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
# Loading the configuration
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


def _kernel_choice(console: Console, value: str | None) -> str | None:
    """A ``--kernel`` value ("docker" or "unsafe-local"); exit 2 for anything else (the retired
    "local" with its own message)."""
    if value is None:
        return None
    choice = value.strip().lower()
    if choice not in VALID_KERNEL_RUNTIMES:
        from hailer.config import runtime_problem

        console.print(runtime_problem(choice, "--kernel"), style="red", markup=False)
        raise typer.Exit(code=2)
    return choice


# --------------------------------------------------------------------------- #
# The chat's terminal: the prompt and the turn display
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
