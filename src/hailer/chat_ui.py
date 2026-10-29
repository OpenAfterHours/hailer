"""Inline terminal chat: native scrollback above a persistent, editable composer.

Only the writer task touches the transcript while the application is running.
The controller can keep using synchronous ``console.print`` calls (including
from a notebook worker thread); the facade queues them for that writer.

The composer is a rounded frame whose colour says the state (cyan ready, grey busy or starting,
red when startup failed), with key hints in its bottom border. While work runs, a live row above
it shows the step in progress in the columns its finished row will use; a footer below it shows
the notebook, the kernel, the model and usage. Every row but the composer's is one window that
draws its whole line, so a redraw stays cheap (the spinner redraws often during a turn).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from typing import Any, ClassVar

from prompt_toolkit.application import Application, run_in_terminal
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text.utils import fragment_list_width
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.margins import Margin
from prompt_toolkit.layout.processors import Processor, Transformation, TransformationInput
from prompt_toolkit.output import create_output
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import TextArea
from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
from rich.markdown import CodeBlock, Heading, Markdown, TableElement
from rich.segment import Segment
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from hailer.log import redact
from hailer.models import KERNEL_RUNTIME_DOCKER, KERNEL_RUNTIME_UNSAFE_LOCAL, AgentEvent, TurnSummary
from hailer.session import EXIT_COMMANDS, parse_command
from hailer.steps import Receipt, StepTracker, plural, single_line
from hailer.transcript import (
    ACCENT,
    LABEL_WIDTH,
    MUTED,
    SESSION_LABEL_WIDTH,
    UNICODE,
    FooterData,
    Glyphs,
    Gutter,
    callout,
    console_glyphs,
    seconds,
    simple_head,
    step_row,
    tokens,
    turn_footer,
    turn_stats,
    user_message,
)

# ChatUI's own activity strings (and the controller's) in the live row's words; "" hides the row.
_LABELS = {
    "Ready": "",
    "Working...": "Working",
    "Thinking...": "Thinking",
    "Preparing chat...": "Starting",
    "Cancelling...": "Cancelling",
}
_FRAME_SECONDS = 0.08  # spinner frame, and the redraw interval during a turn
# While the chat prepares, agent imports run on a worker thread and every redraw takes the GIL from
# them (LEARNINGS §10): nothing animates then, the live row's timer redraws once a second.
_STARTUP_TICK_SECONDS = 1.0
_FLASH_SECONDS = 2.5
# During a turn Esc alone cancels, so it must act quickly: ttimeoutlen for the terminal to finish an
# escape sequence (arrows arrive in one write), timeoutlen for an Enter that makes it Esc Enter.
# While idle prompt_toolkit's own, longer timeouts stay, so a slow Esc, Enter is still a newline.
_TURN_TIMEOUTS = (0.1, 0.3)

_STYLE = Style.from_dict({
    "frame.idle": "ansicyan",
    "frame.busy": "ansibrightblack",
    "frame.failed": "ansired",
    "prompt.idle": "ansicyan bold",
    "prompt.busy": "ansibrightblack",
    "placeholder": "ansibrightblack",
    "auto-suggestion": "ansibrightblack",
    "spinner": "ansicyan",
    "activity": "",
    "muted": "ansibrightblack",
    "hint": "ansibrightblack",
    "hint.flash": "ansiyellow",
    "ok": "ansigreen",
    "warn": "ansiyellow",
})


def _clip(text: str, cells: int) -> str:
    """The longest start of ``text`` that fits in ``cells`` terminal cells."""
    out, used = "", 0
    for char in text:
        used += get_cwidth(char)
        if used > cells:
            break
        out += char
    return out


def _hanging(text: str) -> RenderableType | str:
    """A plain one-line string that starts with spaces, as a renderable whose wrapped lines keep
    that indent."""
    body = text.lstrip(" ")
    indent = len(text) - len(body)
    return Gutter(Text(body), " " * indent) if indent and body and "\n" not in text else text


def _label(text: str) -> str:
    text = single_line(text)
    return _LABELS.get(text, text.removesuffix("...").removesuffix("…"))


# --------------------------------------------------------------------------- #
# The reply's typography
# --------------------------------------------------------------------------- #

# ANSI names only, so the terminal's theme (light or dark) decides the colours.
_REPLY_THEME = Theme({
    "markdown.h1": "bold",
    "markdown.h2": "bold",
    "markdown.h3": "bold",
    "markdown.h4": "bold",
    "markdown.code": ACCENT,
    "markdown.link": "cyan underline",
    "markdown.link_url": MUTED,
    "markdown.item.bullet": MUTED,
    "markdown.item.number": MUTED,
    "markdown.block_quote": "italic",
    "markdown.hr": MUTED,
    "markdown.table.header": "bold",
    "markdown.table.border": MUTED,
})


class _Heading(Heading):
    LEVEL_ALIGN: ClassVar[dict[str, Any]] = {}  # Rich centres h1; a reply reads left to right


class _Code(CodeBlock):
    """A fenced block behind a thin rule, its language above it; no background."""

    glyphs: Glyphs = UNICODE

    @classmethod
    def create(cls, markdown: Any, token: Any) -> _Code:
        block = super().create(markdown, token)
        block.glyphs = getattr(markdown, "glyphs", UNICODE)
        return block  # type: ignore[return-value]

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        code = str(self.text).rstrip()
        syntax = Syntax(code, self.lexer_name, theme="ansi_dark", word_wrap=True, background_color="default")
        if self.lexer_name not in ("", "text", "default"):
            yield Text(self.lexer_name, style=MUTED)
        yield Gutter(syntax, f"{self.glyphs.bar} ", style=MUTED)


class _Table(TableElement):
    """A table with only a rule under its header, keeping each column's alignment."""

    glyphs: Glyphs = UNICODE

    @classmethod
    def create(cls, markdown: Any, token: Any) -> _Table:
        table = cls()
        table.glyphs = getattr(markdown, "glyphs", UNICODE)
        return table

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        table = Table(
            box=simple_head(self.glyphs), show_edge=False, pad_edge=False,
            border_style="markdown.table.border", header_style="markdown.table.header",
        )
        if self.header is not None and self.header.row is not None:
            for cell in self.header.row.cells:
                justify = cell.content.justify if cell.content.justify not in (None, "default") else "left"
                table.add_column(cell.content.copy(), justify=justify)
        if self.body is not None:
            for row in self.body.rows:
                table.add_row(*(cell.content for cell in row.cells))
        yield table


class ReplyMarkdown(Markdown):
    """Rich Markdown with left-aligned headings, simple tables and quiet code."""

    elements: ClassVar[dict[str, type[Any]]] = {
        **Markdown.elements, "heading_open": _Heading, "fence": _Code, "code_block": _Code, "table_open": _Table,
    }

    def __init__(self, markup: str, glyphs: Glyphs = UNICODE) -> None:
        self.glyphs = glyphs
        super().__init__(markup)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        # Rendered whole while the reply's theme is in place, so no other output sees it.
        with console.use_theme(_REPLY_THEME):
            lines = list(Segment.split_lines(console.render(_Plain(super().__rich_console__(console, options)), options)))
        while lines and not "".join(segment.text for segment in lines[0]).strip():
            lines.pop(0)  # Rich starts a table (and so a reply that opens with one) with a blank line
        for line in lines:
            yield from line
            yield Segment.line()


class _Plain:
    """Render an already-made render result (a generator) as a renderable."""

    def __init__(self, result: RenderResult) -> None:
        self.result = result

    def __rich_console__(self, _console: Console, _options: ConsoleOptions) -> RenderResult:
        yield from self.result


def reply(glyphs: Glyphs, markdown: str) -> RenderableType:
    """The final answer behind the reply mark."""
    final = (markdown or "").strip()
    body: RenderableType = ReplyMarkdown(final, glyphs) if final else Text("(no reply)", style=MUTED)
    return Gutter(body, f"{glyphs.reply} ", style=ACCENT)


# --------------------------------------------------------------------------- #
# Terminal plumbing
# --------------------------------------------------------------------------- #


class _PersistentPasteOutput:
    """Delegate output, retaining paste mode across ``run_in_terminal`` redraws.

    prompt_toolkit normally disables bracketed paste when temporarily erasing
    the prompt for output. This application never hands control to a subprocess,
    so paste belongs to the entire session. The real mode is restored in close.
    No private renderer state is modified.
    """

    def __init__(self, output: Any) -> None:
        self.output = output
        self.paste_enabled = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.output, name)

    @property
    def responds_to_cpr(self) -> bool:
        return False

    def ask_for_cpr(self) -> None:
        pass

    def enable_bracketed_paste(self) -> None:
        if not self.paste_enabled:
            self.output.enable_bracketed_paste()
            self.paste_enabled = True

    def disable_bracketed_paste(self) -> None:
        # Renderer.reset calls this during every transcript write.
        pass

    def close(self) -> None:
        try:
            if self.paste_enabled:
                self.output.disable_bracketed_paste()
        finally:
            self.paste_enabled = False
            self.output.show_cursor()
            self.output.flush()


class _Status:
    """The small subset of Rich Status used by existing slash commands: it drives the live row."""

    def __init__(self, ui: ChatUI, status: Any) -> None:
        self.ui = ui
        self.status = status

    def start(self) -> None:
        self.update(self.status)

    def update(self, status: Any, **_kwargs: Any) -> None:
        self.status = status
        self.ui.set_activity(str(status))

    def stop(self) -> None:
        self.ui.set_activity("Working..." if self.ui.busy else "Ready")

    def __enter__(self) -> _Status:
        self.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.stop()


class _QueuedConsole:
    def __init__(self, ui: ChatUI, console: Console) -> None:
        self._ui = ui
        self._console = console

    def __getattr__(self, name: str) -> Any:
        return getattr(self._console, name)

    def print(self, *objects: Any, **kwargs: Any) -> None:
        self._ui._write(objects, kwargs)

    def clear(self, *args: Any, **kwargs: Any) -> None:
        self._ui._enqueue(lambda: self._ui._clear_now(args, kwargs))

    def status(self, status: Any, **_kwargs: Any) -> _Status:
        return _Status(self._ui, status)


class _LogStream:
    """Route already-formatted logs, including tracebacks, through the writer."""

    def __init__(self, ui: ChatUI, original: Any) -> None:
        self.ui = ui
        self.original = original

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)

    def write(self, text: str) -> int:
        if text:
            self.ui.console.print(redact(text), markup=False, end="", style="dim")
        return len(text)

    def flush(self) -> None:
        pass


class _Placeholder(Processor):
    """Grey text in the empty composer that says what Enter would do now, on one line."""

    def __init__(self, text: Callable[[], str], ellipsis: str) -> None:
        self.text = text
        self.ellipsis = ellipsis

    def apply_transformation(self, ti: TransformationInput) -> Transformation:
        if ti.document.text or ti.lineno != 0:
            return Transformation(ti.fragments)
        text, room = self.text(), ti.width - 3  # the prompt takes two cells, and a full line would wrap
        if get_cwidth(text) > room:
            text = _clip(text, room - get_cwidth(self.ellipsis)).rstrip() + self.ellipsis if room > 4 else ""
        return Transformation([*ti.fragments, ("class:placeholder", text)])


class _FrameSide(Margin):
    """A side of the composer's frame (``"│ "`` or ``" │"``) beside every line, in the frame's colour."""

    def __init__(self, ui: ChatUI, text: str) -> None:
        self.ui = ui
        self.text = text

    def get_width(self, get_ui_content: Callable[[], Any]) -> int:
        return get_cwidth(self.text)

    def create_margin(self, window_render_info: Any, width: int, height: int) -> list[tuple[str, str]]:
        style = self.ui._border_style()
        fragments: list[tuple[str, str]] = []
        for row in range(height):
            fragments += [("", "\n")] if row else []
            fragments.append((style, self.text))
        return fragments


# --------------------------------------------------------------------------- #
# The composer
# --------------------------------------------------------------------------- #


class ChatUI:
    """One live composer and at most one asynchronous command or model turn.

    ``on_submit`` owns session work and expected error presentation. Cancellation
    propagates back here only after controller/agent cleanup, so a fresh request
    cannot overlap it. ``footer`` returns the footer's :class:`~hailer.transcript.FooterData`;
    it is called on every redraw, so it must only read state the controller already has.
    """

    def __init__(
        self,
        console: Console,
        on_submit: Callable[[str], Awaitable[None]],
        footer: Callable[[], FooterData],
        *,
        pt_input: Any = None,
        pt_output: Any = None,
    ) -> None:
        self._raw_console = console
        self.console = _QueuedConsole(self, console)
        self._on_submit = on_submit
        self._footer = footer
        #: One symbol set for the transcript and the composer (ASCII when the console cannot encode Unicode).
        self.glyphs = console_glyphs(console)
        self._queue: asyncio.Queue[Callable[[], None] | None] = asyncio.Queue()
        self._pending: deque[Callable[[], None]] = deque()
        self._pending_lock = threading.Lock()
        self._draining_pending = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._writer: asyncio.Task[None] | None = None
        self._writer_error: BaseException | None = None
        self._active_task: asyncio.Task[None] | None = None
        self._startup_task: asyncio.Task[None] | None = None
        self._ticker: asyncio.Task[None] | None = None
        self._prepare: Callable[[], Awaitable[bool]] | None = None
        self._on_input_ready: Callable[[], None] | None = None
        self._preparing = False
        self._startup_failed = False
        self._pending_submission: str | None = None
        self._input_ready = False
        self._executing = False
        self._cancelling = False
        self._exit_requested = False
        # The live row: the activity label (when no step runs) and since when; the turn's steps.
        self._label = ""
        self._label_since = time.monotonic()
        self._steps = StepTracker()
        self._steps_cancelled = False
        # Session work in progress (startup's notebook and agent, a notebook switch): (label, detail,
        # since) in the order begun; the latest is shown, in the session rows' columns, when no step
        # runs (at startup the agent begins first and the notebook, which the user acts on, next).
        self._tasks: dict[object, tuple[str, str, float]] = {}
        self._submitted_at = time.monotonic()
        self._flash: tuple[str, float] | None = None
        self._wake = asyncio.Event()  # the ticker's sleep, cut short when a turn starts
        # What the transcript ends with ("blank", "row" or "text"), tracked by the writer itself.
        self._tail = "blank"
        self._output = _PersistentPasteOutput(pt_output or create_output())

        self.composer = TextArea(
            multiline=True,
            history=InMemoryHistory(),
            height=Dimension(min=1, max=6),
            dont_extend_height=True,
            wrap_lines=True,
            get_line_prefix=self._line_prefix,
            input_processors=[_Placeholder(self._placeholder, self.glyphs.ellipsis)],
        )
        self.buffer = self.composer.buffer
        self.application: Application[None] = Application(
            layout=Layout(self._container(), focused_element=self.composer),
            key_bindings=self._key_bindings(),
            full_screen=False,
            mouse_support=False,
            erase_when_done=True,
            input=pt_input,
            output=self._output,
            after_render=self._after_render,
            style=_STYLE,
        )
        self._idle_timeouts = (self.application.ttimeoutlen, self.application.timeoutlen)

    # -- state --------------------------------------------------------------- #

    @property
    def busy(self) -> bool:
        return (
            self._active_task is not None and not self._active_task.done()
        ) or (self._preparing and self._pending_submission is not None)

    @property
    def activity(self) -> str:
        """What the live row says is happening ("" while idle): the oldest running step's verb, else
        the latest session task's label (:meth:`begin_task`), else the latest activity
        (``set_activity`` or a ``status`` event)."""
        current = self._steps.current()
        if current is not None:
            return current[0].live
        return next(reversed(self._tasks.values()))[0] if self._tasks else self._label

    @property
    def flash_text(self) -> str:
        """The short hint shown in the frame's border instead of the key hints, or ""."""
        if self._flash is not None and time.monotonic() < self._flash[1]:
            return self._flash[0]
        return ""

    def _state(self) -> str:
        if self._startup_failed:
            return "failed"
        return "busy" if self.busy or self._preparing else "idle"

    def _moving(self) -> bool:
        return self.busy or self._preparing

    def configure_startup(
        self,
        prepare: Callable[[], Awaitable[bool]],
        *,
        on_input_ready: Callable[[], None] | None = None,
    ) -> None:
        """Prepare after rendering, while the composer accepts drafts and one message.

        The callback reports startup errors and returns false on failure. The UI
        owns its lifetime so it settles before the controller closes resources.
        """
        self._prepare = prepare
        self._on_input_ready = on_input_ready
        self._preparing = True
        self.set_activity("Preparing chat...")

    def _after_render(self, _application: Any) -> None:
        if self._input_ready:
            return
        self._input_ready = True
        if self._on_input_ready is not None:
            self._on_input_ready()
        if self._prepare is not None:
            self._startup_task = asyncio.create_task(self._prepare_chat())
        self._animate()

    async def _prepare_chat(self) -> None:
        assert self._prepare is not None
        ready = False
        try:
            ready = await self._prepare()
        except asyncio.CancelledError:
            # A startup operation can itself be cancelled; do not leave the
            # composer waiting forever for a readiness signal that cannot come.
            if not self._exit_requested:
                self.console.print(callout(self.glyphs, "warn", "Chat preparation cancelled."))
                self.exit()
            raise
        except Exception as exc:
            self.console.print(callout(self.glyphs, "fail", f"Unexpected error: {type(exc).__name__}: {exc}"))
        finally:
            self._preparing = False
            self._flash = None  # a "queued" hint is over either way
        if self._exit_requested:
            return
        self.gap()  # the startup rows end with a blank line, as a turn does
        self._startup_failed = not ready
        if not ready:
            self.set_activity("Ready")
            if self._pending_submission is not None:
                self.console.print(
                    Text("Your queued message was not sent and is kept in history (Up to recall).", style=MUTED)
                )
            return
        if self._pending_submission is not None:
            text, self._pending_submission = self._pending_submission, None
            self._launch_submission(text)
        else:
            self.set_activity("Ready")

    # -- keys ---------------------------------------------------------------- #

    def _key_bindings(self) -> KeyBindings:
        bindings = KeyBindings()

        @bindings.add("enter")
        def submit(_event: Any) -> None:
            self.submit()

        @bindings.add("escape", "enter")
        def newline(event: Any) -> None:
            event.current_buffer.insert_text("\n")

        @bindings.add("c-j")
        def ctrl_j(event: Any) -> None:
            # A newline key that never involves Esc (terminals without Option as Meta).
            event.current_buffer.insert_text("\n")

        @bindings.add("escape", filter=Condition(lambda: self.busy))
        def escape(_event: Any) -> None:
            # The same path as Ctrl+C during work; while idle Esc does nothing.
            self.cancel()

        @bindings.add("up")
        def up(event: Any) -> None:
            event.current_buffer.auto_up()

        @bindings.add("down")
        def down(event: Any) -> None:
            event.current_buffer.auto_down()

        @bindings.add("c-c")
        def interrupt(_event: Any) -> None:
            if self.busy:
                self.cancel()
            else:
                self.exit()

        @bindings.add("c-d")
        def eof(event: Any) -> None:
            if not event.current_buffer.text:
                if not self.busy or self._preparing:
                    self.exit()
            else:
                event.current_buffer.delete()

        @bindings.add("c-z")
        def windows_eof(event: Any) -> None:
            event.current_buffer.insert_text("\x1a")

        return bindings

    # -- submissions ----------------------------------------------------------- #

    def submit(self) -> None:
        """Submit the draft once, retaining it unchanged if work is still active."""
        text = self.buffer.text.strip()
        command = parse_command(text)
        if (self._preparing or self._startup_failed) and (
            text.startswith("\x1a") or command is not None and command.name in EXIT_COMMANDS
        ):
            self.exit()
            return
        g = self.glyphs
        if self.busy:
            self.flash(
                g.join("Starting up", "your first message is queued", "your draft is kept")
                if self._preparing else g.join("Still working", "esc cancels", "your draft is kept")
            )
            return
        if self._startup_failed:
            self.flash(g.join("Startup failed", "your draft is kept", "/exit to leave"))
            return
        if text.startswith("\x1a"):
            self.exit()
            return
        if not text:
            return
        self.buffer.reset(append_to_history=True)
        self.gap()
        self.console.print(user_message(g, text))
        self.console.print()
        if self._preparing:
            self._pending_submission = text
            self.flash(g.join("Queued", "it sends once startup finishes"))
            return
        self._launch_submission(text)

    def _launch_submission(self, text: str) -> None:
        self._cancelling = False
        self._executing = True
        self._steps = StepTracker()
        self._steps_cancelled = False
        self._submitted_at = time.monotonic()
        self.application.ttimeoutlen, self.application.timeoutlen = _TURN_TIMEOUTS
        self.set_activity("Thinking..." if not text.startswith("/") else "Working...")
        self._active_task = asyncio.create_task(self._submit(text))
        self._animate()
        self._wake.set()  # a ticker left from startup redraws at the turn's pace now

    async def _submit(self, text: str) -> None:
        try:
            await self._on_submit(text)
        except asyncio.CancelledError:
            self._on_ui_thread(self._report_cancelled)
        except Exception as exc:  # Controller normally formats its own errors.
            message = f"Unexpected error: {type(exc).__name__}: {exc}"

            def report() -> None:
                self._settle_steps()
                self.console.print(callout(self.glyphs, "fail", message))

            self._on_ui_thread(report)
        finally:
            self._executing = False

            def settle() -> None:
                self._settle_steps()  # steps a failed turn left open: effect unknown
                self.gap()

            self._on_ui_thread(settle)
            await self.flush()
            self._cancelling = False
            self._flash = None  # "Still working" is over
            self.application.ttimeoutlen, self.application.timeoutlen = self._idle_timeouts
            self.set_activity("Ready")
            if self._exit_requested:
                self._exit_application()

    def cancel(self) -> None:
        if self._preparing and self._pending_submission is not None:
            text, self._pending_submission = self._pending_submission, None
            if not self.buffer.text:
                self.buffer.insert_text(text)
            self.console.print(
                Text("Queued message cancelled before sending; kept in history (Up to recall).", style=MUTED)
            )
            self.set_activity("Preparing chat...")
            return
        if self.busy and self._executing and not self._cancelling:
            self._cancelling = True
            # The running steps end here for the user: the calls may still finish (and change the
            # notebook), so their rows say the effect is unknown.
            self._on_ui_thread(self._cancel_steps)
            self.set_activity("Cancelling...")
            assert self._active_task is not None
            # Give a just-created task its first tick so its cancellation and
            # finally blocks also run for a quick Enter followed by Ctrl+C.
            task = self._active_task

            def cancel_work() -> None:
                # A quick command may already be flushing its final output.
                # Cancelling that finalizer could prevent /exit from exiting.
                if self._active_task is task and self._executing:
                    task.cancel()

            asyncio.get_running_loop().call_soon(cancel_work)

    def _exit_application(self) -> None:
        if self.application.is_running and self.application.future is not None and not self.application.future.done():
            self.application.exit()

    def exit(self) -> None:
        """Exit after the current callback and pending output have finished."""
        if self._exit_requested:
            return
        self._exit_requested = True
        if not self.busy or self._preparing:
            # The writer calls this after all preceding transcript output.
            # Keeping exit in the queue avoids an unowned flush task if the
            # terminal fails while writing the farewell.
            self._enqueue(self._exit_application)

    # -- what the controller and the agent report -------------------------------- #

    def set_activity(self, text: str) -> None:
        def update() -> None:
            self._label = _label(text)
            self._label_since = time.monotonic()
            self._animate()
            self.application.invalidate()

        self._on_ui_thread(update)

    def begin_task(self, key: object, label: str, detail: str = "") -> None:
        """Show session work (``Notebook  opening analysis.py in your browser``) on the live row, in
        the session rows' columns, until :meth:`end_task` with the same ``key``; from any thread.
        Begun again, it keeps its start time and place. The latest task is shown; a running step
        goes first."""

        def update() -> None:
            since = self._tasks[key][2] if key in self._tasks else time.monotonic()
            self._tasks[key] = (single_line(label), single_line(detail), since)
            self._animate()
            self.application.invalidate()

        self._on_ui_thread(update)

    def end_task(self, key: object) -> None:
        def update() -> None:
            self._tasks.pop(key, None)
            self.application.invalidate()

        self._on_ui_thread(update)

    def flash(self, text: str, seconds: float = _FLASH_SECONDS) -> None:
        """Show ``text`` in place of the key hints for a moment (the activity stays as it is)."""

        def update() -> None:
            self._flash = (single_line(text), time.monotonic() + seconds)
            self.application.invalidate()
            if self._loop is not None:
                self._loop.call_later(seconds + 0.05, self._redraw)  # one redraw takes it away

        self._on_ui_thread(update)

    def _redraw(self) -> None:
        if self.application.is_running:
            self.application.invalidate()

    def on_event(self, event: AgentEvent) -> None:
        """A step starts on ``tool_call`` (keyed by its id) and is committed as one row on its
        ``tool_result``; deltas only change the live row; the answer is printed by ``finish``."""
        self._on_ui_thread(lambda: self._event(event))

    def _event(self, event: AgentEvent) -> None:
        if self._cancelling:
            return  # the open steps are already shown as cancelled and the row says "Cancelling"
        if event.kind == "tool_call":
            self._steps.call(event)
        elif event.kind == "tool_result":
            receipt = self._steps.result(event)
            if receipt is not None:
                self._commit(receipt)
            if not self._steps.running:
                self._set_label("Thinking")
        elif event.kind == "message_delta":
            if self._label != "Writing the answer":
                self._set_label("Writing the answer")
        elif event.kind == "status" and event.text:
            self._set_label(event.text)
        self._animate()
        self.application.invalidate()

    def _set_label(self, text: str) -> None:
        self._label = _label(text)
        self._label_since = time.monotonic()

    def _commit(self, receipt: Receipt) -> None:
        self._write((step_row(self.glyphs, receipt),), {}, kind="row")

    def settle_steps(self) -> None:
        """End the steps still open as "no result, effect unknown": the controller calls it when a
        turn fails, before it prints the error, so the rows come first."""
        self._on_ui_thread(self._settle_steps)

    def _settle_steps(self) -> None:
        for receipt in self._steps.finish():
            self._commit(receipt)

    def _cancel_steps(self) -> None:
        receipts = self._steps.cancel()
        self._steps_cancelled = self._steps_cancelled or bool(receipts)
        for receipt in receipts:
            self._commit(receipt)

    def _report_cancelled(self) -> None:
        self._cancel_steps()
        hints = ["The conversation continues; the model is told the last step's effect is unknown."] if self._steps_cancelled else []
        self._write((callout(self.glyphs, "warn", "Cancelled.", hints),), {}, kind="attached")

    def finish(self, summary: TurnSummary) -> None:
        """Commit the completed answer once, then the turn's steps, time and tokens; event deltas
        only affect the live row."""
        self._on_ui_thread(lambda: self._finish(summary))

    def _finish(self, summary: TurnSummary) -> None:
        self._settle_steps()
        g = self.glyphs
        elapsed = summary.duration_ms / 1000 if summary.duration_ms is not None else time.monotonic() - self._submitted_at
        stats = turn_stats(
            g, steps=summary.steps, elapsed=elapsed, input_tokens=summary.input_tokens, output_tokens=summary.output_tokens,
        )
        footer = self._footer_data()
        self.gap()
        self.console.print(reply(g, summary.final_response))
        self.console.print()
        self.console.print(turn_footer(g, stats, single_line(footer.notebook), footer.notebook_url))
        self.console.print()

    # -- the transcript writer ---------------------------------------------------- #

    def gap(self) -> None:
        """A blank line, unless the transcript already ends with one."""
        self._enqueue(lambda: self._print_now((), {}, "gap"))

    def _write(self, objects: tuple[Any, ...], kwargs: dict[str, Any], kind: str = "text") -> None:
        self._enqueue(lambda: self._print_now(objects, kwargs, kind))

    def _print_now(self, objects: tuple[Any, ...], kwargs: dict[str, Any], kind: str) -> None:
        """Print on the writer, in transcript order. ``kind``: "row" (a step), "attached" (follows
        steps directly), "gap" (a blank unless there is one) or "text" (a blank line first when it
        follows steps).

        Rich wraps what the composer prints at word boundaries inside the terminal width, so
        gutters indent wrapped lines, and so does a plain indented line (an error's hints). The
        console itself stays soft-wrapped for plain mode and pipes. A string with a URL is printed
        soft-wrapped: Rich would cut the URL into lines that no longer open or copy as one."""
        if kind == "gap":
            if self._tail != "blank":
                self._raw_console.print()
                self._tail = "blank"
            return
        if kind == "text" and objects and self._tail == "row":
            self._raw_console.print()
        if "soft_wrap" not in kwargs:
            url = any(isinstance(item, str) and "://" in item for item in objects)
            # crop=False: Rich lays renderables out inside the width anyway, and a line a Gutter
            # writes whole (a URL) must reach the terminal whole.
            kwargs = {"crop": False, **kwargs, "soft_wrap": url}
            if not url and kwargs.get("markup") is False:
                objects = tuple(_hanging(item) if isinstance(item, str) else item for item in objects)
        self._raw_console.print(*objects, **kwargs)
        self._tail = "blank" if not objects else ("row" if kind == "row" else "text")

    def _clear_now(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self._raw_console.clear(*args, **kwargs)
        self._tail = "blank"

    # -- layout ---------------------------------------------------------------- #

    def _container(self) -> HSplit:
        # One window per row, each drawing its whole line at the terminal's width: a row of
        # VSplits costs a width negotiation on every redraw. The frame's sides are margins of the
        # composer's own window.
        bar = self.glyphs.bar
        self.composer.window.left_margins = [_FrameSide(self, f"{bar} ")]
        self.composer.window.right_margins = [_FrameSide(self, f" {bar}")]

        def row(text: Callable[[], list[tuple[str, str]]]) -> Window:
            return Window(FormattedTextControl(text), height=1)

        # The live row lines up with the step rows above it, so a running step turns into its
        # finished row in place. It takes no space while idle.
        live = ConditionalContainer(
            row(self._live_line), filter=Condition(lambda: (self.busy or self._preparing) and bool(self.activity)),
        )
        return HSplit([live, row(self._top_border), self.composer, row(self._bottom_border), row(self._footer_row)])

    def _border_style(self) -> str:
        return f"class:frame.{self._state()}"

    def _top_border(self) -> list[tuple[str, str]]:
        tl, tr, _bl, _br = self.glyphs.corners
        return [(self._border_style(), tl + self.glyphs.rule * max(0, self._columns() - 2) + tr)]

    def _bottom_border(self) -> list[tuple[str, str]]:
        """The frame's bottom edge with the key hints (or a flash) set into it."""
        _tl, _tr, bl, br = self.glyphs.corners
        style, rule = self._border_style(), self.glyphs.rule
        hints = self._hints()
        rest = max(0, self._columns() - 3 - fragment_list_width(hints))
        return [(style, bl + rule), *hints, (style, rule * rest + br)]

    def _line_prefix(self, line: int, wrap: int) -> list[tuple[str, str]]:
        if line == 0 and wrap == 0:
            return [(f"class:prompt.{'idle' if self._state() == 'idle' else 'busy'}", f"{self.glyphs.prompt} ")]
        return [("", "  ")]

    def _placeholder(self) -> str:
        if self._startup_failed:
            return "Startup failed; /exit to leave"
        if self._preparing and self._pending_submission is None:
            return "Ask about your data; your first message waits for startup"
        if self.busy:
            return "Draft your next message"
        return "Ask about your data, or type / for commands"

    def _live(self) -> tuple[str, str, float, int]:
        """(label, detail, since, label width) for the live row."""
        current = self._steps.current()
        if current is not None:
            step, started, others = current
            return step.live, self.glyphs.join(step.target, f"+{others} more" if others else ""), started, LABEL_WIDTH
        if self._tasks:
            return (*next(reversed(self._tasks.values())), SESSION_LABEL_WIDTH)
        return self._label, "", self._label_since, LABEL_WIDTH

    def _live_line(self) -> list[tuple[str, str]]:
        """Spinner, activity and detail in the step rows' (or session rows') columns; the timer at
        the right edge."""
        label, detail, since, width = self._live()
        now = time.monotonic()
        frames = self.glyphs.spinner
        # While starting nothing animates: the spinner moves with the timer, once a second.
        tick = now / _FRAME_SECONDS if self._executing else now - since
        head = f"  {frames[int(tick) % len(frames)]} "
        timer = seconds(now - since) if now - since >= 1 else ""
        room = self._columns() - get_cwidth(head) - (get_cwidth(timer) + 1 if timer else 0)
        label = label.ljust(width)
        detail = f" {detail}" if detail else ""
        if get_cwidth(label) > room:
            label, detail = _clip(label, max(0, room)), ""
        elif get_cwidth(label + detail) > room:
            detail = _clip(detail, room - get_cwidth(label) - get_cwidth(self.glyphs.ellipsis)) + self.glyphs.ellipsis
        gap = max(0, room - get_cwidth(label + detail)) + (1 if timer else 0)
        return [
            ("class:spinner", head), ("class:activity", label), ("class:muted", detail),
            ("", " " * gap if timer else ""), ("class:muted", timer),
        ]

    def _columns(self) -> int:
        try:
            return self.application.output.get_size().columns
        except Exception:  # noqa: BLE001 - a closed terminal must not break a redraw
            return 80

    def _hints(self) -> list[tuple[str, str]]:
        g = self.glyphs
        flash = self.flash_text
        if flash:
            style, text = "class:hint.flash", flash
        else:
            style = "class:hint"
            if self._cancelling:
                text = "cancelling"
            elif self.busy and self._executing:
                text = g.join("esc cancel", "your draft is kept")
            elif self._preparing:
                text = g.join("starting", "your message is queued" if self._pending_submission is not None else "you can type now")
            elif self._startup_failed:
                text = g.join("startup failed", "/exit to leave")
            elif self.buffer.text:
                text = g.join("enter send", "alt+enter new line")
            else:
                text = g.join("/ commands", f"{g.up} history", "ctrl+c exit")
        room = max(0, self._columns() - 6)
        if get_cwidth(text) > room:
            text = _clip(text, room - get_cwidth(g.ellipsis)).rstrip() + g.ellipsis
        return [(style, f" {text} ")] if text else []

    def _footer_data(self) -> FooterData:
        try:
            data = self._footer()
        except Exception:  # noqa: BLE001 - the footer must never break a redraw
            return FooterData()
        return data if isinstance(data, FooterData) else FooterData()

    def _footer_row(self) -> list[tuple[str, str]]:
        """Notebook, kernel and model on the left, usage (or the turn's time) on the right. When
        the terminal is narrow, usage goes first, then the model, then a docker kernel; the
        unsafe-local warning stays."""
        g, data = self.glyphs, self._footer_data()
        notebook = single_line(data.notebook)
        warning = data.kernel == KERNEL_RUNTIME_UNSAFE_LOCAL
        if warning:
            kernel = [("class:warn", f"{g.warn} unsafe-local {g.sep} not isolated")]
        elif data.kernel == KERNEL_RUNTIME_DOCKER:
            kernel = [("class:muted", f"docker {g.sep} isolated")]
        else:
            kernel = [("class:muted", single_line(data.kernel))] if data.kernel else []
        dot = {True: [("class:ok", f"{g.on} ")], False: [("class:muted", f"{g.off} ")]}.get(data.notebook_open, [])
        groups: dict[str, list[tuple[str, str]]] = {}
        if notebook:
            groups["notebook"] = [*dot, ("", notebook)]
        if kernel:
            groups["kernel"] = kernel
        if data.model:
            groups["model"] = [("class:muted", single_line(data.model))]
        if self.busy and self._executing:
            right = [f"working {seconds(time.monotonic() - self._submitted_at)}"]
        else:
            right = [plural(data.turns, "turn")] + ([f"{tokens(data.tokens)} tokens"] if data.tokens else [])
        width = self._columns() - 2
        sep = ("class:muted", f"  {g.sep}  ")

        def left() -> list[tuple[str, str]]:
            return [part for index, group in enumerate(groups.values()) for part in ([sep] if index else []) + group]

        def length(fragments: list[tuple[str, str]]) -> int:
            return fragment_list_width(fragments)  # terminal cells, so wide characters count twice

        while length(left()) + get_cwidth(g.join(*right)) + (2 if right else 0) > width:
            if right:
                right.pop()
            elif "model" in groups:
                del groups["model"]
            elif "kernel" in groups and not warning:
                del groups["kernel"]
            elif "notebook" in groups:
                room = width - (length(left()) - get_cwidth(notebook))
                if room >= 4 and get_cwidth(notebook) > room:
                    groups["notebook"] = [*dot, ("", _clip(notebook, room - get_cwidth(g.ellipsis)) + g.ellipsis)]
                    break
                del groups["notebook"]
            else:
                break
        tail = g.join(*right)
        gap = max(2, width - length(left()) - get_cwidth(tail))
        return [("", " "), *left(), ("", " " * gap if tail else ""), ("class:muted", tail)]

    # -- the redraw ticker --------------------------------------------------------- #

    def _animate(self) -> None:
        """Redraw while work runs: every spinner frame during a turn, once a second while the chat
        prepares; nothing ticks while idle (a flash schedules its own redraw)."""
        if self._ticker is not None or self._loop is None or not self._moving():
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        self._ticker = asyncio.ensure_future(self._tick())

    async def _tick(self) -> None:
        try:
            while self._moving():
                if self.application.is_running:
                    self.application.invalidate()
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), _FRAME_SECONDS if self._executing else _STARTUP_TICK_SECONDS)
                except TimeoutError:
                    pass
            if self.application.is_running:
                self.application.invalidate()  # the frame, hints and live row settle
        finally:
            self._ticker = None

    # -- the UI-thread bridge ---------------------------------------------------- #

    def _on_ui_thread(self, callback: Callable[[], None]) -> None:
        # Commit order before scheduling the loop notification: a worker's
        # output/status must not be overtaken by a later main-thread callback
        # merely because its call_soon_threadsafe wake has not been delivered.
        with self._pending_lock:
            self._pending.append(callback)
        loop = self._loop
        if loop is not None:
            try:
                same_loop = asyncio.get_running_loop() is loop
            except RuntimeError:
                same_loop = False
            if same_loop:
                self._drain_pending()
            else:
                loop.call_soon_threadsafe(self._drain_pending)
        # Before startup or after shutdown, flush() delivers pending callbacks.

    def _drain_pending(self) -> None:
        """Apply the ordered bridge on the UI thread, never on a worker."""
        if self._draining_pending:
            return
        self._draining_pending = True
        try:
            while True:
                with self._pending_lock:
                    if not self._pending:
                        return
                    callback = self._pending.popleft()
                callback()
        finally:
            self._draining_pending = False

    def _enqueue(self, callback: Callable[[], None]) -> None:
        self._on_ui_thread(lambda: self._queue.put_nowait(callback))

    def _print_transcript(self, callback: Callable[[], None]) -> None:
        # run_in_terminal normally enters cooked mode for a subprocess. We
        # only print output: retain raw input while a large reply is written,
        # so typing/pasting cannot echo escape markers or enter canonical input
        # buffering. The temporarily detached reader consumes those bytes when
        # run_in_terminal reattaches it immediately afterwards.
        if self.application.is_running:
            with self.application.input.raw_mode():
                callback()
        else:
            callback()

    async def _write_transcript(self) -> None:
        try:
            while True:
                batch = [await self._queue.get()]
                # A user message, a final answer or a table often comes from
                # several console.print calls. Commit them in one redraw.
                while len(batch) < 100 and not self._queue.empty() and batch[-1] is not None:
                    batch.append(self._queue.get_nowait())
                try:
                    def write_batch() -> None:
                        for callback in batch:
                            if callback is not None:
                                callback()

                    if any(callback is not None for callback in batch):
                        await run_in_terminal(lambda: self._print_transcript(write_batch))
                    if batch[-1] is None:
                        return
                finally:
                    for _callback in batch:
                        self._queue.task_done()
        except Exception as exc:
            self._writer_error = exc
            # Release flush waiters even when the terminal has disappeared.
            while not self._queue.empty():
                self._queue.get_nowait()
                self._queue.task_done()
            if self.application.is_running and self.application.future is not None and not self.application.future.done():
                self.application.exit(exception=exc)

    async def flush(self) -> None:
        """Wait for queued output, or print it directly before/after UI startup."""
        # Include callbacks already posted by workers even when their loop
        # notifications are still pending; queue.join alone cannot see them.
        self._drain_pending()
        if self._writer_error is not None:
            raise self._writer_error
        if self._writer is None:
            while not self._queue.empty():
                callback = self._queue.get_nowait()
                try:
                    if callback is not None:
                        callback()
                finally:
                    self._queue.task_done()
        else:
            await self._queue.join()
            if self._writer_error is not None:
                raise self._writer_error

    @contextmanager
    def _route_logs(self):
        # Existing handlers hold their own stderr reference; patch_stdout alone
        # would not catch them. Preserve formatters, filters and file handlers.
        restored: list[tuple[logging.StreamHandler, Any]] = []
        loggers = [logging.getLogger(), *logging.Logger.manager.loggerDict.values()]
        handlers = {
            handler
            for logger in loggers if isinstance(logger, logging.Logger)
            for handler in logger.handlers
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
        }
        original_last_resort = logging.lastResort
        try:
            for handler in handlers:
                original = handler.stream
                handler.setStream(_LogStream(self, original))
                restored.append((handler, original))
            if original_last_resort is not None:
                replacement = logging.StreamHandler(_LogStream(self, getattr(original_last_resort, "stream", None)))
                replacement.setLevel(original_last_resort.level)
                replacement.setFormatter(original_last_resort.formatter)
                for log_filter in original_last_resort.filters:
                    replacement.addFilter(log_filter)
                logging.lastResort = replacement
            yield
        finally:
            logging.lastResort = original_last_resort
            for handler, original in restored:
                handler.setStream(original)

    async def _shutdown(self) -> None:
        failure: BaseException | None = None
        self._exit_requested = True
        if self._startup_task is not None:
            if not self._startup_task.done():
                self._startup_task.cancel()
            try:
                await self._startup_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                failure = exc
        if self._active_task is not None:
            if self.busy:
                self.cancel()
            try:
                # Retrieve failures even if the task finished before shutdown
                # began (for example, a terminal error while flushing a reply).
                await self._active_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                failure = exc
        self.gap()  # what follows the composer (the CLI's exit summary) starts after a blank line
        try:
            await self.flush()
        except Exception as exc:
            failure = failure or exc
        finally:
            if self._writer is not None:
                if not self._writer.done():
                    self._queue.put_nowait(None)
                await self._writer
                self._writer = None
        if failure is not None:
            raise failure

    async def _finish_shutdown(self) -> None:
        # A second process cancellation must not interrupt agent/history repair
        # or strand the terminal writer. Finish the whole cleanup, then preserve
        # the caller's cancellation rather than pretending run() succeeded.
        cleanup = asyncio.create_task(self._shutdown())
        interrupted = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                interrupted = True
        cleanup.result()
        if interrupted:
            raise asyncio.CancelledError

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()

        def ready() -> None:
            # pre_run inherits the application's context; run_in_terminal can
            # then find the correct application from this writer task.
            self._drain_pending()
            self._writer = asyncio.create_task(self._write_transcript())

        try:
            with self._route_logs():
                try:
                    await self.application.run_async(pre_run=ready, set_exception_handler=False)
                except EOFError:
                    pass  # input closed: the CLI's exit summary follows
                finally:
                    await self._finish_shutdown()
        finally:
            self._loop = None
            ticker, self._ticker = self._ticker, None
            if ticker is not None:
                ticker.cancel()
            self._output.close()
