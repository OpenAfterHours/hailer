"""Inline terminal chat: native scrollback above a persistent, editable composer.

Only the writer task touches the transcript while the application is running.
The controller can keep using synchronous ``console.print`` calls (including
from a notebook worker thread); the facade queues them for that writer.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from typing import Any

from prompt_toolkit.application import Application, run_in_terminal
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.output import create_output
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame, TextArea
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from hailer.log import redact
from hailer.models import AgentEvent


def _single_line(text: str) -> str:
    """Keep externally supplied labels from adding terminal controls or rows."""
    return " ".join("".join(c for c in text if c.isprintable() or c.isspace()).split())


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
    """The small subset of Rich Status used by existing slash commands."""

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
        self._ui._enqueue(lambda: self._console.print(*objects, **kwargs))

    def clear(self, *args: Any, **kwargs: Any) -> None:
        self._ui._enqueue(lambda: self._console.clear(*args, **kwargs))

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


class ChatUI:
    """One live composer and at most one asynchronous command or model turn.

    ``on_submit`` owns session work and expected error presentation. Cancellation
    propagates back here only after controller/agent cleanup, so a fresh request
    cannot overlap it. ``context`` supplies a short, current context strip.
    """

    def __init__(
        self,
        console: Console,
        on_submit: Callable[[str], Awaitable[None]],
        context: Callable[[], str],
        *,
        pt_input: Any = None,
        pt_output: Any = None,
    ) -> None:
        self._raw_console = console
        self.console = _QueuedConsole(self, console)
        self._on_submit = on_submit
        self._context = context
        self._queue: asyncio.Queue[Callable[[], None] | None] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._writer: asyncio.Task[None] | None = None
        self._writer_error: BaseException | None = None
        self._active_task: asyncio.Task[None] | None = None
        self._executing = False
        self._cancelling = False
        self._exit_requested = False
        self.activity = "Ready"
        self._output = _PersistentPasteOutput(pt_output or create_output())

        self.composer = TextArea(
            multiline=True,
            history=InMemoryHistory(),
            height=Dimension(min=1, max=6),
            dont_extend_height=True,
            wrap_lines=True,
        )
        self.buffer = self.composer.buffer
        context_line = Window(
            FormattedTextControl(lambda: [("class:context", _single_line(self._context()))]),
            height=1,
        )
        activity_line = Window(
            FormattedTextControl(lambda: [("class:activity", self.activity)]), height=1,
        )
        help_line = Window(
            FormattedTextControl(
                "Enter send | Alt+Enter newline | /help commands | Ctrl+C cancel/exit"
            ),
            height=1,
            style="class:help",
        )
        self.application: Application[None] = Application(
            layout=Layout(HSplit([context_line, activity_line, Frame(self.composer), help_line])),
            key_bindings=self._key_bindings(),
            full_screen=False,
            mouse_support=False,
            erase_when_done=True,
            input=pt_input,
            output=self._output,
            style=Style.from_dict({
                "context": "ansibrightblack",
                "activity": "ansicyan",
                "help": "ansibrightblack",
                "frame.border": "ansibrightblack",
            }),
        )

    @property
    def busy(self) -> bool:
        return self._active_task is not None and not self._active_task.done()

    def _key_bindings(self) -> KeyBindings:
        bindings = KeyBindings()

        @bindings.add("enter")
        def submit(_event: Any) -> None:
            self.submit()

        @bindings.add("escape", "enter")
        def newline(event: Any) -> None:
            event.current_buffer.insert_text("\n")

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
                self.console.print("Bye.", markup=False)
                self.exit()

        @bindings.add("c-d")
        def eof(event: Any) -> None:
            if not event.current_buffer.text:
                if not self.busy:
                    self.console.print("Bye.", markup=False)
                    self.exit()
            else:
                event.current_buffer.delete()

        @bindings.add("c-z")
        def windows_eof(event: Any) -> None:
            event.current_buffer.insert_text("\x1a")

        return bindings

    def submit(self) -> None:
        """Submit the draft once, retaining it unchanged if work is still active."""
        if self.busy:
            self.set_activity("Still working. Ctrl+C cancels; your draft is kept.")
            return
        text = self.buffer.text.strip()
        if text.startswith("\x1a"):
            self.console.print("Bye.", markup=False)
            self.exit()
            return
        if not text:
            return
        self.buffer.reset(append_to_history=True)
        self.console.print(Text("You", style="bold cyan"))
        self.console.print(text, markup=False)
        self.console.print()
        self._cancelling = False
        self._executing = True
        self.set_activity("Thinking..." if not text.startswith("/") else "Working...")
        self._active_task = asyncio.create_task(self._submit(text))

    async def _submit(self, text: str) -> None:
        try:
            await self._on_submit(text)
        except asyncio.CancelledError:
            self.console.print("Interrupted.", markup=False)
        except Exception as exc:  # Controller normally formats its own errors.
            self.console.print(f"Unexpected error: {type(exc).__name__}: {exc}", style="red", markup=False)
        finally:
            self._executing = False
            await self.flush()
            self._cancelling = False
            self.set_activity("Ready")
            if self._exit_requested:
                self._exit_application()

    def cancel(self) -> None:
        if self.busy and self._executing and not self._cancelling:
            self._cancelling = True
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
        if not self.busy:
            # The writer calls this after all preceding transcript output.
            # Keeping exit in the queue avoids an unowned flush task if the
            # terminal fails while writing the farewell.
            self._enqueue(self._exit_application)

    def set_activity(self, text: str) -> None:
        def update() -> None:
            self.activity = _single_line(text)
            self.application.invalidate()

        self._on_ui_thread(update)

    def on_event(self, event: AgentEvent) -> None:
        if event.kind == "message_delta":
            self.set_activity("Writing reply...")
        elif event.kind == "tool_call":
            self.set_activity(f"Using: {event.text}")
        elif event.kind == "status" and event.text:
            self.set_activity(event.text)

    def finish(self, final_response: str) -> None:
        """Commit the completed answer once; event deltas only affect activity."""
        self.console.print(Text("Hailer", style="bold"))
        final = (final_response or "").strip()
        self.console.print(Markdown(final) if final else Text("(no reply)", style="dim"))
        self.console.print()

    def _on_ui_thread(self, callback: Callable[[], None]) -> None:
        if self._loop is not None:
            try:
                same_loop = asyncio.get_running_loop() is self._loop
            except RuntimeError:
                same_loop = False
            if not same_loop:
                self._loop.call_soon_threadsafe(callback)
                return
        callback()

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
            self._writer = asyncio.create_task(self._write_transcript())

        try:
            with self._route_logs():
                try:
                    await self.application.run_async(pre_run=ready, set_exception_handler=False)
                except EOFError:
                    self.console.print("Bye.", markup=False)
                finally:
                    await self._finish_shutdown()
        finally:
            self._loop = None
            self._output.close()
