"""The live composer uses pipe input and a recorded VT100 terminal; no network."""

from __future__ import annotations

import asyncio
import io
import logging
import threading
from contextlib import contextmanager

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output.vt100 import Vt100_Output
from rich.console import Console

from hailer.chat_ui import ChatUI
from hailer.models import AgentEvent

PASTE_ON, PASTE_OFF = "\x1b[?2004h", "\x1b[?2004l"


@contextmanager
def terminal(on_submit, context=lambda: "Notebook: analysis.py | Model: test | Context: 2 files"):
    with create_pipe_input() as keys:
        screen, transcript = io.StringIO(), io.StringIO()
        size = [Size(rows=24, columns=100)]
        output = Vt100_Output(screen, lambda: size[0], term="xterm-256color")
        console = Console(file=transcript, force_terminal=False, color_system=None, width=90)
        ui = ChatUI(console, on_submit, context, pt_input=keys, pt_output=output)
        yield ui, keys, screen, transcript, size


async def until(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.01)


async def start(ui):
    running = asyncio.create_task(ui.run())
    await until(lambda: ui.application.is_running and ui.application.render_counter > 0)
    return running


def test_slow_turn_accepts_draft_and_paste_stays_enabled_through_output():
    async def scenario():
        release = asyncio.Event()
        submissions = []

        async def submit(text):
            submissions.append(text)
            ui.on_event(AgentEvent("tool_call", "marimo_execute"))
            await release.wait()
            ui.on_event(AgentEvent("message_delta", "interim text must not be printed"))
            ui.finish("**A completed reply**\n\n- one\n- two")

        with terminal(submit) as (ui, keys, screen, transcript, _size):
            ui.console.print("Conversation resumed.", markup=False)
            running = await start(ui)
            keys.send_text("\x1b[200~first line\r\nsecond line\x1b[201~\r")
            await until(lambda: len(submissions) == 1)
            await ui.flush()
            assert submissions == ["first line\nsecond line"]
            assert PASTE_ON in screen.getvalue() and PASTE_OFF not in screen.getvalue()
            keys.send_text("next draft\r")
            await until(lambda: "draft is kept" in ui.activity)
            assert ui.buffer.text == "next draft"
            assert len(submissions) == 1
            release.set()
            await until(lambda: not ui.busy)
            assert ui.buffer.text == "next draft"
            assert PASTE_OFF not in screen.getvalue()
            result = transcript.getvalue()
            assert result.index("Conversation resumed.") < result.index("You") < result.index("Hailer")
            assert result.count("first line") == result.count("A completed reply") == 1
            assert "interim text" not in result
            ui.exit()
            await asyncio.wait_for(running, 5)
            assert screen.getvalue().count(PASTE_ON) == screen.getvalue().count(PASTE_OFF) == 1

    asyncio.run(scenario())


def test_cancellation_waits_for_cleanup_ignores_repeat_and_allows_followup():
    async def scenario():
        cleanup_started, release_cleanup = asyncio.Event(), asyncio.Event()
        submissions = []

        async def submit(text):
            submissions.append(text)
            if len(submissions) == 1:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cleanup_started.set()
                    await release_cleanup.wait()
                    raise
            ui.finish("Follow-up succeeded")

        with terminal(submit) as (ui, keys, _screen, transcript, _size):
            running = await start(ui)
            keys.send_text("slow\r")
            await until(lambda: len(submissions) == 1)
            keys.send_text("follow-up\x03")
            await asyncio.wait_for(cleanup_started.wait(), 5)
            keys.send_text("\x03\r")
            await until(lambda: "draft is kept" in ui.activity)
            assert ui.busy and ui.buffer.text == "follow-up"
            assert submissions == ["slow"]
            release_cleanup.set()
            await until(lambda: not ui.busy)
            assert transcript.getvalue().count("Interrupted.") == 1
            keys.send_text("\r")
            await until(lambda: len(submissions) == 2 and not ui.busy)
            assert submissions == ["slow", "follow-up"]
            assert "Follow-up succeeded" in transcript.getvalue()
            keys.send_text("\x04")
            await asyncio.wait_for(running, 5)

    asyncio.run(scenario())


def test_failure_preserves_draft_and_restores_terminal_modes():
    async def scenario():
        release = asyncio.Event()

        async def submit(_text):
            await release.wait()
            raise ValueError("simulated failure")

        with terminal(submit) as (ui, keys, screen, transcript, _size):
            running = await start(ui)
            keys.send_text("fail\r")
            await until(lambda: ui.busy)
            keys.send_text("my draft")
            await until(lambda: ui.buffer.text == "my draft")
            release.set()
            await until(lambda: not ui.busy)
            assert ui.buffer.text == "my draft"
            assert "ValueError: simulated failure" in transcript.getvalue()
            keys.send_text("\x03")
            await asyncio.wait_for(running, 5)
            output = screen.getvalue()
            assert output.rfind(PASTE_OFF) > output.rfind(PASTE_ON)
            assert "\x1b[?1049h" not in output
            assert "\x1b[?1000h" not in output and "\x1b[?1003h" not in output
            assert "\x1b[6n" not in output

    asyncio.run(scenario())


def test_multiline_history_newline_resize_and_context_refresh():
    async def scenario():
        submissions = []
        context = ["Notebook: first.py | Model: test"]

        async def submit(text):
            submissions.append(text)
            context[0] = "Notebook: next.py | Model: changed"

        with terminal(submit, lambda: context[0]) as (ui, keys, screen, _transcript, size):
            running = await start(ui)
            keys.send_text("one\x1b\rtwo\r")
            await until(lambda: len(submissions) == 1 and not ui.busy)
            assert submissions == ["one\ntwo"]
            await until(lambda: "next.py" in screen.getvalue())
            keys.send_text("\x1b[A")
            await until(lambda: ui.buffer.text == "one\ntwo")
            # Up within a multiline message moves the cursor before recalling
            # earlier history. Down at the bottom restores the unsent draft.
            keys.send_text("\x1b[A")
            await until(lambda: ui.buffer.document.cursor_position_row == 0)
            assert ui.buffer.text == "one\ntwo"
            keys.send_text("\x1b[B\x1b[B")
            await until(lambda: ui.buffer.text == "")
            pasted = "\n".join(f"line {index} Unicode \u03bb" for index in range(30))
            keys.send_text(f"\x1b[200~{pasted}\x1b[201~")
            await until(lambda: ui.buffer.text == pasted)
            size[0] = Size(rows=15, columns=32)
            ui.application.invalidate()
            await until(lambda: ui.composer.window.render_info is not None and ui.composer.window.render_info.window_width < 32)
            assert ui.composer.window.render_info.window_height <= 6
            assert ui.buffer.text == pasted
            keys.send_text("\x03")
            await asyncio.wait_for(running, 5)

    asyncio.run(scenario())


@pytest.mark.parametrize("exit_keys", ["\x03", "\x04", "\x1a\r"])
def test_idle_keyboard_exits(exit_keys):
    async def scenario():
        async def submit(_text):
            pytest.fail("An exit gesture must not submit a request")

        with terminal(submit) as (ui, keys, screen, transcript, _size):
            running = await start(ui)
            keys.send_text(exit_keys)
            await asyncio.wait_for(running, 5)
            assert "Bye." in transcript.getvalue()
            assert PASTE_OFF in screen.getvalue()

    asyncio.run(scenario())


def test_logs_from_worker_are_redacted_serialized_and_handler_restored(monkeypatch):
    async def scenario():
        original_stream = io.StringIO()
        handler = logging.StreamHandler(original_stream)
        logger = logging.getLogger("hailer.test_chat_ui")
        logger.addHandler(handler)
        old_propagate, old_level = logger.propagate, logger.level
        logger.propagate, logger.level = False, logging.DEBUG
        try:
            async def submit(_text):
                pass

            with terminal(submit) as (ui, keys, _screen, transcript, _size):
                running = await start(ui)
                keys.send_text("retained")
                await until(lambda: ui.buffer.text == "retained")
                await asyncio.to_thread(logger.warning, "Token: secret-value-123")
                ui.console.print("After log", markup=False)
                await ui.flush()
                assert "secret-value-123" not in transcript.getvalue()
                assert transcript.getvalue().index("<redacted>") < transcript.getvalue().index("After log")
                assert original_stream.getvalue() == ""
                assert ui.buffer.text == "retained"
                keys.send_text("\x03")
                await asyncio.wait_for(running, 5)
                assert handler.stream is original_stream
        finally:
            logger.removeHandler(handler)
            logger.propagate, logger.level = old_propagate, old_level

    monkeypatch.setenv("TEST_UI_KEY", "secret-value-123")
    asyncio.run(scenario())


def test_worker_output_precedes_main_output_before_thread_notification_is_delivered():
    async def scenario():
        async def submit(_text):
            pass

        with terminal(submit) as (ui, _keys, _screen, transcript, _size):
            running = await start(ui)

            def worker_print(text):
                worker = threading.Thread(target=lambda: ui.console.print(text, markup=False))
                worker.start()
                # Hold this event-loop turn until the worker has enqueued its
                # output. Its call_soon_threadsafe notification cannot run yet.
                worker.join(timeout=5)
                assert not worker.is_alive()

            worker_print("Worker first")
            ui.console.print("Main second", markup=False)
            await ui.flush()
            assert transcript.getvalue() == "Worker first\nMain second\n"

            worker_print("Flush must observe this worker output")
            # No main-thread print wakes the writer: flush must account for
            # already-enqueued output whose notification is still pending.
            await ui.flush()
            assert transcript.getvalue().endswith("Flush must observe this worker output\n")
            ui.exit()
            await asyncio.wait_for(running, 5)

    asyncio.run(scenario())


def test_delayed_worker_status_cannot_overwrite_later_ready_status():
    async def scenario():
        async def submit(_text):
            pass

        with terminal(submit) as (ui, _keys, _screen, _transcript, _size):
            running = await start(ui)
            worker = threading.Thread(target=lambda: ui.set_activity("Working..."))
            worker.start()
            worker.join(timeout=5)
            assert not worker.is_alive()
            ui.set_activity("Ready")
            # Let the earlier worker's delayed notification run afterwards.
            await asyncio.sleep(0)
            assert ui.activity == "Ready"
            ui.exit()
            await asyncio.wait_for(running, 5)

    asyncio.run(scenario())


def test_flush_before_run_handles_startup_notices():
    async def submit(_text):
        pass

    with terminal(submit) as (ui, _keys, screen, transcript, _size):
        ui.console.print("Startup warning", markup=False)
        asyncio.run(ui.flush())
        assert transcript.getvalue() == "Startup warning\n"
        assert PASTE_ON not in screen.getvalue()


def test_terminal_write_failure_exits_without_hanging_and_restores_modes(monkeypatch):
    async def scenario():
        async def submit(_text):
            pass

        with terminal(submit) as (ui, _keys, screen, _transcript, _size):
            running = await start(ui)

            def failed_write(*_args, **_kwargs):
                raise OSError("terminal closed")

            monkeypatch.setattr(ui._raw_console, "print", failed_write)
            ui.console.print("cannot write")
            ui.console.print("also queued")
            with pytest.raises(OSError, match="terminal closed"):
                await asyncio.wait_for(running, 5)
            assert PASTE_OFF in screen.getvalue()

    asyncio.run(scenario())


def test_closing_input_during_turn_cleans_up_task():
    async def scenario():
        cleaned = asyncio.Event()

        async def submit(_text):
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        with terminal(submit) as (ui, keys, screen, _transcript, _size):
            running = await start(ui)
            keys.send_text("slow\r")
            await until(lambda: ui.busy)
            keys.close()
            await asyncio.wait_for(running, 5)
            assert cleaned.is_set()
            assert not ui.busy
            assert PASTE_OFF in screen.getvalue()

    asyncio.run(scenario())


def test_transcript_rendering_retains_raw_input_mode(monkeypatch):
    async def scenario():
        async def submit(_text):
            pass

        with terminal(submit) as (ui, keys, _screen, _transcript, _size):
            modes = []

            @contextmanager
            def mode(name):
                modes.append(name)
                try:
                    yield
                finally:
                    modes.pop()

            monkeypatch.setattr(keys, "raw_mode", lambda: mode("raw"))
            monkeypatch.setattr(keys, "cooked_mode", lambda: mode("cooked"))
            running = await start(ui)
            observed = []
            ui._enqueue(lambda: observed.append(modes[-1]))
            await ui.flush()
            assert observed == ["raw"]
            ui.exit()
            await asyncio.wait_for(running, 5)

    asyncio.run(scenario())


def test_rapid_interrupt_does_not_cancel_exit_output_finalizer():
    async def scenario():
        async def submit(text):
            assert text == "/exit"
            ui.console.print("Bye.")
            ui.exit()

        with terminal(submit) as (ui, keys, screen, transcript, _size):
            running = await start(ui)
            keys.send_text("/exit\r\x03")
            await asyncio.wait_for(running, 5)
            assert "Bye." in transcript.getvalue()
            assert PASTE_OFF in screen.getvalue()

    asyncio.run(scenario())


def test_reply_write_failure_retrieves_completed_turn_exception(monkeypatch):
    async def scenario():
        errors = []
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _loop, context: errors.append(context))

        async def submit(_text):
            ui.finish("The final reply")

        with terminal(submit) as (ui, keys, screen, _transcript, _size):
            running = await start(ui)

            def failed_write(*_args, **_kwargs):
                raise OSError("terminal closed during reply")

            monkeypatch.setattr(ui._raw_console, "print", failed_write)
            keys.send_text("hello\r")
            with pytest.raises(OSError, match="terminal closed during reply"):
                await asyncio.wait_for(running, 5)
            assert ui._active_task is not None
            assert not ui._active_task._log_traceback
            assert not errors
            assert PASTE_OFF in screen.getvalue()

    asyncio.run(scenario())


def test_repeated_external_cancellation_finishes_async_turn_cleanup():
    async def scenario():
        cleanup_started, release_cleanup, cleaned = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def submit(_text):
            try:
                await asyncio.Event().wait()
            finally:
                cleanup_started.set()
                await release_cleanup.wait()
                cleaned.set()

        with terminal(submit) as (ui, keys, screen, _transcript, _size):
            running = await start(ui)
            keys.send_text("slow\r")
            await until(lambda: ui.busy)
            running.cancel()
            await asyncio.wait_for(cleanup_started.wait(), 5)
            running.cancel()
            await asyncio.sleep(0)
            assert not cleaned.is_set()
            assert not running.done()
            release_cleanup.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(running, 5)
            assert cleaned.is_set()
            assert not ui.busy
            assert ui._writer is None
            assert PASTE_OFF in screen.getvalue()

    asyncio.run(scenario())


def test_idle_exit_write_failure_has_no_unowned_task(monkeypatch):
    async def scenario():
        errors = []
        asyncio.get_running_loop().set_exception_handler(lambda _loop, context: errors.append(context))

        async def submit(_text):
            pass

        with terminal(submit) as (ui, keys, screen, _transcript, _size):
            running = await start(ui)

            def failed_write(*_args, **_kwargs):
                raise OSError("terminal closed during exit")

            monkeypatch.setattr(ui._raw_console, "print", failed_write)
            keys.send_text("\x03")
            with pytest.raises(OSError, match="terminal closed during exit"):
                await asyncio.wait_for(running, 5)
            await asyncio.sleep(0)
            assert errors == []
            assert PASTE_OFF in screen.getvalue()

    asyncio.run(scenario())
