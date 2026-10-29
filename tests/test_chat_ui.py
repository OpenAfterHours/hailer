"""The live composer uses pipe input and a recorded VT100 terminal; no network."""

from __future__ import annotations

import asyncio
import io
import logging
import re
import threading
from contextlib import contextmanager

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output.vt100 import Vt100_Output
from rich.console import Console

from hailer.chat_ui import ChatUI
from hailer.models import AgentEvent, TurnSummary
from hailer.transcript import ASCII, UNICODE, FooterData, glyphs_for

PASTE_ON, PASTE_OFF = "\x1b[?2004h", "\x1b[?2004l"
FOOTER = FooterData(notebook="analysis.py", notebook_open=True, kernel="docker", model="test", turns=2, tokens=1500)


@contextmanager
def terminal(on_submit, footer=lambda: FOOTER, console=None):
    with create_pipe_input() as keys:
        screen, transcript = io.StringIO(), io.StringIO()
        size = [Size(rows=24, columns=100)]
        output = Vt100_Output(screen, lambda: size[0], term="xterm-256color")
        console = console or Console(file=transcript, force_terminal=False, color_system=None, width=90)
        ui = ChatUI(console, on_submit, footer, pt_input=keys, pt_output=output)
        yield ui, keys, screen, transcript, size


def summary(text, **kwargs):
    return TurnSummary(final_response=text, thread_id="thread", **kwargs)


def call(tool, call_id, **args):
    return AgentEvent("tool_call", tool, {"arguments": str(args)[:80], "args": args, "id": call_id})


def result(tool, call_id, summary="", seconds=0.5, ok=True):
    return AgentEvent("tool_result", tool, {"id": call_id, "ok": ok, "summary": summary, "seconds": seconds})


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
            ui.on_event(call("marimo_execute", "call_1", code="df.head()"))
            await release.wait()
            ui.on_event(result("marimo_execute", "call_1", "5 rows"))
            ui.on_event(AgentEvent("message_delta", "interim text must not be printed"))
            ui.finish(summary("**A completed reply**\n\n- one\n- two", steps=1))

        with terminal(submit) as (ui, keys, screen, transcript, _size):
            ui.console.print("Conversation resumed.", markup=False)
            running = await start(ui)
            keys.send_text("\x1b[200~first line\r\nsecond line\x1b[201~\r")
            await until(lambda: len(submissions) == 1)
            await ui.flush()
            assert submissions == ["first line\nsecond line"]
            assert PASTE_ON in screen.getvalue() and PASTE_OFF not in screen.getvalue()
            keys.send_text("next draft\r")
            await until(lambda: "draft is kept" in ui.flash_text)
            assert ui.buffer.text == "next draft"
            assert len(submissions) == 1
            release.set()
            await until(lambda: not ui.busy)
            assert ui.buffer.text == "next draft"
            assert PASTE_OFF not in screen.getvalue()
            output = transcript.getvalue()
            assert (
                output.index("Conversation resumed.") < output.index("▌ first line") < output.index("✓ Ran code")
                < output.index("● A completed reply")
            )
            assert output.count("first line") == output.count("A completed reply") == 1
            assert "interim text" not in output
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
            ui.finish(summary("Follow-up succeeded"))

        with terminal(submit) as (ui, keys, _screen, transcript, _size):
            running = await start(ui)
            keys.send_text("slow\r")
            await until(lambda: len(submissions) == 1)
            keys.send_text("follow-up\x03")
            await asyncio.wait_for(cleanup_started.wait(), 5)
            keys.send_text("\x03\r")
            await until(lambda: "draft is kept" in ui.flash_text)
            assert ui.busy and ui.buffer.text == "follow-up"
            assert submissions == ["slow"]
            release_cleanup.set()
            await until(lambda: not ui.busy)
            assert transcript.getvalue().count("Cancelled.") == 1
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


def test_multiline_history_newline_resize_and_footer_refresh():
    async def scenario():
        submissions = []
        footer = [FooterData(notebook="first.py", model="test")]

        async def submit(text):
            submissions.append(text)
            footer[0] = FooterData(notebook="next.py", model="changed")

        with terminal(submit, lambda: footer[0]) as (ui, keys, screen, _transcript, size):
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
            assert ui.activity == ""  # ready: the live row is hidden, not "Working"
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
            ui.finish(summary("The final reply"))

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


# --------------------------------------------------------------------------- #
# Steps, receipts, the live row, the footer and the glyphs
# --------------------------------------------------------------------------- #


def rows(output):
    """The transcript's lines without trailing padding."""
    return [line.rstrip() for line in output.splitlines()]


def test_receipts_follow_ids_for_parallel_and_failed_steps_and_the_reply_prints_once():
    async def scenario():
        live = []

        async def submit(_text):
            ui.on_event(AgentEvent("status", "Thinking"))
            ui.on_event(call("notebook_cells", "a"))
            ui.on_event(call("marimo_execute", "b", code="x = 1\ny = 2\n"))
            live.append(ui._live()[:2])
            ui.on_event(result("marimo_execute", "b", "NameError", 1.3, ok=False))  # the later call ends first
            live.append(ui._live()[:2])
            ui.on_event(result("notebook_cells", "a", "6 cells", 0.4))
            ui.on_event(result("notebook_cells", "a", "a second answer", 9.0))  # already closed: no row
            ui.on_event(call("load_skill", "c", name="example-skill"))
            ui.on_event(result("load_skill", "c", "example-skill, 3 files", 0.2))  # repeats the target
            live.append(ui.activity)
            for _ in range(3):
                ui.on_event(AgentEvent("message_delta", "streamed words are not printed"))
            live.append(ui.activity)
            ui.finish(summary("**Done.**", steps=3, duration_ms=2500, input_tokens=3140, output_tokens=612))

        with terminal(submit) as (ui, keys, _screen, transcript, _size):
            running = await start(ui)
            keys.send_text("go\r")
            await until(lambda: "Done." in transcript.getvalue() and not ui.busy)
            assert live == [
                ("Reading the notebook", "+1 more"), ("Reading the notebook", ""), "Thinking", "Writing the answer",
            ]
            lines = rows(transcript.getvalue())
            failed = next(i for i, line in enumerate(lines) if "✗ Ran code" in line)
            assert re.fullmatch(r"  ✗ Ran code\s+2 lines · NameError\s+1\.3s", lines[failed])
            assert re.fullmatch(r"  ✓ Read the notebook\s+6 cells\s+0\.4s", lines[failed + 1])
            assert re.fullmatch(r"  ✓ Loaded a skill\s+example-skill · 3 files\s+0\.2s", lines[failed + 2])
            output = transcript.getvalue()
            assert "a second answer" not in output and "streamed words" not in output
            assert output.count("Done.") == 1 and "● Done." in output
            assert "  3 steps · 2.5s · 3.1k in · 612 out" in output
            assert output.index("1.3s") < output.index("● Done.") < output.index("3 steps")
            ui.exit()
            await asyncio.wait_for(running, 5)

    asyncio.run(scenario())


def test_escape_cancels_a_turn_like_ctrl_c_and_is_harmless_while_idle():
    async def scenario():
        submissions, cleaned = [], asyncio.Event()

        async def submit(text):
            submissions.append(text)
            ui.on_event(call("fetch_page", "a", url="https://docs.pola.rs/api/python/"))
            ui.on_event(call("marimo_execute", "b", code="print(1)\n"))
            ui.on_event(result("fetch_page", "a", "12 KB of text", 0.3))
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        with terminal(submit) as (ui, keys, _screen, transcript, _size):
            running = await start(ui)
            keys.send_text("draft\x1b")
            await asyncio.sleep(0.7)  # a slow Esc, Enter while idle is still Esc Enter: a newline
            keys.send_text("\r")
            await until(lambda: ui.buffer.text == "draft\n")
            keys.send_text("\x1b")
            await asyncio.sleep(1.7)  # past prompt_toolkit's idle timeouts: a lone Esc did nothing
            assert ui.application.is_running and ui.buffer.text == "draft\n" and not submissions
            ui.buffer.text = "draft"
            keys.send_text("\r")
            await until(lambda: ui.activity == "Running code")
            keys.send_text("next\x1b\r")  # Alt+Enter (Esc Enter) still inserts a newline while busy
            await until(lambda: ui.buffer.text == "next\n")
            keys.send_text("\n")  # Ctrl+J: a newline key that never involves Esc
            await until(lambda: ui.buffer.text == "next\n\n")
            await asyncio.sleep(0.5)
            assert ui.busy and not cleaned.is_set()
            keys.send_text("\x1b")
            await asyncio.wait_for(cleaned.wait(), 1)  # the turn's short timeouts: Esc acts at once
            await until(lambda: not ui.busy)
            assert ui.buffer.text == "next\n\n" and submissions == ["draft"]
            assert (ui.application.ttimeoutlen, ui.application.timeoutlen) == (0.5, 1.0)  # idle again
            lines = rows(transcript.getvalue())
            done = next(i for i, line in enumerate(lines) if "Fetched a page" in line)
            assert re.fullmatch(r"  ✓ Fetched a page\s+docs\.pola\.rs · 12 KB of text\s+0\.3s", lines[done])
            assert re.fullmatch(r"  ▲ Ran code\s+1 line · cancelled, effect unknown\s+\d+\.\ds", lines[done + 1])
            assert lines[done + 2] == "  ▲ Cancelled."
            assert "the model is told the last step's effect is unknown" in lines[done + 3]
            ui.exit()
            await asyncio.wait_for(running, 5)

    asyncio.run(scenario())


def test_cancel_without_a_running_step_does_not_claim_an_unknown_effect():
    async def scenario():
        async def submit(_text):
            await asyncio.Event().wait()

        with terminal(submit) as (ui, keys, _screen, transcript, _size):
            running = await start(ui)
            keys.send_text("slow\r")
            await until(lambda: ui.busy)
            keys.send_text("\x03")
            await until(lambda: not ui.busy)
            assert "▲ Cancelled." in transcript.getvalue()
            assert "effect unknown" not in transcript.getvalue()
            ui.exit()
            await asyncio.wait_for(running, 5)

    asyncio.run(scenario())


def test_enter_while_busy_flashes_a_hint_without_replacing_the_activity_and_the_ticker_stops():
    async def scenario():
        release = asyncio.Event()

        async def submit(_text):
            ui.on_event(call("marimo_execute", "a", code="slow()\n"))
            await release.wait()
            ui.on_event(result("marimo_execute", "a", "", 2.0))
            ui.finish(summary("Finished."))

        with terminal(submit) as (ui, keys, screen, _transcript, _size):
            running = await start(ui)
            assert ui._ticker is None  # idle: nothing redraws on a timer
            keys.send_text("slow\r")
            await until(lambda: ui.activity == "Running code")
            assert ui._ticker is not None
            assert "working" in "".join(text for _style, text in ui._footer_row())
            keys.send_text("my draft\r")
            await until(lambda: ui.flash_text == "Still working · esc cancels · your draft is kept")
            await until(lambda: "Still working · esc cancels · your draft is kept" in screen.getvalue())
            assert ui.activity == "Running code" and ui.buffer.text == "my draft"
            release.set()
            await until(lambda: not ui.busy)
            assert ui.flash_text == ""  # the hint ends with the turn
            await until(lambda: ui._ticker is None)
            assert ui.buffer.text == "my draft"
            ui.exit()
            await asyncio.wait_for(running, 5)

    asyncio.run(scenario())


def test_frame_hints_and_placeholder_follow_the_state():
    async def submit(_text):
        pass

    with terminal(submit) as (ui, _keys, _screen, _transcript, _size):
        def hints():
            return "".join(text for _style, text in ui._hints()).strip()

        assert (ui._state(), hints()) == ("idle", "/ commands · ↑ history · ctrl+c exit")
        assert ui._placeholder() == "Ask about your data, or type / for commands"
        ui.buffer.text = "draft"
        assert hints() == "enter send · alt+enter new line"
        ui._preparing = True
        assert (ui._state(), hints()) == ("busy", "starting · you can type now")
        assert ui._placeholder() == "Ask about your data; your first message waits for startup"
        ui._preparing, ui._startup_failed = False, True
        assert (ui._state(), hints(), ui._border_style()) == ("failed", "startup failed · /exit to leave", "class:frame.failed")


@pytest.mark.parametrize(
    ("columns", "shown", "dropped"),
    [
        (100, ["● analysis.py", "▲ unsafe-local · not isolated", "gpt-5.5", "4 turns · 18.6k tokens"], []),
        (72, ["● analysis.py", "▲ unsafe-local · not isolated", "gpt-5.5", "4 turns"], ["tokens"]),
        (60, ["● analysis.py", "▲ unsafe-local · not isolated"], ["turns", "gpt-5.5"]),
        (34, ["▲ unsafe-local · not isolated"], ["analysis", "gpt-5.5", "turns"]),
    ],
)
def test_footer_drops_usage_then_the_model_but_never_the_unsafe_local_warning(columns, shown, dropped):
    data = FooterData(notebook="analysis.py", notebook_open=True, kernel="unsafe-local", model="gpt-5.5", turns=4, tokens=18_600)

    async def submit(_text):
        pass

    with terminal(submit, lambda: data) as (ui, _keys, _screen, _transcript, size):
        size[0] = Size(rows=24, columns=columns)
        text = "".join(text for _style, text in ui._footer_row())
        assert all(part in text for part in shown) and not any(part in text for part in dropped)
        assert len(text.rstrip()) < columns


def test_footer_shows_the_notebook_state_and_drops_a_docker_kernel_before_the_notebook():
    data = [FooterData(notebook="analysis.py", notebook_open=False, kernel="docker", model="gpt-5.5")]

    async def submit(_text):
        pass

    with terminal(submit, lambda: data[0]) as (ui, _keys, _screen, _transcript, size):
        def footer():
            return "".join(text for _style, text in ui._footer_row())

        assert "○ analysis.py  ·  docker · isolated  ·  gpt-5.5" in footer() and "0 turns" in footer()
        data[0] = FooterData(notebook="analysis.py", notebook_open=None, kernel="docker", model="gpt-5.5")
        assert footer().startswith(" analysis.py")  # not known: no mark
        size[0] = Size(rows=24, columns=30)
        assert "analysis.py" in footer() and "docker" not in footer()
        data[0] = FooterData()
        assert footer().strip() == "0 turns"  # a broken or empty footer never breaks a redraw


def test_ascii_glyphs_when_the_console_cannot_encode_unicode():
    assert glyphs_for("cp1252") is ASCII and glyphs_for("utf-8") is UNICODE and glyphs_for(None) is ASCII

    async def scenario():
        async def submit(_text):
            ui.on_event(call("list_periods", "a", name="sales"))
            ui.on_event(result("list_periods", "a", "6 months", 0.5))
            ui.finish(summary("| a | b |\n|--|--|\n| 1 | 2 |", steps=1, duration_ms=900))

        raw = io.BytesIO()
        console = Console(file=io.TextIOWrapper(raw, encoding="cp1252"), force_terminal=False, color_system=None, width=80)
        with terminal(submit, console=console) as (ui, keys, screen, _transcript, _size):
            assert ui.glyphs is ASCII
            running = await start(ui)
            keys.send_text("hi\r")
            await until(lambda: ui._active_task is not None and not ui.busy)
            ui.exit()
            await asyncio.wait_for(running, 5)
            console.file.flush()
            output = raw.getvalue().decode("cp1252")
            assert "| hi" in output and "+ Scanned the data" in output and "* a" in output
            assert "1 step - 0.9s" in output
            assert not any(char in output for char in "▌●✓·─")
            assert "╭" not in screen.getvalue() and "+-" in screen.getvalue()

    asyncio.run(scenario())


def test_reply_footer_links_the_notebook_without_its_token():
    async def scenario():
        url = "http://127.0.0.1:2718/?file=/work/notebooks/analysis.py&view-as=present"
        data = FooterData(notebook="analysis.py", kernel="docker", notebook_url=url)

        async def submit(_text):
            ui.finish(summary("Answer", steps=2, duration_ms=7700, input_tokens=3140, output_tokens=612))

        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=True, color_system="standard", width=90)
        with terminal(submit, lambda: data, console=console) as (ui, keys, _screen, _transcript, _size):
            running = await start(ui)
            keys.send_text("hi\r")
            await until(lambda: "612 out" in buffer.getvalue() and not ui.busy)
            output = buffer.getvalue()
            assert "2 steps · 7.7s · 3.1k in · 612 out" in output and "↗ analysis.py" in output
            assert "\x1b]8;id=" in output and url in output and "access_token" not in output
            ui.exit()
            await asyncio.wait_for(running, 5)

    asyncio.run(scenario())


def test_composer_lets_rich_wrap_with_indents_but_keeps_urls_whole():
    async def scenario():
        url = "http://127.0.0.1:2718/?file=/work/notebooks/analysis.py&access_token=" + "t" * 40

        async def submit(_text):
            ui.console.print("    a hint line that is much longer than the narrow transcript is wide", markup=False)
            ui.console.print(f"Open {url} in your browser.", markup=False)

        console = Console(file=io.StringIO(), force_terminal=False, color_system=None, width=40)
        with terminal(submit, console=console) as (ui, keys, _screen, _transcript, _size):
            running = await start(ui)
            keys.send_text("a message with enough words to wrap twice inside forty columns\r")
            await until(lambda: "browser" in console.file.getvalue() and not ui.busy)
            lines = rows(console.file.getvalue())
            echo = [line for line in lines if line.startswith("▌")]
            assert len(echo) == 2 and all(len(line) <= 40 for line in echo)
            hint = [line for line in lines if "hint" in line or "transcript" in line]
            assert len(hint) == 2 and all(line.startswith("    ") and len(line) <= 40 for line in hint)
            assert any(url in line for line in lines)  # one line: the terminal wraps it, nothing is cut
            ui.exit()
            await asyncio.wait_for(running, 5)

    asyncio.run(scenario())


def test_bracketed_names_print_as_text_never_as_markup():
    """Notebook, skill and file names, errors and warnings come from outside: "q[/].py" would be a
    Rich markup error in the writer and end the app."""
    from hailer.transcript import NotebookRow, callout, detail_card, notebook_table, status_card

    names = ["q[/].py", "[bold]x.py", "[red]warn[/red]"]

    async def scenario():
        async def submit(_text):
            g = ui.glyphs
            ui.console.print(notebook_table(g, [NotebookRow(name, "[today]", "[1 KB]", active=True) for name in names], "[/] note"))
            ui.console.print(detail_card(g, [("[Skills]", [names[1]])], ["[/]warning"]))
            ui.console.print(status_card(g, [("ok", "[Model]", names[2])], title="[/]"))
            ui.console.print(callout(g, "fail", "Unknown notebook q[/].py", ["[bold]hint[/bold]"]))

        with terminal(submit) as (ui, keys, _screen, transcript, _size):
            running = await start(ui)
            keys.send_text("/notebook list\r")
            await until(lambda: "hint" in transcript.getvalue() and not ui.busy)
            output = transcript.getvalue()
            for text in [*names, "[today]", "[1 KB]", "[/] note", "[Skills]", "[/]warning", "[Model]", "[/]", "Unknown notebook q[/].py", "[bold]hint[/bold]"]:
                assert text in output, text
            assert ui.application.is_running and not running.done()
            ui.exit()
            await asyncio.wait_for(running, 5)

    asyncio.run(scenario())


def test_a_url_in_a_callout_stays_on_one_line_while_the_rest_wraps_in_the_gutter():
    async def scenario():
        url = "https://docs.example.com/api/python/stable/reference/a/very/long/path.html?query=1"

        async def submit(_text):
            ui.console.print(callout(ui.glyphs, "fail", f"Could not fetch {url}", ["A hint that is long enough to wrap inside forty columns."]))

        console = Console(file=io.StringIO(), force_terminal=False, color_system=None, width=40)
        with terminal(submit, console=console) as (ui, keys, _screen, _transcript, _size):
            running = await start(ui)
            keys.send_text("go\r")
            await until(lambda: "columns." in console.file.getvalue() and not ui.busy)
            lines = rows(console.file.getvalue())
            assert f"  ✗ Could not fetch {url}" in lines  # whole: the terminal wraps it, nothing cuts the URL
            hint = [line for line in lines if "hint" in line or "columns." in line]
            assert len(hint) == 2 and all(line.startswith("    ") and len(line) <= 40 for line in hint)
            ui.exit()
            await asyncio.wait_for(running, 5)

    from hailer.transcript import callout

    asyncio.run(scenario())


def test_turn_footer_keeps_its_stats_and_shortens_the_link_when_narrow():
    from hailer.transcript import turn_footer

    stats = "4 steps · 6.5s · 3.1k in · 612 out"
    for width, link in ((100, "↗ analysis.py"), (46, "↗ analy…")):
        console = Console(file=io.StringIO(), force_terminal=False, color_system=None, width=width)
        console.print(turn_footer(UNICODE, stats, "analysis.py", "http://127.0.0.1:2718/?file=analysis.py"))
        line = console.file.getvalue().rstrip()
        assert line.startswith(f"  {stats}  ") and line.endswith(link) and len(line) <= width


def test_placeholder_stays_on_one_line_in_a_narrow_terminal():
    async def scenario():
        async def submit(_text):
            pass

        with terminal(submit) as (ui, _keys, _screen, _transcript, size):
            running = await start(ui)
            size[0] = Size(rows=24, columns=30)
            ui.application.invalidate()
            await until(lambda: ui.composer.window.render_info is not None and ui.composer.window.render_info.window_width < 30)
            assert ui.composer.window.render_info.window_height == 1
            ui.exit()
            await asyncio.wait_for(running, 5)

    asyncio.run(scenario())


def test_events_after_a_cancel_leave_the_row_saying_cancelling():
    async def scenario():
        cleanup, release = asyncio.Event(), asyncio.Event()

        async def submit(_text):
            ui.on_event(call("marimo_execute", "a", code="slow()\n"))
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleanup.set()
                await release.wait()
                raise

        with terminal(submit) as (ui, keys, _screen, transcript, _size):
            running = await start(ui)
            keys.send_text("slow\r")
            await until(lambda: ui.activity == "Running code")
            keys.send_text("\x03")
            await asyncio.wait_for(cleanup.wait(), 5)
            for event in (AgentEvent("message_delta", "late words"), AgentEvent("status", "Thinking"), call("load_skill", "b", name="x")):
                ui.on_event(event)
            assert ui.activity == "Cancelling"
            release.set()
            await until(lambda: not ui.busy)
            assert "Loaded a skill" not in transcript.getvalue()
            ui.exit()
            await asyncio.wait_for(running, 5)

    asyncio.run(scenario())


def test_nothing_animates_while_the_chat_prepares():
    async def scenario():
        ready = asyncio.Event()

        async def prepare():
            await ready.wait()
            return True

        async def submit(_text):
            pass

        with terminal(submit) as (ui, _keys, _screen, _transcript, _size):
            ui.configure_startup(prepare)
            running = await start(ui)
            await asyncio.sleep(0.2)
            before = ui.application.render_counter
            await asyncio.sleep(2.1)
            assert ui.application.render_counter - before <= 3  # the timer, once a second
            assert "".join(text for _style, text in ui._live_line()).rstrip().endswith("s")  # e.g. "2.3s"
            ready.set()
            await until(lambda: not ui._preparing)
            await until(lambda: ui._ticker is None)
            ui.exit()
            await asyncio.wait_for(running, 5)

    asyncio.run(scenario())
