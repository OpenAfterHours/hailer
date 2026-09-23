"""Shared chat operations through the async presentation; no terminal or model endpoint needed."""

from __future__ import annotations

import asyncio
import io
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from rich.console import Console

import hailer.cli as cli
from hailer.cli import chat as cli_chat
from hailer.cli import common
from hailer import notebooks
from hailer.errors import HailerError
from hailer.models import ContextBundle, SessionState, SkillInfo
from hailer.session import load_session, save_session
from test_cli import FakeAgent, Harness, harness, runner, url_for, write_notebook  # noqa: F401 - shared offline fixture


class AsyncFakeAgent(FakeAgent):
    """Fail if interactive code accidentally enters the synchronous agent facade."""

    def start(self, **kwargs: Any) -> str:
        raise AssertionError("interactive code called synchronous start")

    def new_thread(self) -> str:
        raise AssertionError("interactive code called synchronous new_thread")

    def run_turn(self, text: str, **kwargs: Any) -> Any:
        raise AssertionError("interactive code called synchronous run_turn")

    def close(self) -> None:
        raise AssertionError("interactive code called synchronous close")

    async def astart(self, **kwargs: Any) -> str:
        return FakeAgent.start(self, **kwargs)

    async def anew_thread(self) -> str:
        return FakeAgent.new_thread(self)

    async def arun_turn(self, text: str, **kwargs: Any) -> Any:
        return FakeAgent.run_turn(self, text, **kwargs)

    async def aclose(self) -> None:
        FakeAgent.close(self)


def test_composer_keeps_docker_connection_across_switches(harness):
    from dataclasses import replace
    from test_cli import DOCKER_SERVER, TOKEN
    from hailer.models import KernelConfig

    harness.server = DOCKER_SERVER
    harness.config = replace(harness.config, kernel=KernelConfig(runtime="docker"))
    harness.agent = AsyncFakeAgent()
    write_notebook(harness.config, "other")
    sandbox = harness.sandbox

    async def scenario(controller, ui):
        assert harness.agent_sandboxes == [sandbox]
        await ui.submit("/new")
        await ui.submit("/model changed-model")
        await ui.submit("/notebook open other")
        await ui.submit("/status")
        await ui.submit("inspect it")
        assert controller.sandbox is sandbox
        assert harness.opened[-1] == (
            f"{DOCKER_SERVER.url}/?file=/work/notebooks/other.py&view-as=present&access_token={TOKEN}"
        )
        assert all(TOKEN not in (preamble or "") for preamble in harness.agent.preambles)

    _controller, _ui, output = run_session(harness, scenario, sandbox=sandbox)
    assert "docker (hailer-kernel" in output and "no network; data read-only" in output


class StubUI:
    def __init__(self, console: Console, submit: Any, context: Any, scenario: Any) -> None:
        self.console = console
        self.submit = submit
        self.context = context
        self.scenario = scenario
        self.answers: list[str] = []
        self.events: list[Any] = []
        self.activities: list[str] = []
        self.exited = False
        self.flushed = False
        self.prepare: Any = None

    def configure_startup(self, prepare: Any, *, on_input_ready: Any = None) -> None:
        self.prepare = prepare
        self.on_input_ready = on_input_ready

    async def run(self) -> None:
        if self.on_input_ready is not None:
            self.on_input_ready()
        if not await self.prepare():
            return
        await self.scenario(self)

    def finish(self, text: str) -> None:
        self.answers.append(text)
        self.console.print(text, markup=False)

    def on_event(self, event: Any) -> None:
        self.events.append(event)

    def set_activity(self, text: str) -> None:
        self.activities.append(text)

    def exit(self) -> None:
        self.exited = True

    async def flush(self) -> None:
        self.flushed = True


def run_session(
    h: Harness,
    scenario: Callable[[Any, StubUI], Awaitable[None]],
    *,
    new_thread: bool = False,
    sandbox: Any = None,
) -> tuple[Any, StubUI, str]:
    output = io.StringIO()
    console = Console(file=output, force_terminal=False, width=120, highlight=False)
    # The chat's own kernel: the sandbox given, else the harness's (what its fake start hands a chat).
    controller = cli_chat.ChatLoop(console, h.config, common.CliOptions(new_thread=new_thread), sandbox=sandbox or h.sandbox)
    made: list[StubUI] = []

    def factory(console: Console, submit: Any, context: Any) -> StubUI:
        ui = StubUI(console, submit, context, lambda ui: scenario(controller, ui))
        made.append(ui)
        return ui

    asyncio.run(controller.run_interactive(ui_factory=factory))
    assert controller.console is console
    assert h.agent.closed and made[0].flushed
    return controller, made[0], output.getvalue()


@pytest.mark.parametrize("new_thread", [False, True], ids=["resume", "fresh"])
def test_async_start_persists_resumed_or_new_session(harness, new_thread):
    harness.agent = AsyncFakeAgent()
    save_session(harness.config.workspace, SessionState(thread_id="old-thread", turns=3, input_tokens=40, output_tokens=9))

    async def scenario(controller: Any, ui: StubUI) -> None:
        saved = load_session(harness.config.workspace)
        assert saved.thread_id == ("thread-1" if new_thread else "old-thread")
        assert saved.turns == (0 if new_thread else 3)
        assert saved.model == harness.config.model.name
        assert saved.provider == harness.config.model.provider
        await ui.submit("/exit")

    _controller, ui, output = run_session(harness, scenario, new_thread=new_thread)
    assert harness.agent.started == ([None] if new_thread else ["old-thread"])
    assert harness.agent.forgotten == (["old-thread"] if new_thread else [None])
    assert ui.exited and "Bye." in output
    assert ("Resumed conversation (3 turns so far)" in output) is not new_thread


def test_async_model_and_new_reset_thread_once_and_persist_counters(harness):
    harness.agent = AsyncFakeAgent()

    async def scenario(controller: Any, ui: StubUI) -> None:
        await ui.submit("one")
        assert controller.state.turns == 1
        await ui.submit("/model gpt-new")
        assert harness.agent.new_threads == 1
        assert harness.agent.model == ("gpt-new", None)
        assert "Model: gpt-new" in ui.context()
        saved = load_session(harness.config.workspace)
        assert (saved.turns, saved.input_tokens, saved.output_tokens) == (0, 0, 0)
        assert saved.model == "gpt-new"
        await ui.submit("two")
        await ui.submit("/new")
        assert harness.agent.new_threads == 2
        saved = load_session(harness.config.workspace)
        assert (saved.turns, saved.input_tokens, saved.output_tokens) == (0, 0, 0)

    _controller, ui, output = run_session(harness, scenario)
    assert harness.agent.turns == [("one", None), ("two", None)]
    assert len(ui.answers) == 2
    assert output.count("Model set to gpt-new") == 1
    assert output.count("New conversation started.") == 1


def test_async_prompt_skill_and_reload_share_normal_turn_accounting(harness):
    harness.agent = AsyncFakeAgent(stream=True)
    skill = SkillInfo(name="reconcile", description="Reconcile data", path=harness.config.skills_dir / "reconcile")
    harness.bundle = ContextBundle(skills=[skill], prompts={"report": harness.config.prompts_dir / "report.md"})

    async def scenario(controller: Any, ui: StubUI) -> None:
        await ui.submit("/prompt report March")
        await ui.submit("/skill reconcile compare months")
        assert controller.state.turns == 2
        harness.bundle = ContextBundle(context_text="Updated instructions", context_files=[harness.config.context_dir / "a.md"])
        await ui.submit("/reload")
        assert harness.agent.bundle is harness.bundle
        assert "Context: 1 files" in ui.context()
        await ui.submit("next")
        saved = load_session(harness.config.workspace)
        assert (saved.turns, saved.input_tokens, saved.output_tokens) == (3, 30, 15)

    _controller, ui, output = run_session(harness, scenario)
    assert harness.agent.turns == [("PROMPT[report](March)", None), ("compare months", "reconcile"), ("next", None)]
    assert ui.answers == ["Answer to: PROMPT[report](March)", "Answer to: compare months", "Answer to: next"]
    assert all(output.count(answer) == 1 for answer in ui.answers)
    assert "Checking the notebook..." not in output  # interim deltas remain activity, never duplicate the final
    assert harness.context_loads == 2


@pytest.mark.parametrize("cancelled", [False, True], ids=["failed", "cancelled"])
def test_async_turn_syncs_notebook_after_failure_or_cancellation_and_preserves_notice_rules(harness, cancelled):
    harness.agent = AsyncFakeAgent()
    other = write_notebook(harness.config, "other")
    harness.agent.on_turn = lambda: notebooks.save_active_notebook(harness.config, "other.py")
    harness.agent.fail_with = asyncio.CancelledError() if cancelled else HailerError("endpoint failed", hint="retry")

    async def scenario(controller: Any, ui: StubUI) -> None:
        notice = "[Hailer] The active notebook changed before this turn."
        controller._pending_preamble = notice
        if cancelled:
            with pytest.raises(asyncio.CancelledError):
                await ui.submit("switch notebook")
        else:
            await ui.submit("switch notebook")
        assert controller.notebook == "other.py"
        assert "Notebook: other.py" in ui.context()
        assert controller._pending_preamble == (None if cancelled else notice)
        assert not ui.answers and controller.state.turns == 0
        harness.agent.fail_with = None
        await ui.submit("continue")
        assert harness.agent.preambles == [notice, None if cancelled else notice]
        assert controller._pending_preamble is None and controller.state.turns == 1

    _controller, ui, output = run_session(harness, scenario)
    assert output.count("Active notebook is now other.py.") == 1
    assert harness.opened == [url_for(other)]
    assert ui.answers == ["Answer to: continue"]
    assert ("endpoint failed" in output) is not cancelled


def test_async_notebook_command_queues_notice_for_next_turn(harness):
    harness.agent = AsyncFakeAgent()
    write_notebook(harness.config, "other")

    async def scenario(controller: Any, ui: StubUI) -> None:
        await ui.submit("/notebook open other")
        assert controller.notebook == "other.py"
        assert "Notebook: other.py" in ui.context()
        await ui.submit("inspect it")
        await ui.submit("another question")

    _controller, ui, output = run_session(harness, scenario)
    assert harness.agent.preambles[0].startswith("[Hailer] The active notebook is now other.py")
    assert harness.agent.preambles[1] is None
    assert "Active notebook: other.py." in output
    assert len(ui.answers) == 2


def test_cancelled_notebook_session_wait_stops_worker_and_preserves_switch_notice(harness, monkeypatch):
    harness.agent = AsyncFakeAgent()
    other = write_notebook(harness.config, "other")
    entered, stopped = threading.Event(), threading.Event()

    def wait_session(client: Any, notebook: Any, timeout: float, *, should_stop: Any = None) -> None:
        entered.set()
        assert should_stop is not None
        deadline = time.monotonic() + 5
        while not should_stop():
            if time.monotonic() > deadline:
                raise AssertionError("session wait never received its cancellation signal")
            time.sleep(0.001)
        stopped.set()
        return None

    monkeypatch.setattr(common, "_wait_for_session", wait_session)

    async def scenario(controller: Any, ui: StubUI) -> None:
        operation = asyncio.create_task(ui.submit("/notebook open other"))
        async with asyncio.timeout(5):
            while not entered.is_set():
                await asyncio.sleep(0.001)
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(operation, timeout=3)
        assert stopped.is_set()
        assert controller.notebook == "other.py"
        assert notebooks.load_active_notebook(harness.config) == "other.py"
        assert "Notebook: other.py" in ui.context()
        assert controller._pending_preamble.startswith("[Hailer] The active notebook is now other.py")
        assert "not open in a browser yet" in controller._pending_preamble
        await ui.submit("inspect it")
        assert harness.agent.preambles[0].startswith("[Hailer] The active notebook is now other.py")
        assert controller._pending_preamble is None

    _controller, ui, output = run_session(harness, scenario)
    assert harness.opened == [url_for(other)]
    assert "Active notebook: other.py." in output
    assert ui.answers == ["Answer to: inspect it"]


@pytest.mark.parametrize("worker_fails", [False, True], ids=["worker-completes", "worker-fails"])
def test_blocking_command_cancellation_waits_for_worker_before_returning(harness, worker_fails):
    harness.agent = AsyncFakeAgent()
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def worker() -> str:
        entered.set()
        try:
            assert release.wait(timeout=5)
            if worker_fails:
                raise HailerError("worker failed")
            return "complete"
        finally:
            finished.set()

    async def scenario(controller: Any, ui: StubUI) -> None:
        operation = asyncio.create_task(controller._blocking(worker))
        try:
            async with asyncio.timeout(5):
                while not entered.is_set():
                    await asyncio.sleep(0.001)
            operation.cancel()
            await asyncio.sleep(0)
            operation.cancel()
            await asyncio.sleep(0)
            assert not operation.done() and not finished.is_set()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(operation, timeout=5)
        assert finished.is_set()
        await ui.submit("still usable")

    _controller, ui, _output = run_session(harness, scenario)
    assert ui.answers == ["Answer to: still usable"]


@pytest.mark.parametrize("failure", [HailerError("startup failed"), asyncio.CancelledError()])
def test_interactive_start_failure_or_cancellation_closes_agent_and_flushes(harness, failure):
    class FailedAgent(AsyncFakeAgent):
        async def astart(self, **kwargs: Any) -> str:
            raise failure

    harness.agent = FailedAgent()
    output = io.StringIO()
    console = Console(file=output)
    controller = cli_chat.ChatLoop(console, harness.config, common.CliOptions())
    ui = StubUI(console, controller.ahandle, controller.context_line, None)
    if isinstance(failure, asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(controller.run_interactive(ui_factory=lambda *_args: ui))
    else:
        asyncio.run(controller.run_interactive(ui_factory=lambda *_args: ui))
        assert "startup failed" in output.getvalue()
    assert harness.agent.closed and ui.flushed and controller.console is console


def test_failed_final_flush_still_restores_controller_console_and_closes_agent(harness):
    harness.agent = AsyncFakeAgent()
    original = Console(file=io.StringIO())
    replacement = Console(file=io.StringIO())
    controller = cli_chat.ChatLoop(original, harness.config, common.CliOptions())

    async def scenario(_ui: StubUI) -> None:
        assert controller.console is replacement

    class BrokenOutputUI(StubUI):
        async def flush(self) -> None:
            raise OSError("terminal output closed")

    ui = BrokenOutputUI(replacement, controller.ahandle, controller.context_line, scenario)
    with pytest.raises(OSError, match="terminal output closed"):
        asyncio.run(controller.run_interactive(ui_factory=lambda *_args: ui))
    assert harness.agent.closed
    assert controller.console is original


def test_first_message_waits_for_both_startups_while_composer_accepts_paste_and_draft(harness):
    from test_chat_ui import terminal, until
    from hailer.startup import StartupTimings

    async def scenario():
        agent_entered, notebook_entered = asyncio.Event(), asyncio.Event()
        release_agent, release_notebook = asyncio.Event(), asyncio.Event()

        class SlowAgent(AsyncFakeAgent):
            async def astart(self, **kwargs: Any) -> str:
                agent_entered.set()
                await release_agent.wait()
                return await super().astart(**kwargs)

        async def prepare_notebook():
            notebook_entered.set()
            await release_notebook.wait()

        harness.agent = SlowAgent()
        controller = cli_chat.ChatLoop(Console(file=io.StringIO()), harness.config, common.CliOptions())
        timings = StartupTimings()
        with terminal(controller.ahandle, controller.context_line) as (ui, keys, screen, transcript, _size):
            running = asyncio.create_task(controller.run_interactive(
                ui_factory=lambda *_args: ui, prepare_notebook=prepare_notebook, timings=timings,
            ))
            await asyncio.wait_for(agent_entered.wait(), 5)
            await asyncio.wait_for(notebook_entered.wait(), 5)
            assert ui.application.is_running and ui.application.render_counter > 0
            assert set(timings.milestones) == {"input_ready"}
            assert ui.activity == "Preparing chat..."
            keys.send_text("\x1b[200~first line\r\nsecond line\x1b[201~\r")
            await until(lambda: ui._pending_submission is not None)
            keys.send_text("next draft\r\r")
            await until(lambda: "draft is kept" in ui.activity)
            assert ui.buffer.text == "next draft"
            assert harness.agent.turns == []
            release_agent.set()
            await until(lambda: "agent_ready" in timings.milestones)
            assert harness.agent.turns == []
            release_notebook.set()
            await until(lambda: len(harness.agent.turns) == 1 and not ui.busy)
            assert harness.agent.turns == [("first line\nsecond line", None)]
            assert ui.buffer.text == "next draft"
            assert transcript.getvalue().count("You") == 1
            assert "\x1b[?2004l" not in screen.getvalue()
            assert timings.milestones["ready_to_answer"] >= timings.milestones["notebook_wait_complete"]
            assert timings.milestones["agent_ready"] > timings.milestones["input_ready"]
            ui.exit()
            await asyncio.wait_for(running, 5)
            assert harness.agent.closed

    asyncio.run(scenario())


@pytest.mark.parametrize("failed_part", ["agent", "notebook"])
def test_startup_failure_retains_queued_message_and_new_draft_and_settles_sibling(harness, failed_part):
    from test_chat_ui import terminal, until

    async def scenario():
        entered = {name: asyncio.Event() for name in ("agent", "notebook")}
        settled = {name: asyncio.Event() for name in ("agent", "notebook")}
        fail = asyncio.Event()

        async def preparation(name):
            entered[name].set()
            try:
                if name == failed_part:
                    await fail.wait()
                    raise HailerError("startup failed", hint="check configuration")
                await asyncio.Event().wait()
            finally:
                settled[name].set()

        class FailedAgent(AsyncFakeAgent):
            async def astart(self, **_kwargs: Any) -> str:
                await preparation("agent")
                raise AssertionError("unreachable")

            async def aclose(self) -> None:
                assert all(event.is_set() for event in settled.values())
                await super().aclose()

        harness.agent = FailedAgent()
        controller = cli_chat.ChatLoop(Console(file=io.StringIO()), harness.config, common.CliOptions())
        with terminal(controller.ahandle, controller.context_line) as (ui, keys, _screen, transcript, _size):
            running = asyncio.create_task(controller.run_interactive(
                ui_factory=lambda *_args: ui, prepare_notebook=lambda: preparation("notebook"),
            ))
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered.values())), 5)
            keys.send_text("keep queued message\r")
            await until(lambda: ui._pending_submission is not None)
            keys.send_text("keep next draft")
            await until(lambda: ui.buffer.text == "keep next draft")
            fail.set()
            await until(lambda: ui._startup_failed)
            await ui.flush()
            assert ui._pending_submission == "keep queued message"
            assert ui.buffer.text == "keep next draft"
            assert "keep queued message" in ui.buffer.history.get_strings()
            assert "startup failed" in transcript.getvalue()
            assert "was not sent" in transcript.getvalue()
            assert harness.agent.turns == []
            assert all(event.is_set() for event in settled.values())
            assert controller.startup_failed
            keys.send_text("\r")
            await asyncio.sleep(0)
            assert ui.buffer.text == "keep next draft"
            ui.buffer.text = "/exit"
            keys.send_text("\r")
            await asyncio.wait_for(running, 5)
            assert harness.agent.closed

    asyncio.run(scenario())


@pytest.mark.parametrize("gesture", ["interrupt", "double_interrupt", "exit", "eof", "input_closed", "external_cancel"])
def test_exit_during_startup_settles_preparation_before_closing_agent(harness, gesture):
    from test_chat_ui import terminal, until

    async def scenario():
        entered, cleanup_started, release_cleanup, settled = (
            asyncio.Event(), asyncio.Event(), asyncio.Event(), asyncio.Event()
        )

        class SlowAgent(AsyncFakeAgent):
            async def astart(self, **_kwargs: Any) -> str:
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cleanup_started.set()
                    await release_cleanup.wait()
                    settled.set()

            async def aclose(self) -> None:
                assert settled.is_set()
                await super().aclose()

        harness.agent = SlowAgent()
        original = Console(file=io.StringIO())
        controller = cli_chat.ChatLoop(original, harness.config, common.CliOptions())
        with terminal(controller.ahandle, controller.context_line) as (ui, keys, screen, transcript, _size):
            running = asyncio.create_task(controller.run_interactive(ui_factory=lambda *_args: ui))
            await asyncio.wait_for(entered.wait(), 5)
            if gesture != "interrupt":
                keys.send_text("queued\r")
                await until(lambda: ui._pending_submission == "queued")
            if gesture == "input_closed":
                keys.close()
            elif gesture == "external_cancel":
                running.cancel()
            else:
                keys.send_text({"interrupt": "\x03", "double_interrupt": "\x03\x03", "exit": "/exit\r", "eof": "\x04"}[gesture])
            await asyncio.wait_for(cleanup_started.wait(), 5)
            if gesture == "external_cancel":
                running.cancel()
            await asyncio.sleep(0)
            assert not running.done() and not harness.agent.closed
            release_cleanup.set()
            if gesture == "external_cancel":
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(running, 5)
            else:
                await asyncio.wait_for(running, 5)
            assert harness.agent.closed and settled.is_set()
            assert not controller.startup_failed
            assert harness.agent.turns == []
            assert controller.console is original
            assert screen.getvalue().count("\x1b[?2004h") == screen.getvalue().count("\x1b[?2004l") == 1
            if gesture == "double_interrupt":
                assert transcript.getvalue().count("Queued message cancelled") == 1
                assert ui.buffer.text == "queued"

    asyncio.run(scenario())


@pytest.mark.parametrize("args", [["--plain"], ["notebook", "--plain"], ["--plain", "notebook"]])
def test_plain_option_reaches_default_and_notebook_sessions(harness, monkeypatch, args):
    sessions: list[common.CliOptions] = []
    monkeypatch.setattr(cli_chat, "_run_session", lambda _console, _config, opts, **_kwargs: sessions.append(opts))
    result = runner.invoke(cli.app, args, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert len(sessions) == 1 and sessions[0].plain


@pytest.mark.parametrize(
    ("args", "term"),
    [(["--plain"], "xterm-256color"), (["notebook", "--plain"], "xterm-256color"), ([], "dumb")],
    ids=["plain-default", "plain-notebook", "dumb-fallback"],
)
def test_plain_terminal_session_emits_no_cursor_controls(harness, monkeypatch, args, term):
    class TtyOutput(io.StringIO):
        def isatty(self) -> bool:
            return True

    output = TtyOutput()
    base_console = Console(file=output, force_terminal=True, color_system="standard", width=120)
    assert base_console.is_terminal
    monkeypatch.setattr(common, "console_factory", lambda: base_console)
    monkeypatch.setattr(common, "_stdio_is_terminal", lambda: True)
    monkeypatch.setenv("TERM", term)

    def interactive_input_must_not_start(_reader: Any) -> None:
        pytest.fail("plain terminal chat must use noninteractive input")

    monkeypatch.setattr(common._LineReader, "_prompt_session", interactive_input_must_not_start)
    result = runner.invoke(cli.app, args, input="hello\n/exit\n", catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert harness.agent.turns == [("hello", None)]
    assert harness.agent.closed
    rendered = output.getvalue()
    assert "Answer to: hello" in rendered and "Bye." in rendered
    assert "\x1b" not in rendered


@pytest.mark.parametrize(
    ("tty", "term", "plain", "expected"),
    [(True, "xterm-256color", False, True), (True, "", False, True), (False, "xterm", False, False),
     (True, "dumb", False, False), (True, "unknown", False, False), (True, "xterm", True, False)],
)
def test_terminal_selection_keeps_pipes_plain_and_honours_override(monkeypatch, tty, term, plain, expected):
    monkeypatch.setattr(common, "_stdio_is_terminal", lambda: tty)
    monkeypatch.setenv("TERM", term)
    assert cli_chat._use_composer(common.CliOptions(plain=plain)) is expected
