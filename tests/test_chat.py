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
    from hailer.kernel import attach_runtime, docker_paths, kernel_state_path

    server = replace(DOCKER_SERVER, paths=docker_paths(harness.config))
    harness.config = attach_runtime(harness.config, server)
    harness.agent = AsyncFakeAgent()
    harness.server = None  # discovery cannot supply the connection after startup
    write_notebook(harness.config, "other")

    async def scenario(controller, ui):
        assert harness.agent_servers == [server]
        kernel_state_path(harness.config.workspace).unlink(missing_ok=True)
        await ui.submit("/new")
        await ui.submit("/model changed-model")
        await ui.submit("/notebook open other")
        await ui.submit("/status")
        await ui.submit("inspect it")
        assert controller.server is server
        assert harness.opened[-1] == (
            f"{server.url}/?file=/work/notebooks/other.py&view-as=present&access_token={TOKEN}"
        )
        assert all(TOKEN not in (preamble or "") for preamble in harness.agent.preambles)

    _controller, _ui, output = run_session(harness, scenario, server=server)
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

    async def run(self) -> None:
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
    server: Any = None,
) -> tuple[Any, StubUI, str]:
    output = io.StringIO()
    console = Console(file=output, force_terminal=False, width=120, highlight=False)
    controller = cli.ChatLoop(console, h.config, cli.CliOptions(new_thread=new_thread), server=server)
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
    harness.agent.on_turn = lambda: notebooks.save_active_notebook(harness.config, other)
    harness.agent.fail_with = asyncio.CancelledError() if cancelled else HailerError("endpoint failed", hint="retry")

    async def scenario(controller: Any, ui: StubUI) -> None:
        notice = "[Hailer] The active notebook changed before this turn."
        controller._pending_preamble = notice
        if cancelled:
            with pytest.raises(asyncio.CancelledError):
                await ui.submit("switch notebook")
        else:
            await ui.submit("switch notebook")
        assert controller.config.notebook == other
        assert "Notebook: notebooks/other.py" in ui.context()
        assert controller._pending_preamble == (None if cancelled else notice)
        assert not ui.answers and controller.state.turns == 0
        harness.agent.fail_with = None
        await ui.submit("continue")
        assert harness.agent.preambles == [notice, None if cancelled else notice]
        assert controller._pending_preamble is None and controller.state.turns == 1

    _controller, ui, output = run_session(harness, scenario)
    assert output.count("Active notebook is now notebooks/other.py.") == 1
    assert harness.opened == [url_for(other)]
    assert ui.answers == ["Answer to: continue"]
    assert ("endpoint failed" in output) is not cancelled


def test_async_notebook_command_queues_notice_for_next_turn(harness):
    harness.agent = AsyncFakeAgent()
    other = write_notebook(harness.config, "other")

    async def scenario(controller: Any, ui: StubUI) -> None:
        await ui.submit("/notebook open other")
        assert controller.config.notebook == other
        assert "Notebook: notebooks/other.py" in ui.context()
        await ui.submit("inspect it")
        await ui.submit("another question")

    _controller, ui, output = run_session(harness, scenario)
    assert harness.agent.preambles[0].startswith("[Hailer] The active notebook is now notebooks/other.py")
    assert harness.agent.preambles[1] is None
    assert "Active notebook: notebooks/other.py." in output
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

    monkeypatch.setattr(cli, "_wait_for_session", wait_session)

    async def scenario(controller: Any, ui: StubUI) -> None:
        operation = asyncio.create_task(ui.submit("/notebook open other"))
        async with asyncio.timeout(5):
            while not entered.is_set():
                await asyncio.sleep(0.001)
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(operation, timeout=3)
        assert stopped.is_set()
        assert controller.config.notebook == other
        assert notebooks.load_active_notebook(harness.config) == other
        assert "Notebook: notebooks/other.py" in ui.context()
        assert controller._pending_preamble.startswith("[Hailer] The active notebook is now notebooks/other.py")
        assert "not open in a browser yet" in controller._pending_preamble
        await ui.submit("inspect it")
        assert harness.agent.preambles[0].startswith("[Hailer] The active notebook is now notebooks/other.py")
        assert controller._pending_preamble is None

    _controller, ui, output = run_session(harness, scenario)
    assert harness.opened == [url_for(other)]
    assert "Active notebook: notebooks/other.py." in output
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
    controller = cli.ChatLoop(console, harness.config, cli.CliOptions())
    ui = StubUI(console, controller.ahandle, controller.context_line, None)
    with pytest.raises(type(failure)):
        asyncio.run(controller.run_interactive(ui_factory=lambda *_args: ui))
    assert harness.agent.closed and ui.flushed and controller.console is console


def test_failed_final_flush_still_restores_controller_console_and_closes_agent(harness):
    harness.agent = AsyncFakeAgent()
    original = Console(file=io.StringIO())
    replacement = Console(file=io.StringIO())
    controller = cli.ChatLoop(original, harness.config, cli.CliOptions())

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


@pytest.mark.parametrize("args", [["--plain"], ["notebook", "--plain"], ["--plain", "notebook"]])
def test_plain_option_reaches_default_and_notebook_sessions(harness, monkeypatch, args):
    sessions: list[cli.CliOptions] = []
    monkeypatch.setattr(cli, "_run_session", lambda _console, _config, opts, **_kwargs: sessions.append(opts))
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
    monkeypatch.setattr(cli, "console_factory", lambda: base_console)
    monkeypatch.setattr(cli, "_stdio_is_terminal", lambda: True)
    monkeypatch.setenv("TERM", term)

    def interactive_input_must_not_start(_reader: Any) -> None:
        pytest.fail("plain terminal chat must use noninteractive input")

    monkeypatch.setattr(cli._LineReader, "_prompt_session", interactive_input_must_not_start)
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
    monkeypatch.setattr(cli, "_stdio_is_terminal", lambda: tty)
    monkeypatch.setenv("TERM", term)
    assert cli._use_composer(cli.CliOptions(plain=plain)) is expected
