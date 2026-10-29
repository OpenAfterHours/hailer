"""Shared chat operations through the async presentation; no terminal or model endpoint needed."""

from __future__ import annotations

import asyncio
import io
import re
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
from hailer.models import AgentEvent, ContextBundle, SessionState, SkillInfo
from hailer.session import load_session, save_session
from hailer.transcript import FooterData
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
        self.summaries: list[Any] = []
        self.links: list[Any] = []
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

    def finish(self, summary: Any) -> None:
        self.answers.append(summary.final_response)
        self.summaries.append(summary)
        self.links.append(self.context().notebook_url if callable(self.context) else None)  # what the reply would link
        self.console.print(summary.final_response, markup=False)

    def on_event(self, event: Any) -> None:
        self.events.append(event)

    def settle_steps(self) -> None:
        self.console.print("[steps settled]", markup=False)

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
        assert ui.context().model == "gpt-new"
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
        assert controller.bundle is harness.bundle and len(controller.bundle.context_files) == 1
        await ui.submit("next")
        saved = load_session(harness.config.workspace)
        assert (saved.turns, saved.input_tokens, saved.output_tokens) == (3, 30, 15)

    _controller, ui, output = run_session(harness, scenario)
    assert harness.agent.turns == [("PROMPT[report](March)", None), ("compare months", "reconcile"), ("next", None)]
    assert ui.answers == ["Answer to: PROMPT[report](March)", "Answer to: compare months", "Answer to: next"]
    assert all(output.count(answer) == 1 for answer in ui.answers)
    assert "Checking the notebook..." not in output  # interim deltas remain activity, never duplicate the final
    assert harness.context_loads == 2


def test_footer_data_comes_from_held_state_and_its_link_has_no_token(harness, monkeypatch):
    harness.agent = AsyncFakeAgent()

    async def scenario(controller: Any, ui: StubUI) -> None:
        made: list[tuple[str, dict]] = []
        original = controller.sandbox.notebook_url
        monkeypatch.setattr(
            controller.sandbox, "notebook_url", lambda name, **kwargs: made.append((name, kwargs)) or original(name, **kwargs),
        )
        footer = ui.context()
        assert isinstance(footer, FooterData)
        assert (footer.notebook, footer.kernel, footer.model) == (controller.notebook, harness.config.kernel.runtime, harness.config.model.name)
        assert footer.notebook_open is None  # nothing learned yet (the stub skips the CLI's notebook wait)
        assert footer.notebook_url.startswith(harness.server.url) and "access_token" not in footer.notebook_url
        for _ in range(5):
            ui.context()
        assert made == [(controller.notebook, {})]  # once per name, never signed: redraws ask nothing
        await ui.submit("hello")
        assert (ui.context().turns, ui.context().tokens) == (1, 15)
        await ui.submit("/status")
        assert ui.context().notebook_open is True  # /status learned the session
        await ui.submit("/notebook close")
        assert ui.context().notebook_open is False

    run_session(harness, scenario)


@pytest.mark.parametrize("opened", [True, False, None], ids=["open", "not-open", "stopped"])
def test_startup_notebook_wait_sets_the_footer_notebook_state(harness, opened):
    harness.agent = AsyncFakeAgent()
    controller = cli_chat.ChatLoop(Console(file=io.StringIO()), harness.config, common.CliOptions())
    seen: list[Any] = []

    async def prepare_notebook() -> bool | None:
        return opened

    async def scenario(ui: StubUI) -> None:
        seen.append(ui.context().notebook_open)
        await ui.submit("/exit")

    def factory(console: Console, submit: Any, footer: Any) -> StubUI:
        return StubUI(console, submit, footer, scenario)

    asyncio.run(controller.run_interactive(ui_factory=factory, prepare_notebook=prepare_notebook))
    assert seen == [opened]


def test_composer_cards_keep_every_piece_of_plain_information(harness):
    harness.agent = AsyncFakeAgent()
    harness.agent.key_source = "keyring"
    workspace = harness.config.workspace
    (workspace / "ctx.md").write_text("hello", encoding="utf-8")
    harness.bundle = ContextBundle(
        context_text="x",
        context_files=[workspace / "ctx.md"],
        skills=[SkillInfo("recon", "Reconcile months", harness.config.skills_dir / "recon")],
        prompts={"monthly": harness.config.prompts_dir / "monthly.md"},
        warnings=["ctx.md looks like it contains a secret"],
    )
    write_notebook(harness.config, "other")

    async def scenario(controller: Any, ui: StubUI) -> None:
        await ui.submit("hello")
        for command in ("/status", "/help", "/notebook list", "/context"):
            await ui.submit(command)

    controller, ui, output = run_session(harness, scenario)
    assert ui.summaries[0].final_response == "Answer to: hello" and ui.summaries[0].steps == 1  # the whole summary
    config = harness.config
    status = output[output.index("Session"):output.index("Conversation")]
    for value in (
        config.model.name, common._provider_line(config), "OPENAI_API_KEY from keyring", harness.server.url,
        common._kernel_line(config), controller.notebook, "session", "thread-1", "1 turn", "10 in / 5 out tokens",
        common._relative(config.notebooks_root, config.workspace), common._web_line(config),
    ):
        assert value in status, value
    for label in ("Model", "Kernel", "Marimo", "Notebook", "Notebooks", "Thread", "Web access"):
        assert re.search(rf"[●▲○ ] {label} ", status), label
    for text in ("Conversation", "/model <name>", "Switch model; starts a new conversation", "/notebook [list|new|open|close]", "esc cancel"):
        assert text in output, text
    assert re.search(r"Notebook\s+Modified\s+Size", output)
    assert re.search(rf"{re.escape(controller.notebook)}\s+today \d\d:\d\d\s+[\d.]+ KB\s+● open\s+active", output)
    assert re.search(r"other\.py\s+today", output) and "notebooks in" in output
    assert "ctx.md" in output and "(5 bytes)" in output and "Reconcile months" in output and "monthly" in output
    assert "▲ ctx.md looks like it contains a secret" in output


def test_help_card_lists_every_command_once():
    from hailer.session import COMMAND_GROUPS, COMMAND_USAGE, COMMANDS

    assert set(COMMAND_USAGE) == set(COMMANDS) - {"quit"}  # /quit is /exit's alias
    grouped = [name for names in COMMAND_GROUPS.values() for name in names]
    assert sorted(grouped) == sorted(COMMAND_USAGE)


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
        assert ui.context().notebook == "other.py"
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


def test_reply_links_the_notebook_a_tool_switched_to_and_the_switch_is_reported_after_it(harness):
    harness.agent = AsyncFakeAgent()
    other = write_notebook(harness.config, "other")
    harness.agent.on_turn = lambda: notebooks.save_active_notebook(harness.config, "other.py")

    async def scenario(controller: Any, ui: StubUI) -> None:
        await ui.submit("make another notebook")
        assert controller.notebook == "other.py" and ui.context().notebook == "other.py"

    _controller, ui, output = run_session(harness, scenario)
    assert ui.links == [harness.sandbox.notebook_url("other.py")]  # not the notebook from before the turn
    assert "access_token" not in ui.links[0]
    assert output.count("Active notebook is now other.py.") == 1
    assert output.index("Answer to: make another notebook") < output.index("Active notebook is now other.py.")
    assert harness.opened == [url_for(other)]  # still opened once, after the answer


def test_composer_ends_open_steps_before_a_failed_turn_prints_its_error(harness):
    from test_chat_ui import terminal, until

    class FailingAgent(AsyncFakeAgent):
        async def arun_turn(self, text: str, *, on_event: Any = None, **_kwargs: Any) -> Any:
            self.turns.append((text, None))
            on_event(AgentEvent("tool_call", "marimo_execute", {"args": {"code": "x = 1\n"}, "id": "a"}))
            raise HailerError("The model endpoint failed.", hint="Try again.")

    async def scenario():
        harness.agent = FailingAgent()
        controller = cli_chat.ChatLoop(Console(file=io.StringIO()), harness.config, common.CliOptions())
        with terminal(controller.ahandle, controller.footer) as (ui, keys, _screen, transcript, _size):
            running = asyncio.create_task(controller.run_interactive(ui_factory=lambda *_args: ui))
            await until(lambda: ui.application.is_running and not ui._preparing)
            keys.send_text("fail\r")
            await until(lambda: "Try again." in transcript.getvalue() and not ui.busy)
            out = transcript.getvalue()
            assert re.search(r"▲ Ran code\s+1 line · no result, effect unknown", out)
            assert out.index("no result, effect unknown") < out.index("✗ The model endpoint failed.")  # as in plain mode
            assert out.count("effect unknown") == 1
            ui.exit()
            await asyncio.wait_for(running, 5)

    asyncio.run(scenario())


def test_composer_cards_print_bracketed_names_as_text(harness):
    """Names from the workspace and the user are never Rich markup in the cards."""
    harness.agent = AsyncFakeAgent()
    workspace = harness.config.workspace
    (workspace / "[red]ctx.md").write_text("hello", encoding="utf-8")
    harness.bundle = ContextBundle(
        context_text="x",
        context_files=[workspace / "[red]ctx.md"],
        skills=[SkillInfo("[bold]recon", "Reconcile [/] months", harness.config.skills_dir / "recon")],
        prompts={"[i]monthly": harness.config.prompts_dir / "monthly.md"},
        warnings=["[/]ctx.md looks like it contains a secret"],
    )
    write_notebook(harness.config, "q[/]")
    write_notebook(harness.config, "[bold]x")

    async def scenario(controller: Any, ui: StubUI) -> None:
        await ui.submit("/model [bold]gpt")
        await ui.submit("/model")
        for command in ("/status", "/notebook list", "/context"):
            await ui.submit(command)

    _controller, _ui, output = run_session(harness, scenario)
    for text in ("q[/].py", "[bold]x.py", "[red]ctx.md", "[bold]recon", "Reconcile [/] months", "[i]monthly", "[/]ctx.md looks like", "Model: [bold]gpt"):
        assert text in output, text
    assert re.search(r"Model\s+\[bold\]gpt", output)


def test_async_notebook_command_queues_notice_for_next_turn(harness):
    harness.agent = AsyncFakeAgent()
    write_notebook(harness.config, "other")

    async def scenario(controller: Any, ui: StubUI) -> None:
        await ui.submit("/notebook open other")
        assert controller.notebook == "other.py"
        assert ui.context().notebook == "other.py"
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
        assert ui.context().notebook == "other.py"
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
    ui = StubUI(console, controller.ahandle, controller.footer, None)
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

    ui = BrokenOutputUI(replacement, controller.ahandle, controller.footer, scenario)
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
        with terminal(controller.ahandle, controller.footer) as (ui, keys, screen, transcript, _size):
            running = asyncio.create_task(controller.run_interactive(
                ui_factory=lambda *_args: ui, prepare_notebook=prepare_notebook, timings=timings,
            ))
            await asyncio.wait_for(agent_entered.wait(), 5)
            await asyncio.wait_for(notebook_entered.wait(), 5)
            assert ui.application.is_running and ui.application.render_counter > 0
            assert set(timings.milestones) == {"input_ready"}
            assert ui.activity == "Starting"
            keys.send_text("\x1b[200~first line\r\nsecond line\x1b[201~\r")
            await until(lambda: ui._pending_submission is not None)
            keys.send_text("next draft\r\r")
            await until(lambda: "draft is kept" in ui.flash_text)
            assert ui.buffer.text == "next draft"
            assert harness.agent.turns == []
            release_agent.set()
            await until(lambda: "agent_ready" in timings.milestones)
            assert harness.agent.turns == []
            release_notebook.set()
            await until(lambda: len(harness.agent.turns) == 1 and not ui.busy)
            assert harness.agent.turns == [("first line\nsecond line", None)]
            assert ui.buffer.text == "next draft"
            assert transcript.getvalue().count("▌ first line") == 1
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
        with terminal(controller.ahandle, controller.footer) as (ui, keys, _screen, transcript, _size):
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
        with terminal(controller.ahandle, controller.footer) as (ui, keys, screen, transcript, _size):
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


# --------------------------------------------------------------------------- #
# The command menu's data and the input history file
# --------------------------------------------------------------------------- #


def test_the_menus_notebook_names_are_listed_off_the_ui_loop_and_never_by_the_completer(harness, tmp_path):
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document

    from hailer.completion import SlashCompleter

    harness.agent = AsyncFakeAgent()
    harness.bundle = ContextBundle(
        skills=[SkillInfo("example-skill", "How this team writes reports", tmp_path)], prompts={"first-look": tmp_path / "p.md"},
    )
    write_notebook(harness.config, "other")
    sandbox = harness.sandbox
    listed_on: list[threading.Thread] = []
    list_notebooks = sandbox.list_notebooks

    def listing() -> Any:
        listed_on.append(threading.current_thread())
        return list_notebooks()

    sandbox.list_notebooks = listing  # type: ignore[method-assign]

    async def scenario(controller: Any, ui: StubUI) -> None:
        data = controller.completions()
        assert "other.py" in data.notebooks and data.active_notebook == controller.notebook, "listed at startup"
        assert data.skills == (("example-skill", "How this team writes reports"),) and data.prompts == ("first-look",)
        assert data.model == "gpt-5.5" and "openai" in data.providers
        listings = len(listed_on)
        completer = SlashCompleter(controller.completions)
        for text in ("/notebook open ", "/notebook close o", "/model ", "/skill ", "/prompt "):
            assert list(completer.get_completions(Document(text), CompleteEvent()))
        assert len(listed_on) == listings, "the completer reads the cached names"
        write_notebook(harness.config, "made-by-hand")
        await ui.submit("/notebook")
        assert "made-by-hand.py" in controller.completions().notebooks, "refreshed after a /notebook command"
        write_notebook(harness.config, "made-by-a-tool")
        await ui.submit("a question")
        assert "made-by-a-tool.py" in controller.completions().notebooks, "refreshed after a turn"

    run_session(harness, scenario, sandbox=sandbox)
    assert listed_on and threading.main_thread() not in listed_on, "never on the UI loop's thread"


@pytest.mark.parametrize("on", [True, False], ids=["history-on", "history-off"])
def test_the_composer_and_the_plain_prompt_keep_history_in_the_file_while_the_setting_is_on(harness, monkeypatch, on):
    from dataclasses import replace

    import hailer.chat_ui as chat_ui

    harness.agent = AsyncFakeAgent()
    harness.config = replace(harness.config, input_history=on)
    options: dict[str, Any] = {}

    class Composer(StubUI):
        def __init__(self, console: Console, submit: Any, footer: Any, **kwargs: Any) -> None:
            options.update(kwargs)

            async def scenario(ui: StubUI) -> None:
                await ui.submit("/exit")

            super().__init__(console, submit, footer, scenario)

    monkeypatch.setattr(chat_ui, "ChatUI", Composer)
    controller = cli_chat.ChatLoop(Console(file=io.StringIO()), harness.config, common.CliOptions())
    expected = harness.config.workspace / ".hailer" / "input_history" if on else None
    assert controller.reader.history_file == expected
    asyncio.run(controller.run_interactive())
    assert options == {"completions": controller.completions, "history_file": expected}
