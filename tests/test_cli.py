"""CLI tests: every collaborator is faked; no marimo, model endpoint, keyring or network."""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import quote

import pytest
from rich.console import Console
from typer.testing import CliRunner

import hailer.cli as cli
from fake_sandbox import FolderSandbox, folder_sandbox, with_folder_files
from hailer import __version__, kernel_image, notebooks
from hailer.errors import ConfigError, CredentialsError, HailerError, NoSessionError
from hailer.models import (
    AgentEvent,
    ContextBundle,
    ExecResult,
    HailerConfig,
    KernelConfig,
    MarimoServer,
    MarimoSession,
    ModelConfig,
    ProviderConfig,
    SessionState,
    SkillInfo,
    TurnSummary,
    WebConfig,
)
from hailer.session import save_session, session_path

runner = CliRunner()
# The real kernel start, captured before the harness replaces it.
REAL_START_KERNEL = cli._start_kernel
SERVER = MarimoServer(url="http://127.0.0.1:2718")  # the kernel the harness's fake start hands the chat
TOKEN = "test-token-0123456789"  # what the nb fixture's runtime hands the servers it starts


def server_command(port, folder="notebooks"):
    """The command LocalRuntime runs for the nb fixture's workspace: Hailer's interpreter, the token on stdin."""
    return [
        sys.executable, "-m", "marimo", "edit", folder, "--token-password-file", "-",
        "--headless", "--port", str(port), "--skip-update-check",
    ]


def signed(url: str) -> str:
    """``url`` as the CLI prints and opens it for a server Hailer started (signed in with its token)."""
    return f"{url}&access_token={TOKEN}"
NOTEBOOK_SOURCE = "import marimo\n\napp = marimo.App()\n\n\n@app.cell\ndef _():\n    return\n"
INTERNAL = ProviderConfig(
    id="internal",
    base_url="https://llm.example.internal/v1",
    env_key="INTERNAL_MODEL_API_KEY",
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


def url_for(notebook: Path) -> str:
    """The URL the harness's local kernel gives the notebook at ``notebook`` (its host path): the
    notebooks folder's file key plus the name, app view."""
    from hailer.marimo_client import notebook_file_key

    return f"{SERVER.url}/?file={quote(notebook_file_key(Path(notebook)), safe='/:')}&view-as=present"


@dataclass
class FakeAgent:
    thread_id: str = "thread-1"
    stream: bool = False
    fail_with: BaseException | None = None
    turns: list = field(default_factory=list)
    preambles: list = field(default_factory=list)
    started: list = field(default_factory=list)
    forgotten: list = field(default_factory=list)
    model: tuple | None = None
    closed: bool = False
    bundle: ContextBundle | None = None
    key_source: str | None = None
    new_threads: int = 0
    on_turn: object = None  # callable run inside run_turn, e.g. to mimic a notebook switch made by the model

    def start(self, *, resume_thread_id=None, forget_thread_id=None):
        self.started.append(resume_thread_id)
        self.forgotten.append(forget_thread_id)
        return resume_thread_id or self.thread_id

    def run_turn(self, text, *, on_event=None, skill=None, preamble=None):
        self.turns.append((text, skill.name if skill else None))
        self.preambles.append(preamble)
        if self.on_turn is not None:
            self.on_turn()
        if self.fail_with is not None:
            raise self.fail_with  # for Ctrl+C the real agent has cancelled the turn before re-raising
        final = f"Answer to: {text}"
        if on_event:
            on_event(AgentEvent("tool_call", "marimo_execute"))
            if self.stream:
                # a model can stream interim commentary and then the final answer
                for chunk in ("Checking the ", "notebook...", "Answer ", "to: ", text):
                    on_event(AgentEvent("message_delta", chunk))
        return TurnSummary(final_response=final, thread_id=self.thread_id, input_tokens=10, output_tokens=5)

    def new_thread(self):
        self.new_threads += 1
        self.thread_id = "thread-2"
        return self.thread_id

    def set_model(self, name, provider=None):
        self.model = (name, provider)

    def close(self):
        self.closed = True


class FakeClient:
    """Session-aware stand-in for MarimoClient: a notebook (a name) has a session only when one was
    listed for it (or granted later, the way a browser tab would), matched by the real rule.
    Sessions given with a host path get their name from the notebooks folder of the sandbox that
    made the client (``root``, bound by :meth:`Harness.bound_client`)."""

    def __init__(self, *, healthy=True, session="default", exec_result=None, other_sessions=()):
        self.healthy = healthy
        self.root: str | None = None
        # the configured notebook's session
        self.session = MarimoSession("s1", "analysis.py", "/nb/analysis.py", name="analysis.py") if session == "default" else session
        self.other_sessions = list(other_sessions)  # sessions for notebooks other than the active one
        self.exec_result = exec_result
        self.codes: list[str] = []
        self.closed: list[str] = []
        self.granted: list[str] = []

    def health(self):
        return self.healthy

    def _named(self, session):
        if session.name is not None or not self.root:
            return session
        from hailer.marimo_client import name_in_folder

        native = os.name == "nt" and not self.root.startswith("/")
        return replace(session, name=name_in_folder(self.root, session.path, native=native))

    def sessions(self):
        return [self._named(s) for s in ([self.session] if self.session is not None else []) + self.other_sessions]

    def grant(self, notebook, session_id="s9"):
        """Give ``notebook`` (a name) a session (what happens once its browser tab loads)."""
        self.granted.append(notebook)
        session = MarimoSession(session_id, notebook.rsplit("/", 1)[-1], f"{self.root}/{notebook}", name=notebook)
        self.other_sessions.append(session)
        return session

    def resolve_session(self, notebook):
        from hailer.marimo_client import match_session

        session = match_session(self.sessions(), notebook) if notebook is not None else None
        if session is None:
            url = f"{SERVER.url}/?file={self.root}/{notebook}" if notebook is not None else SERVER.url
            raise NoSessionError("The notebook is not open in a browser.", hint=f"Open {url} in your browser.")
        return session

    def shutdown_session(self, session_id):
        self.closed.append(session_id)
        if self.session is not None and self.session.session_id == session_id:
            self.session = None
        self.other_sessions = [s for s in self.other_sessions if s.session_id != session_id]

    def execute(self, code, *, session_id=None, notebook=None, on_stdout=None, on_stderr=None, timeout=600.0):
        if session_id is None:
            self.resolve_session(notebook)  # raises NoSessionError like the real client
        self.codes.append(code)
        result = self.exec_result or ExecResult(True, stdout="out\n", output="42")
        if on_stdout and result.stdout:
            on_stdout(result.stdout)
        if on_stderr and result.stderr:
            on_stderr(result.stderr)
        return result


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def make_config(
    ws: Path, *, notebook_exists=True, provider="openai", providers=None, web=None, notebooks_dir="notebooks"
) -> HailerConfig:
    nb = ws / notebooks_dir / "analysis.py"
    if notebook_exists:
        nb.parent.mkdir(parents=True, exist_ok=True)
        nb.write_text(NOTEBOOK_SOURCE, encoding="utf-8")
    elif nb.exists():
        nb.unlink()
    (ws / "data").mkdir(exist_ok=True)
    cfg_dir = ws / ".config" / "hailer"
    return HailerConfig(
        workspace=ws,
        notebook=nb,
        notebooks_dir=ws / notebooks_dir,
        data_dir=ws / "data",
        context_dir=cfg_dir / "context",
        skills_dir=cfg_dir / "skills",
        prompts_dir=cfg_dir / "prompts",
        model=ModelConfig(name="gpt-5.5", provider=provider),
        providers=providers or {},
        web=web or WebConfig(),
        config_path=ws / "hailer.toml",
    )


def write_notebook(config: HailerConfig, name: str, source: str = NOTEBOOK_SOURCE) -> Path:
    """Add another notebook to the notebooks folder and return its path."""
    path = config.notebooks_root / f"{name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def active_state(config: HailerConfig) -> str | None:
    """The 'active' entry of .hailer/notebook.json, or None when the file does not exist."""
    path = notebooks.state_path(config.workspace)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("active")


@dataclass
class Harness:
    config: HailerConfig
    agent: FakeAgent
    client: FakeClient
    opened: list = field(default_factory=list)
    context_loads: int = 0
    stored: list = field(default_factory=list)
    deleted: list = field(default_factory=list)
    key_source: tuple = ("x", "env")
    server: MarimoServer = SERVER  # the endpoint of the kernel the fake start hands the chat
    bundle: ContextBundle = field(default_factory=ContextBundle)
    session_waits: list = field(default_factory=list)  # the notebooks (names) waited for
    waited_session: object = "default"  # what _wait_for_session returns ("default" -> session s9)
    agent_sandboxes: list = field(default_factory=list)  # the sandbox each agent was handed
    warmups: list = field(default_factory=list)
    kernel_starts: list = field(default_factory=list)  # the sandbox of every kernel the chat started
    kernels: list = field(default_factory=list)  # what `hailer status` lists as running
    real_clients: bool = False  # the sandbox's clients talk to ``server`` for real (a fake_marimo)

    @property
    def url(self) -> str:
        return url_for(self.config.notebook)

    @property
    def sandbox(self) -> FolderSandbox:
        """A sandbox like the one the fake start hands the chat: ``server``, the notebooks folder,
        ``client`` for every kernel request (a docker server gets the container's paths)."""
        box = folder_sandbox(self.config, self.server, docker=self.server.runtime == "docker")
        if not self.real_clients:
            box.client_factory = lambda **kw: self.bound_client(box)
        return box

    def bound_client(self, box) -> FakeClient:
        """``client``, knowing the notebooks folder of the sandbox that made it."""
        self.client.root = box.notebooks_path
        return self.client


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "120")
    monkeypatch.delenv("HAILER_NOTEBOOK", raising=False)
    h = Harness(config=make_config(tmp_path), agent=FakeAgent(), client=FakeClient())

    def load_context(config):
        h.context_loads += 1
        return h.bundle

    def wait_session(client, notebook, timeout, *, should_stop=None):
        h.session_waits.append(notebook)
        if h.waited_session == "default":
            return client.grant(notebook)  # the tab loaded: the notebook now has a session
        if isinstance(h.waited_session, BaseException):
            raise h.waited_session  # e.g. Ctrl+C while waiting
        return h.waited_session

    monkeypatch.setattr(cli, "console_factory", lambda: Console(force_terminal=False, width=120, highlight=False, soft_wrap=True, color_system=None))
    monkeypatch.setattr(cli, "_load_config", lambda opts: h.config)
    monkeypatch.setattr(cli, "_validate_config", lambda config: [])
    monkeypatch.setattr(cli, "_setup_logging", lambda config, opts: None)
    monkeypatch.setattr(cli, "_wait_for_session", wait_session)
    monkeypatch.setattr(cli, "_workspace_kernels", lambda config: h.kernels)
    monkeypatch.setattr(cli, "_resolve_key", lambda provider: h.key_source)
    monkeypatch.setattr(cli, "_store_key", lambda provider, value: h.stored.append((provider.id, value)))
    monkeypatch.setattr(cli, "_delete_key", lambda provider: (h.deleted.append(provider.id) or True))
    monkeypatch.setattr(cli, "_load_context", load_context)
    monkeypatch.setattr(cli, "_render_prompt", lambda config, name, args: f"PROMPT[{name}]({args})")
    monkeypatch.setattr(cli, "_make_agent", lambda config, bundle, sandbox=None: (h.agent_sandboxes.append(sandbox) or h.agent))
    monkeypatch.setattr(cli, "_start_dependency_warmup", lambda: h.warmups.append("started"))
    monkeypatch.setattr(cli, "_open_browser", lambda url: h.opened.append(url))

    def start_kernel(console, runtime, port, *, verbose):
        """The chat's own kernel, started without a process (the nb fixture runs the real start)."""
        running = h.sandbox
        running.log_hint = "the log"
        h.kernel_starts.append(running)
        return running

    monkeypatch.setattr(cli, "_start_kernel", start_kernel)
    return h


def chat(args=(), input_text="/exit\n"):
    return runner.invoke(cli.app, list(args), input=input_text, catch_exceptions=False)


# --------------------------------------------------------------------------- #
# Chat startup and REPL
# --------------------------------------------------------------------------- #


def test_startup_panel_then_exit(harness):
    result = chat()
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Hailer" in out
    assert "gpt-5.5" in out
    assert "openai" in out
    assert "analysis.py" in out
    assert str(harness.config.workspace) in out
    assert "Type /help for commands." in out
    assert "Bye." in out
    assert harness.agent.closed


def test_interactive_session_enters_composer_before_waiting_for_browser(harness, monkeypatch):
    """The CLI defers notebook preparation and sends its output through the UI console."""
    harness.client = FakeClient(session=None)
    ui_output = io.StringIO()
    ui_console = Console(file=ui_output, force_terminal=False, color_system=None)
    milestones = {}

    def no_status(*args, **kwargs):
        raise AssertionError("a background Rich spinner would fight the composer")

    async def run_interactive(self, ui_factory=None, *, prepare_notebook=None, timings=None):
        assert harness.opened == [] and harness.session_waits == []
        assert self.sandbox is harness.kernel_starts[0] and self.sandbox.server is SERVER
        assert set(timings.milestones) == {"checks_complete", "kernel_ready", "panel_shown"}
        self.console = ui_console
        timings.mark("input_ready")
        assert prepare_notebook is not None
        await prepare_notebook()
        milestones.update(timings.milestones)

    monkeypatch.setattr(ui_console, "status", no_status)
    monkeypatch.setattr(cli, "_use_composer", lambda opts: True)
    monkeypatch.setattr(cli.ChatLoop, "run_interactive", run_interactive)
    result = chat()
    assert result.exit_code == 0, result.output
    assert harness.opened == [harness.url]
    assert harness.session_waits == ["analysis.py"]
    assert "Notebook is open (session s9)." in ui_output.getvalue()
    assert "Notebook is open" not in result.output
    assert milestones["input_ready"] >= milestones["panel_shown"]


def test_interactive_startup_failure_exits_one_without_reporting_twice(harness, monkeypatch):
    async def failed_startup(self, ui_factory=None, *, prepare_notebook=None, timings=None):
        self.startup_failed = True
        self.console.print("Could not prepare chat dependencies.")
        self.console.print("Bye.")

    monkeypatch.setattr(cli, "_use_composer", lambda opts: True)
    monkeypatch.setattr(cli.ChatLoop, "run_interactive", failed_startup)
    result = chat()
    assert result.exit_code == 1, result.output
    assert result.output.count("Could not prepare chat dependencies.") == 1
    assert "Unexpected error" not in result.output
    assert "Bye." in result.output


def test_notebook_preparation_cancel_during_resolve_does_not_open_browser(harness, monkeypatch):
    """A request already in flight settles before cancellation finishes, with no later side effects."""
    entered, release = threading.Event(), threading.Event()
    output = io.StringIO()

    def resolve(notebook):
        entered.set()
        assert release.wait(5), "test did not release session resolution"
        raise NoSessionError("not open")

    monkeypatch.setattr(harness.client, "resolve_session", resolve)

    async def scenario():
        task = asyncio.create_task(cli._prepare_notebook(
            Console(file=output), harness.sandbox, "analysis.py", open_browser=True,
        ))
        try:
            async with asyncio.timeout(5):
                while not entered.is_set():
                    await asyncio.sleep(0.01)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done(), "the in-flight worker must settle before shutdown"
            task.cancel()  # repeated Ctrl+C cannot abandon the worker either
            await asyncio.sleep(0)
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)

    asyncio.run(scenario())
    assert harness.opened == []
    assert harness.session_waits == []
    assert output.getvalue() == ""


def test_notebook_preparation_cancel_stops_real_session_poll(harness, monkeypatch):
    from hailer.marimo_client import wait_for_session

    harness.client = FakeClient(session=None)
    polling = threading.Event()
    output = io.StringIO()

    def wait_session(client, notebook, timeout, *, should_stop=None):
        polling.set()
        return wait_for_session(client, notebook, timeout, interval=30, should_stop=should_stop)

    monkeypatch.setattr(cli, "_wait_for_session", wait_session)

    async def scenario():
        task = asyncio.create_task(cli._prepare_notebook(
            Console(file=output), harness.sandbox, "analysis.py", open_browser=True,
        ))
        async with asyncio.timeout(5):
            while not polling.is_set():
                await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)

    asyncio.run(scenario())
    assert harness.opened == [harness.url]
    assert "No kernel session yet" not in output.getvalue(), "cancellation is not a timeout"


def test_turn_prints_answer_and_persists_session(harness):
    result = chat(input_text="hello there\n/exit\n")
    assert result.exit_code == 0, result.output
    assert "Hailer >" in result.output
    assert "Answer to: hello there" in result.output
    assert harness.agent.turns == [("hello there", None)]
    saved = json.loads(session_path(harness.config.workspace).read_text())
    assert saved["turns"] == 1
    assert saved["thread_id"] == "thread-1"
    assert saved["input_tokens"] == 10 and saved["output_tokens"] == 5


def test_streamed_answer_is_not_printed_twice(harness):
    harness.agent.stream = True
    result = chat(input_text="hi\n/exit\n")
    assert result.output.count("Answer to: hi") == 1


def test_eof_exits_cleanly(harness):
    result = chat(input_text="")
    assert result.exit_code == 0
    assert "Bye." in result.output


def test_blank_lines_are_ignored(harness):
    result = chat(input_text="\n   \n/exit\n")
    assert result.exit_code == 0
    assert harness.agent.turns == []


def test_help_status_new_and_unknown(harness):
    result = chat(input_text="/help\n/status\n/new\n/unknown\n/\n/exit\n")
    assert result.exit_code == 0, result.output
    out = result.output
    assert "/model" in out and "/skill" in out
    assert "Thread" in out and "thread-1" in out
    assert "New conversation started." in out
    assert "Unknown command /unknown" in out
    assert "Type /help for commands." in out
    assert harness.agent.thread_id == "thread-2"


def test_model_switch_validates_provider(harness):
    harness.config = make_config(harness.config.workspace, providers={"internal": INTERNAL})
    result = chat(input_text="/model internal:foo\n/model nope:x\n/model\n/exit\n")
    assert result.exit_code == 0, result.output
    assert harness.agent.model == ("foo", "internal")
    assert "Model set to foo (internal)" in result.output
    assert "Unknown provider 'nope'" in result.output
    assert "Usage: /model" in result.output
    saved = json.loads(session_path(harness.config.workspace).read_text())
    assert saved["model"] == "foo" and saved["provider"] == "internal"


def test_notebook_and_context_commands(harness):
    harness.bundle = ContextBundle(
        context_text="x",
        context_files=[harness.config.workspace / "ctx.md"],
        skills=[SkillInfo("recon", "Reconcile months", harness.config.skills_dir / "recon")],
        prompts={"monthly": harness.config.prompts_dir / "monthly.md"},
        warnings=["ctx.md looks like it contains a secret"],
    )
    (harness.config.workspace / "ctx.md").write_text("hello", encoding="utf-8")
    result = chat(input_text="/notebook\n/context\n/exit\n")
    out = result.output
    assert "Launch:" not in out and "--foreground" not in out
    assert harness.url in out
    assert "ctx.md" in out and "(5 bytes)" in out
    assert "recon: Reconcile months" in out
    assert "Prompts: monthly" in out
    assert "looks like it contains a secret" in out


def test_skill_and_prompt_commands(harness):
    harness.bundle = ContextBundle(
        skills=[SkillInfo("recon", "Reconcile months", harness.config.skills_dir / "recon")],
        prompts={"monthly": harness.config.prompts_dir / "monthly.md"},
    )
    result = chat(input_text="/skill recon do it\n/skill nope\n/skill\n/prompt monthly 2025-03\n/prompt\n/exit\n")
    assert result.exit_code == 0, result.output
    assert ("do it", "recon") in harness.agent.turns
    assert ("PROMPT[monthly](2025-03)", None) in harness.agent.turns
    assert "Unknown skill 'nope'" in result.output
    assert "Usage: /skill" in result.output
    assert "Usage: /prompt" in result.output


def test_reload_reloads_context(harness):
    result = chat(input_text="/reload\n/exit\n")
    assert "Reloaded" in result.output
    assert harness.context_loads == 2
    assert harness.agent.bundle is harness.bundle


def test_resume_and_new_flag(harness):
    save_session(harness.config.workspace, SessionState(thread_id="old-thread", turns=4))
    result = chat()
    assert "Resumed conversation (4 turns so far)" in result.output
    assert harness.agent.started == ["old-thread"]

    harness.agent = FakeAgent()
    result = chat(args=["--new"])
    assert result.exit_code == 0
    assert harness.agent.started == [None]
    assert "Resumed" not in result.output


def test_resume_fallback_message(harness):
    save_session(harness.config.workspace, SessionState(thread_id="gone", turns=2))
    harness.agent = FakeAgent()
    harness.agent.start = lambda *, resume_thread_id=None, forget_thread_id=None: "fresh"  # the thread is not in the store
    result = chat()
    assert "could not be resumed" in result.output
    saved = json.loads(session_path(harness.config.workspace).read_text())
    assert saved["thread_id"] == "fresh" and saved["turns"] == 0


def test_hailer_error_in_turn_keeps_loop(harness):
    harness.agent.fail_with = CredentialsError("Endpoint rejected the API key (401).", hint="Run: uvx hailer login openai")
    result = chat(input_text="hello\n/exit\n")
    assert result.exit_code == 0
    assert "rejected the API key" in result.output
    assert "hailer login openai" in result.output
    assert "Bye." in result.output


def test_unexpected_error_in_turn_keeps_loop(harness):
    harness.agent.fail_with = RuntimeError("kaboom")
    result = chat(input_text="hello\n/exit\n")
    assert result.exit_code == 0
    assert "Unexpected error: RuntimeError: kaboom" in result.output
    assert "--verbose" in result.output
    assert "Bye." in result.output


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #


def test_config_error_exits_1(harness, monkeypatch):
    def boom(opts):
        raise ConfigError("hailer.toml line 3: bad value", hint="Fix the value or delete the line.")

    monkeypatch.setattr(cli, "_load_config", boom)
    result = chat()
    assert result.exit_code == 1
    assert "line 3" in result.output and "Fix the value" in result.output


def test_missing_notebook_is_fatal(harness):
    harness.config = make_config(harness.config.workspace, notebook_exists=False)
    result = chat()
    assert result.exit_code == 1
    assert "notebook: not found" in result.output
    assert "[hailer].notebook" in result.output


def test_missing_key_for_custom_provider_is_fatal(harness):
    harness.config = make_config(harness.config.workspace, provider="internal", providers={"internal": INTERNAL})
    harness.key_source = (None, "missing")
    result = chat()
    assert result.exit_code == 1
    assert "INTERNAL_MODEL_API_KEY is not set" in result.output
    assert "uvx hailer login internal" in result.output


def test_missing_openai_key_is_fatal_too(harness):
    """The built-in provider needs OPENAI_API_KEY like any other endpoint needs its key."""
    harness.key_source = (None, "missing")
    result = chat()
    assert result.exit_code == 1
    assert harness.warmups == [] and harness.kernel_starts == [], "no kernel for a session that cannot start"
    assert "OPENAI_API_KEY is not set" in result.output
    assert "uvx hailer login openai" in result.output
    assert harness.agent.turns == []


def test_every_chat_starts_its_own_kernel_opens_the_notebook_and_stops_the_kernel(harness):
    """Bare hailer: its own kernel (nothing is looked for or attached to), the notebook opened and
    waited for, and the kernel stopped when the chat ends."""
    harness.client = FakeClient(session=None)
    result = chat()
    assert result.exit_code == 0, result.output
    assert [k.stopped for k in harness.kernel_starts] == [1]
    assert harness.agent_sandboxes == harness.kernel_starts, "the agent's tools get the kernel in memory"
    assert harness.opened == [harness.url]
    assert harness.session_waits == ["analysis.py"]
    assert "Notebook is open (session s9)." in result.output
    assert "Stopped marimo." in result.output
    result = chat()
    assert [k.stopped for k in harness.kernel_starts] == [1, 1], "the next chat starts another"


def test_invalid_config_problem_is_fatal(harness, monkeypatch):
    monkeypatch.setattr(cli, "_validate_config", lambda config: ["[model_providers.internal].base_url is missing", "Warning: data directory not found"])
    result = chat()
    assert result.exit_code == 1
    assert "base_url is missing" in result.output


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #


def test_exec_slash_command_runs_in_the_chats_own_kernel(harness):
    result = chat(input_text="/exec print(1)\n/help\n/exit\n")
    assert result.exit_code == 0, result.output
    assert harness.client.codes == ["print(1)"]
    assert "out" in result.output and "42" in result.output
    assert harness.agent.turns == [], "nothing goes to the model"
    assert "/exec" in result.output, "listed by /help"


def test_exec_slash_command_dedents_an_indented_paste():
    from io import StringIO

    from hailer.chat import ChatController

    class Client(FakeClient):
        pass

    client = Client()
    services = type("S", (), {})()
    services._marimo_state = lambda sandbox, notebook: (sandbox, MarimoSession("s1", "a.py", "a.py", name="a.py"), None)
    controller = ChatController.__new__(ChatController)
    controller.console, controller._cli, controller.config = Console(file=StringIO()), services, None
    controller.sandbox, controller.notebook = type("Box", (), {"client": lambda self, **kw: client})(), "a.py"
    controller._stop_event = None
    controller._exec("    x = 1\n    print(x)")
    assert client.codes == ["x = 1\nprint(x)"]


def test_exec_slash_command_failure_usage_and_no_session(harness):
    harness.client = FakeClient(exec_result=ExecResult(False, stderr="Traceback: boom"))
    result = chat(input_text="/exec 1/0\n/exec\n/exit\n")
    assert "boom" in result.output and "(the code failed)" in result.output
    assert "Usage: /exec <python code>" in result.output and harness.client.codes == ["1/0"]
    harness.client = FakeClient(session=None)
    harness.waited_session = None
    result = chat(input_text="/exec 1\n/exit\n")
    assert "not open in a browser" in result.output and harness.url in result.output
    assert harness.client.codes == []


def test_exec_slash_command_talks_to_the_real_server_with_its_token(harness, monkeypatch):
    from fake_marimo import serving

    with serving(token=TOKEN) as srv:
        srv.sessions = {"s1": {"filename": "notebooks/analysis.py", "path": str(harness.config.notebook)}}
        harness.server, harness.real_clients = MarimoServer(url=srv.url, token=TOKEN), True
        result = chat(input_text="/exec print('x')\n/exit\n")
    assert result.exit_code == 0, result.output
    assert "hello world" in result.output and "42" in result.output
    execute = [r for r in srv.requests if r["path"] == "/api/kernel/execute"][-1]
    assert execute["headers"]["Authorization"] == f"Bearer {TOKEN}" and execute["headers"]["Marimo-Session-Id"] == "s1"
    assert execute["body"] == {"code": "print('x')"}


def test_doctor_table_never_starts_or_probes_a_kernel(harness):
    result = runner.invoke(cli.app, ["doctor"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "Hailer doctor" in result.output
    assert "OK" in result.output
    assert "code mode" not in result.output and "marimo" not in result.output.lower().split("hailer doctor")[1]
    assert harness.kernel_starts == [] and harness.client.codes == []


def test_doctor_fails_when_notebook_missing(harness):
    harness.config = make_config(harness.config.workspace, notebook_exists=False)
    result = runner.invoke(cli.app, ["doctor"], catch_exceptions=False)
    assert result.exit_code == 1
    assert "FAIL" in result.output


def test_status_subcommand(harness):
    result = runner.invoke(cli.app, ["status"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "Credentials: OPENAI_API_KEY from env" in result.output
    assert "Kernels:     none running for this workspace (each uvx hailer session starts its own)" in result.output
    assert "Context:     0 file(s)" in result.output
    harness.kernels = ["local marimo at http://127.0.0.1:2718 (process 7, started by Hailer process 6)", "docker hailer-kernel-x"]
    result = runner.invoke(cli.app, ["status"], catch_exceptions=False)
    assert "Kernels:     local marimo at http://127.0.0.1:2718 (process 7, started by Hailer process 6)" in result.output
    assert "             docker hailer-kernel-x" in result.output
    assert harness.kernel_starts == [], "status starts nothing"


def test_login_and_logout(harness):
    harness.config = make_config(harness.config.workspace, providers={"internal": INTERNAL})
    result = runner.invoke(cli.app, ["login", "internal"], input="s3cret\n", catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert harness.stored == [("internal", "s3cret")]
    assert "s3cret" not in result.output
    result = runner.invoke(cli.app, ["logout", "internal"], catch_exceptions=False)
    assert result.exit_code == 0 and "Removed." in result.output
    assert harness.deleted == ["internal"]
    result = runner.invoke(cli.app, ["login", "nope"], input="x\n", catch_exceptions=False)
    assert result.exit_code == 1
    assert "Unknown provider 'nope'" in result.output


# --------------------------------------------------------------------------- #
# hailer notebook: the one-command session
# --------------------------------------------------------------------------- #


class FakeProc:
    """Stands in for the marimo subprocess."""

    def __init__(self, exit_code=None):
        self.pid = 4242
        self.returncode = exit_code
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


class ForegroundProc(FakeProc):
    """marimo attached to the terminal: waiting on it returns once the user stops it."""

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


@dataclass
class NotebookHarness:
    spawned: list = field(default_factory=list)
    foreground: list = field(default_factory=list)
    health_waits: list = field(default_factory=list)
    session_waits: list = field(default_factory=list)
    stdin: list = field(default_factory=list)  # what each started server read on stdin (the token)
    envs: list = field(default_factory=list)  # the environment each started server got
    proc: FakeProc = field(default_factory=FakeProc)
    foreground_proc: FakeProc = field(default_factory=ForegroundProc)
    healthy: bool = True
    waited_session: object = "default"
    free_port: int | None = None


@pytest.fixture
def nb(harness, monkeypatch):
    monkeypatch.setattr(cli, "_start_kernel", REAL_START_KERNEL)
    """Notebook-command collaborators on top of the chat harness: the real LocalRuntime with its
    process layer faked (nothing is spawned or killed). Marimo is not running by default."""
    from hailer.kernel import LocalProcesses, LocalRuntime

    h = NotebookHarness()

    def spawn(cmd, cwd, log_path, *, env=None, stdin_text=None):
        h.spawned.append((cmd, cwd, log_path))
        h.stdin.append(stdin_text)
        h.envs.append(env)
        return h.proc

    def attach(cmd, cwd, *, env=None, stdin_text=None):
        h.foreground.append((cmd, cwd))
        h.stdin.append(stdin_text)
        h.envs.append(env)
        return h.foreground_proc

    def wait_health(url, timeout, token=None, should_stop=None):
        h.health_waits.append(url)
        return h.healthy

    def wait_session(client, notebook, timeout, *, should_stop=None):
        h.session_waits.append(notebook)
        if h.waited_session == "default":
            return MarimoSession("s9", "analysis.py", "/nb/analysis.py", name="analysis.py")
        return h.waited_session

    procs = LocalProcesses(
        spawn=spawn,
        attach=attach,
        kill_tree=lambda pid: None,
        wait_for_health=wait_health,
    )

    class Runtime(LocalRuntime):
        """The real start; the sandbox it returns serves the notebooks folder without a marimo
        server behind it, and its clients are the harness's."""

        def start(self, port, *, foreground=False):
            running = super().start(port, foreground=foreground)
            return with_folder_files(running, self.config.notebooks_root, lambda **kw: harness.bound_client(running))

    monkeypatch.setattr(cli, "_runtime_for", lambda config: Runtime(config, procs=procs, token_factory=lambda: TOKEN))
    monkeypatch.setattr(cli, "_find_free_port", lambda preferred: h.free_port if h.free_port is not None else preferred)
    monkeypatch.setattr(cli, "_wait_for_session", wait_session)
    return h


def notebook_cmd(args=(), input_text="/exit\n"):
    return runner.invoke(cli.app, ["notebook", *args], input=input_text, catch_exceptions=False)


def test_notebook_starts_marimo_runs_chat_and_stops_it(harness, nb, monkeypatch):
    from hailer.kernel import local_log_path

    ws = harness.config.workspace
    monkeypatch.setenv("OPENAI_API_KEY", "sk-parent-shell-key-123456")
    result = notebook_cmd(["--port", "2718"], input_text="hi\n/exit\n")
    assert result.exit_code == 0, result.output
    cmd, cwd, log_path = nb.spawned[0]
    assert cwd == ws
    assert log_path == local_log_path(ws)
    assert cmd == server_command(2718), "headless, token on stdin (never --no-token, never on the command line)"
    assert TOKEN not in " ".join(cmd) and nb.stdin == [TOKEN]
    assert "OPENAI_API_KEY" not in nb.envs[0], "the kernel's environment has no API key"
    assert nb.health_waits == ["http://127.0.0.1:2718"]
    # the notebook already had a session, so no browser and no wait
    assert harness.opened == [] and nb.session_waits == []
    assert "Notebook is open (session s1)" in result.output
    assert "Kernel:     local (runs as you; not isolated)" in result.output
    assert harness.agent.started == [None], "the chat ran in the same terminal"
    assert nb.proc.terminated
    assert "Stopped marimo." in result.output
    assert harness.agent.closed


def test_notebook_opens_browser_and_waits_for_session(harness, nb):
    harness.client = FakeClient(session=None)  # server up, nobody has the tab open yet
    result = notebook_cmd()
    assert result.exit_code == 0, result.output
    assert harness.opened == [signed(harness.url)], "browser opened exactly once, signed in with the server's token"
    assert nb.session_waits == ["analysis.py"]
    assert "Notebook is open (session s9)" in result.output
    assert "app view" in result.output and "Ctrl+." in result.output, "tells the user how to reach the code"


def test_notebook_no_browser_still_waits_and_continues_without_session(harness, nb):
    harness.client = FakeClient(session=None)
    nb.waited_session = None
    result = notebook_cmd(["--no-browser"])
    assert result.exit_code == 0, result.output
    assert harness.opened == []
    assert f"Open {signed(harness.url)} in your browser." in result.output
    assert "No kernel session yet" in result.output
    assert harness.agent.started == [None], "chat still starts; the agent reports the notebook state"
    assert nb.proc.terminated


def test_dependency_warmup_starts_before_kernel_start(harness, nb, monkeypatch):
    order = []
    start_kernel = cli._start_kernel

    def start(*args, **kwargs):
        order.append("kernel")
        return start_kernel(*args, **kwargs)

    monkeypatch.setattr(cli, "_start_dependency_warmup", lambda: order.append("dependencies"))
    monkeypatch.setattr(cli, "_start_kernel", start)
    result = notebook_cmd()
    assert result.exit_code == 0, result.output
    assert order == ["dependencies", "kernel"]


def test_startup_exit_settles_notebook_worker_before_stopping_owned_kernel(harness, nb, monkeypatch):
    harness.client = FakeClient(session=None)
    polling, cleanup_started, release = threading.Event(), threading.Event(), threading.Event()

    def wait_session(client, notebook, timeout, *, should_stop=None):
        assert should_stop is not None
        polling.set()
        deadline = time.monotonic() + 5
        while not should_stop():
            assert time.monotonic() < deadline, "startup cancellation did not stop polling"
            time.sleep(0.01)
        cleanup_started.set()
        assert release.wait(5), "test did not release cleanup"
        assert not nb.proc.terminated, "the kernel must outlive its startup worker"
        return None

    async def run_interactive(self, ui_factory=None, *, prepare_notebook=None, timings=None):
        task = asyncio.create_task(prepare_notebook())
        try:
            async with asyncio.timeout(5):
                while not polling.is_set():
                    await asyncio.sleep(0.01)
            task.cancel()
            async with asyncio.timeout(5):
                while not cleanup_started.is_set():
                    await asyncio.sleep(0.01)
            assert not task.done() and not nb.proc.terminated
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)

    monkeypatch.setattr(cli, "_use_composer", lambda opts: True)
    monkeypatch.setattr(cli, "_wait_for_session", wait_session)
    monkeypatch.setattr(cli.ChatLoop, "run_interactive", run_interactive)
    result = notebook_cmd()
    assert result.exit_code == 0, result.output
    assert nb.proc.terminated
    assert "Stopped marimo." in result.output
    assert "No kernel session yet" not in result.output


def test_notebook_stops_marimo_even_when_chat_fails(harness, nb):
    harness.agent.fail_with = HailerError("boom", hint="fix it")
    result = notebook_cmd(input_text="hello\n/exit\n")
    assert "boom" in result.output and "fix it" in result.output
    assert nb.proc.terminated and "Stopped marimo." in result.output


def test_notebook_reports_early_exit_with_log_tail(harness, nb, tmp_path):
    from hailer.kernel import local_log_path

    nb.healthy = False
    nb.proc = FakeProc(exit_code=1)
    log = local_log_path(harness.config.workspace)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("\n".join(f"line {i}" for i in range(20)), encoding="utf-8")
    result = notebook_cmd()
    assert result.exit_code == 1
    assert "Marimo exited early (code 1)" in result.output
    assert "Last lines of its log:" in result.output
    assert "line 19" in result.output and "line 4" not in result.output, "last 15 lines only"
    assert harness.agent.started == []
    assert not log.exists(), "the tail is in the message; the log does not stay behind"


def test_ctrl_c_right_after_the_kernel_started_still_stops_it(harness, nb, monkeypatch):
    """Between the runtime returning the kernel and the chat's guarded block, Ctrl+C must not leak it."""
    real_runtime_for = cli._runtime_for
    started: list = []

    class Interrupted:
        def __init__(self, running):
            self.running = running

        @property
        def server(self):
            raise KeyboardInterrupt  # the first thing _start_kernel does with it: print its URL

        def stop(self):
            started.append("stopped")
            self.running.stop()

    class Runtime:
        def __init__(self, config):
            self.inner = real_runtime_for(config)
            self.name = self.inner.name

        def prepare(self, say=None):
            self.inner.prepare(say)

        def start(self, port, **kw):
            return Interrupted(self.inner.start(port, **kw))

    monkeypatch.setattr(cli, "_runtime_for", Runtime)
    with pytest.raises(KeyboardInterrupt):
        cli._start_kernel(Console(file=io.StringIO()), Runtime(harness.config), 2718, verbose=False)
    assert started == ["stopped"] and nb.proc.terminated


def test_notebook_picks_a_free_port_when_busy(harness, nb):
    nb.free_port = 2731
    result = notebook_cmd(["--port", "2718"])
    assert result.exit_code == 0, result.output
    assert "Port 2718 is busy; using 2731." in result.output
    assert nb.health_waits == ["http://127.0.0.1:2731"]


def test_every_notebook_session_starts_its_own_server(harness, nb):
    """A server that is already up (another terminal's) is never looked for: each run starts one."""
    for _ in range(2):
        result = notebook_cmd()
        assert result.exit_code == 0, result.output
        assert "Using the running marimo" not in result.output and "Stopped marimo." in result.output
    assert len(nb.spawned) == 2 and nb.proc.terminated


def test_notebook_fatal_local_check_exits_before_starting_marimo(harness, nb):
    harness.config = make_config(harness.config.workspace, notebook_exists=False)
    result = notebook_cmd()
    assert result.exit_code == 1
    assert "notebook: not found" in result.output
    assert "Run uvx hailer init to create it" in result.output
    assert nb.spawned == []


def test_notebook_foreground_runs_marimo_attached_without_chat(harness, nb, monkeypatch):
    result = notebook_cmd(["--foreground", "--port", "2718"])
    assert result.exit_code == 0, result.output
    assert nb.foreground == [(server_command(2718), harness.config.workspace)], "Hailer's interpreter, not uv run"
    assert nb.stdin == [TOKEN], "the token goes to marimo's stdin in the foreground too"
    assert nb.spawned == [] and harness.agent.started == []
    # Hailer opens marimo's home page signed in
    assert harness.opened == [f"http://127.0.0.1:2718/?access_token={TOKEN}"]
    assert "Kernel:     local (runs as you; not isolated)" in result.output
    assert "Marimo is running at http://127.0.0.1:2718. Ctrl+C stops it." in result.output


def test_notebook_foreground_no_browser_does_not_open_one(harness, nb):
    result = notebook_cmd(["--foreground", "--no-browser", "--port", "2720"])
    assert result.exit_code == 0, result.output
    assert nb.foreground == [(server_command(2720), harness.config.workspace)]
    assert harness.opened == []
    assert f"Open http://127.0.0.1:2720/?access_token={TOKEN} in your browser." in result.output


def test_notebook_foreground_reports_a_server_that_does_not_start(harness, nb):
    nb.healthy = False
    nb.foreground_proc = ForegroundProc(exit_code=2)
    result = notebook_cmd(["--foreground"])
    assert result.exit_code == 1
    assert "Marimo exited early (code 2)." in result.output
    assert harness.opened == []


def test_notebook_new_flag_starts_fresh_thread(harness, nb):
    save_session(harness.config.workspace, SessionState(thread_id="old-thread", turns=3))
    result = notebook_cmd(["--new"])
    assert result.exit_code == 0, result.output
    assert harness.agent.started == [None]


def test_doctor_and_status_never_start_a_kernel(harness):
    """doctor checks config, Docker, the image and folders; status lists running kernels."""
    result = runner.invoke(cli.app, ["doctor"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "No marimo server" not in result.output and "session" not in result.output
    result = runner.invoke(cli.app, ["status"], catch_exceptions=False)
    assert "Kernels:     none running for this workspace" in result.output
    assert harness.kernel_starts == []


def init_cmd(ws: Path, *args: str):
    return runner.invoke(cli.app, ["--workspace", str(ws), "init", *args], catch_exceptions=False)


def test_init_writes_config_and_skeleton(harness, tmp_path, monkeypatch):
    ws = tmp_path / "fresh"
    ws.mkdir()
    monkeypatch.setattr(cli, "_example_config_dir", lambda: None)
    result = init_cmd(ws)
    assert result.exit_code == 0, result.output
    assert (ws / "hailer.toml").exists()
    for sub in ("context", "skills", "prompts"):
        assert (ws / ".config" / "hailer" / sub / "README.md").exists()
    assert "Next steps" in result.output
    assert "Put the files you want to analyse in data/ (CSV, Parquet, JSON or Excel, any name)." in result.output
    result = init_cmd(ws)
    assert "already exists" in result.output


def test_init_creates_the_notebook_and_data_folder_so_notebook_can_start(harness, tmp_path, monkeypatch):
    """Regression: `hailer init` in an empty folder, then `hailer notebook`, failed with "Notebook not found"."""
    from hailer.config import load_config, validate

    ws = tmp_path / "fresh"
    ws.mkdir()
    monkeypatch.setattr(cli, "_example_config_dir", lambda: None)
    result = init_cmd(ws)
    assert result.exit_code == 0, result.output
    notebook = ws / "notebooks" / "analysis.py"
    assert "import marimo" in notebook.read_text(encoding="utf-8") and "marimo.App" in notebook.read_text(encoding="utf-8")
    assert "# Hailer workspace" in notebook.read_text(encoding="utf-8")
    assert (ws / "data").is_dir()
    assert "Created notebooks/analysis.py (starter notebook), data/" in result.output
    assert validate(load_config(workspace=ws)) == [], "the checks `hailer notebook` runs first now pass"
    assert "uvx hailer login openai" in result.output and "uvx hailer notebook" in result.output
    assert "uv run" not in result.output, "no project .venv is needed"


def test_init_again_keeps_the_notebook_and_fills_in_what_is_missing(harness, tmp_path, monkeypatch):
    ws = tmp_path / "fresh"
    ws.mkdir()
    monkeypatch.setattr(cli, "_example_config_dir", lambda: None)
    init_cmd(ws)
    notebook = ws / "notebooks" / "analysis.py"
    notebook.write_text(NOTEBOOK_SOURCE, encoding="utf-8")
    (ws / "data").rmdir()
    result = init_cmd(ws, "--force")
    assert result.exit_code == 0, result.output
    assert notebook.read_text(encoding="utf-8") == NOTEBOOK_SOURCE, "--force only replaces hailer.toml"
    assert "Created data/" in result.output and (ws / "data").is_dir()
    result = init_cmd(ws)
    assert "notebooks/analysis.py already exists." in result.output


def test_init_uses_the_notebook_and_data_dir_from_an_existing_config(harness, tmp_path, monkeypatch):
    ws = tmp_path / "custom"
    ws.mkdir()
    (ws / "hailer.toml").write_text(
        '[hailer]\nnotebook = "work/q2.py"\ndata_dir = "inputs"\n\n[model]\nname = "m"\nprovider = "internal"\n\n'
        '[model_providers.internal]\nbase_url = "https://llm.example.internal/v1"\nenv_key = "INTERNAL_MODEL_API_KEY"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "_example_config_dir", lambda: None)
    result = init_cmd(ws)
    assert result.exit_code == 0, result.output
    assert "already exists" in result.output, "the existing hailer.toml is kept"
    assert "marimo.App" in (ws / "work" / "q2.py").read_text(encoding="utf-8")
    assert (ws / "inputs").is_dir() and not (ws / "notebooks").exists() and not (ws / "data").exists()
    assert "uvx hailer login internal" in result.output
    assert "inputs/" in result.output


def test_init_with_a_broken_config_skips_the_notebook(harness, tmp_path, monkeypatch):
    ws = tmp_path / "broken"
    ws.mkdir()
    (ws / "hailer.toml").write_text("[hailer\nnotebook = ", encoding="utf-8")
    monkeypatch.setattr(cli, "_example_config_dir", lambda: None)
    result = init_cmd(ws)
    assert result.exit_code == 0, result.output
    assert "Skipped the notebook and data folder" in result.output
    assert not (ws / "notebooks").exists()
    assert "uvx hailer login <provider>" in result.output


def test_version_flag():
    result = runner.invoke(cli.app, ["--version"], catch_exceptions=False)
    assert result.exit_code == 0
    assert f"hailer {__version__}" in result.output


# --------------------------------------------------------------------------- #
# Review fixes: markup, encoding, interrupts, status, model switch, staleness
# --------------------------------------------------------------------------- #


def test_validation_message_with_brackets_is_printed_verbatim(harness, monkeypatch):
    monkeypatch.setattr(
        cli,
        "_validate_config",
        lambda config: ["[web].allowed_domains entry '*' is invalid", '[model_providers.internal].wire_api must be "responses"'],
    )
    result = chat()
    assert result.exit_code == 1
    assert "[web].allowed_domains entry '*' is invalid" in result.output
    assert '[model_providers.internal].wire_api must be "responses"' in result.output


def test_error_hint_with_brackets_is_printed_verbatim(harness):
    harness.agent.fail_with = HailerError("Bad [web] section", hint="Fix [web].allowed_domains in hailer.toml")
    result = chat(input_text="hello\n/exit\n")
    assert "Bad [web] section" in result.output
    assert "Fix [web].allowed_domains in hailer.toml" in result.output


def test_reconfigure_streams_makes_cp1252_pipe_safe(monkeypatch):
    raw = io.BytesIO()
    pipe = io.TextIOWrapper(raw, encoding="cp1252")  # strict, like a redirected stdout on Windows
    monkeypatch.setattr(sys, "stdout", pipe)
    monkeypatch.setattr(sys, "stderr", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))
    assert pipe.errors == "strict"
    cli._reconfigure_streams()
    assert sys.stdout.errors == "replace" and sys.stderr.errors == "replace"
    console = Console(file=sys.stdout, force_terminal=False, width=80, highlight=False, color_system=None)
    console.print("┌─┐ → 🙂 shape: (6, 2)", markup=False)  # Polars frames and emoji must not crash
    sys.stdout.flush()
    assert b"shape: (6, 2)" in raw.getvalue()


def test_ctrl_c_during_turn_interrupts_and_keeps_loop(harness):
    harness.agent.fail_with = KeyboardInterrupt()
    result = chat(input_text="long running\n/exit\n")
    assert result.exit_code == 0, result.output
    assert "Interrupted." in result.output
    assert "Bye." in result.output  # the loop survived and took the next command


def test_ctrl_c_at_prompt_exits_cleanly(harness, monkeypatch):
    def raise_interrupt(self, prompt="", **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(Console, "input", raise_interrupt)
    result = chat(input_text="")
    assert result.exit_code == 0
    assert "Bye." in result.output
    assert harness.agent.closed


def test_verbose_prints_traceback_for_hailer_error(harness):
    harness.agent.fail_with = CredentialsError("Endpoint rejected the API key (401).", hint="Run: uvx hailer login openai")
    result = chat(args=["--verbose"], input_text="hello\n/exit\n")
    assert "rejected the API key" in result.output
    assert "Traceback" in result.output


def test_unknown_prompt_reports_available(harness, monkeypatch):
    def render(config, name, args):
        raise HailerError(f"Unknown prompt {name!r}.", hint="Available prompts: monthly-pack")

    monkeypatch.setattr(cli, "_render_prompt", render)
    result = chat(input_text="/prompt nope\n/exit\n")
    assert "Unknown prompt 'nope'" in result.output
    assert "Available prompts: monthly-pack" in result.output
    assert harness.agent.turns == []


def test_login_openai_then_status_reports_keyring(harness, monkeypatch):
    stored: dict[str, str] = {}
    monkeypatch.setattr(cli, "_store_key", lambda provider, value: stored.__setitem__(provider.id, value))
    monkeypatch.setattr(cli, "_resolve_key", lambda provider: ("v", "keyring") if provider.id in stored else (None, "missing"))
    result = runner.invoke(cli.app, ["status"], catch_exceptions=False)
    assert "Credentials: OPENAI_API_KEY missing (run: uvx hailer login openai)" in result.output
    result = runner.invoke(cli.app, ["login", "openai"], input="sk-test\n", catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert stored == {"openai": "sk-test"}
    assert "sk-test" not in result.output
    result = runner.invoke(cli.app, ["status"], catch_exceptions=False)
    assert "Credentials: OPENAI_API_KEY from keyring" in result.output
    result = runner.invoke(cli.app, ["doctor"], catch_exceptions=False)
    assert "OPENAI_API_KEY from keyring" in result.output


def test_status_command_shows_endpoint_and_credentials(harness):
    harness.config = make_config(harness.config.workspace, provider="internal", providers={"internal": INTERNAL})
    harness.agent.key_source = "keyring"
    result = chat(input_text="/status\n/exit\n")
    assert result.exit_code == 0, result.output
    assert "internal (https://llm.example.internal/v1)" in result.output
    assert "INTERNAL_MODEL_API_KEY from keyring" in result.output


def test_status_command_marks_chat_completions_providers(harness):
    chat_provider = ProviderConfig(
        id="internal", base_url="https://llm.example.internal/v1", wire_api="chat", env_key="INTERNAL_MODEL_API_KEY"
    )
    harness.config = make_config(harness.config.workspace, provider="internal", providers={"internal": chat_provider})
    harness.agent.key_source = "env"
    result = chat(input_text="/status\n/exit\n")
    assert result.exit_code == 0, result.output
    assert "internal (https://llm.example.internal/v1, chat completions)" in result.output


def test_status_command_marks_non_streaming_chat_providers(harness):
    chat_provider = ProviderConfig(
        id="internal", base_url="https://llm.example.internal/v1", wire_api="chat", stream=False, env_key="INTERNAL_MODEL_API_KEY"
    )
    harness.config = make_config(harness.config.workspace, provider="internal", providers={"internal": chat_provider})
    harness.agent.key_source = "env"
    result = chat(input_text="/status\n/exit\n")
    assert result.exit_code == 0, result.output
    assert "internal (https://llm.example.internal/v1, chat completions, no streaming)" in result.output


def test_model_switch_to_another_provider_starts_exactly_one_thread(harness):
    harness.config = make_config(harness.config.workspace, providers={"internal": INTERNAL})
    result = chat(input_text="/model internal:foo\n/exit\n")
    assert result.exit_code == 0, result.output
    assert harness.agent.model == ("foo", "internal")
    assert harness.agent.new_threads == 1
    assert result.output.count("started a new thread") == 1
    saved = json.loads(session_path(harness.config.workspace).read_text())
    assert (saved["thread_id"], saved["model"], saved["provider"]) == ("thread-2", "foo", "internal")


def test_resume_has_no_stale_prompt_warning(harness):
    """The instructions are sent with every turn, so a resumed conversation always uses the current ones."""
    save_session(harness.config.workspace, SessionState(thread_id="old-thread", turns=1))
    result = chat()
    assert "Resumed conversation (1 turns so far)" in result.output
    assert "use /new to apply" not in result.output
    assert harness.agent.started == ["old-thread"] and harness.agent.forgotten == [None]


def test_new_flag_starts_fresh_and_has_the_stored_conversation_deleted(harness):
    save_session(harness.config.workspace, SessionState(thread_id="old-thread", turns=4))
    result = chat(args=["--new"])
    assert result.exit_code == 0, result.output
    assert "Resumed conversation" not in result.output
    assert harness.agent.started == [None] and harness.agent.forgotten == ["old-thread"]
    saved = json.loads(session_path(harness.config.workspace).read_text())
    assert (saved["thread_id"], saved["turns"]) == ("thread-1", 0)


def test_commentary_deltas_are_not_printed_and_answer_appears_once(harness):
    harness.agent.stream = True
    result = chat(input_text="hi\n/exit\n")
    assert result.output.count("Answer to: hi") == 1
    assert "Checking the notebook" not in result.output


def test_progress_line_shows_the_tool_in_use(harness):
    console = Console(force_terminal=False, width=100, highlight=False, color_system=None)
    display = cli._TurnDisplay(console)
    display(cli.AgentEvent("tool_call", "marimo_execute", {"arguments": "{'code': 'df.head()'}"}))
    assert display.last_activity == "marimo_execute"
    display(cli.AgentEvent("tool_call", "notebook_open\nq2"))
    assert display.last_activity == "notebook_open q2"  # one line, whatever the event carries


# --------------------------------------------------------------------------- #
# Notebooks: /notebook, switches made in the chat or by the model, the preamble
# --------------------------------------------------------------------------- #


def test_load_config_keeps_the_configured_notebook_and_hailer_notebook_is_saved_as_a_name(tmp_path, monkeypatch):
    """config.notebook stays the configured notebook; the active one is a name in the state file.
    An explicit HAILER_NOTEBOOK is written there, so the tools (which read only the file) and the
    next session start on the same notebook."""
    base = make_config(tmp_path)
    write_notebook(base, "other")
    monkeypatch.delenv("HAILER_NOTEBOOK", raising=False)
    monkeypatch.setattr("hailer.config.load_config", lambda workspace=None, config_path=None: base)
    notebooks.save_active_notebook(base, "other.py")
    assert cli._load_config(cli.CliOptions()) is base
    assert notebooks.load_active_notebook(base) == "other.py"
    monkeypatch.setenv("HAILER_NOTEBOOK", "explicit")
    assert cli._load_config(cli.CliOptions()).notebook == base.notebook
    assert active_state(base) == "analysis.py" and notebooks.load_active_notebook(base) == "analysis.py"


def test_a_session_converts_an_old_state_file_and_falls_back_from_a_deleted_notebook(harness):
    """notebook.json from before names held host paths: the first session converts it. An active
    notebook the sandbox no longer has falls back to the configured one."""
    write_notebook(harness.config, "q3/review")
    state = notebooks.state_path(harness.config.workspace)
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({"active": "notebooks/q3/review.py", "recent": ["notebooks/q3/review.py", "../x.py"]}), encoding="utf-8")
    result = chat()
    assert result.exit_code == 0, result.output
    assert "Notebook:   q3/review.py" in result.output
    assert json.loads(state.read_text(encoding="utf-8")) == {"version": 2, "active": "q3/review.py", "recent": ["q3/review.py"]}
    (harness.config.notebooks_root / "q3" / "review.py").unlink()
    result = chat()
    assert "Notebook:   analysis.py" in result.output and active_state(harness.config) == "analysis.py"


def test_notebook_command_shows_active_folder_and_usage(harness):
    result = chat(input_text="/notebook\n/exit\n")
    out = result.output
    assert "Notebook:  analysis.py (active)" in out
    assert "Notebooks: notebooks (1 notebook; /notebook list)" in out
    assert "Marimo:    http://127.0.0.1:2718 (session s1)" in out
    assert harness.url in out
    assert "Usage: /notebook [list | new <name> [--empty] | open <name> | close [name]]" in out
    result = chat(input_text="/notebook wat\n/exit\n")
    assert "Unknown /notebook subcommand 'wat'" in result.output


def test_notebook_list_marks_active_and_open_and_skips_non_notebooks(harness):
    write_notebook(harness.config, "other")
    (harness.config.notebooks_root / "helper.py").write_text("print('not a notebook')\n", encoding="utf-8")
    cache = harness.config.notebooks_root / "__marimo__"
    cache.mkdir()
    (cache / "cached.py").write_text(NOTEBOOK_SOURCE, encoding="utf-8")
    result = chat(input_text="/notebook list\n/exit\n")
    lines = [ln for ln in result.output.splitlines() if ".py " in ln]
    assert any("analysis.py " in ln and "active, open" in ln for ln in lines), lines
    assert any("other.py " in ln and "active" not in ln and "open" not in ln for ln in lines), lines
    assert "helper.py" not in result.output and "cached.py" not in result.output
    assert "2 notebooks in notebooks." in result.output


def test_notebook_list_when_folder_is_empty(harness):
    harness.agent.on_turn = lambda: harness.config.notebook.unlink()  # the file disappears mid-session
    result = chat(input_text="hi\n/notebook list\n/exit\n")
    assert "No notebooks in notebooks yet. Create one with /notebook new <name>." in result.output


def test_notebook_new_creates_switches_and_sends_preamble_once(harness):
    result = chat(input_text="/notebook new Q2 Churn\nhello\nagain\n/exit\n")
    assert result.exit_code == 0, result.output
    created = harness.config.notebooks_root / "q2_churn.py"
    assert created.exists()
    source = created.read_text(encoding="utf-8")
    assert "marimo.App" in source and "hailer.periods" in source and "# Q2 Churn" in source
    assert "Created q2_churn.py from the starter template." in result.output
    assert "Active notebook: q2_churn.py." in result.output
    assert active_state(harness.config) == "q2_churn.py"
    # a freshly created notebook has no session: its app-view URL is opened and the tab is waited for
    assert harness.opened == [url_for(created)]
    assert harness.session_waits == ["q2_churn.py"]
    assert harness.client.granted == ["q2_churn.py"]
    assert "Notebook is open (session s9)." in result.output
    assert harness.agent.turns == [("hello", None), ("again", None)]
    assert harness.agent.preambles == [
        "[Hailer] The active notebook is now q2_churn.py (created from the starter template, 1 cell). "
        "Call notebook_cells before editing.",
        None,
    ]
    assert harness.client.codes and "cm.get_context()" in harness.client.codes[0], "cell count via the list-cells snippet"


def test_notebook_new_empty_template_and_status_panel(harness):
    result = chat(input_text="/notebook new scratch --empty\n/status\n/exit\n")
    assert result.exit_code == 0, result.output
    source = (harness.config.notebooks_root / "scratch.py").read_text(encoding="utf-8")
    assert "marimo.App" in source and "hailer.periods" not in source
    assert "Created scratch.py from the empty template." in result.output
    assert harness.agent.preambles == [], "no turn was sent, so the notice is still pending"
    status = result.output[result.output.index("Model"):]
    assert "scratch.py" in status and "Notebooks" in status and "notebooks" in status


def test_notebook_new_existing_name_is_an_error_and_keeps_active(harness):
    result = chat(input_text="/notebook new analysis\n/notebook new\n/exit\n")
    assert result.exit_code == 0, result.output
    assert "already exists" in result.output
    assert "Usage: /notebook new <name> [--empty]" in result.output
    assert active_state(harness.config) == "analysis.py"
    assert harness.agent.preambles == []


def test_notebook_open_by_name_opens_browser_and_waits_when_no_session(harness):
    other = write_notebook(harness.config, "other")
    harness.client = FakeClient(session=None)
    result = chat(input_text="/notebook open other\nhi\n/exit\n")
    assert result.exit_code == 0, result.output
    # startup opened the (then active) analysis notebook; the switch opened other.py and waited for it
    assert harness.opened == [url_for(harness.config.notebook), url_for(other)]
    assert harness.session_waits == ["analysis.py", "other.py"]
    assert "Notebook is open (session s9)." in result.output
    assert "Active notebook: other.py." in result.output
    assert active_state(harness.config) == "other.py"
    assert harness.agent.preambles == [
        "[Hailer] The active notebook is now other.py (reopened, 1 cell). Call notebook_cells before editing."
    ]


def test_notebook_open_without_session_after_wait_says_so(harness):
    other = write_notebook(harness.config, "other")
    harness.client = FakeClient(session=None)
    harness.waited_session = None
    result = chat(input_text="/notebook open other.py\nhi\n/exit\n")
    assert "No kernel session yet" in result.output
    assert harness.agent.preambles == [
        "[Hailer] The active notebook is now other.py (reopened, not open in a browser yet). "
        "Call notebook_cells before editing."
    ]
    assert harness.opened[-1] == url_for(other)


def test_notebook_open_active_notebook_is_a_no_op(harness):
    result = chat(input_text="/notebook open analysis\n/exit\n")
    assert "analysis.py is already the active notebook." in result.output
    assert active_state(harness.config) == "analysis.py"
    assert harness.agent.preambles == []


def test_notebook_open_unknown_lists_available(harness):
    result = chat(input_text="/notebook open nope\n/notebook open\n/exit\n")
    assert "No notebook named 'nope' in the notebooks folder." in result.output
    assert "Notebooks: analysis.py." in result.output
    assert "Usage: /notebook open <name>" in result.output
    assert active_state(harness.config) == "analysis.py"


def test_notebook_open_outside_folder_is_refused(harness):
    outside = harness.config.workspace / "elsewhere.py"
    outside.write_text(NOTEBOOK_SOURCE, encoding="utf-8")
    result = chat(input_text="/notebook open ../elsewhere.py\n/notebook open elsewhere.py\n/exit\n")
    assert "outside the notebooks folder" in result.output
    assert "No notebook named 'elsewhere.py'" in result.output, "a name is only looked for inside the folder"
    assert active_state(harness.config) == "analysis.py"


def test_notebook_close_shuts_down_the_session(harness, nb):
    result = chat(input_text="/notebook close\n/notebook close\n/exit\n")
    assert result.exit_code == 0, result.output
    assert harness.client.closed == ["s1"]
    assert "Closed analysis.py (session s1)" in result.output
    assert "It stays the active notebook; /notebook open analysis reopens it." in result.output
    assert "analysis.py is not open (no kernel session)." in result.output


def test_switch_made_by_the_model_is_detected_after_the_turn(harness):
    other = write_notebook(harness.config, "other")
    harness.client = FakeClient(session=None)
    harness.agent.on_turn = lambda: notebooks.save_active_notebook(harness.config, "other.py")
    result = chat(input_text="make a new notebook\n/status\n/exit\n")
    assert result.exit_code == 0, result.output
    assert "Active notebook is now other.py." in result.output
    assert harness.opened == [url_for(harness.config.notebook), url_for(other)], "opened once the switch was seen"
    assert harness.agent.preambles == [None], "the model already knows; no notice is queued"
    status = result.output[result.output.index("Active notebook is now"):]
    assert "other.py" in status


def test_switch_made_by_the_model_with_session_does_not_open_browser(harness):
    other = write_notebook(harness.config, "other")
    harness.client = FakeClient(other_sessions=[MarimoSession("s2", "other.py", str(other))])
    harness.agent.on_turn = lambda: notebooks.save_active_notebook(harness.config, "other.py")
    result = chat(input_text="make a new notebook\n/exit\n")
    assert "Active notebook is now other.py." in result.output
    assert harness.opened == []


def test_notebook_command_creates_folder_and_reports_missing_notebook_in_foreground(harness, nb):
    harness.config = make_config(harness.config.workspace, notebook_exists=False, notebooks_dir="nbs")
    assert not harness.config.notebooks_root.exists()
    result = notebook_cmd(["--foreground"])
    assert result.exit_code == 0, result.output
    assert harness.config.notebooks_root.is_dir()
    assert "Notebook not found:" in result.output and "/notebook new" in result.output
    assert "uvx hailer init" in result.output
    assert "marimo will create it" not in result.output
    assert nb.foreground == [(server_command(2718, folder="nbs"), harness.config.workspace)]


def test_notebook_command_opens_the_active_notebook_when_another_is_open(harness, nb):
    other = write_notebook(harness.config, "other")
    harness.client = FakeClient(session=None, other_sessions=[MarimoSession("s2", "other.py", str(other))])
    result = notebook_cmd()
    assert result.exit_code == 0, result.output
    assert len(nb.spawned) == 1, "its own server"
    assert harness.opened == [signed(harness.url)], "the active notebook is opened on it"
    assert nb.session_waits == ["analysis.py"]


def test_notebook_command_ignores_sessions_outside_the_folder(harness, nb):
    elsewhere = harness.config.workspace / "elsewhere.py"
    elsewhere.write_text(NOTEBOOK_SOURCE, encoding="utf-8")
    harness.client = FakeClient(session=None, other_sessions=[MarimoSession("s2", "elsewhere.py", str(elsewhere))])
    result = notebook_cmd()
    assert result.exit_code == 0, result.output
    assert len(nb.spawned) == 1


def test_doctor_names_the_active_notebook(harness):
    result = runner.invoke(cli.app, ["doctor"], catch_exceptions=False)
    assert "(active)" in result.output


def test_missing_notebook_hint_points_at_the_notebooks_folder(harness):
    harness.config = make_config(harness.config.workspace, notebook_exists=False)
    result = chat()
    assert result.exit_code == 1
    assert "[hailer].notebook" in result.output and "/notebook new" in result.output


def test_switch_notice_survives_ctrl_c_during_the_wait(harness):
    """Ctrl+C while waiting for the tab: the switch has happened, so the notice is still queued."""
    other = write_notebook(harness.config, "other")
    harness.waited_session = KeyboardInterrupt()
    result = chat(input_text="/notebook open other\nhi\n/exit\n")
    assert result.exit_code == 0, result.output
    assert "Interrupted." in result.output
    assert "Active notebook: other.py." in result.output
    assert active_state(harness.config) == "other.py"
    assert harness.opened[-1] == url_for(other)
    assert harness.agent.preambles == [
        "[Hailer] The active notebook is now other.py (reopened, not open in a browser yet). "
        "Call notebook_cells before editing."
    ]


def test_switch_notice_survives_marimo_error_during_the_wait(harness):
    write_notebook(harness.config, "other")
    harness.waited_session = HailerError("marimo went away", hint="restart it")
    result = chat(input_text="/notebook open other\nhi\n/exit\n")
    assert result.exit_code == 0, result.output
    assert "marimo went away" in result.output and "restart it" in result.output
    assert active_state(harness.config) == "other.py"
    assert harness.agent.preambles[0].startswith("[Hailer] The active notebook is now other.py (reopened, not open in a browser yet)")


def test_switch_made_by_the_model_is_detected_after_ctrl_c(harness):
    """A notebook_open tool call that completed before Ctrl+C is picked up straight away."""
    other = write_notebook(harness.config, "other")
    harness.client = FakeClient(session=None)
    harness.agent.on_turn = lambda: notebooks.save_active_notebook(harness.config, "other.py")
    harness.agent.fail_with = KeyboardInterrupt()
    result = chat(input_text="open the other notebook\n/notebook\n/exit\n")
    assert result.exit_code == 0, result.output
    assert "Interrupted." in result.output
    assert "Active notebook is now other.py." in result.output
    assert harness.opened[-1] == url_for(other), "opened once the switch was seen"
    assert "Notebook:  other.py (active)" in result.output


def test_switch_made_by_the_model_is_detected_after_a_failed_turn(harness):
    write_notebook(harness.config, "other")
    harness.agent.on_turn = lambda: notebooks.save_active_notebook(harness.config, "other.py")
    harness.agent.fail_with = HailerError("the model endpoint hung up", hint="retry")
    result = chat(input_text="open the other notebook\n/status\n/exit\n")
    assert result.exit_code == 0, result.output
    assert "the model endpoint hung up" in result.output
    assert "Active notebook is now other.py." in result.output
    status = result.output[result.output.index("Model"):]
    assert "other.py" in status


def test_notebook_and_status_commands_read_fresh_state(harness, monkeypatch):
    """A switch written to the state file behind the CLI's back (a late tool call) is applied
    before /notebook and /status act, without opening a browser tab for it."""
    write_notebook(harness.config, "other")
    load_context = cli._load_context

    def late_switch(config):  # once the chat has read the state: a late tool call switches
        notebooks.save_active_notebook(harness.config, "other.py")
        return load_context(config)

    monkeypatch.setattr(cli, "_load_context", late_switch)
    result = chat(input_text="/status\n/notebook\n/exit\n")
    assert result.exit_code == 0, result.output
    assert result.output.count("Active notebook is now other.py.") == 1
    assert "Notebook:  other.py (active)" in result.output
    assert harness.opened == [], "a stale-state refresh never opens a browser"


def test_notebook_command_lists_recent_notebooks(harness):
    write_notebook(harness.config, "alpha")
    write_notebook(harness.config, "beta")
    result = chat(input_text="/notebook open alpha\n/notebook open beta\n/notebook open analysis\n/notebook\n/exit\n")
    assert result.exit_code == 0, result.output
    shown = result.output[result.output.rindex("Notebook:  analysis.py (active)"):]
    assert "Recent:    beta.py, alpha.py" in shown
    assert "analysis.py" not in shown.split("Recent:")[1].splitlines()[0]


def test_session_matching_is_by_path_not_filename(harness):
    """Two notebooks called report.py in different sub-folders never share a session."""
    sub_a = write_notebook(harness.config, "a/report")
    sub_b = write_notebook(harness.config, "b/report")
    harness.client = FakeClient(session=None, other_sessions=[MarimoSession("s2", "report.py", str(sub_a))])
    result = chat(input_text="/notebook list\n/notebook open b/report.py\n/exit\n")
    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.splitlines() if "report.py" in ln]
    assert any("a/report.py " in ln and "open" in ln for ln in lines), lines
    assert any("b/report.py " in ln and "open" not in ln.split("report.py", 1)[1] for ln in lines), lines
    assert harness.opened[-1] == url_for(sub_b), "b/report.py had no session of its own, so it was opened"
    assert harness.session_waits == ["analysis.py", "b/report.py"]


# --------------------------------------------------------------------------- #
# Kernel runtimes: the Kernel line, doctor, docker fails closed, attaching, tokens
# --------------------------------------------------------------------------- #

LOCAL_LINE = "local (runs as you; not isolated)"
DOCKER_SERVER = MarimoServer(url="http://127.0.0.1:2731", token=TOKEN, runtime="docker")
CONTRACT = kernel_image.contract_tag()
IMAGE = f"ghcr.io/openafterhours/hailer-kernel:{CONTRACT}"


def docker_config(config: HailerConfig) -> HailerConfig:
    return replace(config, kernel=KernelConfig(runtime="docker"))


@pytest.fixture
def docker(harness, monkeypatch):
    """The real DockerRuntime (and ``hailer kernel ...``) driving a scripted docker CLI with state:
    nothing runs. Local configs still get whatever runtime was set up before (list ``nb`` first
    to fake it)."""
    from fake_docker import FakeDocker
    from hailer import kernel_docker

    fake = FakeDocker()
    local_runtime_for = cli._runtime_for

    def runtime_for(config):
        if config.kernel.runtime != "docker":
            return local_runtime_for(config)
        return Runtime(
            config, runner=fake, token_factory=lambda: TOKEN, health=lambda url, timeout, token=None, should_stop=None: True, user=lambda: None
        )  # fmt: skip

    class Runtime(kernel_docker.DockerRuntime):
        """The real start; its sandbox serves the notebooks folder and its clients are the harness's."""

        def start(self, port, *, foreground=False):
            running = super().start(port, foreground=foreground)
            return with_folder_files(running, self.config.notebooks_root, lambda **kw: harness.bound_client(running))

    monkeypatch.setattr(cli, "_runtime_for", runtime_for)
    monkeypatch.setattr(cli, "_docker_runner", lambda: fake)
    monkeypatch.setattr(kernel_docker, "_sleep", lambda seconds: None)  # network rm retries
    return fake


def test_kernel_line_in_the_startup_panel_status_and_slash_status(harness):
    result = chat(input_text="/status\n/exit\n")
    assert result.exit_code == 0, result.output
    assert f"Kernel:     {LOCAL_LINE}" in result.output, "startup panel"
    assert any(line.startswith("Kernel ") and LOCAL_LINE in line for line in result.output.splitlines()), "/status"
    result = runner.invoke(cli.app, ["status"], catch_exceptions=False)
    assert f"Kernel:     {LOCAL_LINE}" in result.output


def test_doctor_has_a_kernel_row(harness):
    result = runner.invoke(cli.app, ["doctor"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert any("kernel" in line and "OK" in line and "local (runs as you" in line for line in result.output.splitlines())


def test_doctor_reports_a_runtime_it_cannot_use(harness, docker):
    harness.config = docker_config(harness.config)
    docker.installed = False
    result = runner.invoke(cli.app, ["doctor"], catch_exceptions=False)
    assert result.exit_code == 1
    assert any("docker" in line and "FAIL" in line and "Docker is not installed." in line for line in result.output.splitlines())
    assert 'runtime = "local"' in result.output


@pytest.mark.parametrize("args", [["notebook", "--help"], ["kernel", "--help"], ["kernel", "build", "--help"], ["init", "--help"]])
def test_help_texts_keep_the_kernel_section_name(args):
    """Typer renders help as Rich markup, which ate an unescaped [kernel]."""
    result = runner.invoke(cli.app, args, catch_exceptions=False, env={"COLUMNS": "250", "NO_COLOR": "1"})
    assert result.exit_code == 0 and "[kernel]" in result.output, result.output


def test_notebook_with_docker_requested_never_starts_a_local_server(harness, nb, docker):
    harness.config = docker_config(harness.config)
    for fake_state in ({"installed": False}, {"engine": None}):
        for name, value in fake_state.items():
            setattr(docker, name, value)
        for args in ([], ["--foreground"]):
            result = notebook_cmd(args)
            assert result.exit_code == 1
            assert "Docker is not installed." in result.output or "Docker is not running." in result.output
        docker.installed, docker.engine = True, "29.4.3 linux"
    assert nb.spawned == [] and nb.foreground == [] and harness.agent.started == []
    assert not docker.commands("run") and not docker.commands("create")


def test_notebook_hands_its_chat_the_sandbox_it_started(harness, nb):
    """The chat keeps the sandbox (endpoint, token, files) in memory; nothing reads it from a file."""
    seen: list = []

    def on_turn():
        seen.append(harness.agent_sandboxes[0])

    harness.agent.on_turn = on_turn
    result = notebook_cmd(["--port", "2718"], input_text="hi\n/status\n/exit\n")
    assert result.exit_code == 0, result.output
    pinned = seen[0].server
    assert pinned.url == "http://127.0.0.1:2718" and pinned.token == TOKEN, "tools get it in memory"
    assert re.search(r"^Marimo\s+http://127\.0\.0\.1:2718 \(session", result.output, re.M), "/status still knows the server"


def test_the_prompt_never_carries_the_servers_token(harness, nb, monkeypatch):
    """Everything the agent gets from `hailer notebook` (config and server become the system prompt).
    The check is sensitive: a server URL with the token in it would show up in the prompt."""
    from hailer.agent import system_prompt

    captured: list = []
    monkeypatch.setattr(cli, "_make_agent", lambda config, bundle, sandbox=None: (captured.append((config, sandbox.server)) or harness.agent))
    result = notebook_cmd()
    assert result.exit_code == 0, result.output
    config, server = captured[0]
    assert server.token == TOKEN, "the tools hold the token (for the server, never for the model)"
    text = system_prompt(config, ContextBundle(), server)
    assert "- Marimo URL: http://127.0.0.1:2718" in text and TOKEN not in text and "access_token" not in text
    leaky = system_prompt(config, ContextBundle(), replace(server, url=f"{server.url}/?access_token={TOKEN}"))
    assert TOKEN in leaky, "the probe would catch a tokenised URL"


def test_cli_clients_carry_the_servers_token_and_the_kernels_notebooks_folder(tmp_path):
    from hailer.sandbox import MarimoSandbox

    sandbox = MarimoSandbox(MarimoServer(url="http://127.0.0.1:2731", token=TOKEN, runtime="docker"), notebooks_path="/work/notebooks")
    client = sandbox.client(notebook="q3/review.py", token_in_links=True)
    assert client.token == TOKEN and client.token_in_links, "the CLI prints its hints for the user"
    assert client.notebooks_path == "/work/notebooks" and client.notebook == "q3/review.py"
    assert sandbox.notebook_url("analysis.py", with_token=True) == (
        f"http://127.0.0.1:2731/?file=/work/notebooks/analysis.py&view-as=present&access_token={TOKEN}"
    )
    assert sandbox.home_url(with_token=True) == f"http://127.0.0.1:2731/?access_token={TOKEN}"


def test_config_warnings_show_when_a_session_starts(harness, nb, monkeypatch):
    """A runtime line uncommented without its [kernel] line lands under [model]: say so, never silently local."""
    warning = "Warning: [model].runtime in hailer.toml is ignored: runtime belongs under [kernel]. Uncomment the [kernel] line above it too."
    monkeypatch.setattr(cli, "_validate_config", lambda config: [warning])
    result = notebook_cmd()
    assert result.exit_code == 0, result.output
    assert f"warn  config: {warning}" in result.output


def test_several_config_problems_are_listed_one_per_line(harness, monkeypatch):
    problems = ['Invalid [kernel].memory "4 GB"; write a size such as "4g"', "Invalid [kernel].cpus 0; use a number above 0"]
    monkeypatch.setattr(cli, "_validate_config", lambda config: problems)
    result = chat()
    assert result.exit_code == 1
    assert "FAIL  config: 2 problems in the configuration:" in result.output
    assert f"        - {problems[0]}" in result.output and f"        - {problems[1]}" in result.output


# --------------------------------------------------------------------------- #
# The docker runtime from the CLI: --kernel, the image before the spinner, kernel pull/build/stop
# --------------------------------------------------------------------------- #


def test_notebook_kernel_docker_runs_the_chat_against_a_container(harness, nb, docker, monkeypatch):
    from hailer.kernel_docker import LABEL_OWNER, owner_alive, workspace_id

    agent_configs: list = []
    labels: list = []
    monkeypatch.setattr(cli, "_make_agent", lambda config, bundle, sandbox=None: (agent_configs.append((config, sandbox)) or harness.agent))
    harness.agent.on_turn = lambda: labels.extend(owner_alive(harness.config.workspace, c.labels[LABEL_OWNER]) for c in docker.containers.values())
    harness.client = FakeClient(session=None)  # nobody has the tab open yet
    result = notebook_cmd(["--kernel", "docker", "--port", "2731"], input_text="hi\n/exit\n")
    assert result.exit_code == 0, result.output
    assert nb.spawned == [] and nb.foreground == [], "no local server"
    run = docker.commands("run")[0]
    kernel_name, network_name = run[5], run[7]
    assert re.fullmatch(rf"hailer-kernel-{workspace_id(harness.config.workspace)}-[0-9a-f]{{6}}", kernel_name), "unique per start"
    assert run[:8] == ["run", "-d", "--pull", "never", "--name", kernel_name, "--network", network_name]
    assert docker.commands("create")[0][5:7] == ["-p", "127.0.0.1:2731:2718"]
    assert f"Marimo is running at http://127.0.0.1:2731  (log: docker logs {kernel_name})" in result.output
    assert harness.opened == [f"http://127.0.0.1:2731/?file=/work/notebooks/analysis.py&view-as=present&access_token={TOKEN}"]
    assert f"Kernel:     docker (hailer-kernel {CONTRACT}; no network; data read-only)" in result.output
    config, sandbox = agent_configs[0]
    assert config.kernel.runtime == "docker"
    assert sandbox.server.runtime == "docker" and sandbox.server.token == TOKEN, "the chat holds the container"
    assert sandbox.notebooks_path == "/work/notebooks" and sandbox.data_path == "/work/data"
    assert labels and all(labels), "owned by this live session while the chat runs"
    assert not list((harness.config.workspace / ".hailer").glob("owner-*.lock")), "its lock goes with the kernel"
    assert docker.containers == {} and docker.networks == {}, "removed when the chat ends"
    assert "Stopped marimo." in result.output


def test_notebook_downloads_a_missing_image_before_the_spinner_without_warning_first(harness, nb, docker, monkeypatch):
    """docker's progress bars and a spinner line would fight over the terminal; and no "image
    missing, run kernel pull" warning right before the pull that fixes it."""
    events: list[str] = []

    class RecordingConsole(Console):
        def status(self, *args, **kwargs):
            events.append("spinner")
            return super().status(*args, **kwargs)

    monkeypatch.setattr(cli, "console_factory", lambda: RecordingConsole(force_terminal=False, width=120, highlight=False, soft_wrap=True, color_system=None))
    monkeypatch.setattr(cli, "_chat_console", lambda opts: cli.console_factory())
    docker.image_contract = None
    docker.stream_hook = lambda args: events.append(args[0])
    result = notebook_cmd(["--kernel", "docker"])
    assert result.exit_code == 0, result.output
    assert events[0] == "pull" and events[1] == "spinner", events
    assert f"Downloading the kernel image {IMAGE} (first use; this can take a few minutes) ..." in result.output
    assert "not on this machine yet" not in result.output and "uvx hailer kernel pull downloads it now" not in result.output
    assert docker.streams == [["pull", IMAGE]], "one download"


def test_notebook_kernel_flag_beats_the_file_and_rejects_other_values(harness, nb, docker):
    harness.config = docker_config(harness.config)
    result = notebook_cmd(["--kernel", "LOCAL"])
    assert result.exit_code == 0, result.output
    assert len(nb.spawned) == 1 and not docker.commands("run"), "--kernel local ran a local server"
    result = notebook_cmd(["--kernel", "podman"])
    assert result.exit_code == 2 and '--kernel must be "local" or "docker"' in result.output


def test_notebook_foreground_in_docker_mode_follows_the_log_and_removes_the_containers(harness, nb, docker):
    def ctrl_c(args):
        if args[:2] == ["logs", "-f"]:
            raise KeyboardInterrupt

    docker.stream_hook = ctrl_c
    result = notebook_cmd(["--kernel", "docker", "--foreground", "--no-browser", "--port", "2731"])
    assert result.exit_code == 0, result.output
    assert "Starting marimo in Docker on http://127.0.0.1:2731 (Ctrl+C stops it) ..." in result.output
    assert f"Kernel:     docker (hailer-kernel {CONTRACT}; no network; data read-only)" in result.output
    assert f"Open http://127.0.0.1:2731/?access_token={TOKEN} in your browser." in result.output
    kernel_id = docker.streams[0][-1]
    assert docker.streams == [["logs", "-f", "--tail", "0", kernel_id]] and ["rm", "-f", kernel_id] in docker.calls, "by id"
    assert docker.containers == {} and docker.networks == {}, "removed after Ctrl+C"
    assert nb.foreground == []


def test_notebook_foreground_says_why_the_kernel_went_away(harness, nb, docker):
    """Removed from another terminal (uvx hailer kernel stop): say so instead of exiting 137 silently."""

    def removed_elsewhere(args):
        if args[:2] == ["logs", "-f"]:
            docker.containers.clear()

    docker.stream_hook = removed_elsewhere
    result = notebook_cmd(["--kernel", "docker", "--foreground", "--no-browser"])
    assert "The kernel container was removed from outside this terminal (uvx hailer kernel stop, or docker rm)." in result.output


def test_doctor_in_docker_mode_has_docker_image_and_data_rows(harness, docker):
    harness.config = docker_config(harness.config)
    result = runner.invoke(cli.app, ["doctor"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    for name, text in (("kernel", "docker (hailer-kernel"), ("docker", "Docker 29.4.3 (Linux engine)"), ("image", IMAGE), ("data", "/work/data")):
        assert any(f" {name} " in line and "OK" in line for line in lines), name
        assert text in result.output, text
    assert not docker.commands("run") and harness.client.codes == [], "no kernel is started or probed"


def test_kernel_pull(harness, docker):
    result = runner.invoke(cli.app, ["kernel", "pull"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert docker.streams == [["pull", IMAGE]], "pulled even though a copy is here: it may be stale"
    assert f"Downloading {IMAGE} ..." in result.output and f"{IMAGE} is ready (kernel contract {CONTRACT})." in result.output
    docker.pull_code, docker.pull_error = 1, "dial tcp: lookup ghcr.io: no such host"
    result = runner.invoke(cli.app, ["kernel", "pull"], catch_exceptions=False)
    assert result.exit_code == 1 and "Could not download the kernel image" in result.output and "uvx hailer kernel build" in result.output
    docker.installed = False
    result = runner.invoke(cli.app, ["kernel", "pull"], catch_exceptions=False)
    assert result.exit_code == 1 and "Docker is not installed." in result.output


def test_kernel_pull_of_an_image_that_is_not_published(harness, docker):
    docker.pull_code, docker.pull_error = 1, "Error response from daemon: error from registry: denied\ndenied"
    result = runner.invoke(cli.app, ["kernel", "pull"], catch_exceptions=False)
    assert result.exit_code == 1
    assert f"The kernel image {CONTRACT} is not published (or not visible to you): {IMAGE}." in result.output
    assert "Build it on this machine with: uvx hailer kernel build" in result.output


def test_kernel_build(harness, docker, isolated_package_settings):
    from hailer.kernel_image import build_args

    result = runner.invoke(cli.app, ["kernel", "build"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    build = docker.streams[0]
    assert build[:3] == ["build", "--tag", IMAGE]
    assert build[3:-1] == [part for name, value in build_args().items() for part in ("--build-arg", f"{name}={value}")]
    assert f"Building {IMAGE} for kernel contract {CONTRACT} (marimo 0.24.2, Polars " in result.output and f"Built {IMAGE}." in result.output
    result = runner.invoke(cli.app, ["kernel", "build", "--tag", "hailer-kernel:dev"], catch_exceptions=False)
    assert docker.streams[-1][:3] == ["build", "--tag", "hailer-kernel:dev"]
    assert 'set image = "hailer-kernel:dev" under [kernel] in hailer.toml' in result.output
    docker.build_code = 1
    result = runner.invoke(cli.app, ["kernel", "build"], catch_exceptions=False)
    assert result.exit_code == 1 and "docker build failed (exit code 1)." in result.output
    docker.engine = None
    streams = len(docker.streams)
    result = runner.invoke(cli.app, ["kernel", "build"], catch_exceptions=False)
    assert result.exit_code == 1 and "Docker is not running." in result.output and len(docker.streams) == streams


def test_kernel_build_with_corporate_mirrors(harness, docker, tmp_path, isolated_package_settings):
    import csv

    config = tmp_path / "pip.ini"
    config.write_text("[global]\nindex-url = https://user:private-token@packages.example/simple\n")
    cert = tmp_path / "company.pem"
    cert.write_text("test-ca")
    result = runner.invoke(cli.app, [
        "kernel", "build", "--tag", "company/hailer:dev", "--base-image", "company/python:3.13",
        "--pip-config", str(config), "--pip-cert", str(cert), "--no-cache",
    ], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    build = docker.streams[-1]
    assert "BASE_IMAGE=company/python:3.13" in build
    assert "--no-cache" in build
    secrets = [next(csv.reader([build[i + 1]])) for i, arg in enumerate(build) if arg == "--secret"]
    assert secrets == [
        ["type=file", "id=pip_config", f"src={config.resolve()}"],
        ["type=file", "id=pip_cert", f"src={cert.resolve()}"],
    ]
    assert "private-token" not in result.output and "private-token" not in str(build)
    assert 'set image = "company/hailer:dev"' in result.output


def test_kernel_build_discovers_mirror_and_can_opt_out(harness, docker, isolated_package_settings, monkeypatch):
    monkeypatch.setenv("PIP_INDEX_URL", "https://user:private-token@mirror/simple")
    result = runner.invoke(cli.app, ["kernel", "build"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "Using discovered host package settings" in result.output
    assert "--secret" in docker.streams[-1]
    assert "private-token" not in result.output and "private-token" not in str(docker.streams[-1])
    result = runner.invoke(cli.app, ["kernel", "build", "--no-host-config"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "--secret" not in docker.streams[-1]


@pytest.mark.parametrize("option", ["--pip-config", "--pip-cert"])
@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_kernel_build_rejects_invalid_secret_files(harness, docker, tmp_path, option, kind):
    path = tmp_path / "missing.ini" if kind == "missing" else tmp_path
    result = runner.invoke(cli.app, ["kernel", "build", option, str(path)], catch_exceptions=False)
    assert result.exit_code == 2
    assert not docker.streams


def test_kernel_stop(harness, docker):
    result = runner.invoke(cli.app, ["kernel", "stop"], catch_exceptions=False)
    assert result.exit_code == 0 and "No kernel is running for this workspace." in result.output
    workspace = str(harness.config.workspace)
    # a running kernel of a live session goes too: kernel stop removes every labelled object
    from hailer.kernel_docker import LABEL_OWNER, acquire_owner_lock

    lock = acquire_owner_lock(harness.config.workspace)
    owner = {LABEL_OWNER: lock.id}
    kernel_c = docker.add_container("hailer-kernel-a-1", workspace=workspace, **owner)
    forwarder_c = docker.add_container("hailer-fwd-a-1", workspace=workspace, role="forwarder", **owner)
    network = docker.add_network("hailer-net-a-1", workspace=workspace)
    other = docker.add_container("hailer-kernel-b-1", workspace=str(harness.config.workspace / "elsewhere"))
    result = runner.invoke(cli.app, ["kernel", "stop"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert f"Removed {kernel_c.name}, {forwarder_c.name}, {network.name}." in result.output
    assert list(docker.containers.values()) == [other], "another workspace's container is not touched"
    assert lock.path.exists(), "the live session's lock stays with it"
    lock.release()
    help_text = runner.invoke(cli.app, ["kernel", "--help"], catch_exceptions=False).output
    assert all(command in help_text for command in ("pull", "build", "stop"))


def test_kernel_stop_with_docker_not_running_says_so(harness, docker):
    docker.engine = None
    result = runner.invoke(cli.app, ["kernel", "stop"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "Docker is not running, so containers a docker kernel may have left were not checked" in result.output
    assert "No kernel is running" not in result.output


def test_kernel_stop_that_cannot_remove_everything_exits_1(harness, docker):
    network = docker.add_network("hailer-net-a-1", workspace=str(harness.config.workspace))
    docker.fail[("network", "rm")] = (1, "Error response from daemon: network has active endpoints")
    result = runner.invoke(cli.app, ["kernel", "stop"], catch_exceptions=False)
    assert result.exit_code == 1
    assert f"Could not remove {network.name}: docker said: Error response from daemon: network has active endpoints" in result.output


def test_init_kernel_docker_switches_the_section_on(harness, tmp_path, monkeypatch):
    from hailer.config import load_config, validate

    monkeypatch.delenv("HAILER_KERNEL", raising=False)
    monkeypatch.setattr(cli, "_example_config_dir", lambda: None)
    monkeypatch.setattr(cli, "_docker_on_path", lambda: False)
    ws = tmp_path / "fresh"
    ws.mkdir()
    result = init_cmd(ws, "--kernel", "docker")
    assert result.exit_code == 0, result.output
    assert '[kernel]\nruntime = "docker"' in (ws / "hailer.toml").read_text(encoding="utf-8")
    config = load_config(workspace=ws, env={})
    assert config.kernel.runtime == "docker" and validate(config) == []
    assert f'Wrote {ws / "hailer.toml"} ([kernel] runtime = "docker")' in result.output
    assert "uvx hailer kernel pull does it now" in result.output and "uvx hailer kernel build builds it on this machine" in result.output
    assert "Docker was not found on this machine" in result.output
    assert "uv run" not in result.output


def test_init_mentions_the_docker_kernel_only_when_docker_is_installed(harness, tmp_path, monkeypatch):
    monkeypatch.delenv("HAILER_KERNEL", raising=False)
    monkeypatch.setattr(cli, "_example_config_dir", lambda: None)
    for found in (True, False):
        ws = tmp_path / f"found-{found}"
        ws.mkdir()
        monkeypatch.setattr(cli, "_docker_on_path", lambda found=found: found)
        result = init_cmd(ws)
        assert result.exit_code == 0, result.output
        assert "# [kernel]" in (ws / "hailer.toml").read_text(encoding="utf-8"), "commented out without the flag"
        hint = 'uncomment the [kernel] and runtime lines in hailer.toml and set runtime = "docker"'
        assert (hint in result.output) is found, "both lines: uncommenting only runtime would land under [model]"
        assert ("uvx hailer notebook --kernel docker" in result.output) is found


def test_init_kernel_flag_with_an_existing_config(harness, tmp_path, monkeypatch):
    from hailer.config import load_config

    monkeypatch.delenv("HAILER_KERNEL", raising=False)
    monkeypatch.setattr(cli, "_example_config_dir", lambda: None)
    monkeypatch.setattr(cli, "_docker_on_path", lambda: True)
    ws = tmp_path / "fresh"
    ws.mkdir()
    init_cmd(ws)
    result = init_cmd(ws, "--kernel", "docker")
    assert "already exists" in result.output
    assert 'hailer.toml was kept, so [kernel] runtime is still "local". To change it, set runtime = "docker" under [kernel]' in result.output
    assert "uvx hailer init --kernel docker --force" in result.output
    assert "Optional: to isolate notebook code" not in result.output, "the kept advice is not repeated"
    result = init_cmd(ws, "--kernel", "docker", "--force")
    assert result.exit_code == 0 and load_config(workspace=ws, env={}).kernel.runtime == "docker"
    result = init_cmd(tmp_path / "other", "--kernel", "vm")
    assert result.exit_code == 2 and not (tmp_path / "other" / "hailer.toml").exists()


# --------------------------------------------------------------------------- #
# Final fix pass: one rule set on every start path, planted files, status, wording
# --------------------------------------------------------------------------- #


def test_foreground_refuses_data_inside_the_notebooks_folder(harness, nb, docker):
    """--foreground never runs validate(): the start itself refuses, before any container, instead of
    starting a kernel whose "read-only" data sits in the writable notebooks mount."""
    config = docker_config(harness.config)
    harness.config = replace(config, data_dir=config.notebooks_root / "data")
    result = notebook_cmd(["--foreground", "--no-browser"])
    assert result.exit_code == 1
    assert f"[hailer].data_dir ({config.notebooks_root / 'data'}) is inside the notebooks folder" in result.output
    assert not docker.commands("run") and not docker.commands("network", "create") and not docker.streams


def test_foreground_refuses_a_notebooks_folder_that_is_a_git_repository(harness, nb, docker):
    harness.config = docker_config(harness.config)
    (harness.config.notebooks_root / ".git").mkdir()
    result = notebook_cmd(["--foreground", "--no-browser"])
    assert result.exit_code == 1 and "is a git repository (it has .git at its top level)" in result.output
    assert not docker.commands("run")


def test_a_chat_that_stops_its_docker_kernel_warns_about_planted_files(harness, nb, docker):
    def plant():
        (harness.config.notebooks_root / ".vscode").mkdir(exist_ok=True)  # what notebook code could do
        (harness.config.notebooks_root / ".git").mkdir(exist_ok=True)

    harness.agent.on_turn = plant
    result = notebook_cmd(["--kernel", "docker"], input_text="hi\n/exit\n")
    assert result.exit_code == 0, result.output
    stopped = result.output.index("Stopped marimo.")
    warning = f"WARNING: the notebooks folder {harness.config.notebooks_root} contains .git, .vscode."
    assert warning in result.output[stopped:], result.output
    assert "delete them before you run git in that folder, open it in an editor" in result.output


def test_foreground_end_warns_about_planted_files(harness, nb, docker):
    def plant_then_ctrl_c(args):
        if args[:2] == ["logs", "-f"]:
            (harness.config.notebooks_root / ".devcontainer").mkdir()
            raise KeyboardInterrupt

    docker.stream_hook = plant_then_ctrl_c
    result = notebook_cmd(["--kernel", "docker", "--foreground", "--no-browser"])
    assert result.exit_code == 0, result.output
    assert f"WARNING: the notebooks folder {harness.config.notebooks_root} contains .devcontainer." in result.output


def test_kernel_stop_prints_the_planted_files_warning(harness, docker):
    docker.add_container("hailer-kernel-a-1", workspace=str(harness.config.workspace))
    (harness.config.notebooks_root / ".idea").mkdir()
    result = runner.invoke(cli.app, ["kernel", "stop"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert f"WARNING: the notebooks folder {harness.config.notebooks_root} contains .idea." in result.output


def test_a_local_foreground_server_ended_from_outside_says_so(harness, nb):
    nb.foreground_proc = ForegroundProc(exit_code=1)  # e.g. `uvx hailer kernel stop` in another terminal
    result = notebook_cmd(["--foreground", "--no-browser"])
    assert result.exit_code == 1
    assert "marimo stopped (exit code 1): it was ended from outside this terminal (uvx hailer kernel stop" in result.output


def test_kernel_flag_values_are_quoted_like_other_messages(harness, nb):
    result = notebook_cmd(["--kernel", "podman"])
    assert result.exit_code == 2 and '--kernel must be "local" or "docker", not "podman".' in result.output


def test_doctor_fails_a_unc_data_folder_with_the_start_wording(harness, docker, monkeypatch):
    """doctor showed WARN while a start refused: same rule, same wording, FAIL."""
    import hailer.config as config_module
    from hailer import kernel_docker
    from hailer.config import docker_mount_problems

    monkeypatch.setattr(config_module, "_on_windows", lambda: True)
    monkeypatch.setattr(kernel_docker, "_on_windows", lambda: True)
    monkeypatch.setattr(cli, "_validate_config", docker_mount_problems)  # validate()'s own checks would wait on the share
    share = Path(r"\\fileserver\team\sales")
    harness.config = replace(docker_config(harness.config), data_dir=share)
    result = runner.invoke(cli.app, ["doctor"], catch_exceptions=False, env={"COLUMNS": "400"})
    assert result.exit_code == 1
    lines = result.output.splitlines()
    assert any(" config " in line and "FAIL" in line and "is on a network share (UNC path), which Docker" in line for line in lines), result.output
    assert not any(line.lstrip("│| ").startswith("data ") for line in lines), "no second row saying something else"


# --------------------------------------------------------------------------- #
# Bare hailer: the same session as hailer notebook, with its own kernel
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("args", [[], ["notebook"]], ids=["hailer", "hailer-notebook"])
def test_bare_hailer_and_hailer_notebook_start_their_own_kernel_and_stop_it(harness, nb, args):
    """Bare hailer does exactly what hailer notebook does: start marimo on the notebooks folder, chat, stop it."""
    from hailer.kernel import local_log_path

    result = chat(args=args)
    assert result.exit_code == 0, result.output
    assert len(nb.spawned) == 1
    cmd, cwd, log_path = nb.spawned[0]
    assert cmd == server_command(2718) and cwd == harness.config.workspace
    assert log_path == local_log_path(harness.config.workspace)
    assert "Marimo is running at http://127.0.0.1:2718" in result.output
    assert harness.agent.started == [None], "the chat ran"
    assert nb.proc.terminated
    assert "Stopped marimo." in result.output


# --------------------------------------------------------------------------- #
# The prompt: prompt_toolkit in a terminal, bracketed paste, plain input otherwise
# --------------------------------------------------------------------------- #

PASTE_ON, PASTE_OFF = "\x1b[?2004h", "\x1b[?2004l"


@contextmanager
def terminal():
    """A prompt_toolkit reader on a pipe (what the keyboard or the host types) and a VT100 screen
    that records every byte written, escape sequences included."""
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output.vt100 import Vt100_Output

    with create_pipe_input() as keys:
        screen = io.StringIO()
        output = Vt100_Output(screen, lambda: Size(rows=24, columns=100), term="xterm-256color")
        console = Console(file=io.StringIO(), force_terminal=False, color_system=None)
        yield cli._LineReader(console, interactive=True, pt_input=keys, pt_output=output), keys, screen


def last_paste_mode(screen: io.StringIO) -> str | None:
    text = screen.getvalue()
    on, off = text.rfind(PASTE_ON), text.rfind(PASTE_OFF)
    if on < 0 and off < 0:
        return None
    return "on" if on > off else "off"


def test_prompt_bracketed_paste_is_one_submission_with_its_newlines():
    with terminal() as (reader, keys, _screen):
        keys.send_text("\x1b[200~first line\r\nsecond line\nthird\x1b[201~\r")
        assert reader.read() == "first line\nsecond line\nthird"
        keys.send_text("next\r")
        assert reader.read() == "next", "the paste produced exactly one submission"


def test_prompt_turns_bracketed_paste_on_while_waiting_and_off_before_returning():
    with terminal() as (reader, keys, screen):
        result: list[str] = []
        thread = threading.Thread(target=lambda: result.append(reader.read()), daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        while "You > " not in screen.getvalue() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert "You > " in screen.getvalue(), "the prompt is drawn"
        assert last_paste_mode(screen) == "on", "waiting at the prompt: DECSET 2004 is on"
        keys.send_text("hello\r")
        thread.join(10)
        assert result == ["hello"]
        assert last_paste_mode(screen) == "off", "DECRST 2004 is written before read() returns"
        out = screen.getvalue()
        assert "\x1b[?1049h" not in out, "no alternate screen"
        assert "\x1b[?1000h" not in out and "\x1b[?1003h" not in out, "no mouse capture"
        assert "\x1b[6n" not in out, "no cursor position request"


def test_prompt_ctrl_c_ctrl_d_and_ctrl_z_enter():
    with terminal() as (reader, keys, _screen):
        keys.send_text("\x03")
        with pytest.raises(KeyboardInterrupt):
            reader.read()
        keys.send_text("\x04")
        with pytest.raises(EOFError):
            reader.read()
        keys.send_text("\x1a\r")  # Ctrl+Z then Enter: end of input on Windows
        with pytest.raises(EOFError):
            reader.read()


def type_at_prompt(reader, keys, screen, text: str) -> str:
    """read() with ``text`` typed once the prompt is on screen, as a person would type it."""
    shown = screen.getvalue().count("You > ")
    result: list[str] = []
    thread = threading.Thread(target=lambda: result.append(reader.read()), daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while screen.getvalue().count("You > ") <= shown and time.monotonic() < deadline:
        time.sleep(0.02)
    time.sleep(0.1)  # history loads in the background when the prompt starts
    keys.send_text(text)
    thread.join(10)
    return result[0]


def test_prompt_up_arrow_recalls_earlier_messages():
    with terminal() as (reader, keys, screen):
        assert type_at_prompt(reader, keys, screen, "first\r") == "first"
        assert type_at_prompt(reader, keys, screen, "second\r") == "second"
        assert type_at_prompt(reader, keys, screen, "\x1b[A\x1b[A\r") == "first"  # Up, Up, Enter


def test_prompt_falls_back_to_plain_input_without_a_terminal(monkeypatch):
    monkeypatch.setattr(cli, "_stdio_is_terminal", lambda: False)
    monkeypatch.setattr(Console, "input", lambda self, prompt="", **kw: "typed")
    reader = cli._LineReader(Console(file=io.StringIO()))
    assert reader.interactive is False
    assert reader.read() == "typed"
    assert reader._session is None, "prompt_toolkit is never started on a pipe"


def test_chat_with_the_prompt_sends_a_pasted_block_as_one_turn(harness, monkeypatch):
    with terminal() as (reader, keys, screen):
        monkeypatch.setattr(cli, "_make_line_reader", lambda console: reader)
        modes_at_turn: list[str | None] = []
        harness.agent.on_turn = lambda: modes_at_turn.append(last_paste_mode(screen))
        keys.send_text("\x1b[200~sum revenue\nby region\x1b[201~\r/exit\r")
        result = chat(input_text="")
    assert result.exit_code == 0, result.output
    assert harness.agent.turns == [("sum revenue\nby region", None)]
    assert modes_at_turn == ["off"], "bracketed paste is off while the turn runs"
    assert "Bye." in result.output


def test_chat_with_the_prompt_ctrl_c_in_a_turn_cancels_only_that_turn(harness, monkeypatch):
    with terminal() as (reader, keys, _screen):
        monkeypatch.setattr(cli, "_make_line_reader", lambda console: reader)
        harness.agent.fail_with = KeyboardInterrupt()  # the agent cancels the turn and re-raises
        keys.send_text("long question\r/exit\r")
        result = chat(input_text="")
    assert result.exit_code == 0, result.output
    assert "Interrupted." in result.output and "Bye." in result.output


def test_chat_with_the_prompt_ctrl_c_at_the_prompt_exits(harness, monkeypatch):
    with terminal() as (reader, keys, _screen):
        monkeypatch.setattr(cli, "_make_line_reader", lambda console: reader)
        keys.send_text("\x03")
        result = chat(input_text="")
    assert result.exit_code == 0, result.output
    assert "Bye." in result.output and harness.agent.turns == []
    assert harness.agent.closed
