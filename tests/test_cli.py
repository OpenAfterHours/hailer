"""CLI tests: every collaborator is faked; no marimo, model endpoint, keyring or network."""

from __future__ import annotations

import io
import json
import socket
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import quote

import pytest
from fake_marimo import running
from rich.console import Console
from typer.testing import CliRunner

import hailer.cli as cli
from hailer import notebooks, __version__
from hailer.errors import ConfigError, CredentialsError, HailerError, NoSessionError
from hailer.models import (
    AgentEvent,
    ContextBundle,
    ExecResult,
    HailerConfig,
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
# The real discovery and client, captured before the harness replaces them.
REAL_FIND_SERVER = cli._find_server
REAL_MAKE_CLIENT = cli._make_client
SERVER = MarimoServer(url="http://127.0.0.1:2718", server_id="127.0.0.1:2718", version="0.24.2", source="config")
LAUNCH = ["uvx", "hailer", "notebook", "--foreground"]


def server_command(port, headless=True):
    """What the nb fixture's _marimo_server_command returns (the real one runs sys.executable -m marimo)."""
    return [sys.executable, "-m", "marimo", "edit", "notebooks", "--no-token", *(["--headless"] if headless else []), "--port", str(port), "--skip-update-check"]
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
    """The URL the real open_notebook_url produces: absolute posix file key, app view."""
    return f"{SERVER.url}/?file={quote(Path(notebook).resolve().as_posix(), safe='/:')}&view-as=present"


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
    """Session-aware stand-in for MarimoClient: a notebook has a session only when one was listed
    for its path (or granted later, the way a browser tab would), matched by the real rule.
    ``folder`` is what ``root()`` reports: the folder marimo was started on (None: another folder
    or a single-file server)."""

    def __init__(self, *, healthy=True, session="default", exec_result=None, other_sessions=(), workspace=None, folder=None):
        self.healthy = healthy
        self.folder = folder
        self.workspace = workspace  # bound by the harness's _make_client; relative session paths resolve against it
        # the configured notebook's session, relative to the workspace like a hand-written entry
        self.session = MarimoSession("s1", "analysis.py", "notebooks/analysis.py") if session == "default" else session
        self.other_sessions = list(other_sessions)  # sessions for notebooks other than the active one
        self.exec_result = exec_result
        self.codes: list[str] = []
        self.closed: list[str] = []
        self.granted: list[Path] = []

    def health(self):
        return self.healthy

    def sessions(self):
        return ([self.session] if self.session is not None else []) + self.other_sessions

    def root(self):
        return str(self.folder) if self.folder is not None else None

    def grant(self, notebook, session_id="s9"):
        """Give ``notebook`` a session (what happens once its browser tab loads)."""
        self.granted.append(Path(notebook))
        session = MarimoSession(session_id, Path(notebook).name, str(Path(notebook).resolve()))
        self.other_sessions.append(session)
        return session

    def resolve_session(self, notebook):
        from hailer.marimo_client import match_session

        session = match_session(self.sessions(), notebook, self.workspace) if notebook is not None else None
        if session is None:
            url = url_for(notebook) if notebook is not None else SERVER.url
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
    server: MarimoServer | None = SERVER
    bundle: ContextBundle = field(default_factory=ContextBundle)
    session_waits: list = field(default_factory=list)
    waited_session: object = "default"  # what _wait_for_session returns ("default" -> session s9)

    @property
    def url(self) -> str:
        return url_for(self.config.notebook)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "120")
    monkeypatch.delenv("HAILER_NOTEBOOK", raising=False)
    h = Harness(config=make_config(tmp_path), agent=FakeAgent(), client=FakeClient())

    def load_context(config):
        h.context_loads += 1
        return h.bundle

    def wait_session(client, notebook, timeout):
        h.session_waits.append(notebook)
        if h.waited_session == "default":
            return client.grant(notebook)  # the tab loaded: the notebook now has a session
        if isinstance(h.waited_session, BaseException):
            raise h.waited_session  # e.g. Ctrl+C while waiting
        return h.waited_session

    def make_client(server, config):
        if getattr(h.client, "workspace", None) is None:
            h.client.workspace = config.workspace
        return h.client

    monkeypatch.setattr(cli, "console_factory", lambda: Console(force_terminal=False, width=120, highlight=False, soft_wrap=True, color_system=None))
    monkeypatch.setattr(cli, "_load_config", lambda opts: h.config)
    monkeypatch.setattr(cli, "_validate_config", lambda config: [])
    monkeypatch.setattr(cli, "_setup_logging", lambda config, opts: None)
    monkeypatch.setattr(cli, "_find_server", lambda config: h.server)
    monkeypatch.setattr(cli, "_make_client", make_client)
    monkeypatch.setattr(cli, "_wait_for_session", wait_session)
    monkeypatch.setattr(cli, "_launch_command", lambda: LAUNCH)
    monkeypatch.setattr(cli, "_cm_help_code", lambda: "import marimo._code_mode as cm; help(cm)")
    monkeypatch.setattr(cli, "_resolve_key", lambda provider: h.key_source)
    monkeypatch.setattr(cli, "_store_key", lambda provider, value: h.stored.append((provider.id, value)))
    monkeypatch.setattr(cli, "_delete_key", lambda provider: (h.deleted.append(provider.id) or True))
    monkeypatch.setattr(cli, "_load_context", load_context)
    monkeypatch.setattr(cli, "_render_prompt", lambda config, name, args: f"PROMPT[{name}]({args})")
    monkeypatch.setattr(cli, "_make_agent", lambda config, bundle: h.agent)
    monkeypatch.setattr(cli, "_open_browser", lambda url: h.opened.append(url))

    def no_spawn(cmd, cwd, log_path):
        raise AssertionError("marimo was started; use the nb fixture when a test expects that")

    monkeypatch.setattr(cli, "_spawn_marimo", no_spawn)
    return h


def owned_client(config: HailerConfig, **kwargs) -> FakeClient:
    """A client for a server started on this workspace's notebooks folder (reused, not replaced)."""
    return FakeClient(folder=config.notebooks_root, **kwargs)


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
    assert "Launch:    uvx hailer notebook --foreground" in out
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
    assert "OPENAI_API_KEY is not set" in result.output
    assert "uvx hailer login openai" in result.output
    assert harness.agent.turns == []


def test_owned_server_without_session_is_reused_and_the_notebook_opened(harness):
    """Bare hailer, this workspace's server is up but the notebook tab is not: open it and wait, like hailer notebook."""
    harness.client = owned_client(harness.config, session=None)
    result = chat()
    assert result.exit_code == 0, result.output
    assert f"Using the running marimo at {SERVER.url}." in result.output
    assert harness.opened == [harness.url]
    assert harness.session_waits == [harness.config.notebook]
    assert "Notebook is open (session s9)." in result.output


def test_invalid_config_problem_is_fatal(harness, monkeypatch):
    monkeypatch.setattr(cli, "_validate_config", lambda config: ["[model_providers.internal].base_url is missing", "Warning: data directory not found"])
    result = chat()
    assert result.exit_code == 1
    assert "base_url is missing" in result.output


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #


def test_exec_inline_code(harness):
    result = runner.invoke(cli.app, ["exec", "-c", "print(1)"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "out" in result.output and "42" in result.output
    assert harness.client.codes == ["print(1)"]


def test_exec_failure_exit_code(harness):
    harness.client = FakeClient(exec_result=ExecResult(False, stderr="Traceback: boom"))
    result = runner.invoke(cli.app, ["exec", "-c", "1/0"], catch_exceptions=False)
    assert result.exit_code == 1
    assert "boom" in result.output


def test_exec_from_file_and_stdin(harness, tmp_path):
    script = tmp_path / "snippet.py"
    script.write_text("print('file')", encoding="utf-8")
    result = runner.invoke(cli.app, ["exec", str(script)], catch_exceptions=False)
    assert result.exit_code == 0
    result = runner.invoke(cli.app, ["exec", "-"], input="print('stdin')", catch_exceptions=False)
    assert result.exit_code == 0
    assert harness.client.codes == ["print('file')", "print('stdin')"]


def test_exec_without_code_and_marimo_down(harness):
    result = runner.invoke(cli.app, ["exec"], catch_exceptions=False)
    assert result.exit_code == 2
    harness.server = None
    result = runner.invoke(cli.app, ["exec", "-c", "1"], catch_exceptions=False)
    assert result.exit_code == 1
    assert "uvx hailer notebook --foreground" in result.output


def test_doctor_table_and_code_mode_probe(harness):
    harness.client = FakeClient(exec_result=ExecResult(True, stdout="Help on module ... get_context ..."))
    result = runner.invoke(cli.app, ["doctor"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "Hailer doctor" in result.output
    assert "OK" in result.output
    assert "code mode" in result.output


def test_doctor_fails_when_notebook_missing(harness):
    harness.config = make_config(harness.config.workspace, notebook_exists=False)
    result = runner.invoke(cli.app, ["doctor"], catch_exceptions=False)
    assert result.exit_code == 1
    assert "FAIL" in result.output


def test_status_subcommand(harness):
    result = runner.invoke(cli.app, ["status"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "Credentials: OPENAI_API_KEY from env" in result.output
    assert "Marimo:      http://127.0.0.1:2718 (session s1)" in result.output
    assert "Context:     0 file(s)" in result.output


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


@dataclass
class NotebookHarness:
    spawned: list = field(default_factory=list)
    foreground: list = field(default_factory=list)
    removed: list = field(default_factory=list)
    health_waits: list = field(default_factory=list)
    session_waits: list = field(default_factory=list)
    proc: FakeProc = field(default_factory=FakeProc)
    healthy: bool = True
    waited_session: object = "default"
    free_port: int | None = None


@pytest.fixture
def nb(harness, monkeypatch):
    """Notebook-command collaborators on top of the chat harness. Marimo is not running by default."""
    h = NotebookHarness()
    harness.server = None  # nothing discovered → hailer notebook starts its own server

    def spawn(cmd, cwd, log_path):
        h.spawned.append((cmd, cwd, log_path))
        return h.proc

    def wait_health(url, timeout, should_stop=None):
        h.health_waits.append(url)
        return h.healthy

    def wait_session(client, notebook, timeout):
        h.session_waits.append(notebook)
        if h.waited_session == "default":
            return MarimoSession("s9", "analysis.py", "notebooks/analysis.py")
        return h.waited_session

    monkeypatch.setattr(cli, "_spawn_marimo", spawn)
    monkeypatch.setattr(cli, "_run_foreground", lambda cmd, cwd: (h.foreground.append((cmd, cwd)) or 0))
    monkeypatch.setattr(cli, "_find_free_port", lambda preferred: h.free_port if h.free_port is not None else preferred)
    monkeypatch.setattr(cli, "_marimo_server_command", lambda config, port, headless=True: server_command(port, headless))
    monkeypatch.setattr(cli, "_wait_for_health", wait_health)
    monkeypatch.setattr(cli, "_wait_for_session", wait_session)
    monkeypatch.setattr(cli, "_registry_remove", lambda url: (h.removed.append(url) or True))
    monkeypatch.setattr(cli, "_kill_tree", lambda pid: None)
    return h


def notebook_cmd(args=(), input_text="/exit\n"):
    return runner.invoke(cli.app, ["notebook", *args], input=input_text, catch_exceptions=False)


def test_notebook_starts_marimo_runs_chat_and_stops_it(harness, nb):
    result = notebook_cmd(["--port", "2718"])
    assert result.exit_code == 0, result.output
    cmd, cwd, log_path = nb.spawned[0]
    assert cwd == harness.config.workspace
    assert log_path == harness.config.workspace / ".hailer" / "marimo.log"
    assert "--headless" in cmd and "--no-token" in cmd and "--skip-update-check" in cmd
    assert cmd[cmd.index("--port") + 1] == "2718"
    assert nb.health_waits == ["http://127.0.0.1:2718"]
    # the notebook already had a session, so no browser and no wait
    assert harness.opened == [] and nb.session_waits == []
    assert "Notebook is open (session s1)" in result.output
    assert harness.agent.started == [None], "the chat ran in the same terminal"
    assert nb.proc.terminated and nb.removed == ["http://127.0.0.1:2718"]
    assert "Stopped marimo." in result.output
    assert harness.agent.closed


def test_notebook_opens_browser_and_waits_for_session(harness, nb):
    harness.client = FakeClient(session=None)  # server up, nobody has the tab open yet
    result = notebook_cmd()
    assert result.exit_code == 0, result.output
    assert harness.opened == [harness.url], "browser opened exactly once"
    assert nb.session_waits == [harness.config.notebook]
    assert "Notebook is open (session s9)" in result.output
    assert "app view" in result.output and "Ctrl+." in result.output, "tells the user how to reach the code"


def test_notebook_no_browser_still_waits_and_continues_without_session(harness, nb):
    harness.client = FakeClient(session=None)
    nb.waited_session = None
    result = notebook_cmd(["--no-browser"])
    assert result.exit_code == 0, result.output
    assert harness.opened == []
    assert f"Open {harness.url} in your browser." in result.output
    assert "No kernel session yet" in result.output
    assert harness.agent.started == [None], "chat still starts; the agent reports the notebook state"
    assert nb.proc.terminated


def test_notebook_keep_marimo_leaves_server_running(harness, nb):
    result = notebook_cmd(["--keep-marimo"])
    assert result.exit_code == 0, result.output
    assert not nb.proc.terminated and nb.removed == []
    assert "still running at http://127.0.0.1:2718" in result.output
    assert "Stopped marimo." not in result.output


def test_notebook_stops_marimo_even_when_chat_fails(harness, nb):
    harness.agent.fail_with = HailerError("boom", hint="fix it")
    result = notebook_cmd(input_text="hello\n/exit\n")
    assert "boom" in result.output and "fix it" in result.output
    assert nb.proc.terminated and "Stopped marimo." in result.output


def test_notebook_reports_early_exit_with_log_tail(harness, nb, tmp_path):
    nb.healthy = False
    nb.proc = FakeProc(exit_code=1)
    log = harness.config.workspace / ".hailer" / "marimo.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("\n".join(f"line {i}" for i in range(20)), encoding="utf-8")
    result = notebook_cmd()
    assert result.exit_code == 1
    assert "Marimo exited early (code 1)" in result.output
    assert str(log) in result.output
    assert "line 19" in result.output and "line 4" not in result.output, "last 15 lines only"
    assert harness.agent.started == []


def test_notebook_picks_a_free_port_when_busy(harness, nb):
    nb.free_port = 2731
    result = notebook_cmd(["--port", "2718"])
    assert result.exit_code == 0, result.output
    assert "Port 2718 is busy; using 2731." in result.output
    assert nb.health_waits == ["http://127.0.0.1:2731"]
    assert nb.removed == ["http://127.0.0.1:2731"]


def test_notebook_reuses_running_server_with_session(harness, nb):
    harness.server = SERVER  # discovered, healthy, has our notebook open
    result = notebook_cmd()
    assert result.exit_code == 0, result.output
    assert nb.spawned == [], "no second server"
    assert f"Using the running marimo at {SERVER.url}." in result.output
    assert "Stopped marimo." not in result.output and nb.removed == []


def test_notebook_starts_own_server_when_live_server_has_no_session(harness, nb):
    harness.server = SERVER
    harness.client = FakeClient(session=None)
    result = notebook_cmd()
    assert result.exit_code == 0, result.output
    assert len(nb.spawned) == 1, "a live server without our notebook is not reused"


def test_notebook_fatal_local_check_exits_before_starting_marimo(harness, nb):
    harness.config = make_config(harness.config.workspace, notebook_exists=False)
    result = notebook_cmd()
    assert result.exit_code == 1
    assert "notebook: not found" in result.output
    assert "Run uvx hailer init to create it" in result.output
    assert nb.spawned == []


def test_notebook_foreground_runs_marimo_attached_without_chat(harness, nb):
    result = notebook_cmd(["--foreground", "--port", "2718"])
    assert result.exit_code == 0, result.output
    assert nb.foreground == [(server_command(2718, headless=False), harness.config.workspace)], "Hailer's interpreter, not uv run"
    assert nb.spawned == [] and harness.agent.started == []


def test_notebook_foreground_no_browser_runs_headless(harness, nb):
    result = notebook_cmd(["--foreground", "--no-browser", "--port", "2720"])
    assert result.exit_code == 0, result.output
    assert nb.foreground == [(server_command(2720, headless=True), harness.config.workspace)]


def test_notebook_new_flag_starts_fresh_thread(harness, nb):
    save_session(harness.config.workspace, SessionState(thread_id="old-thread", turns=3))
    result = notebook_cmd(["--new"])
    assert result.exit_code == 0, result.output
    assert harness.agent.started == [None]


def test_spawn_marimo_uses_background_flags_and_log(tmp_path, monkeypatch):
    calls = []

    class Popen:
        def __init__(self, cmd, **kwargs):
            calls.append((cmd, kwargs))
            self.pid = 1

    monkeypatch.setattr(cli.subprocess, "Popen", Popen)
    log = tmp_path / ".hailer" / "marimo.log"
    log.parent.mkdir()  # _start_marimo creates it (with its .gitignore) before spawning
    cli._spawn_marimo(["python", "-m", "marimo"], tmp_path, log)
    cmd, kwargs = calls[0]
    assert cmd == ["python", "-m", "marimo"] and kwargs["cwd"] == str(tmp_path)
    assert log.is_file(), "the child logs there"
    assert kwargs["stderr"] == cli.subprocess.STDOUT and kwargs["stdin"] == cli.subprocess.DEVNULL
    expected = getattr(cli.subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if cli.os.name == "nt" else 0
    assert kwargs["creationflags"] == expected, "own process group on Windows so Ctrl+C in the chat is not delivered to marimo"


def test_stop_process_terminates_then_kills():
    proc = FakeProc()
    cli._stop_process(proc)
    assert proc.terminated

    class Stubborn(FakeProc):
        def terminate(self):
            self.terminated = True  # stays alive

        def wait(self, timeout=None):
            if self.returncode is None:
                raise cli.subprocess.TimeoutExpired("marimo", timeout)
            return self.returncode

    stubborn = Stubborn()
    cli._stop_process(stubborn, timeout=0.01)
    assert stubborn.terminated and stubborn.killed


def test_doctor_and_status_without_a_server_for_this_workspace_point_at_hailer(harness):
    """No server serves this workspace: a warning that says how to get one, not "Marimo is not running"."""
    harness.server = None
    result = runner.invoke(cli.app, ["doctor"], catch_exceptions=False)
    assert result.exit_code == 0, "hailer starts a server itself, so this is not a failure"
    assert "No marimo server is running for this workspace." in result.output and "WARN" in result.output
    assert "uvx hailer" in result.output and "uvx hailer notebook" in result.output
    assert "Marimo is not running" not in result.output
    result = runner.invoke(cli.app, ["status"], catch_exceptions=False)
    assert "Marimo:      none for this workspace (uvx hailer starts one)" in result.output


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
    assert "Put the files you want to analyse in data/ (CSV, Parquet or JSON, any name)." in result.output
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
    assert notebooks.is_marimo_notebook(notebook)
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
    assert notebooks.is_marimo_notebook(ws / "work" / "q2.py")
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


def test_exec_without_session_shows_url(harness):
    harness.client = FakeClient(session=None)
    result = runner.invoke(cli.app, ["exec", "-c", "1"], catch_exceptions=False)
    assert result.exit_code == 1
    assert "not open in a browser" in result.output
    assert harness.url in result.output


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


def test_load_config_applies_active_notebook_from_state(tmp_path, monkeypatch):
    """The real _load_config returns the state file's notebook as config.notebook."""
    base = make_config(tmp_path)
    other = write_notebook(base, "other")
    monkeypatch.delenv("HAILER_NOTEBOOK", raising=False)
    monkeypatch.setattr("hailer.config.load_config", lambda workspace=None, config_path=None: base)
    assert cli._load_config(cli.CliOptions()).notebook == base.notebook, "no state file: the configured notebook"
    notebooks.save_active_notebook(base, other)
    config = cli._load_config(cli.CliOptions())
    assert config.notebook == other
    assert config.notebooks_root == base.notebooks_root
    # An explicit HAILER_NOTEBOOK wins and is written to the state file, so the tool server (which
    # reads only the file) and the next session start on the same notebook.
    monkeypatch.setenv("HAILER_NOTEBOOK", "explicit")
    assert cli._load_config(cli.CliOptions()).notebook == base.notebook, "an explicit override wins"
    assert active_state(base) == "notebooks/analysis.py"
    assert notebooks.load_active_notebook(base) == base.notebook


def test_notebook_command_shows_active_folder_and_usage(harness):
    result = chat(input_text="/notebook\n/exit\n")
    out = result.output
    assert "Notebook:  notebooks/analysis.py (active)" in out
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
    lines = [ln for ln in result.output.splitlines() if "notebooks/" in ln]
    assert any("notebooks/analysis.py" in ln and "active, open" in ln for ln in lines), lines
    assert any("notebooks/other.py" in ln and "active" not in ln and "open" not in ln for ln in lines), lines
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
    assert "Created notebooks/q2_churn.py from the starter template." in result.output
    assert "Active notebook: notebooks/q2_churn.py." in result.output
    assert active_state(harness.config) == "notebooks/q2_churn.py"
    # a freshly created notebook has no session: its app-view URL is opened and the tab is waited for
    assert harness.opened == [url_for(created)]
    assert harness.session_waits == [created]
    assert harness.client.granted == [created]
    assert "Notebook is open (session s9)." in result.output
    assert harness.agent.turns == [("hello", None), ("again", None)]
    assert harness.agent.preambles == [
        "[Hailer] The active notebook is now notebooks/q2_churn.py (created from the starter template, 1 cell). "
        "Call notebook_cells before editing.",
        None,
    ]
    assert harness.client.codes and "cm.get_context()" in harness.client.codes[0], "cell count via the list-cells snippet"


def test_notebook_new_empty_template_and_status_panel(harness):
    result = chat(input_text="/notebook new scratch --empty\n/status\n/exit\n")
    assert result.exit_code == 0, result.output
    source = (harness.config.notebooks_root / "scratch.py").read_text(encoding="utf-8")
    assert "marimo.App" in source and "hailer.periods" not in source
    assert "Created notebooks/scratch.py from the empty template." in result.output
    assert harness.agent.preambles == [], "no turn was sent, so the notice is still pending"
    status = result.output[result.output.index("Model"):]
    assert "scratch.py" in status and "Notebooks" in status and "notebooks" in status


def test_notebook_new_existing_name_is_an_error_and_keeps_active(harness):
    result = chat(input_text="/notebook new analysis\n/notebook new\n/exit\n")
    assert result.exit_code == 0, result.output
    assert "already exists" in result.output
    assert "Usage: /notebook new <name> [--empty]" in result.output
    assert active_state(harness.config) is None
    assert harness.agent.preambles == []


def test_notebook_open_by_name_opens_browser_and_waits_when_no_session(harness):
    other = write_notebook(harness.config, "other")
    harness.client = owned_client(harness.config, session=None)
    result = chat(input_text="/notebook open other\nhi\n/exit\n")
    assert result.exit_code == 0, result.output
    # startup opened the (then active) analysis notebook; the switch opened other.py and waited for it
    assert harness.opened == [url_for(harness.config.notebook), url_for(other)]
    assert harness.session_waits == [harness.config.notebook, other]
    assert "Notebook is open (session s9)." in result.output
    assert "Active notebook: notebooks/other.py." in result.output
    assert active_state(harness.config) == "notebooks/other.py"
    assert harness.agent.preambles == [
        "[Hailer] The active notebook is now notebooks/other.py (reopened, 1 cell). Call notebook_cells before editing."
    ]


def test_notebook_open_without_session_after_wait_says_so(harness):
    other = write_notebook(harness.config, "other")
    harness.client = owned_client(harness.config, session=None)
    harness.waited_session = None
    result = chat(input_text="/notebook open other.py\nhi\n/exit\n")
    assert "No kernel session yet" in result.output
    assert harness.agent.preambles == [
        "[Hailer] The active notebook is now notebooks/other.py (reopened, not open in a browser yet). "
        "Call notebook_cells before editing."
    ]
    assert harness.opened[-1] == url_for(other)


def test_notebook_open_active_notebook_is_a_no_op(harness):
    result = chat(input_text="/notebook open analysis\n/exit\n")
    assert "notebooks/analysis.py is already the active notebook." in result.output
    assert active_state(harness.config) is None
    assert harness.agent.preambles == []


def test_notebook_open_unknown_lists_available(harness):
    result = chat(input_text="/notebook open nope\n/notebook open\n/exit\n")
    assert "No notebook named 'nope' in notebooks." in result.output
    assert "Notebooks in notebooks: analysis." in result.output
    assert "Usage: /notebook open <name>" in result.output
    assert active_state(harness.config) is None


def test_notebook_open_outside_folder_is_refused(harness):
    outside = harness.config.workspace / "elsewhere.py"
    outside.write_text(NOTEBOOK_SOURCE, encoding="utf-8")
    result = chat(input_text="/notebook open elsewhere.py\n/exit\n")
    assert "outside the notebooks folder" in result.output
    assert active_state(harness.config) is None


def test_notebook_close_shuts_down_the_session(harness, nb):
    harness.server = SERVER  # nb starts with no server; the second chat below runs without one
    result = chat(input_text="/notebook close\n/notebook close\n/exit\n")
    assert result.exit_code == 0, result.output
    assert harness.client.closed == ["s1"]
    assert "Closed notebooks/analysis.py (session s1)" in result.output
    assert "It stays the active notebook; /notebook open analysis reopens it." in result.output
    assert "notebooks/analysis.py is not open (no kernel session)." in result.output
    harness.server = None
    result = chat(input_text="/notebook close\n/exit\n")
    assert "Marimo is not running." in result.output


def test_switch_made_by_the_model_is_detected_after_the_turn(harness):
    other = write_notebook(harness.config, "other")
    harness.client = owned_client(harness.config, session=None)
    harness.agent.on_turn = lambda: notebooks.save_active_notebook(harness.config, other)
    result = chat(input_text="make a new notebook\n/status\n/exit\n")
    assert result.exit_code == 0, result.output
    assert "Active notebook is now notebooks/other.py." in result.output
    assert harness.opened == [url_for(harness.config.notebook), url_for(other)], "opened once the switch was seen"
    assert harness.agent.preambles == [None], "the model already knows; no notice is queued"
    status = result.output[result.output.index("Active notebook is now"):]
    assert "other.py" in status


def test_switch_made_by_the_model_with_session_does_not_open_browser(harness):
    other = write_notebook(harness.config, "other")
    harness.client = FakeClient(other_sessions=[MarimoSession("s2", "other.py", str(other))])
    harness.agent.on_turn = lambda: notebooks.save_active_notebook(harness.config, other)
    result = chat(input_text="make a new notebook\n/exit\n")
    assert "Active notebook is now notebooks/other.py." in result.output
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
    assert nb.foreground == [(server_command(2718, headless=False), harness.config.workspace)]


def test_notebook_command_reuses_server_hosting_another_notebook(harness, nb):
    other = write_notebook(harness.config, "other")
    harness.server = SERVER
    harness.client = FakeClient(session=None, other_sessions=[MarimoSession("s2", "other.py", str(other))])
    result = notebook_cmd()
    assert result.exit_code == 0, result.output
    assert nb.spawned == [], "one server hosts every notebook in the folder"
    assert f"Using the running marimo at {SERVER.url}." in result.output
    assert harness.opened == [harness.url], "the active notebook is opened on that server"
    assert nb.session_waits == [harness.config.notebook]


def test_notebook_command_ignores_sessions_outside_the_folder(harness, nb):
    elsewhere = harness.config.workspace / "elsewhere.py"
    elsewhere.write_text(NOTEBOOK_SOURCE, encoding="utf-8")
    harness.server = SERVER
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
    assert "Active notebook: notebooks/other.py." in result.output
    assert active_state(harness.config) == "notebooks/other.py"
    assert harness.opened[-1] == url_for(other)
    assert harness.agent.preambles == [
        "[Hailer] The active notebook is now notebooks/other.py (reopened, not open in a browser yet). "
        "Call notebook_cells before editing."
    ]


def test_switch_notice_survives_marimo_error_during_the_wait(harness):
    other = write_notebook(harness.config, "other")
    harness.waited_session = HailerError("marimo went away", hint="restart it")
    result = chat(input_text="/notebook open other\nhi\n/exit\n")
    assert result.exit_code == 0, result.output
    assert "marimo went away" in result.output and "restart it" in result.output
    assert active_state(harness.config) == "notebooks/other.py"
    assert harness.agent.preambles[0].startswith("[Hailer] The active notebook is now notebooks/other.py (reopened, not open in a browser yet)")


def test_switch_made_by_the_model_is_detected_after_ctrl_c(harness):
    """A notebook_open tool call that completed before Ctrl+C is picked up straight away."""
    other = write_notebook(harness.config, "other")
    harness.client = owned_client(harness.config, session=None)
    harness.agent.on_turn = lambda: notebooks.save_active_notebook(harness.config, other)
    harness.agent.fail_with = KeyboardInterrupt()
    result = chat(input_text="open the other notebook\n/notebook\n/exit\n")
    assert result.exit_code == 0, result.output
    assert "Interrupted." in result.output
    assert "Active notebook is now notebooks/other.py." in result.output
    assert harness.opened[-1] == url_for(other), "opened once the switch was seen"
    assert "Notebook:  notebooks/other.py (active)" in result.output


def test_switch_made_by_the_model_is_detected_after_a_failed_turn(harness):
    other = write_notebook(harness.config, "other")
    harness.agent.on_turn = lambda: notebooks.save_active_notebook(harness.config, other)
    harness.agent.fail_with = HailerError("the model endpoint hung up", hint="retry")
    result = chat(input_text="open the other notebook\n/status\n/exit\n")
    assert result.exit_code == 0, result.output
    assert "the model endpoint hung up" in result.output
    assert "Active notebook is now notebooks/other.py." in result.output
    status = result.output[result.output.index("Model"):]
    assert "other.py" in status


def test_notebook_and_status_commands_read_fresh_state(harness):
    """A switch written to the state file behind the CLI's back (a late tool call) is applied
    before /notebook and /status act, without opening a browser tab for it."""
    other = write_notebook(harness.config, "other")
    notebooks.save_active_notebook(harness.config, other)  # after _load_config ran (it is faked)
    result = chat(input_text="/status\n/notebook\n/exit\n")
    assert result.exit_code == 0, result.output
    assert result.output.count("Active notebook is now notebooks/other.py.") == 1
    assert "Notebook:  notebooks/other.py (active)" in result.output
    assert harness.opened == [], "a stale-state refresh never opens a browser"


def test_notebook_command_lists_recent_notebooks(harness):
    write_notebook(harness.config, "alpha")
    write_notebook(harness.config, "beta")
    result = chat(input_text="/notebook open alpha\n/notebook open beta\n/notebook open analysis\n/notebook\n/exit\n")
    assert result.exit_code == 0, result.output
    shown = result.output[result.output.rindex("Notebook:  notebooks/analysis.py (active)"):]
    assert "Recent:    notebooks/beta.py, notebooks/alpha.py" in shown
    assert "notebooks/analysis.py" not in shown.split("Recent:")[1].splitlines()[0]


def test_session_matching_is_by_path_not_filename(harness):
    """Two notebooks called report.py in different sub-folders never share a session."""
    sub_a = write_notebook(harness.config, "a/report")
    sub_b = write_notebook(harness.config, "b/report")
    harness.client = FakeClient(session=None, other_sessions=[MarimoSession("s2", "report.py", str(sub_a))])
    result = chat(input_text="/notebook list\n/notebook open b/report.py\n/exit\n")
    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.splitlines() if "report.py" in ln]
    assert any("notebooks/a/report.py" in ln and "open" in ln for ln in lines), lines
    assert any("notebooks/b/report.py" in ln and "open" not in ln for ln in lines), lines
    assert harness.opened[-1] == url_for(sub_b), "b/report.py had no session of its own, so it was opened"
    assert harness.session_waits == [harness.config.notebook, sub_b]


# --------------------------------------------------------------------------- #
# Bare hailer: reuse this workspace's marimo or start one (the same session as hailer notebook)
# --------------------------------------------------------------------------- #


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def real_discovery(harness, monkeypatch, tmp_path):
    """The real find_server and MarimoClient over a private registry folder; returns a function that
    registers FakeMarimo servers in it the way marimo does."""
    registry = tmp_path / "registry"
    monkeypatch.setattr("hailer.marimo_client.registry_dir", lambda: registry)
    monkeypatch.setattr(cli, "_find_server", REAL_FIND_SERVER)
    monkeypatch.setattr(cli, "_make_client", REAL_MAKE_CLIENT)

    def register(*servers):
        registry.mkdir(exist_ok=True)
        for srv in servers:
            port = srv.server_address[1]
            entry = {"server_id": f"127.0.0.1:{port}", "pid": 1, "host": "127.0.0.1", "port": port, "base_url": "", "started_at": "", "version": "0.24.2"}
            (registry / f"127.0.0.1_{port}.json").write_text(json.dumps(entry), encoding="utf-8")

    return register


@pytest.mark.parametrize("args", [[], ["notebook"]], ids=["hailer", "hailer-notebook"])
def test_zero_servers_starts_one_and_stops_it_on_exit(harness, nb, args):
    """Bare hailer does exactly what hailer notebook does: start marimo on the notebooks folder, chat, stop it."""
    result = chat(args=args)
    assert result.exit_code == 0, result.output
    assert len(nb.spawned) == 1
    cmd, cwd, log_path = nb.spawned[0]
    assert cmd == server_command(2718) and cwd == harness.config.workspace
    assert log_path == harness.config.workspace / ".hailer" / "marimo.log"
    assert "Marimo is running at http://127.0.0.1:2718" in result.output
    assert harness.agent.started == [None], "the chat ran"
    assert nb.proc.terminated and nb.removed == ["http://127.0.0.1:2718"]
    assert "Stopped marimo." in result.output


def test_two_live_servers_attaches_to_the_one_owned_by_this_workspace(harness, real_discovery):
    ws = harness.config.workspace
    with running() as foreign, running() as ours:
        foreign.root = str(ws.parent / "other-worktree" / "notebooks")  # another worktree of the same repo
        ours.sessions = {"s1": {"filename": "analysis.py", "path": str(harness.config.notebook)}}
        real_discovery(foreign, ours)
        result = chat()
    assert result.exit_code == 0, result.output
    assert f"Using the running marimo at {ours.url}." in result.output
    assert "Notebook is open (session s1)." in result.output
    assert harness.opened == [], "the notebook was already open there"
    assert "Stopped marimo." not in result.output, "a server Hailer did not start is left alone"


def test_two_live_servers_attaches_to_ours_before_its_tab_is_open(harness, real_discovery):
    """No session anywhere: the server started on this workspace's notebooks folder is recognised by its root."""
    ws = harness.config.workspace
    with running() as foreign, running() as ours:
        foreign.root = str(ws.parent / "other-worktree" / "notebooks")
        ours.root = str(harness.config.notebooks_root)
        real_discovery(foreign, ours)
        harness.waited_session = MarimoSession("s9", "analysis.py", str(harness.config.notebook))  # the tab loaded
        result = chat()
    assert result.exit_code == 0, result.output
    assert f"Using the running marimo at {ours.url}." in result.output
    assert harness.session_waits == [harness.config.notebook]


def test_one_live_server_owned_by_another_workspace_is_not_attached_to(harness, nb, real_discovery):
    ws = harness.config.workspace
    with running() as foreign:
        foreign.root = str(ws.parent / "other-worktree" / "notebooks")
        foreign.sessions = {"s1": {"filename": "analysis.py", "path": str(ws.parent / "other-worktree" / "notebooks" / "analysis.py")}}
        real_discovery(foreign)
        nb.free_port = free_port()  # the real client must not reach whatever listens on 2718 here
        result = chat()
    assert result.exit_code == 0, result.output
    assert foreign.url not in result.output
    assert len(nb.spawned) == 1, "this workspace got its own server"
    assert nb.proc.terminated and "Stopped marimo." in result.output


@pytest.mark.parametrize("args", [[], ["notebook"]], ids=["hailer", "hailer-notebook"])
def test_pinned_url_that_is_down_is_an_error_and_no_server_is_started(harness, nb, real_discovery, args):
    pinned = f"http://127.0.0.1:{free_port()}"
    harness.config = replace(harness.config, marimo_url=pinned)
    result = chat(args=args)
    assert result.exit_code == 1
    assert f"Marimo is not running at {pinned}." in result.output
    assert "does not start one in its place" in result.output
    assert nb.spawned == [] and harness.agent.started == []
    if args:
        return  # doctor once is enough (a refused loopback connection takes ~2 s on Windows)
    result = runner.invoke(cli.app, ["doctor"], catch_exceptions=False)
    assert result.exit_code == 1, "a pinned server that is down is a failure"
    assert f"not running at {pinned}" in result.output


def test_pinned_url_that_is_up_is_used_even_when_it_serves_another_folder(harness, nb, real_discovery):
    with running() as pinned:
        pinned.root = str(harness.config.workspace.parent / "elsewhere")
        harness.config = replace(harness.config, marimo_url=pinned.url)
        result = chat()
    assert result.exit_code == 0, result.output
    assert f"Using the running marimo at {pinned.url}." in result.output
    assert nb.spawned == []


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
