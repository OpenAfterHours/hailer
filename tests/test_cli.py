"""CLI tests: every collaborator is faked; no marimo, Codex, keyring or network."""

from __future__ import annotations

import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

import hailer.cli as cli
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
SERVER = MarimoServer(url="http://127.0.0.1:2718", server_id="127.0.0.1:2718", version="0.24.2", source="config")
LAUNCH = ["uv", "run", "marimo", "edit", "notebooks/analysis.py", "--no-token"]
URL = "http://127.0.0.1:2718/?file=notebooks/analysis.py&view-as=present"
INTERNAL = ProviderConfig(
    id="internal",
    base_url="https://llm.example.internal/v1",
    env_key="INTERNAL_MODEL_API_KEY",
    requires_openai_auth=False,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


@dataclass
class FakeAgent:
    thread_id: str = "thread-1"
    stream: bool = False
    fail_with: BaseException | None = None
    turns: list = field(default_factory=list)
    started: list = field(default_factory=list)
    model: tuple | None = None
    interrupted: bool = False
    closed: bool = False
    bundle: ContextBundle | None = None
    set_model_restarts: bool = False  # True: set_model starts its own thread (provider change) and returns True
    key_source: str | None = None
    new_threads: int = 0

    def start(self, *, resume_thread_id=None):
        self.started.append(resume_thread_id)
        return resume_thread_id or self.thread_id

    def run_turn(self, text, *, on_event=None, skill=None):
        self.turns.append((text, skill.name if skill else None))
        if self.fail_with is not None:
            if isinstance(self.fail_with, KeyboardInterrupt):
                self.interrupted = True  # the real agent interrupts the turn before re-raising
            raise self.fail_with
        final = f"Answer to: {text}"
        if on_event:
            on_event(AgentEvent("command", "dir data"))
            on_event(AgentEvent("tool_call", "marimo_execute"))
            if self.stream:
                # Codex streams interim commentary and the final answer as separate messages
                for chunk in ("Checking the ", "notebook...", "Answer ", "to: ", text):
                    on_event(AgentEvent("message_delta", chunk))
        return TurnSummary(final_response=final, thread_id=self.thread_id, turn_id="turn-1", input_tokens=10, output_tokens=5)

    def interrupt(self):
        self.interrupted = True

    def new_thread(self):
        self.new_threads += 1
        self.thread_id = "thread-2"
        return self.thread_id

    def set_model(self, name, provider=None):
        self.model = (name, provider)
        if self.set_model_restarts:
            self.thread_id = "thread-restarted"
            return True
        return False

    def close(self):
        self.closed = True


class FakeClient:
    def __init__(self, *, healthy=True, session="default", exec_result=None):
        self.healthy = healthy
        self.session = MarimoSession("s1", "analysis.py", "notebooks/analysis.py") if session == "default" else session
        self.exec_result = exec_result
        self.codes: list[str] = []

    def health(self):
        return self.healthy

    def resolve_session(self, notebook):
        if self.session is None:
            raise NoSessionError("The notebook is not open in a browser.", hint=f"Open {URL} in your browser.")
        return self.session

    def execute(self, code, *, session_id=None, notebook=None, on_stdout=None, on_stderr=None, timeout=600.0):
        if self.session is None and session_id is None:
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


def make_config(ws: Path, *, notebook_exists=True, provider="openai", providers=None, web=None) -> HailerConfig:
    nb = ws / "notebooks" / "analysis.py"
    if notebook_exists:
        nb.parent.mkdir(parents=True, exist_ok=True)
        nb.write_text("import marimo\n", encoding="utf-8")
    elif nb.exists():
        nb.unlink()
    (ws / "data").mkdir(exist_ok=True)
    cfg_dir = ws / ".config" / "hailer"
    return HailerConfig(
        workspace=ws,
        notebook=nb,
        data_dir=ws / "data",
        context_dir=cfg_dir / "context",
        skills_dir=cfg_dir / "skills",
        prompts_dir=cfg_dir / "prompts",
        model=ModelConfig(name="gpt-5.5", provider=provider),
        providers=providers or {},
        web=web or WebConfig(),
        config_path=ws / "hailer.toml",
    )


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
    prompt_hash: str = "hash-1"


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "120")
    h = Harness(config=make_config(tmp_path), agent=FakeAgent(), client=FakeClient())

    def load_context(config):
        h.context_loads += 1
        return h.bundle

    monkeypatch.setattr(cli, "_prompt_hash", lambda config, bundle: h.prompt_hash)

    monkeypatch.setattr(cli, "console_factory", lambda: Console(force_terminal=False, width=120, highlight=False, soft_wrap=True, color_system=None))
    monkeypatch.setattr(cli, "_load_config", lambda opts: h.config)
    monkeypatch.setattr(cli, "_validate_config", lambda config: [])
    monkeypatch.setattr(cli, "_setup_logging", lambda config, opts: None)
    monkeypatch.setattr(cli, "_find_server", lambda config: h.server)
    monkeypatch.setattr(cli, "_make_client", lambda server, config: h.client)
    monkeypatch.setattr(cli, "_notebook_url", lambda server, config: URL)
    monkeypatch.setattr(cli, "_launch_command", lambda config, port=None: LAUNCH + (["--port", str(port)] if port else []))
    monkeypatch.setattr(cli, "_cm_help_code", lambda: "import marimo._code_mode as cm; help(cm)")
    monkeypatch.setattr(cli, "_resolve_key", lambda provider: h.key_source)
    monkeypatch.setattr(cli, "_store_key", lambda provider, value: h.stored.append((provider.id, value)))
    monkeypatch.setattr(cli, "_delete_key", lambda provider: (h.deleted.append(provider.id) or True))
    monkeypatch.setattr(cli, "_load_context", load_context)
    monkeypatch.setattr(cli, "_render_prompt", lambda config, name, args: f"PROMPT[{name}]({args})")
    monkeypatch.setattr(cli, "_make_agent", lambda config, bundle: h.agent)
    monkeypatch.setattr(cli, "_open_browser", lambda url: h.opened.append(url))
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
    assert "Launch:" in out and "--no-token" in out
    assert URL in out
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
    harness.agent.start = lambda *, resume_thread_id=None: "fresh"  # resume failed inside the agent
    result = chat()
    assert "could not be resumed" in result.output
    saved = json.loads(session_path(harness.config.workspace).read_text())
    assert saved["thread_id"] == "fresh" and saved["turns"] == 0


def test_hailer_error_in_turn_keeps_loop(harness):
    harness.agent.fail_with = CredentialsError("Endpoint rejected the API key (401).", hint="Run: uv run hailer login openai")
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
    assert "uv run hailer login internal" in result.output


def test_missing_openai_key_is_only_a_warning(harness):
    harness.key_source = (None, "missing")
    result = chat()
    assert result.exit_code == 0, result.output
    assert "ChatGPT login" in result.output
    assert harness.agent.turns == []


def test_marimo_down_is_fatal_with_launch_hint(harness):
    harness.server = None
    result = chat()
    assert result.exit_code == 1
    assert "Marimo is not running." in result.output
    assert "uv run marimo edit notebooks/analysis.py --no-token" in result.output
    assert "Then run Hailer again" in result.output
    assert "uv run hailer" in result.output


def test_no_session_is_warning_and_opens_browser(harness):
    harness.client = FakeClient(session=None)
    result = chat()
    assert result.exit_code == 0, result.output
    assert "not open in a browser" in result.output
    assert harness.opened == [URL]


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
    assert "--no-token" in result.output


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
    monkeypatch.setattr(cli, "_marimo_server_command", lambda config, port: [sys.executable, "-m", "marimo", "edit", "notebooks/analysis.py", "--no-token", "--headless", "--port", str(port), "--skip-update-check"])
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
    assert harness.opened == [URL], "browser opened exactly once"
    assert nb.session_waits == [harness.config.notebook]
    assert "Notebook is open (session s9)" in result.output
    assert "app view" in result.output and "Ctrl+." in result.output, "tells the user how to reach the code"


def test_notebook_no_browser_still_waits_and_continues_without_session(harness, nb):
    harness.client = FakeClient(session=None)
    nb.waited_session = None
    result = notebook_cmd(["--no-browser"])
    assert result.exit_code == 0, result.output
    assert harness.opened == []
    assert f"Open {URL} in your browser." in result.output
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
    assert nb.spawned == []


def test_notebook_foreground_runs_marimo_attached_without_chat(harness, nb):
    result = notebook_cmd(["--foreground", "--port", "2718"])
    assert result.exit_code == 0, result.output
    assert nb.foreground == [(LAUNCH + ["--port", "2718"], harness.config.workspace)]
    assert nb.spawned == [] and harness.agent.started == []


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
    cli._spawn_marimo(["python", "-m", "marimo"], tmp_path, log)
    cmd, kwargs = calls[0]
    assert cmd == ["python", "-m", "marimo"] and kwargs["cwd"] == str(tmp_path)
    assert log.parent.is_dir(), "log directory created"
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


def test_marimo_not_running_hint_offers_one_command_route(harness):
    harness.server = None
    result = chat()
    assert result.exit_code == 1
    assert "Marimo is not running." in result.output
    assert "Start everything in one go:" in result.output
    assert "uv run hailer notebook" in result.output
    assert "Or start it yourself with:" in result.output
    assert "--no-token" in result.output
    assert "Then run Hailer again:" in result.output


def test_init_writes_config_and_skeleton(harness, tmp_path, monkeypatch):
    ws = tmp_path / "fresh"
    ws.mkdir()
    monkeypatch.setattr(cli, "_example_config_dir", lambda: None)
    result = runner.invoke(cli.app, ["--workspace", str(ws), "init"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert (ws / "hailer.toml").exists()
    for sub in ("context", "skills", "prompts"):
        assert (ws / ".config" / "hailer" / sub / "README.md").exists()
    assert "Next steps" in result.output
    result = runner.invoke(cli.app, ["--workspace", str(ws), "init"], catch_exceptions=False)
    assert "already exists" in result.output


def test_version_flag():
    result = runner.invoke(cli.app, ["--version"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "hailer 0.1.0" in result.output


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
    assert harness.agent.interrupted
    assert "Bye." in result.output


def test_ctrl_c_at_prompt_exits_cleanly(harness, monkeypatch):
    def raise_interrupt(self, prompt="", **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(Console, "input", raise_interrupt)
    result = chat(input_text="")
    assert result.exit_code == 0
    assert "Bye." in result.output
    assert harness.agent.closed


def test_verbose_prints_traceback_for_hailer_error(harness):
    harness.agent.fail_with = CredentialsError("Endpoint rejected the API key (401).", hint="Run: uv run hailer login openai")
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
    assert URL in result.output


def test_login_openai_then_status_reports_keyring(harness, monkeypatch):
    stored: dict[str, str] = {}
    monkeypatch.setattr(cli, "_store_key", lambda provider, value: stored.__setitem__(provider.id, value))
    monkeypatch.setattr(cli, "_resolve_key", lambda provider: ("v", "keyring") if provider.id in stored else (None, "missing"))
    result = runner.invoke(cli.app, ["status"], catch_exceptions=False)
    assert "Credentials: Codex login (no OPENAI_API_KEY set)" in result.output
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


def test_model_switch_does_not_start_a_second_thread(harness):
    harness.config = make_config(harness.config.workspace, providers={"internal": INTERNAL})
    harness.agent.set_model_restarts = True
    result = chat(input_text="/model internal:foo\n/exit\n")
    assert result.exit_code == 0, result.output
    assert harness.agent.new_threads == 0
    assert result.output.count("started a new thread") == 1
    saved = json.loads(session_path(harness.config.workspace).read_text())
    assert saved["thread_id"] == "thread-restarted"


def test_resume_warns_when_prompt_changed(harness):
    save_session(harness.config.workspace, SessionState(thread_id="old-thread", turns=1), prompt_hash="hash-0")
    result = chat()
    assert "Resumed conversation" in result.output
    assert "use /new to apply" in result.output
    # a resume keeps the hash the thread was started with
    assert json.loads(session_path(harness.config.workspace).read_text())["prompt_hash"] == "hash-0"


def test_new_thread_records_current_prompt_hash(harness):
    save_session(harness.config.workspace, SessionState(thread_id="old-thread", turns=1), prompt_hash="hash-0")
    result = chat(input_text="/new\n/exit\n")
    assert result.exit_code == 0
    assert json.loads(session_path(harness.config.workspace).read_text())["prompt_hash"] == "hash-1"
    harness.agent = FakeAgent()
    result = chat()
    assert "use /new to apply" not in result.output


def test_commentary_deltas_are_not_printed_and_answer_appears_once(harness):
    harness.agent.stream = True
    result = chat(input_text="hi\n/exit\n")
    assert result.output.count("Answer to: hi") == 1
    assert "Checking the notebook" not in result.output


@pytest.mark.parametrize(
    "raw, shown",
    [
        (
            '"C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command \'Get-ChildItem data\'',
            "Get-ChildItem data",
        ),
        ('powershell -NoProfile -Command "dir data"', "dir data"),
        ("cmd.exe /d /s /c dir data", "dir data"),
        ("cmd /c type hailer.toml", "type hailer.toml"),
        ("bash -lc 'ls data'", "ls data"),
        ('/bin/sh -c "ls"', "ls"),
        ("uv run pytest -q", "uv run pytest -q"),
    ],
)
def test_display_command_strips_shell_wrappers(raw, shown):
    assert cli._display_command(raw) == shown


def test_progress_line_shows_inner_command(harness):
    console = Console(force_terminal=False, width=100, highlight=False, color_system=None)
    display = cli._TurnDisplay(console)
    display(cli.AgentEvent("command", '"C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command \'dir data\''))
    assert display.last_activity == "dir data"
