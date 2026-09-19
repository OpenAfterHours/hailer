"""CLI tests: every collaborator is faked; no marimo, model endpoint, keyring or network."""

from __future__ import annotations

import io
import json
import os
import re
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import pytest
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
SERVER = MarimoServer(url="http://127.0.0.1:2718", server_id="127.0.0.1:2718", version="0.24.2", source="config")
LAUNCH = ["uvx", "hailer", "notebook", "--foreground"]
TOKEN = "test-token-0123456789"  # what the nb fixture's runtime hands the servers it starts


def server_command(port, headless=True, folder="notebooks"):
    """The command LocalRuntime runs for the nb fixture's workspace: Hailer's interpreter, the token on stdin."""
    return [
        sys.executable, "-m", "marimo", "edit", folder, "--token-password-file", "-",
        *(["--headless"] if headless else []), "--port", str(port), "--skip-update-check",
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
    for its path (or granted later, the way a browser tab would), matched by the real rule."""

    def __init__(self, *, healthy=True, session="default", exec_result=None, other_sessions=(), workspace=None):
        self.healthy = healthy
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
    agent_servers: list = field(default_factory=list)  # the server each agent was pinned to (None: discovered)

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
    monkeypatch.setattr(cli, "_make_agent", lambda config, bundle, server=None: (h.agent_servers.append(server) or h.agent))
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


def test_marimo_down_is_fatal_with_launch_hint(harness):
    harness.server = None
    result = chat()
    assert result.exit_code == 1
    assert "Marimo is not running." in result.output
    assert "uvx hailer notebook --foreground" in result.output, "marimo on its own, with Hailer's interpreter"
    assert "Then run Hailer again" in result.output
    assert "uv run" not in result.output, "no project .venv is needed"


def test_no_session_is_warning_and_opens_browser(harness):
    harness.client = FakeClient(session=None)
    result = chat()
    assert result.exit_code == 0, result.output
    assert "not open in a browser" in result.output
    assert harness.opened == [harness.url]


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
    removed: list = field(default_factory=list)
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
    """Notebook-command collaborators on top of the chat harness: the real LocalRuntime with its
    process layer faked (nothing is spawned or killed). Marimo is not running by default."""
    from hailer.kernel import LocalProcesses, LocalRuntime

    h = NotebookHarness()
    harness.server = None  # nothing discovered → hailer notebook starts its own server

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

    def wait_health(url, timeout, should_stop=None):
        h.health_waits.append(url)
        return h.healthy

    def wait_session(client, notebook, timeout):
        h.session_waits.append(notebook)
        if h.waited_session == "default":
            return MarimoSession("s9", "analysis.py", "notebooks/analysis.py")
        return h.waited_session

    procs = LocalProcesses(
        spawn=spawn,
        attach=attach,
        kill_tree=lambda pid: None,
        kill_pid=lambda pid: None,
        wait_for_health=wait_health,
        remove_registry_entry=lambda url: (h.removed.append(url) or True),
    )
    monkeypatch.setattr(cli, "_runtime_for", lambda config: LocalRuntime(config, procs=procs, token_factory=lambda: TOKEN))
    monkeypatch.setattr(cli, "_find_free_port", lambda preferred: h.free_port if h.free_port is not None else preferred)
    monkeypatch.setattr(cli, "_wait_for_session", wait_session)
    return h


def notebook_cmd(args=(), input_text="/exit\n"):
    return runner.invoke(cli.app, ["notebook", *args], input=input_text, catch_exceptions=False)


def test_notebook_starts_marimo_runs_chat_and_stops_it(harness, nb, monkeypatch):
    from hailer.kernel import kernel_state_path

    monkeypatch.setenv("OPENAI_API_KEY", "sk-parent-shell-key-123456")
    recorded: list = []
    harness.agent.on_turn = lambda: recorded.append(json.loads(kernel_state_path(harness.config.workspace).read_text()))
    result = notebook_cmd(["--port", "2718"], input_text="hi\n/exit\n")
    assert result.exit_code == 0, result.output
    cmd, cwd, log_path = nb.spawned[0]
    assert cwd == harness.config.workspace
    assert log_path == harness.config.workspace / ".hailer" / "marimo.log"
    assert cmd == server_command(2718), "headless, token on stdin (never --no-token, never on the command line)"
    assert TOKEN not in " ".join(cmd) and nb.stdin == [TOKEN]
    assert "OPENAI_API_KEY" not in nb.envs[0], "the kernel's environment has no API key"
    assert nb.health_waits == ["http://127.0.0.1:2718"]
    # the notebook already had a session, so no browser and no wait
    assert harness.opened == [] and nb.session_waits == []
    assert "Notebook is open (session s1)" in result.output
    assert "Kernel:     local (runs as you; not isolated)" in result.output
    assert harness.agent.started == [None], "the chat ran in the same terminal"
    # while the chat ran, .hailer/kernel.json recorded the server so other terminals can attach
    assert recorded and recorded[0]["url"] == "http://127.0.0.1:2718" and recorded[0]["token"] == TOKEN
    assert recorded[0]["runtime"] == "local" and recorded[0]["pid"] == nb.proc.pid
    assert nb.proc.terminated and nb.removed == ["http://127.0.0.1:2718"]
    assert not kernel_state_path(harness.config.workspace).exists(), "deleted with the server"
    assert "Stopped marimo." in result.output
    assert harness.agent.closed


def test_notebook_opens_browser_and_waits_for_session(harness, nb):
    harness.client = FakeClient(session=None)  # server up, nobody has the tab open yet
    result = notebook_cmd()
    assert result.exit_code == 0, result.output
    assert harness.opened == [signed(harness.url)], "browser opened exactly once, signed in with the server's token"
    assert nb.session_waits == [harness.config.notebook]
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


def test_notebook_keep_marimo_leaves_server_running(harness, nb):
    from hailer.kernel import read_kernel_state

    result = notebook_cmd(["--keep-marimo"])
    assert result.exit_code == 0, result.output
    assert not nb.proc.terminated and nb.removed == []
    assert "still running at http://127.0.0.1:2718" in result.output
    assert "process 4242" in result.output and "uvx hailer in this workspace attaches to it" in result.output
    assert "Stopped marimo." not in result.output
    state = read_kernel_state(harness.config.workspace)
    assert state is not None and state.url == "http://127.0.0.1:2718" and state.token == TOKEN, "kept for other terminals"


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


def test_notebook_foreground_runs_marimo_attached_without_chat(harness, nb, monkeypatch):
    from hailer.kernel import kernel_state_path

    recorded: list = []
    nb.foreground_proc.wait = lambda timeout=None: (recorded.append(kernel_state_path(harness.config.workspace).exists()) or 0)
    result = notebook_cmd(["--foreground", "--port", "2718"])
    assert result.exit_code == 0, result.output
    assert nb.foreground == [(server_command(2718), harness.config.workspace)], "Hailer's interpreter, not uv run"
    assert nb.stdin == [TOKEN], "the token goes to marimo's stdin in the foreground too"
    assert nb.spawned == [] and harness.agent.started == []
    # Hailer opens marimo's home page signed in, and records the server while it runs
    assert harness.opened == [f"http://127.0.0.1:2718/?access_token={TOKEN}"]
    assert recorded[0] is True, "uvx hailer in another terminal finds it through .hailer/kernel.json"
    assert not kernel_state_path(harness.config.workspace).exists(), "deleted when marimo stops"
    assert "Kernel:     local (runs as you; not isolated)" in result.output
    assert "Chat with it from another terminal: uvx hailer" in result.output


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


def test_marimo_not_running_hint_offers_one_command_route(harness):
    harness.server = None
    result = chat()
    assert result.exit_code == 1
    assert "Marimo is not running." in result.output
    assert "Start everything in one go:" in result.output
    assert "uvx hailer notebook" in result.output
    assert "Or run marimo on its own in another terminal:" in result.output
    assert "uvx hailer notebook --foreground" in result.output
    assert "Then run Hailer again:" in result.output


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
    harness.client = FakeClient(session=None)
    result = chat(input_text="/notebook open other\nhi\n/exit\n")
    assert result.exit_code == 0, result.output
    # startup opened the (then active) analysis notebook; the switch opened other.py and waited for it
    assert harness.opened == [url_for(harness.config.notebook), url_for(other)]
    assert harness.session_waits == [other]
    assert "Notebook is open (session s9)." in result.output
    assert "Active notebook: notebooks/other.py." in result.output
    assert active_state(harness.config) == "notebooks/other.py"
    assert harness.agent.preambles == [
        "[Hailer] The active notebook is now notebooks/other.py (reopened, 1 cell). Call notebook_cells before editing."
    ]


def test_notebook_open_without_session_after_wait_says_so(harness):
    other = write_notebook(harness.config, "other")
    harness.client = FakeClient(session=None)
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


def test_notebook_close_shuts_down_the_session(harness):
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
    harness.client = FakeClient(session=None)
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
    assert nb.foreground == [(server_command(2718, folder="nbs"), harness.config.workspace)]


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
    harness.client = FakeClient(session=None)
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
    assert harness.session_waits == [sub_b]


# --------------------------------------------------------------------------- #
# Kernel runtimes: the Kernel line, doctor, docker fails closed, attaching, tokens
# --------------------------------------------------------------------------- #

LOCAL_LINE = "local (runs as you; not isolated)"
DOCKER_SERVER = MarimoServer(url="http://127.0.0.1:2731", server_id="127.0.0.1:2731", source="kernel", token=TOKEN, runtime="docker")
IMAGE = f"ghcr.io/openafterhours/hailer-kernel:{__version__}"


def docker_config(config: HailerConfig) -> HailerConfig:
    return replace(config, kernel=KernelConfig(runtime="docker"))


def no_server(url, token):
    return False


@pytest.fixture
def docker(harness, monkeypatch):
    """The real DockerRuntime (and ``hailer kernel ...``) driving a scripted docker CLI with state:
    nothing runs. Local configs still get whatever runtime was set up before (list ``nb`` first
    to fake it). Liveness probes answer no (nothing listens)."""
    from fake_docker import FakeDocker
    from hailer import kernel as kernel_mod
    from hailer import kernel_docker

    fake = FakeDocker()
    local_runtime_for = cli._runtime_for

    def runtime_for(config):
        if config.kernel.runtime != "docker":
            return local_runtime_for(config)
        return kernel_docker.DockerRuntime(
            config, runner=fake, token_factory=lambda: TOKEN, health=lambda url, timeout, should_stop=None: True, user=lambda: None
        )  # fmt: skip

    monkeypatch.setattr(cli, "_runtime_for", runtime_for)
    monkeypatch.setattr(cli, "_docker_runner", lambda: fake)
    monkeypatch.setattr(kernel_mod, "answers_with_token", no_server)  # a test that needs a live server patches it again
    monkeypatch.setattr(kernel_docker, "answers_with_token", no_server)
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


def test_doctor_shows_the_kernel_in_use(harness, docker):
    """Like status: asking for local but attached to a docker kernel Hailer started, doctor checks docker."""
    harness.server = DOCKER_SERVER
    result = runner.invoke(cli.app, ["doctor"], catch_exceptions=False)
    lines = result.output.splitlines()
    assert any(" kernel " in line and "docker (hailer-kernel" in line for line in lines), result.output
    assert any(" docker " in line and "Docker 29.4.3 (Linux engine)" in line for line in lines)
    assert LOCAL_LINE not in result.output


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


def test_chat_attached_to_a_docker_kernel_runs_as_docker(harness, monkeypatch):
    """Asking for local may attach to a docker kernel Hailer started (from .hailer/kernel.json); the
    session then runs as docker, for the Kernel line and everything that follows the runtime."""
    harness.server = DOCKER_SERVER
    agent_configs: list = []
    monkeypatch.setattr(cli, "_make_agent", lambda config, bundle, server=None: (agent_configs.append(config) or harness.agent))
    result = chat(input_text="/status\n/exit\n")
    assert result.exit_code == 0, result.output
    assert "Kernel:     docker (hailer-kernel " in result.output and "no network; data read-only" in result.output
    assert LOCAL_LINE not in result.output
    assert agent_configs[0].kernel.runtime == "docker", "the agent (prompt, tools) gets the effective runtime"
    assert harness.agent_servers == [], "chat-only: the tools discover the server on every call (they follow a restart)"


def test_chat_shows_the_network_setting_of_the_kernel_in_use(harness):
    harness.config = docker_config(harness.config)  # hailer.toml: no network
    harness.server = replace(DOCKER_SERVER, network_access=True)
    result = chat()
    assert "network on: the internet and this machine" in result.output


def test_notebook_reuses_the_server_it_started_earlier_even_without_a_session(harness, nb):
    harness.server = replace(SERVER, source="kernel", token=TOKEN)  # found through .hailer/kernel.json
    harness.client = FakeClient(session=None)
    result = notebook_cmd()
    assert result.exit_code == 0, result.output
    assert nb.spawned == [], "the recorded server hosts every notebook in the folder"
    assert f"Using the running marimo at {SERVER.url}." in result.output
    assert harness.opened == [signed(harness.url)]
    assert "Stopped marimo." not in result.output, "a reused server is left running"
    assert harness.agent_servers == [harness.server], "the chat is pinned to it"


def test_notebook_attached_to_a_docker_kernel_runs_as_docker(harness, nb, monkeypatch):
    harness.server = DOCKER_SERVER
    agent_configs: list = []
    monkeypatch.setattr(cli, "_make_agent", lambda config, bundle, server=None: (agent_configs.append(config) or harness.agent))
    result = notebook_cmd()
    assert result.exit_code == 0, result.output
    assert nb.spawned == []
    assert "Kernel:     docker (hailer-kernel " in result.output
    assert agent_configs[0].kernel.runtime == "docker" and agent_configs[0].marimo_url == DOCKER_SERVER.url


def _record_docker_kernel(config: HailerConfig, **settings) -> None:
    from hailer.kernel import KernelState, write_kernel_state

    mounts = ((str(config.notebooks_root), "/work/notebooks"), (str(config.data_dir), "/work/data"))
    write_kernel_state(
        config.workspace,
        KernelState(runtime="docker", url=DOCKER_SERVER.url, port=2731, token=TOKEN, image=IMAGE, mounts=mounts, memory="4g", cpus=2.0, **settings),
    )


def test_notebook_refuses_a_kernel_started_with_other_settings(harness, nb, docker):
    """Reusing it would silently ignore hailer.toml (here: the network setting)."""
    harness.config = docker_config(harness.config)
    harness.server = DOCKER_SERVER
    _record_docker_kernel(harness.config, network_access=True)
    result = notebook_cmd()
    assert result.exit_code == 1
    assert "The running kernel was started with other settings (network on, hailer.toml: off)." in result.output
    assert "uvx hailer kernel stop" in result.output and "Traceback" not in result.output
    assert harness.agent.started == [] and not docker.commands("run")
    _record_docker_kernel(harness.config)  # the same settings: reused
    result = notebook_cmd()
    assert result.exit_code == 0, result.output
    assert f"Using the running marimo at {DOCKER_SERVER.url}." in result.output


def test_chat_warns_about_a_kernel_started_with_other_settings(harness):
    harness.server = DOCKER_SERVER
    other = harness.config.workspace / "old-notebooks"
    from hailer.kernel import KernelState, write_kernel_state

    write_kernel_state(
        harness.config.workspace,
        KernelState(runtime="docker", url=DOCKER_SERVER.url, port=2731, token=TOKEN, mounts=((str(other), "/work/notebooks"),)),
    )
    result = chat()
    assert result.exit_code == 0, result.output
    assert f"warn  kernel: the running kernel was started with other settings (notebooks folder {other}" in result.output
    assert "uvx hailer kernel stop, then uvx hailer notebook" in result.output


def test_a_notebook_the_kernel_cannot_see_gets_the_home_page_and_a_note(harness, nb):
    """No traceback when the running docker kernel mounts other folders than the notebook's."""
    from hailer.kernel import PathMap

    outside = PathMap(((harness.config.workspace / "old-notebooks", PurePosixPath("/work/notebooks")),))
    harness.server = replace(DOCKER_SERVER, paths=outside)
    harness.client = FakeClient(session=None)
    result = notebook_cmd()
    assert result.exit_code == 0, result.output
    assert "The running kernel cannot see" in result.output and "marimo's home page" in result.output
    assert harness.opened == [f"{DOCKER_SERVER.url}/?access_token={TOKEN}"], "the home page, signed in"
    assert "Traceback" not in result.output


def test_notebook_hands_its_chat_the_server_it_started(harness, nb):
    """The chat keeps the server (URL, token, paths) in memory: a kernel.json rewritten or removed
    underneath it (another terminal) cannot take them away."""
    from hailer.kernel import kernel_state_path

    seen: list = []

    def on_turn():
        kernel_state_path(harness.config.workspace).unlink()  # e.g. `uvx hailer kernel stop` + a new start elsewhere
        seen.append(harness.agent_servers[0])

    harness.agent.on_turn = on_turn
    result = notebook_cmd(["--port", "2718"], input_text="hi\n/status\n/exit\n")
    assert result.exit_code == 0, result.output
    pinned = seen[0]
    assert pinned is not None and pinned.url == "http://127.0.0.1:2718" and pinned.token == TOKEN, "tools get it in memory"
    assert re.search(r"^Marimo\s+http://127\.0\.0\.1:2718 \(session", result.output, re.M), "/status still knows the server"


def test_the_prompt_never_carries_the_servers_token(harness, nb, monkeypatch):
    """Everything the agent gets from `hailer notebook` (its config becomes the system prompt). The
    check is sensitive: a pinned URL with the token in it would show up in the prompt."""
    from hailer.agent import system_prompt

    captured: list = []
    monkeypatch.setattr(cli, "_make_agent", lambda config, bundle, server=None: (captured.append((config, server)) or harness.agent))
    result = notebook_cmd()
    assert result.exit_code == 0, result.output
    config, server = captured[0]
    assert server.token == TOKEN, "the tools hold the token (for the server, never for the model)"
    text = system_prompt(config, ContextBundle())
    assert "http://127.0.0.1:2718" in text and TOKEN not in text and "access_token" not in text
    leaky = system_prompt(replace(config, marimo_url=f"{server.url}/?access_token={TOKEN}"), ContextBundle())
    assert TOKEN in leaky, "the probe would catch a tokenised URL"


def test_cli_clients_carry_the_servers_token_and_paths(tmp_path):
    from hailer.kernel import docker_paths

    config = make_config(tmp_path)
    paths = docker_paths(config)
    server = MarimoServer(url="http://127.0.0.1:2731", source="kernel", token=TOKEN, runtime="docker", paths=paths)
    client = cli._make_client(server, replace(config, marimo_token="user-token"))
    assert client.token == TOKEN and client.paths is paths and client.token_in_links, "the CLI prints its hints for the user"
    assert cli._notebook_url(server, config) == (
        f"http://127.0.0.1:2731/?file=/work/notebooks/analysis.py&view-as=present&access_token={TOKEN}"
    )
    plain = MarimoServer(url="http://127.0.0.1:2718")  # a --no-token server of the user's own
    assert cli._make_client(plain, replace(config, marimo_token="user-token")).token == "user-token"
    assert cli._notebook_url(plain, replace(config, marimo_token="user-token")).endswith("&access_token=user-token")
    assert "access_token" not in cli._notebook_url(plain, config)


def test_exec_in_another_terminal_attaches_through_kernel_json(tmp_path, monkeypatch):
    """`uvx hailer exec` finds the server `hailer notebook --keep-marimo` left running: the record in
    .hailer/kernel.json gives the URL and the token (marimo's registry never lists a server with one)."""
    from fake_marimo import serving
    from hailer.kernel import KernelState, write_kernel_state

    for name in ("HAILER_NOTEBOOK", "HAILER_MARIMO_URL", "HAILER_MARIMO_TOKEN", "HAILER_KERNEL", "HAILER_WORKSPACE", "HAILER_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    ws = tmp_path / "ws"
    notebook = ws / "notebooks" / "analysis.py"
    notebook.parent.mkdir(parents=True)
    notebook.write_text(NOTEBOOK_SOURCE, encoding="utf-8")
    (ws / "hailer.toml").write_text('[hailer]\nnotebook = "notebooks/analysis.py"\n', encoding="utf-8")
    with serving(token=TOKEN) as srv:
        srv.sessions = {"s1": {"filename": "notebooks/analysis.py", "path": str(notebook.resolve())}}
        write_kernel_state(ws, KernelState(runtime="local", url=srv.url, port=srv.server_address[1], token=TOKEN, pid=os.getpid()))
        result = runner.invoke(cli.app, ["--workspace", str(ws), "exec", "-c", "print('x')"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "hello world" in result.output and "42" in result.output
    execute = [r for r in srv.requests if r["path"] == "/api/kernel/execute"][-1]
    assert execute["headers"]["Authorization"] == f"Bearer {TOKEN}" and execute["headers"]["Marimo-Session-Id"] == "s1"


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
    from hailer.kernel import kernel_state_path
    from hailer.kernel_docker import docker_names

    names = docker_names(harness.config.workspace)
    agent_configs: list = []
    recorded: list = []
    monkeypatch.setattr(cli, "_make_agent", lambda config, bundle, server=None: (agent_configs.append((config, server)) or harness.agent))
    harness.agent.on_turn = lambda: recorded.append(json.loads(kernel_state_path(harness.config.workspace).read_text()))
    harness.client = FakeClient(session=None)  # nobody has the tab open yet
    result = notebook_cmd(["--kernel", "docker", "--port", "2731"], input_text="hi\n/exit\n")
    assert result.exit_code == 0, result.output
    assert nb.spawned == [] and nb.foreground == [], "no local server"
    assert docker.commands("run")[0][:8] == ["run", "-d", "--pull", "never", "--name", names.kernel, "--network", names.network]
    assert docker.commands("create")[0][4:7] == [names.forwarder, "-p", "127.0.0.1:2731:2718"]
    assert f"Marimo is running at http://127.0.0.1:2731  (log: docker logs {names.kernel})" in result.output
    assert harness.opened == [f"http://127.0.0.1:2731/?file=/work/notebooks/analysis.py&view-as=present&access_token={TOKEN}"]
    assert f"Kernel:     docker (hailer-kernel {__version__}; no network; data read-only)" in result.output
    config, server = agent_configs[0]
    assert config.kernel.runtime == "docker" and config.marimo_url == "http://127.0.0.1:2731"
    assert server.runtime == "docker" and server.token == TOKEN and server.paths is not None, "the chat is pinned to the container"
    assert recorded and recorded[0]["runtime"] == "docker" and recorded[0]["containers"] == [names.kernel, names.forwarder]
    assert len(recorded[0]["container_ids"]) == 2
    assert docker.containers == {} and docker.networks == {}, "removed when the chat ends"
    assert "Stopped marimo." in result.output and not kernel_state_path(harness.config.workspace).exists()


def test_notebook_keep_marimo_in_docker_mode_says_how_to_stop_it(harness, nb, docker):
    from hailer.kernel_docker import docker_names

    result = notebook_cmd(["--kernel", "docker", "--keep-marimo"])
    assert result.exit_code == 0, result.output
    assert "stop it with uvx hailer kernel stop" in result.output
    assert f"log: docker logs {docker_names(harness.config.workspace).kernel}" in result.output
    assert not docker.commands("rm"), "left running"


def test_notebook_downloads_a_missing_image_before_the_spinner_without_warning_first(harness, nb, docker, monkeypatch):
    """docker's progress bars and a spinner line would fight over the terminal; and no "image
    missing, run kernel pull" warning right before the pull that fixes it."""
    events: list[str] = []

    class RecordingConsole(Console):
        def status(self, *args, **kwargs):
            events.append("spinner")
            return super().status(*args, **kwargs)

    monkeypatch.setattr(cli, "console_factory", lambda: RecordingConsole(force_terminal=False, width=120, highlight=False, soft_wrap=True, color_system=None))
    docker.image_version = None
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


def test_notebook_docker_refuses_to_orphan_a_live_local_server(harness, nb, docker, monkeypatch):
    from hailer import kernel as kernel_mod
    from hailer.kernel import KernelState, read_kernel_state, write_kernel_state

    url = "http://127.0.0.1:2790"
    write_kernel_state(harness.config.workspace, KernelState(runtime="local", url=url, port=2790, token=TOKEN, pid=1))
    monkeypatch.setattr(kernel_mod, "answers_with_token", lambda u, t, timeout=1.0: (u, t) == (url, TOKEN))
    result = notebook_cmd(["--kernel", "docker"])
    assert result.exit_code == 1
    assert f"A local marimo server Hailer started for this workspace is still running at {url}." in result.output
    assert "uvx hailer kernel stop" in result.output
    assert not docker.commands("run") and read_kernel_state(harness.config.workspace).runtime == "local"


def test_local_foreground_refuses_to_start_over_a_live_docker_kernel(harness, nb, monkeypatch):
    """--foreground skips the reuse step, so the runtime's own guard must refuse."""
    from hailer import kernel as kernel_mod
    from hailer.kernel import KernelState, read_kernel_state, write_kernel_state

    write_kernel_state(harness.config.workspace, KernelState(runtime="docker", url=DOCKER_SERVER.url, port=2731, token=TOKEN))
    monkeypatch.setattr(kernel_mod, "answers_with_token", lambda u, t, timeout=1.0: (u, t) == (DOCKER_SERVER.url, TOKEN))
    result = notebook_cmd(["--kernel", "local", "--foreground"])
    assert result.exit_code == 1
    assert f"A docker kernel Hailer started for this workspace is already running at {DOCKER_SERVER.url}." in result.output
    assert nb.foreground == [] and read_kernel_state(harness.config.workspace).runtime == "docker", "its record is kept"


def test_notebook_foreground_in_docker_mode_follows_the_log_and_removes_the_containers(harness, nb, docker):
    from hailer.kernel import kernel_state_path
    from hailer.kernel_docker import docker_names

    names = docker_names(harness.config.workspace)

    def ctrl_c(args):
        if args[:2] == ["logs", "-f"]:
            raise KeyboardInterrupt

    docker.stream_hook = ctrl_c
    result = notebook_cmd(["--kernel", "docker", "--foreground", "--no-browser", "--port", "2731"])
    assert result.exit_code == 0, result.output
    assert "Starting marimo in Docker on http://127.0.0.1:2731 (Ctrl+C stops it) ..." in result.output
    assert f"Kernel:     docker (hailer-kernel {__version__}; no network; data read-only)" in result.output
    assert f"Open http://127.0.0.1:2731/?access_token={TOKEN} in your browser." in result.output
    kernel_id = docker.streams[0][-1]
    assert docker.streams == [["logs", "-f", "--tail", "0", kernel_id]] and ["rm", "-f", kernel_id] in docker.calls, "by id"
    assert docker.containers == {} and docker.networks == {}
    assert not kernel_state_path(harness.config.workspace).exists() and nb.foreground == []
    assert names.kernel not in docker.names(), "removed after Ctrl+C"


def test_notebook_foreground_says_why_the_kernel_went_away(harness, nb, docker):
    """Removed from another terminal (uvx hailer kernel stop): say so instead of exiting 137 silently."""

    def removed_elsewhere(args):
        if args[:2] == ["logs", "-f"]:
            docker.containers.clear()

    docker.stream_hook = removed_elsewhere
    result = notebook_cmd(["--kernel", "docker", "--foreground", "--no-browser"])
    assert "The kernel container was removed from outside this terminal (uvx hailer kernel stop, or docker rm)." in result.output


def test_doctor_in_docker_mode_has_docker_image_and_data_rows_and_probes_the_helpers(harness, docker):
    harness.config = docker_config(harness.config)
    harness.client = FakeClient(exec_result=ExecResult(True, stdout="Help on module ... get_context ..."))
    result = runner.invoke(cli.app, ["doctor"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    for name, text in (("kernel", "docker (hailer-kernel"), ("docker", "Docker 29.4.3 (Linux engine)"), ("image", IMAGE), ("data", "/work/data")):
        assert any(f" {name} " in line and "OK" in line for line in lines), name
        assert text in result.output, text
    assert "ok    notebook helpers: hailer.periods imports in the kernel" in result.output
    assert any("import hailer.periods" in code for code in harness.client.codes)


def test_doctor_warns_when_the_kernel_cannot_import_the_helpers(harness):
    class Client(FakeClient):
        def execute(self, code, **kwargs):
            if "hailer.periods" in code:
                return ExecResult(False, stderr="Traceback ...\nModuleNotFoundError: No module named 'hailer'\n")
            return super().execute(code, **kwargs)

    harness.client = Client(exec_result=ExecResult(True, stdout="get_context"))
    result = runner.invoke(cli.app, ["doctor"], catch_exceptions=False)
    assert "warn  notebook helpers: hailer.periods did not import in the kernel (ModuleNotFoundError: No module named 'hailer')" in result.output


def test_kernel_pull(harness, docker):
    result = runner.invoke(cli.app, ["kernel", "pull"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert docker.streams == [["pull", IMAGE]], "pulled even though a copy is here: it may be stale"
    assert f"Downloading {IMAGE} ..." in result.output and f"{IMAGE} is ready (Hailer {__version__})." in result.output
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
    assert f"The kernel image for Hailer {__version__} is not published (or not visible to you): {IMAGE}." in result.output
    assert "Build it on this machine with: uvx hailer kernel build" in result.output


def test_kernel_build(harness, docker):
    from hailer.kernel_image import build_args

    result = runner.invoke(cli.app, ["kernel", "build"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    build = docker.streams[0]
    assert build[:3] == ["build", "--tag", IMAGE]
    assert build[3:-1] == [part for name, value in build_args().items() for part in ("--build-arg", f"{name}={value}")]
    assert f"Building {IMAGE} for Hailer {__version__} (marimo 0.24.2, Polars " in result.output and f"Built {IMAGE}." in result.output
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


def test_kernel_stop(harness, docker):
    from hailer.kernel import KernelState, read_kernel_state, write_kernel_state
    from hailer.kernel_docker import docker_names

    result = runner.invoke(cli.app, ["kernel", "stop"], catch_exceptions=False)
    assert result.exit_code == 0 and "No kernel is running for this workspace." in result.output
    names = docker_names(harness.config.workspace)
    workspace = str(harness.config.workspace)
    kernel_c = docker.add_container(names.kernel, workspace=workspace)
    forwarder_c = docker.add_container(names.forwarder, workspace=workspace, role="forwarder")
    network = docker.add_network(names.network, workspace=workspace)
    write_kernel_state(
        harness.config.workspace,
        KernelState(
            runtime="docker", url="http://127.0.0.1:2731", port=2731, token=TOKEN, containers=(names.kernel, names.forwarder),
            container_ids=(kernel_c.id, forwarder_c.id), network=names.network, network_id=network.id,
        ),
    )  # fmt: skip
    result = runner.invoke(cli.app, ["kernel", "stop"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert f"Stopped the docker kernel at http://127.0.0.1:2731 (removed {names.kernel}, {names.forwarder}, {names.network})." in result.output
    assert read_kernel_state(harness.config.workspace) is None and docker.containers == {}
    help_text = runner.invoke(cli.app, ["kernel", "--help"], catch_exceptions=False).output
    assert all(command in help_text for command in ("pull", "build", "stop"))


def test_kernel_stop_with_docker_not_running_says_so(harness, docker):
    docker.engine = None
    result = runner.invoke(cli.app, ["kernel", "stop"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "Docker is not running, so containers an earlier docker kernel may have left were not checked" in result.output
    assert "No kernel is running" not in result.output


def test_kernel_stop_that_cannot_remove_everything_exits_1(harness, docker):
    from hailer.kernel_docker import docker_names

    network = docker.add_network(docker_names(harness.config.workspace).network, workspace=str(harness.config.workspace))
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
