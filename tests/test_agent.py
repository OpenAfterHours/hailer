"""Offline tests for hailer.agent using a fake Codex runtime."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from openai_codex.errors import TransportClosedError

from hailer import agent as agent_mod
from hailer.agent import (
    HailerAgent,
    build_child_env,
    build_config_overrides,
    map_exception,
    system_prompt,
    toml_value,
)
from hailer.errors import AgentError, ConfigError, CredentialsError, ProviderError
from hailer.models import (
    AgentEvent,
    ContextBundle,
    HailerConfig,
    ModelConfig,
    ProviderConfig,
    SkillInfo,
    WebConfig,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def make_config(tmp_path: Path, **overrides: Any) -> HailerConfig:
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    base: dict[str, Any] = dict(
        workspace=ws,
        notebook=ws / "notebooks" / "analysis.py",
        data_dir=ws / "data",
        context_dir=ws / ".config" / "hailer" / "context",
        skills_dir=ws / ".config" / "hailer" / "skills",
        prompts_dir=ws / ".config" / "hailer" / "prompts",
        model=ModelConfig(name="internal-analyst", provider="internal", reasoning_effort="medium"),
        providers={
            "internal": ProviderConfig(
                id="internal",
                name="Internal",
                base_url="https://llm.example.internal/v1",
                env_key="INTERNAL_MODEL_API_KEY",
                requires_openai_auth=False,
                http_headers={"X-Team": "risk"},
            )
        },
        config_path=ws / "hailer.toml",
    )
    base.update(overrides)
    return HailerConfig(**base)


CHAT_PROVIDER = ProviderConfig(
    id="internal",
    name="Internal",
    base_url="https://llm.example.internal/v1",
    wire_api="chat",
    env_key="INTERNAL_MODEL_API_KEY",
    http_headers={"X-Team": "risk"},
)

USER_CODEX_CONFIG = {
    "model": "gpt-6-astra",
    "plugins": {"browser@openai-bundled": {"enabled": True}, "sites@openai-bundled": {"enabled": True}},
    "mcp_servers": {"node_repl": {"command": "node_repl.exe"}},
}


class FakeHandle:
    def __init__(self, events: list[Any]) -> None:
        self.events = events
        self.interrupted = False

    def stream(self):
        # Like the real runtime: after an interrupt, remaining work is skipped but the
        # turn still ends with a turn/completed notification (status "interrupted").
        for ev in self.events:
            if self.interrupted and ev.method != "turn/completed":
                continue
            yield ev

    def interrupt(self) -> None:
        self.interrupted = True


class FakeThread:
    def __init__(self, thread_id: str, events: list[Any]) -> None:
        self.id = thread_id
        self.events = events
        self.turn_calls: list[tuple[Any, dict[str, Any]]] = []
        self.last_handle: FakeHandle | None = None

    def turn(self, input: Any, **kwargs: Any) -> FakeHandle:
        self.turn_calls.append((input, kwargs))
        self.last_handle = FakeHandle(self.events)
        return self.last_handle


class FakeCodex:
    def __init__(self, cfg: Any, events: list[Any] | None = None, *, resume_fails: bool = False) -> None:
        self.cfg = cfg
        self.events = events or []
        self.resume_fails = resume_fails
        self.start_calls: list[dict[str, Any]] = []
        self.resume_calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False
        self._n = 0

    def thread_start(self, **kwargs: Any) -> FakeThread:
        self.start_calls.append(kwargs)
        self._n += 1
        return FakeThread(f"thread-{self._n}", self.events)

    def thread_resume(self, thread_id: str, **kwargs: Any) -> FakeThread:
        self.resume_calls.append((thread_id, kwargs))
        if self.resume_fails:
            raise RuntimeError("thread not found")
        return FakeThread(thread_id, self.events)

    def close(self) -> None:
        self.closed = True


def N(method: str, **payload: Any) -> SimpleNamespace:
    """A stand-in for openai_codex.models.Notification (.method / .payload)."""
    return SimpleNamespace(method=method, payload=SimpleNamespace(**payload))


def turn_completed(status: str = "completed", error_message: str | None = None, turn_id: str = "turn-1"):
    error = SimpleNamespace(message=error_message) if error_message else None
    return N("turn/completed", turn=SimpleNamespace(id=turn_id, status=status, error=error, duration_ms=321))


def scripted_events() -> list[Any]:
    cmd_item = SimpleNamespace(type="commandExecution", id="c1", command="dir data", cwd="C:\\ws")
    tool_item = SimpleNamespace(
        type="mcpToolCall", id="m1", server="hailer", tool="marimo_execute", arguments={"code": "print(1)"}
    )
    final_msg = SimpleNamespace(type="agentMessage", id="a1", text="RWA rose 5.6%.", phase="final_answer")
    commentary = SimpleNamespace(type="agentMessage", id="a0", text="Looking...", phase="commentary")
    return [
        N("turn/started", turn=SimpleNamespace(id="turn-1")),
        N("item/started", item=cmd_item, thread_id="t", turn_id="turn-1"),
        N("item/commandExecution/outputDelta", delta="25-01 pra101.parquet\n", item_id="c1"),
        N("item/completed", item=cmd_item, turn_id="turn-1"),
        N("item/started", item=tool_item, turn_id="turn-1"),
        N("item/completed", item=tool_item, turn_id="turn-1"),
        N("item/reasoning/summaryTextDelta", delta="thinking", item_id="r1"),
        N("item/completed", item=commentary, turn_id="turn-1"),
        N("item/agentMessage/delta", delta="RWA rose ", item_id="a1"),
        N("item/agentMessage/delta", delta="5.6%.", item_id="a1"),
        N("item/completed", item=final_msg, turn_id="turn-1"),
        N(
            "thread/tokenUsage/updated",
            token_usage=SimpleNamespace(
                last=SimpleNamespace(input_tokens=1200, output_tokens=80),
                total=SimpleNamespace(input_tokens=1200, output_tokens=80),
            ),
            turn_id="turn-1",
        ),
        turn_completed(),
    ]


@pytest.fixture(autouse=True)
def no_real_keyring(monkeypatch):
    # Never touch the OS credential store from tests.
    monkeypatch.setattr(agent_mod._secrets, "_keyring_get", lambda username: None)


# --------------------------------------------------------------------------- #
# TOML rendering
# --------------------------------------------------------------------------- #


def test_toml_value_rendering():
    assert toml_value(True) == "true"
    assert toml_value(600) == "600"
    assert toml_value("plain") == '"plain"'
    assert toml_value("C:\\Users\\x\\.venv") == '"C:\\\\Users\\\\x\\\\.venv"'
    assert toml_value('say "hi"') == '"say \\"hi\\""'
    assert toml_value(["-m", "hailer.mcp_server"]) == '["-m", "hailer.mcp_server"]'
    assert toml_value({"HAILER_WORKSPACE": "C:\\ws", "X-Team": "risk"}) == '{HAILER_WORKSPACE = "C:\\\\ws", X-Team = "risk"}'
    assert toml_value({"127.0.0.1": "allow"}) == '{"127.0.0.1" = "allow"}'


# --------------------------------------------------------------------------- #
# Overrides
# --------------------------------------------------------------------------- #


def test_overrides_for_custom_provider(tmp_path):
    cfg = make_config(tmp_path)
    ov = build_config_overrides(cfg, USER_CODEX_CONFIG)
    assert ov[0] == 'model="internal-analyst"'
    assert ov[1] == 'model_provider="internal"'
    assert 'model_providers.internal.base_url="https://llm.example.internal/v1"' in ov
    assert 'model_providers.internal.wire_api="responses"' in ov
    assert 'model_providers.internal.env_key="INTERNAL_MODEL_API_KEY"' in ov
    assert "model_providers.internal.requires_openai_auth=false" in ov
    assert 'model_providers.internal.name="Internal"' in ov
    assert 'model_providers.internal.http_headers={X-Team = "risk"}' in ov
    assert 'model_reasoning_effort="medium"' in ov
    # trimming set
    for expected in (
        'web_search="disabled"',
        "features.web_search_request=false",
        "features.multi_agent=false",
        "features.multi_agent_v2=false",
        "features.plugins=false",
        "features.apps=false",
        "features.codex_apps=false",
        "features.connectors=false",
        "features.image_generation=false",
    ):
        assert expected in ov
    # user-config-derived disabling
    assert 'plugins."browser@openai-bundled".enabled=false' in ov
    assert 'plugins."sites@openai-bundled".enabled=false' in ov
    assert "mcp_servers.node_repl.enabled=false" in ov
    # Hailer MCP server, with Windows paths escaped
    exe = toml_value(sys.executable)
    assert f"mcp_servers.hailer.command={exe}" in ov
    assert 'mcp_servers.hailer.args=["-m", "hailer.mcp_server"]' in ov
    env_override = next(o for o in ov if o.startswith("mcp_servers.hailer.env="))
    assert "HAILER_WORKSPACE = " in env_override and "HAILER_CONFIG = " in env_override
    assert "HAILER_LOG_LEVEL = " in env_override
    assert "HAILER_MARIMO_TOKEN" not in env_override  # secrets never go on the command line
    assert toml_value(str(cfg.workspace)) in env_override
    assert "mcp_servers.hailer.tool_timeout_sec=600" in ov
    assert "mcp_servers.hailer.startup_timeout_sec=60" in ov
    assert 'mcp_servers.hailer.default_tools_approval_mode="auto"' in ov
    # no network overrides by default, no sandbox/approval overrides ever
    assert not any(o.startswith(("network.", "features.network_proxy", "sandbox_")) for o in ov)
    assert not any(o.startswith(("sandbox_mode", "approval_policy")) for o in ov)
    # never disable our own server, never emit plugin-provided mcp servers we did not see
    assert "mcp_servers.hailer.enabled=false" not in ov
    assert not any("cua_repl" in o for o in ov)


def test_overrides_for_chat_provider_point_codex_at_the_bridge(tmp_path):
    cfg = make_config(tmp_path, providers={"internal": CHAT_PROVIDER})
    ov = build_config_overrides(cfg, USER_CODEX_CONFIG, {"internal": "http://127.0.0.1:4242/internal"})
    assert 'model_providers.internal.wire_api="responses"' in ov
    assert 'model_providers.internal.base_url="http://127.0.0.1:4242/internal"' in ov
    # credentials and headers stay on the provider so Codex attaches them and the bridge forwards them
    assert 'model_providers.internal.env_key="INTERNAL_MODEL_API_KEY"' in ov
    assert 'model_providers.internal.http_headers={X-Team = "risk"}' in ov
    assert "llm.example.internal" not in " ".join(ov)


def test_agent_starts_and_stops_the_bridge_for_chat_providers(tmp_path, monkeypatch):
    monkeypatch.setenv("INTERNAL_MODEL_API_KEY", "k")
    ag, created = make_agent(tmp_path, providers={"internal": CHAT_PROVIDER})
    assert ag.bridge is None
    ag.start()
    assert ag.bridge is not None and ag.bridge.port
    bridge_url = ag.bridge.urls["internal"]
    (fake,) = created
    assert f'model_providers.internal.base_url="{bridge_url}"' in fake.cfg.config_overrides
    assert 'model_providers.internal.wire_api="responses"' in fake.cfg.config_overrides
    assert 'wire_api="chat"' not in " ".join(fake.cfg.config_overrides)
    assert ag.overrides == fake.cfg.config_overrides  # derived on demand, never a stale copy
    ag.close()
    assert ag.bridge is None


def test_bridge_start_failure_is_a_provider_error(tmp_path, monkeypatch):
    def refuse(self):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(agent_mod.ChatBridge, "start", refuse)
    ag, _ = make_agent(tmp_path, providers={"internal": CHAT_PROVIDER})
    with pytest.raises(ProviderError) as info:
        ag.start()
    assert "Chat Completions bridge" in str(info.value)
    assert "Permission denied" in info.value.hint and "internal" in info.value.hint
    assert ag.bridge is None


def test_child_env_exempts_the_bridge_from_the_proxy(tmp_path):
    cfg = make_config(tmp_path, providers={"internal": CHAT_PROVIDER})
    env = build_child_env(cfg, "k", environ={"HTTPS_PROXY": "http://proxy.corp:3128", "NO_PROXY": ".corp, localhost"})
    assert env["NO_PROXY"] == ".corp,localhost,127.0.0.1"
    assert env["no_proxy"] == ".corp,localhost,127.0.0.1"
    env = build_child_env(cfg, "k", environ={})
    assert env["NO_PROXY"] == "127.0.0.1,localhost"
    # responses-only configs leave the proxy variables alone
    assert "NO_PROXY" not in build_child_env(make_config(tmp_path), "k", environ={"HTTPS_PROXY": "http://p:1"})


def test_map_exception_uses_the_live_provider_after_a_model_switch(tmp_path):
    cfg = make_config(
        tmp_path,
        model=ModelConfig(name="gpt-5.5", provider="openai"),
        providers={"internal": CHAT_PROVIDER},
    )
    mapped = map_exception(RuntimeError("404 Not Found"), cfg, provider="internal")
    assert "https://llm.example.internal/v1/chat/completions" in mapped.hint
    mapped = map_exception(RuntimeError("404 Not Found"), cfg)
    assert "https://api.openai.com/v1/responses" in mapped.hint


def test_map_bridge_unsupported_endpoint_has_its_own_hint(tmp_path):
    cfg = make_config(tmp_path, providers={"internal": CHAT_PROVIDER})
    mapped = map_exception(RuntimeError("404: /responses/compact is not available through the Chat Completions bridge"), cfg)
    assert isinstance(mapped, ProviderError) and "bridge" in str(mapped)
    assert "Only POST /responses is translated" in mapped.hint


def test_map_bridge_dns_failure_is_a_connection_error(tmp_path):
    cfg = make_config(tmp_path, providers={"internal": CHAT_PROVIDER})
    text = "could not reach https://llm.exmaple.internal/v1/chat/completions: dns error ([Errno -2] Name or service not known)"
    mapped = map_exception(RuntimeError(text), cfg)
    assert isinstance(mapped, ProviderError) and "Could not reach the model endpoint" in str(mapped)


def test_agent_has_no_bridge_for_responses_providers(tmp_path, monkeypatch):
    monkeypatch.setenv("INTERNAL_MODEL_API_KEY", "k")
    ag, _ = make_agent(tmp_path)
    ag.start()
    assert ag.bridge is None
    ag.close()


def test_overrides_openai_default_has_no_provider_table(tmp_path):
    cfg = make_config(tmp_path, model=ModelConfig(name="gpt-5.5", provider="openai", reasoning_effort=None), providers={})
    ov = build_config_overrides(cfg, {})
    assert ov[:2] == ('model="gpt-5.5"', 'model_provider="openai"')
    assert not any(o.startswith("model_providers.") for o in ov)
    assert not any(o.startswith("model_reasoning_effort") for o in ov)
    assert not any(o.startswith("plugins.") for o in ov)


def test_overrides_include_all_declared_custom_providers(tmp_path):
    cfg = make_config(tmp_path)
    providers = dict(cfg.providers)
    providers["azure"] = ProviderConfig(id="azure", base_url="https://az.example/v1", env_key="AZURE_KEY", query_params={"api-version": "2025-04-01"})
    cfg = make_config(tmp_path, providers=providers)
    ov = build_config_overrides(cfg, {})
    assert 'model_providers.azure.base_url="https://az.example/v1"' in ov
    assert 'model_providers.azure.query_params={api-version = "2025-04-01"}' in ov


def test_network_overrides_only_when_shell_network_allowed(tmp_path):
    web = WebConfig(allowed_domains=("docs.pola.rs", "**.bankofengland.co.uk"), allow_shell_network=True)
    cfg = make_config(tmp_path, web=web)
    ov = build_config_overrides(cfg, {})
    assert "features.network_proxy.enabled=true" in ov
    assert "network.enabled=true" in ov
    assert 'network.mode="limited"' in ov
    assert "sandbox_workspace_write.network_access=true" in ov
    domains = next(o for o in ov if o.startswith("features.network_proxy.domains="))
    assert '"docs.pola.rs" = "allow"' in domains
    assert '"**.bankofengland.co.uk" = "allow"' in domains
    # "localhost" is a bare TOML key so it is rendered unquoted; "127.0.0.1" must be quoted.
    assert '"127.0.0.1" = "allow"' in domains and 'localhost = "allow"' in domains
    assert next(o for o in ov if o.startswith("network.domains=")).split("=", 1)[1] == domains.split("=", 1)[1]

    cfg2 = make_config(tmp_path, web=WebConfig(allowed_domains=("docs.pola.rs",), allow_shell_network=False))
    assert not any(o.startswith(("network.", "features.network_proxy")) for o in build_config_overrides(cfg2, {}))


def test_overrides_undeclared_provider_is_config_error(tmp_path):
    cfg = make_config(tmp_path, model=ModelConfig(name="x", provider="ghost"))
    with pytest.raises(ConfigError) as err:
        build_config_overrides(cfg, {})
    assert "ghost" in str(err.value) and "hailer.toml" in err.value.hint


# --------------------------------------------------------------------------- #
# Child env
# --------------------------------------------------------------------------- #


def test_child_env_contains_key_under_env_key_only(tmp_path):
    cfg = make_config(tmp_path, codex_home=tmp_path / "codex-home")
    env = build_child_env(cfg, "sekrit", {"AZURE_KEY": "other"})
    assert env["INTERNAL_MODEL_API_KEY"] == "sekrit"
    assert env["AZURE_KEY"] == "other"
    assert env["HAILER_WORKSPACE"] == str(cfg.workspace)
    assert env["HAILER_CONFIG"] == str(cfg.config_path)
    assert env["CODEX_HOME"] == str(tmp_path / "codex-home")
    assert set(env) == {"INTERNAL_MODEL_API_KEY", "AZURE_KEY", "HAILER_WORKSPACE", "HAILER_CONFIG", "CODEX_HOME"}


def test_child_env_without_key(tmp_path):
    cfg = make_config(tmp_path, config_path=None)
    env = build_child_env(cfg, None)
    assert env == {"HAILER_WORKSPACE": str(cfg.workspace)}


# --------------------------------------------------------------------------- #
# System prompt
# --------------------------------------------------------------------------- #


def test_system_prompt_sections(tmp_path):
    cfg = make_config(tmp_path, web=WebConfig(allowed_domains=("docs.pola.rs",)), marimo_url="http://127.0.0.1:2718")
    bundle = ContextBundle(
        context_text="PRA101 is the counterparty credit risk return.",
        skills=[SkillInfo(name="pra101-recon", description="Reconcile PRA101 across months", path=tmp_path)],
    )
    text = system_prompt(cfg, bundle)
    assert "Hailer" in text
    assert "## Project context" in text and "PRA101 is the counterparty" in text
    assert "## Available skills" in text and "pra101-recon — Reconcile PRA101" in text and "load_skill" in text
    assert "## Web access" in text and "docs.pola.rs" in text and "fetch_page" in text
    assert "## Workspace" in text and str(cfg.notebooks_root) in text and "http://127.0.0.1:2718" in text
    assert text == system_prompt(cfg, bundle)  # deterministic


def test_system_prompt_no_web(tmp_path):
    text = system_prompt(make_config(tmp_path), ContextBundle())
    assert "No internet access is available" in text
    assert "## Project context" not in text
    assert "## Available skills" not in text


def test_system_prompt_names_the_notebooks_folder_not_the_notebook(tmp_path):
    ws = tmp_path / "ws"
    cfg = make_config(tmp_path, notebook=ws / "notebooks" / "q2_churn.py", notebooks_dir=ws / "notebooks")
    text = system_prompt(cfg, ContextBundle())
    assert f"- Notebooks folder: {cfg.notebooks_root}" in text
    assert "- Notebook:" not in text
    assert str(cfg.notebook) not in text  # the active notebook changes mid-thread; it must not be baked in


def test_system_prompt_is_stable_across_notebook_switches(tmp_path):
    ws = tmp_path / "ws"
    before = make_config(tmp_path, notebook=ws / "notebooks" / "analysis.py", notebooks_dir=ws / "notebooks")
    after = make_config(tmp_path, notebook=ws / "notebooks" / "q2_churn.py", notebooks_dir=ws / "notebooks")
    bundle = ContextBundle(context_text="PRA101 is the counterparty credit risk return.")
    assert system_prompt(before, bundle) == system_prompt(after, bundle)


def test_system_prompt_folder_falls_back_to_the_notebook_parent(tmp_path):
    cfg = make_config(tmp_path)  # hand-built config: notebooks_dir is None
    assert f"- Notebooks folder: {cfg.notebook.parent}" in system_prompt(cfg, ContextBundle())


def test_packaged_prompt_covers_the_notebook_tools():
    text = agent_mod._base_prompt_text()
    assert text != agent_mod._FALLBACK_SYSTEM_PROMPT, "prompts/system.md must be packaged and non-empty"
    for name in ("notebook_list", "notebook_create", "notebook_open", "notebook_close"):
        assert f"`{name}(" in text, name
    assert "## Notebooks" in text
    assert "[Hailer]" in text and "not the user's words" in text
    assert "active notebook" in text
    # the starter globals are still documented, but scoped to starter-template notebooks
    assert "period_files" in text and "starter template" in text


# --------------------------------------------------------------------------- #
# Agent lifecycle
# --------------------------------------------------------------------------- #


def make_agent(tmp_path, *, events=None, resume_fails=False, env=None, **cfg_overrides):
    cfg = make_config(tmp_path, **cfg_overrides)
    created: list[FakeCodex] = []

    def factory(codex_cfg):
        codex = FakeCodex(codex_cfg, events or scripted_events(), resume_fails=resume_fails)
        created.append(codex)
        return codex

    ag = HailerAgent(
        cfg,
        ContextBundle(),
        codex_factory=factory,
        user_codex_config=USER_CODEX_CONFIG,
        env=env if env is not None else {"INTERNAL_MODEL_API_KEY": "sekrit"},
    )
    return ag, created


def test_start_uses_thread_start_with_expected_kwargs(tmp_path):
    ag, created = make_agent(tmp_path)
    tid = ag.start()
    assert tid == "thread-1" and ag.thread_id == "thread-1"
    codex = created[0]
    assert codex.cfg.cwd == str(ag.config.workspace)
    assert codex.cfg.config_overrides == ag.overrides
    assert codex.cfg.env["INTERNAL_MODEL_API_KEY"] == "sekrit"
    kwargs = codex.start_calls[0]
    assert kwargs["model"] == "internal-analyst" and kwargs["model_provider"] == "internal"
    assert kwargs["sandbox"].value == "workspace-write"
    # auto_review: under deny_all Codex rejects every MCP tool call (verified live)
    assert kwargs["approval_mode"].value == "auto_review"
    assert kwargs["cwd"] == str(ag.config.workspace)
    assert "Hailer" in kwargs["base_instructions"]
    assert ag.key_source == "env"
    assert not hasattr(ag, "_key")


def test_start_resumes_then_falls_back(tmp_path):
    ag, created = make_agent(tmp_path)
    assert ag.start(resume_thread_id="old-thread") == "old-thread"
    assert created[0].resume_calls[0][0] == "old-thread"
    assert created[0].start_calls == []

    ag2, created2 = make_agent(tmp_path, resume_fails=True)
    assert ag2.start(resume_thread_id="old-thread") == "thread-1"
    assert created2[0].resume_calls and created2[0].start_calls


def test_missing_key_for_custom_provider_is_credentials_error(tmp_path):
    ag, created = make_agent(tmp_path, env={})
    assert ag.key_source == "missing"
    with pytest.raises(CredentialsError) as err:
        ag.start()
    assert "INTERNAL_MODEL_API_KEY" in str(err.value)
    assert "hailer login internal" in err.value.hint
    assert created == []  # runtime never launched


def test_missing_key_for_openai_is_not_fatal(tmp_path):
    ag, created = make_agent(
        tmp_path, env={}, model=ModelConfig(name="gpt-5.5", provider="openai"), providers={}
    )
    assert ag.start() == "thread-1"
    assert "OPENAI_API_KEY" not in created[0].cfg.env if created[0].cfg.env else True


def test_run_turn_translates_events_and_collects_summary(tmp_path):
    ag, created = make_agent(tmp_path)
    ag.start()
    events: list[AgentEvent] = []
    summary = ag.run_turn("what's driving the increase?", on_event=events.append)

    kinds = [(e.kind, e.text) for e in events]
    assert ("command", "dir data") in kinds
    assert ("command_output", "25-01 pra101.parquet\n") in kinds
    assert ("tool_call", "hailer.marimo_execute") in kinds
    tool_event = next(e for e in events if e.kind == "tool_call")
    assert "print(1)" in tool_event.detail["arguments"]
    assert ("reasoning", "thinking") in kinds
    assert [e.text for e in events if e.kind == "message_delta"] == ["RWA rose ", "5.6%."]

    assert summary.final_response == "RWA rose 5.6%."
    assert summary.thread_id == "thread-1" and summary.turn_id == "turn-1"
    assert summary.commands == ["dir data"]
    assert summary.tool_calls == ["hailer.marimo_execute"]
    assert (summary.input_tokens, summary.output_tokens) == (1200, 80)
    assert summary.duration_ms == 321 and summary.status == "completed"

    thread = created[0]
    thread_obj = ag._thread
    assert thread_obj.turn_calls[0][0] == "what's driving the increase?"
    assert thread_obj.turn_calls[0][1]["model"] == "internal-analyst"


def test_run_turn_with_skill_input(tmp_path):
    ag, _ = make_agent(tmp_path)
    ag.start()
    skill = SkillInfo(name="pra101-recon", description="d", path=tmp_path / "skills" / "pra101-recon")
    ag.run_turn("reconcile", skill=skill)
    turn_input = ag._thread.turn_calls[0][0]
    assert isinstance(turn_input, list) and len(turn_input) == 2
    assert turn_input[0].name == "pra101-recon" and turn_input[0].path.endswith("SKILL.md")
    assert turn_input[1].text == "reconcile"


NOTICE = "[Hailer] The active notebook is now notebooks/q2_churn.py (reopened, 7 cells). Call notebook_cells before editing."


def test_run_turn_with_preamble_sends_notice_then_message(tmp_path):
    ag, _ = make_agent(tmp_path)
    ag.start()
    ag.run_turn("continue with the churn table", preamble=NOTICE)
    turn_input = ag._thread.turn_calls[0][0]
    assert isinstance(turn_input, list) and len(turn_input) == 2
    assert turn_input[0].text == NOTICE
    assert turn_input[1].text == "continue with the churn table"


def test_run_turn_with_skill_and_preamble_orders_skill_notice_message(tmp_path):
    ag, _ = make_agent(tmp_path)
    ag.start()
    skill = SkillInfo(name="pra101-recon", description="d", path=tmp_path / "skills" / "pra101-recon")
    ag.run_turn("reconcile", skill=skill, preamble=NOTICE)
    turn_input = ag._thread.turn_calls[0][0]
    assert len(turn_input) == 3
    assert turn_input[0].name == "pra101-recon" and turn_input[0].path.endswith("SKILL.md")
    assert turn_input[1].text == NOTICE
    assert turn_input[2].text == "reconcile"


@pytest.mark.parametrize("preamble", [None, "", "   \n"])
def test_run_turn_without_preamble_keeps_plain_string_input(tmp_path, preamble):
    ag, _ = make_agent(tmp_path)
    ag.start()
    ag.run_turn("hello", preamble=preamble)
    assert ag._thread.turn_calls[0][0] == "hello"


def test_final_response_falls_back_to_deltas_when_no_final_item(tmp_path):
    events = [
        N("item/agentMessage/delta", delta="partial ", item_id="a"),
        N("item/agentMessage/delta", delta="answer", item_id="a"),
        turn_completed(),
    ]
    ag, _ = make_agent(tmp_path, events=events)
    ag.start()
    assert ag.run_turn("hi").final_response == "partial answer"


def test_interrupt_calls_handle(tmp_path):
    full = scripted_events()
    events = full[:3] + [turn_completed(status="interrupted")]  # started, command, output, done
    ag, _ = make_agent(tmp_path, events=events)
    ag.start()
    seen: list[str] = []

    def on_event(ev: AgentEvent) -> None:
        seen.append(ev.kind)
        if ev.kind == "command":
            ag.interrupt()

    summary = ag.run_turn("go", on_event=on_event)
    assert ag._thread.last_handle.interrupted is True
    assert "command_output" not in seen  # skipped after the interrupt
    assert summary.status == "interrupted"
    assert ag._handle is None
    ag.interrupt()  # no active turn: must be a no-op


def test_new_thread_and_set_model(tmp_path):
    ag, created = make_agent(tmp_path)
    ag.start()
    assert ag.new_thread() == "thread-2"
    assert ag.set_model("internal-fast") is False  # same provider: no new thread
    ag.run_turn("x")
    assert ag._thread.turn_calls[0][1]["model"] == "internal-fast"
    # provider switch starts a fresh thread automatically
    assert ag.set_model("gpt-5.5", provider="openai") is True  # it started the thread itself
    assert ag.thread_id == "thread-3"
    assert created[0].start_calls[-1]["model_provider"] == "openai"
    with pytest.raises(ConfigError):
        ag.set_model("x", provider="ghost")


def test_close_is_idempotent(tmp_path):
    ag, created = make_agent(tmp_path)
    ag.start()
    ag.close()
    ag.close()
    assert created[0].closed and ag.thread_id == "thread-1" and not ag.started


# --------------------------------------------------------------------------- #
# Error mapping
# --------------------------------------------------------------------------- #


def test_map_transport_config_error(tmp_path):
    cfg = make_config(tmp_path)
    exc = TransportClosedError(
        "Codex process closed stdout. stderr_tail=Error: error loading default config after config error: "
        "invalid transport in `mcp_servers.cua_repl`"
    )
    mapped = map_exception(exc, cfg)
    assert isinstance(mapped, ConfigError) and "config.toml" in mapped.hint


def test_map_transport_other_is_agent_error(tmp_path):
    mapped = map_exception(TransportClosedError("Codex process closed stdout. stderr_tail=panic"), make_config(tmp_path))
    assert isinstance(mapped, AgentError)


def test_map_401_is_credentials_error(tmp_path):
    mapped = map_exception(RuntimeError("HTTP 401 Unauthorized"), make_config(tmp_path))
    assert isinstance(mapped, CredentialsError) and "hailer login internal" in mapped.hint


def test_map_connection_refused_is_provider_error(tmp_path):
    mapped = map_exception(RuntimeError("error sending request: connection refused"), make_config(tmp_path))
    assert isinstance(mapped, ProviderError) and "llm.example.internal" in str(mapped)


def test_map_404_mentions_responses_api(tmp_path):
    mapped = map_exception(RuntimeError("404 Not Found"), make_config(tmp_path))
    assert isinstance(mapped, ProviderError) and "Responses API" in mapped.hint
    assert "https://llm.example.internal/v1/responses" in mapped.hint
    assert 'wire_api = "chat"' in mapped.hint
    assert "{base_url}" not in mapped.hint


def test_map_404_for_chat_provider_names_chat_completions(tmp_path):
    cfg = make_config(tmp_path, providers={"internal": CHAT_PROVIDER})
    mapped = map_exception(RuntimeError("404 Not Found"), cfg)
    assert isinstance(mapped, ProviderError)
    assert "https://llm.example.internal/v1/chat/completions" in mapped.hint
    assert "/responses," not in mapped.hint


def test_failed_turn_is_mapped(tmp_path):
    events = [turn_completed(status="failed", error_message="401 unauthorized")]
    ag, _ = make_agent(tmp_path, events=events)
    ag.start()
    with pytest.raises(CredentialsError):
        ag.run_turn("hi")
    assert ag._handle is None


def test_turn_without_completion_is_agent_error(tmp_path):
    ag, _ = make_agent(tmp_path, events=[N("item/agentMessage/delta", delta="x", item_id="a")])
    ag.start()
    with pytest.raises(AgentError):
        ag.run_turn("hi")


def test_factory_failure_is_mapped(tmp_path):
    cfg = make_config(tmp_path)

    def factory(_cfg):
        raise TransportClosedError("Codex process closed stdout. stderr_tail=error loading default config after config error: bad")

    ag = HailerAgent(cfg, ContextBundle(), codex_factory=factory, user_codex_config={}, env={"INTERNAL_MODEL_API_KEY": "k"})
    with pytest.raises(ConfigError):
        ag.start()


# --------------------------------------------------------------------------- #
# Fix-wave additions: secrets in env only, Ctrl+C, set_model, error classification
# --------------------------------------------------------------------------- #


def test_child_env_carries_marimo_token_but_overrides_never_do(tmp_path):
    cfg = make_config(tmp_path, marimo_token="marimo-secret-token")
    env = build_child_env(cfg, "sekrit")
    assert env["HAILER_MARIMO_TOKEN"] == "marimo-secret-token"
    ov = build_config_overrides(cfg, USER_CODEX_CONFIG)
    assert not any("marimo-secret-token" in o or "HAILER_MARIMO_TOKEN" in o for o in ov)
    assert not any("sekrit" in o for o in ov)


def test_keyboard_interrupt_during_turn_interrupts_and_reraises(tmp_path):
    full = scripted_events()
    events = full[:3] + [turn_completed(status="interrupted")]
    ag, _ = make_agent(tmp_path, events=events)
    ag.start()
    seen: list[str] = []

    def on_event(ev: AgentEvent) -> None:
        seen.append(ev.kind)
        if ev.kind == "command":
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        ag.run_turn("go", on_event=on_event)
    assert ag._thread.last_handle.interrupted is True  # interrupted on the LOCAL handle
    assert ag._handle is None
    assert "command_output" not in seen  # nothing surfaced after the interrupt
    ag.interrupt()  # idempotent when nothing is running

    # the agent is usable again afterwards
    ag._thread.events = scripted_events()
    summary = ag.run_turn("again")
    assert summary.status == "completed" and summary.final_response == "RWA rose 5.6%."


def test_stream_errors_from_the_pump_thread_are_mapped(tmp_path):
    class BoomHandle:
        def stream(self):
            raise RuntimeError("error sending request: connection refused")
            yield  # pragma: no cover

        def interrupt(self) -> None:
            pass

    ag, _ = make_agent(tmp_path)
    ag.start()
    ag._thread.turn = lambda *a, **k: BoomHandle()  # type: ignore[assignment]
    with pytest.raises(ProviderError):
        ag.run_turn("hi")
    assert ag._handle is None


def test_map_unknown_model_is_not_the_proxy_hint(tmp_path):
    exc = RuntimeError("The model `gpt-9` does not exist or you do not have access to it.")
    mapped = map_exception(exc, make_config(tmp_path), model="gpt-9")
    assert isinstance(mapped, ProviderError)
    assert "Unknown model 'gpt-9'" in str(mapped) and "internal" in str(mapped)
    assert "Responses API" not in mapped.hint


def test_map_generic_not_found_is_agent_error(tmp_path):
    mapped = map_exception(RuntimeError("thread not found"), make_config(tmp_path))
    assert isinstance(mapped, AgentError)


def test_map_tool_timeout_is_not_an_endpoint_failure(tmp_path):
    mapped = map_exception(RuntimeError("MCP tool call marimo_execute timed out after 600s"), make_config(tmp_path))
    assert isinstance(mapped, AgentError) and "tool call timed out" in str(mapped).lower()
    mapped = map_exception(RuntimeError("request timed out while connecting"), make_config(tmp_path))
    assert isinstance(mapped, ProviderError) and "Could not reach" in str(mapped)


def test_failed_turn_with_unknown_model_uses_live_model_name(tmp_path):
    events = [turn_completed(status="failed", error_message="model `internal-fast` not found")]
    ag, _ = make_agent(tmp_path, events=events)
    ag.start()
    ag.set_model("internal-fast")
    with pytest.raises(ProviderError) as err:
        ag.run_turn("hi")
    assert "internal-fast" in str(err.value)
