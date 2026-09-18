"""Offline tests for hailer.agent: a scripted chat model for the unit tests, and a strict fake
gateway on loopback for what actually goes over the wire."""

from __future__ import annotations

import _thread
import asyncio
import signal
import threading
import time
from pathlib import Path
from typing import Any

import openai
import pytest
from fake_gateway import FakeGateway
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import ConfigDict, Field

from hailer import agent as agent_mod
from hailer.agent import (
    INTERRUPTED_TOOL_RESULT,
    OPENAI_BASE_URL,
    HailerAgent,
    build_model,
    disable_tracing_unless_opted_in,
    map_exception,
    provider_by_id,
    request_headers,
    system_prompt,
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

KEY_ENV = {"INTERNAL_MODEL_API_KEY": "sk-internal-test-key-000000"}

CHAT_PROVIDER = ProviderConfig(
    id="internal",
    name="Internal",
    base_url="https://llm.example.internal/v1/",
    wire_api="chat",
    env_key="INTERNAL_MODEL_API_KEY",
    http_headers={"X-Team": "risk"},
)


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
        model=ModelConfig(name="internal-analyst", provider="internal", reasoning_effort="medium", summarize_after_tokens=0),
        providers={"internal": CHAT_PROVIDER},
        config_path=ws / "hailer.toml",
    )
    base.update(overrides)
    return HailerConfig(**base)


@pytest.fixture(autouse=True)
def no_real_keyring(monkeypatch):
    """Never touch the developer's OS credential store."""
    monkeypatch.setattr(agent_mod._secrets, "_keyring_get", lambda username: None)


class ScriptedModel(BaseChatModel):
    """A chat model that replays ``script``: messages, exceptions, or callables of the input messages."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    script: list[Any] = Field(default_factory=list)
    seen: list[list[Any]] = Field(default_factory=list)
    bound: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        self.bound[:] = [getattr(t, "name", str(t)) for t in tools]
        return self

    def _next(self, messages: list[Any]) -> Any:
        self.seen.append(list(messages))
        step = self.script.pop(0) if self.script else AIMessage("(script exhausted)")
        return step(messages) if callable(step) and not isinstance(step, AIMessage) else step

    def _generate(self, messages: list[Any], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        step = self._next(messages)
        if isinstance(step, BaseException):
            raise step
        return ChatResult(generations=[ChatGeneration(message=step)])

    async def _agenerate(self, messages: list[Any], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        step = self._next(messages)
        if asyncio.iscoroutine(step):
            step = await step
        if isinstance(step, BaseException):
            raise step
        return ChatResult(generations=[ChatGeneration(message=step)])


def call(name: str, args: dict[str, Any], call_id: str = "call_1", *, text: str = "", tokens: tuple[int, int] = (10, 5)) -> AIMessage:
    return AIMessage(
        content=text,
        tool_calls=[{"name": name, "args": args, "id": call_id}],
        usage_metadata={"input_tokens": tokens[0], "output_tokens": tokens[1], "total_tokens": sum(tokens)},
    )


def say(text: str, tokens: tuple[int, int] | None = (20, 7)) -> AIMessage:
    usage = {"input_tokens": tokens[0], "output_tokens": tokens[1], "total_tokens": sum(tokens)} if tokens else None
    return AIMessage(content=text, usage_metadata=usage)


class Harness:
    """An agent wired to a ScriptedModel and two toy tools; records what the factory was asked for."""

    def __init__(self, tmp_path: Path, script: list[Any], *, env: dict[str, str] | None = None, tool_delay: float = 0.0, **cfg: Any):
        self.config = make_config(tmp_path, **cfg)
        self.model = ScriptedModel(script=list(script))
        self.factory_calls: list[tuple[str, str, str | None]] = []
        self.tool_log: list[str] = []
        self.tool_delay = tool_delay
        harness = self

        @tool
        def echo(text: str) -> str:
            """Return the text."""
            harness.tool_log.append(f"echo:{text}")
            return f"echo: {text}"

        @tool
        async def slow(seconds: float) -> str:
            """Wait, then answer."""
            harness.tool_log.append("slow:start")
            await asyncio.sleep(seconds)
            harness.tool_log.append("slow:done")
            return "slow done"

        self.tools = [echo, slow]
        self.bundle = ContextBundle(context_text="PRA101 is the counterparty credit risk return.")
        self.agent = self.new_agent(env=env)

    def _factory(self, provider: ProviderConfig, model: str, key: str | None) -> Any:
        self.factory_calls.append((provider.id, model, key))
        return self.model

    def new_agent(self, *, env: dict[str, str] | None = None) -> HailerAgent:
        return HailerAgent(
            self.config, self.bundle, model_factory=self._factory, tools=self.tools, env=KEY_ENV if env is None else env
        )


@pytest.fixture
def harness(tmp_path):
    made: list[Harness] = []

    def make(script: list[Any], **kw: Any) -> Harness:
        h = Harness(tmp_path, script, **kw)
        made.append(h)
        return h

    yield make
    for h in made:
        h.agent.close()


def roles(messages: list[Any]) -> list[str]:
    return [type(m).__name__.replace("Message", "").lower() for m in messages]


# --------------------------------------------------------------------------- #
# Providers and the chat model
# --------------------------------------------------------------------------- #


def test_provider_by_id_defaults_openai_and_rejects_undeclared(tmp_path):
    cfg = make_config(tmp_path)
    assert provider_by_id(cfg).id == "internal"
    builtin = provider_by_id(cfg, "openai")
    assert builtin.is_builtin_openai and builtin.env_key == "OPENAI_API_KEY" and builtin.wire_api == "responses"
    with pytest.raises(ConfigError) as err:
        provider_by_id(cfg, "azure")
    assert "[model_providers.azure]" in err.value.hint


def test_request_headers_adds_env_headers_only_when_the_variable_is_set():
    provider = ProviderConfig(
        id="p", base_url="https://x/v1", http_headers={"X-Team": "risk"}, env_http_headers={"X-Client-Id": "CLIENT_ID", "X-Trace": "TRACE_ID"}
    )
    assert request_headers(provider, {"CLIENT_ID": "abc", "TRACE_ID": ""}) == {"X-Team": "risk", "X-Client-Id": "abc"}


def test_build_model_for_a_chat_completions_endpoint():
    provider = ProviderConfig(
        id="internal",
        base_url="https://llm.example.internal/v1/",
        wire_api="chat",
        env_key="K",
        stream_options=False,
        http_headers={"X-Team": "risk"},
        env_http_headers={"X-Client-Id": "CLIENT_ID"},
        query_params={"api-version": "2025-04-01-preview"},
    )
    model = build_model(provider, "corp-gpt", "sk-secret-value-123456", reasoning_effort="low", environ={"CLIENT_ID": "abc"})
    assert model.model_name == "corp-gpt"
    assert model.openai_api_base == "https://llm.example.internal/v1"
    assert model.use_responses_api is False
    assert model.stream_usage is False  # stream_options = false: the field is never sent
    assert model.disable_streaming is False
    assert dict(model.default_headers) == {"X-Team": "risk", "X-Client-Id": "abc"}
    assert dict(model.default_query) == {"api-version": "2025-04-01-preview"}
    assert model.reasoning_effort == "low"
    assert "sk-secret-value-123456" not in repr(model)


def test_build_model_stream_false_and_no_reasoning_effort():
    provider = ProviderConfig(id="internal", base_url="https://x/v1", wire_api="chat", env_key="K", stream=False)
    model = build_model(provider, "corp-gpt", "k", environ={})
    assert model.disable_streaming is True
    assert model.stream_usage is False
    assert model.reasoning_effort is None
    assert model.default_headers is None and model.default_query is None


def test_build_model_for_the_builtin_openai_provider_is_explicit_about_url_and_api():
    model = build_model(ProviderConfig(id="openai", env_key="OPENAI_API_KEY"), "gpt-5.5", "k", environ={"LANGSMITH_GATEWAY": "true"})
    assert model.openai_api_base == OPENAI_BASE_URL  # not the LangSmith gateway
    assert model.use_responses_api is True
    assert model.stream_usage is True


def test_build_model_never_lets_the_model_name_pick_the_wire_api():
    provider = ProviderConfig(id="internal", base_url="https://x/v1", wire_api="chat", env_key="K")
    model = build_model(provider, "gpt-5-codex", "k", environ={})  # langchain-openai would route this name to /responses
    assert model.use_responses_api is False
    assert model._use_responses_api({}) is False


def test_tracing_is_switched_off_unless_opted_in():
    env = {"LANGSMITH_TRACING": "true", "LANGCHAIN_TRACING_V2": "true"}
    assert disable_tracing_unless_opted_in(env) is False
    assert env["LANGSMITH_TRACING"] == "false" and env["LANGCHAIN_TRACING_V2"] == "false"
    opted = {"LANGSMITH_TRACING": "true", "HAILER_TRACING": "1"}
    assert disable_tracing_unless_opted_in(opted) is True
    assert opted["LANGSMITH_TRACING"] == "true"


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
    assert str(cfg.notebook) not in text  # the active notebook changes mid-conversation; it must not be baked in


def test_system_prompt_folder_falls_back_to_the_notebook_parent(tmp_path):
    cfg = make_config(tmp_path)  # hand-built config: notebooks_dir is None
    assert f"- Notebooks folder: {cfg.notebook.parent}" in system_prompt(cfg, ContextBundle())


def test_packaged_prompt_covers_the_tools_the_agent_has():
    text = agent_mod._base_prompt_text()
    assert text != agent_mod._FALLBACK_SYSTEM_PROMPT, "prompts/system.md must be packaged and non-empty"
    from hailer.tools import TOOL_NAMES

    for name in TOOL_NAMES:
        assert f"`{name}(" in text, name
    assert "## Notebooks" in text
    assert "[Hailer]" in text and "not the user's words" in text
    assert "active notebook" in text
    assert "no shell" in text  # the prompt must not promise tools the agent does not have
    # the starter globals are still documented, but scoped to starter-template notebooks
    assert "period_files" in text and "starter template" in text


# --------------------------------------------------------------------------- #
# Error mapping
# --------------------------------------------------------------------------- #


class _Response:
    def __init__(self, status: int) -> None:
        self.status_code = status
        self.request = object()
        self.headers: dict[str, str] = {}


def status_error(status: int, body: Any, message: str = "request failed") -> openai.APIStatusError:
    return openai.APIStatusError(message, response=_Response(status), body=body)  # type: ignore[arg-type]


def test_map_401_is_a_credentials_error_with_the_login_hint(tmp_path):
    err = map_exception(status_error(401, {"message": "Invalid API key"}), make_config(tmp_path))
    assert isinstance(err, CredentialsError)
    assert "provider 'internal'" in str(err)
    assert "hailer login internal" in err.hint and "INTERNAL_MODEL_API_KEY" in err.hint


def test_map_unknown_model_names_the_model_in_use_and_quotes_the_endpoint(tmp_path):
    body = {"detail": [{"type": "string_pattern_mismatch", "loc": ["body", "model"], "msg": "String should match pattern '^corp-'"}]}
    err = map_exception(status_error(422, body), make_config(tmp_path), model="gpt-5.5")
    assert isinstance(err, ProviderError)
    assert str(err) == "Unknown model 'gpt-5.5' for provider 'internal'."
    assert "[model].name" in err.hint and "String should match pattern" in err.hint


def test_map_openai_model_not_found_404(tmp_path):
    body = {"message": "The model `gpt-9` does not exist", "type": "invalid_request_error", "param": None, "code": "model_not_found"}
    err = map_exception(status_error(404, body), make_config(tmp_path), model="gpt-9", provider="openai")
    assert str(err) == "Unknown model 'gpt-9' for provider 'openai'."


def test_map_404_points_at_the_other_wire_api(tmp_path):
    err = map_exception(status_error(404, {"detail": "Not Found"}), make_config(tmp_path))
    assert isinstance(err, ProviderError)
    assert "did not accept the request" in str(err)
    assert "POST https://llm.example.internal/v1/chat/completions" in err.hint
    assert 'wire_api = "responses"' in err.hint and "Not Found" in err.hint
    responses = make_config(tmp_path, providers={"internal": ProviderConfig(id="internal", base_url="https://x/v1", env_key="K")})
    hint = map_exception(status_error(404, {"detail": "Not Found"}), responses).hint
    assert "POST https://x/v1/responses" in hint and 'wire_api = "chat"' in hint


def test_map_rejected_request_names_the_switches_still_on(tmp_path):
    body = {"detail": [{"type": "extra_forbidden", "loc": ["body", "stream_options"], "msg": "Extra inputs are not permitted"}]}
    err = map_exception(status_error(422, body), make_config(tmp_path))
    assert isinstance(err, ProviderError) and "rejected the request (HTTP 422)" in str(err)
    assert "stream = false or stream_options = false" in err.hint
    assert "reasoning_effort" in err.hint and "Extra inputs are not permitted" in err.hint
    off = ProviderConfig(id="internal", base_url="https://x/v1", wire_api="chat", env_key="K", stream=False, stream_options=False)
    hint = map_exception(status_error(400, {"message": "bad field"}), make_config(tmp_path, providers={"internal": off})).hint
    assert "stream = false" not in hint.split("The endpoint said")[0].replace("stream = false, one JSON reply", "")


def test_map_403_429_and_5xx(tmp_path):
    cfg = make_config(tmp_path)
    assert "refused access (HTTP 403)" in str(map_exception(status_error(403, {"message": "forbidden"}), cfg))
    busy = map_exception(status_error(429, {"message": "slow down"}), cfg)
    assert "unavailable (HTTP 429)" in str(busy) and "slow down" in busy.hint
    assert "unavailable (HTTP 503)" in str(map_exception(status_error(503, None, "upstream down"), cfg))


def test_map_context_overflow_suggests_new_or_earlier_summaries(tmp_path):
    body = {"message": "This model's maximum context length is 128000 tokens", "code": "context_length_exceeded"}
    err = map_exception(status_error(400, body), make_config(tmp_path))
    assert "context window" in str(err) and "/new" in err.hint and "summarize_after_tokens" in err.hint


def test_map_connection_and_timeout_errors(tmp_path):
    cfg = make_config(tmp_path)
    unreachable = map_exception(openai.APIConnectionError(request=object()), cfg)  # type: ignore[arg-type]
    assert isinstance(unreachable, ProviderError)
    assert str(unreachable) == "Could not reach the model endpoint at https://llm.example.internal/v1."
    assert "NO_PROXY" in unreachable.hint and "hailer status" in unreachable.hint
    timeout = map_exception(openai.APITimeoutError(request=object()), cfg)  # type: ignore[arg-type]
    assert "did not answer in time" in str(timeout)


def test_map_masks_keys_the_endpoint_echoes_back(tmp_path):
    err = map_exception(status_error(400, {"message": "bad header Authorization: Bearer sk-abcdef0123456789abcdef"}), make_config(tmp_path))
    assert "sk-abcdef0123456789abcdef" not in err.hint


def test_map_generic_failures_and_hailer_errors_pass_through(tmp_path):
    cfg = make_config(tmp_path)
    err = map_exception(RuntimeError("boom"), cfg)
    assert isinstance(err, AgentError) and "RuntimeError: boom" in err.hint
    original = CredentialsError("no key", hint="login")
    assert map_exception(original, cfg) is original


# --------------------------------------------------------------------------- #
# Agent lifecycle
# --------------------------------------------------------------------------- #


def test_start_keeps_an_existing_gitignore_in_hailer(harness):
    h = harness([])
    folder = h.config.workspace / ".hailer"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / ".gitignore").write_text("# mine\n", encoding="utf-8")
    h.agent.start()
    assert (folder / "threads.sqlite").is_file()
    assert (folder / ".gitignore").read_text(encoding="utf-8") == "# mine\n"


def test_undeclared_provider_fails_at_construction(tmp_path):
    cfg = make_config(tmp_path, model=ModelConfig(name="x", provider="azure"))
    with pytest.raises(ConfigError):
        HailerAgent(cfg, ContextBundle(), env=KEY_ENV)


def test_start_creates_the_store_and_a_thread(harness):
    h = harness([])
    assert not h.agent.started
    thread_id = h.agent.start()
    assert thread_id and h.agent.thread_id == thread_id and h.agent.started
    assert (h.config.workspace / ".hailer" / "threads.sqlite").is_file()
    ignore = (h.config.workspace / ".hailer" / ".gitignore").read_text(encoding="utf-8")
    assert "*" in ignore.splitlines(), ".hailer/ ignores itself"
    assert h.factory_calls == [("internal", "internal-analyst", KEY_ENV["INTERNAL_MODEL_API_KEY"])]
    assert h.agent.key_source == "env"


def test_missing_key_is_a_credentials_error_for_every_provider(tmp_path, harness):
    h = harness([], env={})
    assert h.agent.key_source == "missing"
    with pytest.raises(CredentialsError) as err:
        h.agent.start()
    assert "INTERNAL_MODEL_API_KEY is not set" in str(err.value) and "hailer login internal" in err.value.hint
    builtin = harness([], env={}, model=ModelConfig(name="gpt-5.5", provider="openai", summarize_after_tokens=0))
    with pytest.raises(CredentialsError) as err:
        builtin.agent.start()
    assert "OPENAI_API_KEY is not set" in str(err.value) and "hailer login openai" in err.value.hint


def test_key_from_the_credential_store_is_used(harness, monkeypatch):
    monkeypatch.setattr(agent_mod._secrets, "_keyring_get", lambda username: "stored-key" if username == "internal:INTERNAL_MODEL_API_KEY" else None)
    h = harness([], env={})
    h.agent.start()
    assert h.agent.key_source == "keyring"
    assert h.factory_calls[-1] == ("internal", "internal-analyst", "stored-key")


def test_a_conversation_resumes_after_a_restart_and_an_unknown_id_starts_fresh(harness):
    h = harness([say("first answer"), say("second answer")])
    thread_id = h.agent.start()
    h.agent.run_turn("first question")
    h.agent.close()

    again = h.new_agent()
    try:
        assert again.start(resume_thread_id=thread_id) == thread_id
        again.run_turn("second question")
        assert roles(h.model.seen[-1]) == ["system", "human", "ai", "human"]
        fresh = again.start(resume_thread_id="no-such-thread")
        assert fresh != "no-such-thread" and again.thread_id == fresh
    finally:
        again.close()


def test_new_thread_forgets_the_previous_conversation(harness):
    h = harness([say("one"), say("two")])
    first = h.agent.start()
    h.agent.run_turn("hello")
    second = h.agent.new_thread()
    assert second != first
    h.agent.run_turn("hello again")
    assert roles(h.model.seen[-1]) == ["system", "human"]  # nothing carried over
    h.agent.close()
    other = h.new_agent()
    try:
        assert other.start(resume_thread_id=first) != first  # the old thread was deleted, not just left behind
    finally:
        other.close()


def test_start_can_be_told_to_forget_a_conversation(harness):
    """`hailer --new`: the stored conversation is not resumed, so it is deleted rather than left in the file."""
    h = harness([say("one")])
    old = h.agent.start()
    h.agent.run_turn("hello")
    h.agent.close()
    fresh = h.new_agent()
    try:
        assert fresh.start(forget_thread_id=old) != old
        fresh.close()
        assert fresh.start(resume_thread_id=old) != old  # gone
    finally:
        fresh.close()


def test_set_model_rebuilds_the_model_and_rejects_undeclared_providers(harness):
    providers = {"internal": CHAT_PROVIDER, "azure": ProviderConfig(id="azure", base_url="https://az/v1", env_key="AZURE_KEY")}
    h = harness([say("ok")], env={**KEY_ENV, "AZURE_KEY": "az-key"}, providers=providers)
    h.agent.start()
    h.agent.set_model("gpt-x", "azure")
    assert (h.agent.model, h.agent.provider_id) == ("gpt-x", "azure")
    h.agent.run_turn("hello")
    assert h.factory_calls[-1] == ("azure", "gpt-x", "az-key")
    with pytest.raises(ConfigError):
        h.agent.set_model("y", "nope")
    assert h.agent.provider_id == "azure"


def test_close_is_idempotent_and_the_agent_can_start_again(harness):
    h = harness([say("ok")])
    h.agent.close()  # never started
    h.agent.start()
    h.agent.close()
    h.agent.close()
    assert not h.agent.started
    h.agent.start()
    assert h.agent.run_turn("hello").final_response == "ok"


# --------------------------------------------------------------------------- #
# Turns
# --------------------------------------------------------------------------- #


def test_run_turn_runs_tools_reports_events_and_sums_usage(harness):
    h = harness([call("echo", {"text": "hi"}, text="Let me check."), say("All done.")])
    events: list[AgentEvent] = []
    summary = h.agent.run_turn("please echo hi", on_event=events.append)

    assert summary.final_response == "All done."
    assert h.model.bound == ["echo", "slow"]  # the model is offered exactly the agent's tools
    assert summary.tool_calls == ["echo"] and h.tool_log == ["echo:hi"]
    assert (summary.input_tokens, summary.output_tokens) == (30, 12)  # both model calls of the turn
    assert summary.thread_id == h.agent.thread_id and summary.status == "completed" and summary.duration_ms >= 0
    tool_events = [e for e in events if e.kind == "tool_call"]
    assert [(e.text, e.detail["arguments"]) for e in tool_events] == [("echo", "{'text': 'hi'}")]
    assert any(e.kind == "message_delta" for e in events)

    first, second = h.model.seen
    assert isinstance(first[0], SystemMessage) and "PRA101 is the counterparty" in first[0].content
    assert roles(first) == ["system", "human"] and first[1].content == "please echo hi"
    assert roles(second) == ["system", "human", "ai", "tool"] and second[-1].content == "echo: hi"


def test_a_turn_raises_no_warnings_into_the_users_terminal(harness):
    """The suite ignores DeprecationWarning globally; the chat must not print one mid-answer (it did, live)."""
    import warnings

    h = harness([call("echo", {"text": "hi"}, text="Checking."), say("Done.")])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        summary = h.agent.run_turn("please echo hi", on_event=lambda _event: None)
    assert summary.final_response == "Done."


def test_usage_is_none_when_the_endpoint_reports_none(harness):
    h = harness([say("ok", tokens=None)])
    summary = h.agent.run_turn("hello")
    assert summary.input_tokens is None and summary.output_tokens is None


def test_preamble_and_skill_go_above_the_users_text_in_one_message(harness, tmp_path):
    h = harness([say("ok")])
    skill_dir = h.config.skills_dir / "pra101-recon"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: pra101-recon\ndescription: Reconcile\n---\nCompare months pairwise.\n", encoding="utf-8")
    skill = SkillInfo(name="pra101-recon", description="Reconcile", path=skill_dir)

    h.agent.run_turn("reconcile march", skill=skill, preamble="[Hailer] The active notebook is now notebooks/q2.py.")
    assert roles(h.model.seen[0]) == ["system", "human"]  # one user message, not three
    content = h.model.seen[0][1].content
    assert content.startswith("[Hailer] The user attached the skill 'pra101-recon'")
    assert "Compare months pairwise." in content
    assert content.index("Compare months pairwise.") < content.index("[Hailer] The active notebook is now") < content.index("reconcile march")
    assert content.endswith("reconcile march")


def test_blank_preamble_is_ignored(harness):
    h = harness([say("ok")])
    h.agent.run_turn("hello", preamble="   ")
    assert h.model.seen[0][1].content == "hello"


def test_reloaded_context_applies_to_the_next_turn_of_the_same_conversation(harness):
    h = harness([say("one"), say("two")])
    h.agent.run_turn("first")
    h.agent.bundle = ContextBundle(context_text="New rule: report in GBP millions.")
    h.agent.run_turn("second")
    assert "PRA101 is the counterparty" in h.model.seen[0][0].content
    assert "New rule: report in GBP millions." in h.model.seen[1][0].content
    assert roles(h.model.seen[1]) == ["system", "human", "ai", "human"]  # same conversation


def test_a_failed_turn_is_mapped_and_its_message_is_kept_for_the_retry(harness):
    body = {"detail": [{"loc": ["body", "stream"], "msg": "streaming is not supported"}]}
    h = harness([status_error(422, body), say("recovered")])
    with pytest.raises(ProviderError) as err:
        h.agent.run_turn("what changed in March?")
    assert "rejected the request (HTTP 422)" in str(err.value) and "streaming is not supported" in err.value.hint

    summary = h.agent.run_turn("try again")
    assert summary.final_response == "recovered"
    retry = h.model.seen[-1]
    assert roles(retry) == ["system", "human"]  # merged: strict chat templates refuse two user messages in a row
    assert retry[1].content == "what changed in March?\n\ntry again"


def test_summarisation_runs_on_the_configured_model_and_stays_out_of_the_reply(harness):
    def reply(messages: list[Any]) -> AIMessage:
        if any("Messages to summarize" in str(getattr(m, "content", "")) for m in messages):
            return say("SUMMARY OF EARLIER TURNS")
        return say("answer " + "x" * 400)

    h = harness([reply] * 60, model=ModelConfig(name="internal-analyst", provider="internal", summarize_after_tokens=300))
    events: list[AgentEvent] = []
    for i in range(14):
        summary = h.agent.run_turn(f"question {i} " + "y" * 200, on_event=events.append)
        assert summary.final_response.startswith("answer ")
    summary_calls = [seen for seen in h.model.seen if any("Messages to summarize" in str(getattr(m, "content", "")) for m in seen)]
    assert summary_calls, "the conversation passed the threshold, so older turns must have been summarised"
    assert all("SUMMARY OF EARLIER TURNS" not in e.text for e in events if e.kind == "message_delta")
    assert any("SUMMARY OF EARLIER TURNS" in str(m.content) for m in h.model.seen[-1])  # the model now works from the summary
    assert len(h.model.seen[-1]) < 2 * 14  # and no longer receives every turn


# --------------------------------------------------------------------------- #
# Ctrl+C
# --------------------------------------------------------------------------- #


def _press_ctrl_c() -> None:
    """What Ctrl+C does: SIGINT for the main thread.

    On POSIX it must be a real signal. ``_thread.interrupt_main()`` only sets CPython's pending-signal
    flag; it does not interrupt the event loop's ``epoll`` wait, so on Linux the handler ran only when
    the wait timed out (CI: 30 s late). A real SIGINT, which is what a terminal sends, interrupts the
    wait at once. Windows has no ``pthread_kill``; there ``interrupt_main()`` is verified to work.
    """
    if hasattr(signal, "pthread_kill"):
        signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)
    else:
        _thread.interrupt_main()


def interrupt_after(seconds: float) -> threading.Timer:
    timer = threading.Timer(seconds, _press_ctrl_c)
    timer.daemon = True
    timer.start()
    return timer


def test_ctrl_c_during_a_model_call_cancels_the_turn_at_once(harness):
    async def never(_messages: list[Any]) -> AIMessage:
        await asyncio.sleep(30)
        return say("too late")

    h = harness([lambda messages: never(messages), say("after the interrupt")])
    h.agent.start()
    timer = interrupt_after(0.5)
    started = time.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt):
            h.agent.run_turn("a slow question")
    finally:
        timer.cancel()
    assert time.monotonic() - started < 5

    summary = h.agent.run_turn("and now?")
    assert summary.final_response == "after the interrupt"
    assert roles(h.model.seen[-1]) == ["system", "human"]
    assert h.model.seen[-1][1].content == "a slow question\n\nand now?"


def test_ctrl_c_during_a_tool_leaves_a_valid_conversation(harness):
    h = harness([call("slow", {"seconds": 30}, "call_slow"), say("carried on")])
    h.agent.start()
    timer = interrupt_after(0.7)
    started = time.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt):
            h.agent.run_turn("run the slow thing")
    finally:
        timer.cancel()
    assert time.monotonic() - started < 5
    assert h.tool_log == ["slow:start"]

    summary = h.agent.run_turn("what happened?")
    assert summary.final_response == "carried on"
    sent = h.model.seen[-1]
    assert roles(sent) == ["system", "human", "ai", "tool", "human"]  # the dangling call got a result
    assert isinstance(sent[3], ToolMessage) and sent[3].tool_call_id == "call_slow" and sent[3].content == INTERRUPTED_TOOL_RESULT
    assert isinstance(sent[4], HumanMessage) and sent[4].content == "what happened?"


# --------------------------------------------------------------------------- #
# On the wire: a strict Chat-Completions-only gateway (real ChatOpenAI, loopback only)
# --------------------------------------------------------------------------- #

GATEWAY_ENV = {"CORP_KEY": "sk-spike", "CLIENT_ID": "hailer-spike"}


def gateway_config(tmp_path: Path, gateway: FakeGateway, *, model: str = "corp-gpt", effort: str | None = None, **provider: Any) -> HailerConfig:
    settings: dict[str, Any] = dict(
        id="corp",
        base_url=gateway.base_url,
        wire_api="chat",
        env_key="CORP_KEY",
        stream_options=False,
        env_http_headers={"X-Client-Id": "CLIENT_ID"},
        query_params={"api-version": "2025-04-01-preview"},
    )
    settings.update(provider)
    return make_config(
        tmp_path,
        model=ModelConfig(name=model, provider="corp", reasoning_effort=effort, summarize_after_tokens=0),
        providers={"corp": ProviderConfig(**settings)},
    )


@pytest.fixture
def wire(tmp_path, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    agents: list[HailerAgent] = []

    @tool
    def marimo_status() -> str:
        """Report whether marimo is running and which notebook is active."""
        return "marimo running; active notebook notebooks/analysis.py"

    @tool
    def marimo_execute(code: str) -> str:
        """Run Python in the active notebook's kernel and return stdout."""
        return "2"

    def make(gateway: FakeGateway, *, env: dict[str, str] | None = None, **kw: Any) -> HailerAgent:
        agent = HailerAgent(
            gateway_config(tmp_path, gateway, **kw), ContextBundle(), tools=[marimo_status, marimo_execute], env=GATEWAY_ENV if env is None else env
        )
        agents.append(agent)
        return agent

    yield make
    for agent in agents:
        agent.close()


def test_wire_streaming_turn_sends_only_what_a_strict_gateway_accepts(wire):
    with FakeGateway() as gateway:
        events: list[AgentEvent] = []
        summary = wire(gateway).run_turn("what is the status?", on_event=events.append)

    assert summary.tool_calls == ["marimo_status"]
    assert summary.final_response.startswith("Tool said: marimo running")
    assert (summary.input_tokens, summary.output_tokens) == (22, 14)
    assert sum(1 for e in events if e.kind == "message_delta") > 3
    assert {r["path"] for r in gateway.requests} == {"/v1/chat/completions"}
    bodies = [r["body"] for r in gateway.requests]
    assert len(bodies) == 2  # one tool turn = two requests; nothing on the side
    assert all(sorted(b) == ["messages", "model", "stream", "tools"] for b in bodies)  # no stream_options, no parallel_tool_calls
    assert {b["model"] for b in bodies} == {"corp-gpt"}
    assert [m["role"] for m in bodies[0]["messages"]] == ["system", "user"]
    assert [t["function"]["name"] for t in bodies[0]["tools"]] == ["marimo_status", "marimo_execute"]
    first = gateway.requests[0]
    headers = {name.lower(): value for name, value in first["headers"].items()}
    assert headers["x-client-id"] == "hailer-spike" and headers["authorization"] == "Bearer sk-spike"
    assert first["query"] == {"api-version": ["2025-04-01-preview"]}


def test_wire_stream_false_never_asks_for_a_stream(wire):
    with FakeGateway(allow_stream=False) as gateway:
        events: list[AgentEvent] = []
        summary = wire(gateway, stream=False).run_turn("please run it", on_event=events.append)
    assert summary.final_response == "Tool said: 2" and summary.tool_calls == ["marimo_execute"]
    assert [r["body"].get("stream") for r in gateway.requests] == [False, False]
    assert (summary.input_tokens, summary.output_tokens) == (22, 14)
    assert [e.text for e in events if e.kind == "tool_call"] == ["marimo_execute"]


def test_wire_reasoning_effort_is_sent_only_when_configured(wire):
    with FakeGateway() as gateway:
        wire(gateway).run_turn("hello")
        assert "reasoning_effort" not in gateway.requests[-1]["body"]
        wire(gateway, effort="low").run_turn("hello")
        assert gateway.requests[-1]["body"]["reasoning_effort"] == "low"


def test_wire_errors_become_actionable_messages(wire):
    with FakeGateway() as gateway:
        with pytest.raises(ProviderError) as unknown_model:
            wire(gateway, model="gpt-5.5").run_turn("hello")
        assert str(unknown_model.value) == "Unknown model 'gpt-5.5' for provider 'corp'."
        assert "String should match pattern" in unknown_model.value.hint

        with pytest.raises(CredentialsError) as bad_key:
            wire(gateway, env={**GATEWAY_ENV, "CORP_KEY": "sk-wrong"}).run_turn("hello")
        assert "hailer login corp" in bad_key.value.hint

        with pytest.raises(ProviderError) as rejected:
            wire(gateway, stream_options=True).run_turn("hello")
        assert "rejected the request (HTTP 422)" in str(rejected.value)
        assert "stream_options = false" in rejected.value.hint and "Extra inputs are not permitted" in rejected.value.hint

        with pytest.raises(ProviderError) as wrong_api:
            wire(gateway, wire_api="responses").run_turn("hello")
        assert 'wire_api = "chat"' in wrong_api.value.hint
        assert len(gateway.requests) == 4  # one request per failed turn: a 4xx reply is not retried


def test_wire_unreachable_endpoint(wire, tmp_path):
    class Closed:
        base_url = "http://127.0.0.1:9/v1"

    with pytest.raises(ProviderError) as err:
        wire(Closed()).run_turn("hello")  # type: ignore[arg-type]
    assert "Could not reach the model endpoint at http://127.0.0.1:9/v1" in str(err.value)
