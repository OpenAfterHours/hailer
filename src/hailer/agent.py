"""The agent behind Hailer: a LangChain agent talking to an OpenAI-compatible endpoint.

:func:`build_model` turns a provider from ``hailer.toml`` (the built-in OpenAI endpoint or a
custom ``base_url``) into a chat model; :class:`HailerAgent` runs the tool loop with
``langchain.agents.create_agent``, keeps each conversation in a SQLite file under
``<workspace>/.hailer`` so it survives a restart, and reports progress as
:class:`~hailer.models.AgentEvent` values.

Everything runs inside the Hailer process. The API key is read from the environment or the OS
credential store (:mod:`hailer.secrets`) and handed to the HTTP client; it is never logged.

The public methods are synchronous. Inside, one asyncio event loop lives as long as the agent:
``asyncio.Runner.run`` turns Ctrl+C into a cancellation that aborts the HTTP request in flight,
which LangGraph's synchronous ``stream()`` cannot do on Windows (it blocks in an untimed wait and
the interrupt arrived seconds late, after the turn had finished).
"""

from __future__ import annotations

import asyncio
import importlib.resources
import os
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from hailer import secrets as _secrets
from hailer.errors import AgentError, ConfigError, CredentialsError, HailerError, ProviderError
from hailer.log import get_logger, redact
from hailer.models import (
    WIRE_API_RESPONSES,
    AgentEvent,
    ContextBundle,
    HailerConfig,
    ProviderConfig,
    SkillInfo,
    TurnSummary,
)
from hailer.session import SESSION_DIRNAME

log = get_logger("hailer.agent")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

OPENAI_BASE_URL = "https://api.openai.com/v1"
#: Conversations (LangGraph checkpoints), one SQLite file per workspace.
THREADS_FILENAME = "threads.sqlite"
#: How much recent conversation survives a summarisation.
SUMMARY_KEEP_MESSAGES = 20
#: Set to a truthy value to let LangSmith tracing variables in the environment take effect.
TRACING_OPT_IN_ENV = "HAILER_TRACING"
_TRACING_ENV_VARS = ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING", "LANGCHAIN_TRACING_V2")
#: The result recorded for a tool call that was cut short by Ctrl+C.
INTERRUPTED_TOOL_RESULT = "[Interrupted by the user before this tool finished. Its effect is unknown; check before relying on it.]"

_FALLBACK_SYSTEM_PROMPT = (
    "You are Hailer, a conversational local data-analysis assistant. The terminal is for "
    "conversation; a live Marimo notebook is your visual workspace. Prefer Polars for "
    "DataFrames, use DuckDB for SQL over many files, keep raw data local and inspect compact "
    "summaries rather than dumping tables. Change the notebook through the marimo_execute tool "
    "and marimo._code_mode, never by editing the notebook file. One notebook is active at a time: "
    "marimo_status names it, and notebook_list / notebook_create / notebook_open switch to another "
    "notebook in the notebooks folder when the user asks for one. Explain briefly what you changed."
)

ModelFactory = Callable[[ProviderConfig, str, str | None], Any]

# --------------------------------------------------------------------------- #
# Providers and the chat model
# --------------------------------------------------------------------------- #


def provider_by_id(config: HailerConfig, provider_id: str | None = None) -> ProviderConfig:
    """The provider with ``provider_id`` (default: the configured one); ``openai`` needs no declaration."""
    pid = provider_id or config.model.provider
    if pid in config.providers:
        return config.providers[pid]
    if pid == "openai":
        return ProviderConfig(id="openai", env_key="OPENAI_API_KEY")
    raise ConfigError(
        f'Model provider "{pid}" is not declared.',
        hint=f'Add a [model_providers.{pid}] table to hailer.toml or set [model].provider = "openai".',
    )


def provider_base_url(provider: ProviderConfig) -> str:
    return (provider.base_url or OPENAI_BASE_URL).rstrip("/")


def request_headers(provider: ProviderConfig, environ: Mapping[str, str]) -> dict[str, str]:
    """``http_headers`` plus every ``env_http_headers`` entry whose variable is set."""
    headers = dict(provider.http_headers)
    for header, variable in provider.env_http_headers.items():
        value = environ.get(variable)
        if value:
            headers[header] = value
    return headers


def build_model(
    provider: ProviderConfig,
    model: str,
    api_key: str | None,
    *,
    reasoning_effort: str | None = None,
    environ: Mapping[str, str] | None = None,
    http_async_client: Any = None,
) -> Any:
    """The chat model for ``provider``: one ``ChatOpenAI`` covers both wire APIs.

    ``base_url`` and ``use_responses_api`` are always passed explicitly. Left unset,
    langchain-openai routes model names such as ``gpt-5-codex`` to ``/responses`` even on a
    custom ``base_url``, and a ``LANGSMITH_GATEWAY`` variable in the environment would redirect
    the built-in provider's requests (both checked against langchain-openai 1.6.2).
    ``parallel_tool_calls`` is never sent, and ``stream_options`` only when ``stream_usage`` is on,
    so a gateway that rejects unknown fields sees ``model``, ``messages``, ``tools`` and ``stream``.

    ``http_async_client`` is the HTTP client to use (the agent passes one it owns). Without it
    langchain-openai shares one cached client per process, which is bound to the first event
    loop that used it: a second agent, or the same one after ``close()``, then fails with
    "Event loop is closed".
    """
    from langchain_openai import ChatOpenAI

    env = environ if environ is not None else os.environ
    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "base_url": provider_base_url(provider),
        "use_responses_api": provider.wire_api == WIRE_API_RESPONSES,
        # Token counts for streamed replies need stream_options; off when the gateway rejects it.
        "stream_usage": bool(provider.stream and provider.stream_options),
        # No TCP keep-alive transport: langchain-openai's replaces httpx's own and with it the
        # detection of HTTP(S)_PROXY and the system proxy, which users behind a gateway rely on.
        "http_socket_options": (),
    }
    headers = request_headers(provider, env)
    if headers:
        kwargs["default_headers"] = headers
    if provider.query_params:
        kwargs["default_query"] = dict(provider.query_params)
    if not provider.stream:
        kwargs["disable_streaming"] = True  # sends stream: false and reads one JSON reply
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if http_async_client is not None:
        kwargs["http_async_client"] = http_async_client
    return ChatOpenAI(**kwargs)


def disable_tracing_unless_opted_in(environ: Any = None) -> bool:
    """Switch LangSmith tracing off for this process unless ``HAILER_TRACING`` is set.

    LangChain ships the LangSmith client. A ``LANGSMITH_TRACING=true`` left in a developer's
    environment for another project would otherwise send Hailer's prompts and tool results
    (summaries of local data) to a third party. Returns whether tracing was left alone.
    """
    env = environ if environ is not None else os.environ
    if str(env.get(TRACING_OPT_IN_ENV, "")).strip().lower() in ("1", "true", "yes", "on"):
        return True
    for name in _TRACING_ENV_VARS:
        env[name] = "false"
    return False


# --------------------------------------------------------------------------- #
# System prompt
# --------------------------------------------------------------------------- #


def _base_prompt_text() -> str:
    try:
        resource = importlib.resources.files("hailer").joinpath("prompts/system.md")
        text = resource.read_text(encoding="utf-8")
        if text.strip():
            return text.strip()
    except Exception as exc:  # file not written yet / packaging issue
        log.debug("prompts/system.md unavailable (%s); using fallback persona", type(exc).__name__)
    return _FALLBACK_SYSTEM_PROMPT


def system_prompt(config: HailerConfig, bundle: ContextBundle) -> str:
    """The agent's instructions: persona + project context + skills + web + workspace."""
    parts: list[str] = [_base_prompt_text()]

    if bundle.context_text.strip():
        parts.append("## Project context\n\n" + bundle.context_text.strip())

    if bundle.skills:
        lines = [f"- {s.name} — {s.description}".rstrip(" —") for s in bundle.skills]
        parts.append(
            "## Available skills\n\n"
            "Skills are on-demand instructions. When a task matches a skill, call the "
            "`load_skill` tool with its name and follow it before doing the work; use "
            "`read_skill_file` for files it references.\n\n" + "\n".join(lines)
        )

    if config.web.allowed_domains:
        domains = "\n".join(f"- {d}" for d in config.web.allowed_domains)
        parts.append(
            "## Web access\n\n"
            "You may read pages from these domains only, using the `fetch_page` tool. "
            "Any other host is blocked; if you need one, ask the user to add it to "
            "[web].allowed_domains in hailer.toml.\n\n" + domains
        )
    else:
        parts.append(
            "## Web access\n\nNo internet access is available. Work from local data and the "
            "project context only."
        )

    # The active notebook is deliberately absent: it changes with /notebook and the notebook tools
    # during a conversation. The model learns it from marimo_status() and from CLI notices.
    marimo_line = f"- Marimo URL: {config.marimo_url}\n" if config.marimo_url else ""
    parts.append(
        "## Workspace\n\n"
        f"- Workspace: {config.workspace}\n"
        f"- Notebooks folder: {config.notebooks_root}\n"
        f"- Data directory: {config.data_dir}\n" + marimo_line
    )
    return "\n\n".join(p.rstrip() for p in parts) + "\n"


# --------------------------------------------------------------------------- #
# Error mapping
# --------------------------------------------------------------------------- #

_ENDPOINT_SAID_LIMIT = 500
_MODEL_NOT_FOUND_SIGNALS = ("not found", "does not exist", "not exist", "unknown model", "invalid model", "unsupported model", "no such model")


def _login_hint(provider: ProviderConfig) -> str:
    if provider.env_key:
        return (
            f"Run `hailer login {provider.id}` to store the key in the OS credential store, "
            f"or set the {provider.env_key} environment variable."
        )
    return f"Check the credentials for provider '{provider.id}'."


def _endpoint_said(text: str) -> str:
    """One line quoting the endpoint for the end of a hint: secrets masked, whitespace collapsed, trimmed."""
    flat = " ".join(redact(text).split())
    if len(flat) > _ENDPOINT_SAID_LIMIT:
        flat = flat[: _ENDPOINT_SAID_LIMIT - 3] + "..."
    return f"The endpoint said: {flat}"


def _error_body_text(exc: BaseException) -> str:
    """The endpoint's own words from an ``openai.APIStatusError``: the parsed body when there is one."""
    body = getattr(exc, "body", None)
    if body:
        return str(body)
    return str(getattr(exc, "message", "") or exc)


def _names_the_model(exc: BaseException, text: str) -> bool:
    low = text.lower()
    if getattr(exc, "code", None) == "model_not_found" or "model_not_found" in low or getattr(exc, "param", None) == "model":
        return True
    if "'model'" in low and "loc" in low:  # pydantic-style: {"loc": ["body", "model"], ...}
        return True
    return "model" in low and any(signal in low for signal in _MODEL_NOT_FOUND_SIGNALS)


def _request_line(provider: ProviderConfig) -> str:
    base_url = provider_base_url(provider)
    if provider.wire_api == WIRE_API_RESPONSES:
        return f'Hailer sent POST {base_url}/responses (wire_api = "responses").'
    shape = "streaming" if provider.stream else "stream = false, one JSON reply"
    return f'Hailer sent POST {base_url}/chat/completions (wire_api = "chat", {shape}).'


def map_exception(
    exc: BaseException, config: HailerConfig, *, model: str | None = None, provider: str | None = None
) -> Exception:
    """Translate endpoint and runtime failures into actionable Hailer errors.

    ``model`` and ``provider`` are the model name and provider id in use (they default to the
    configured ones) so the message names the right thing after a ``/model`` switch.
    """
    import openai  # local import: keep module import cheap

    if isinstance(exc, HailerError):
        return exc
    try:
        prov = provider_by_id(config, provider)
    except ConfigError as err:
        return err
    base_url = provider_base_url(prov)
    model_name = model or config.model.name

    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        said = _endpoint_said(_error_body_text(exc))
        if status == 401:
            return CredentialsError(
                f"The model endpoint rejected the API key for provider '{prov.id}'.", hint=_login_hint(prov) + "\n" + said
            )
        if _names_the_model(exc, _error_body_text(exc)) and status in (400, 404, 422):
            return ProviderError(
                f"Unknown model '{model_name}' for provider '{prov.id}'.",
                hint=(
                    "Check [model].name in hailer.toml (or the name given to /model). "
                    "Run `hailer status` to see the active provider and endpoint.\n" + said
                ),
            )
        if status == 403:
            return ProviderError(
                f"The endpoint at {base_url} refused access (HTTP 403).",
                hint=(
                    "The key was accepted but is not allowed for this model, route or organisation: check the "
                    "endpoint's access policy and any http_headers / env_http_headers the provider needs.\n" + said
                ),
            )
        if status in (404, 405):
            other = "chat" if prov.wire_api == WIRE_API_RESPONSES else "responses"
            return ProviderError(
                f"The endpoint at {base_url} did not accept the request.",
                hint=(
                    f"{_request_line(prov)} Check base_url in the provider's [model_providers] table, and that the "
                    f'endpoint implements that API; if it offers the other one, set wire_api = "{other}".\n' + said
                ),
            )
        if status == 429 or status >= 500:
            return ProviderError(
                f"The endpoint at {base_url} is unavailable (HTTP {status}).",
                hint="Retry in a moment; if it persists, check the endpoint's status or rate limits.\n" + said,
            )
        if "context_length" in _error_body_text(exc).lower() or "maximum context" in _error_body_text(exc).lower():
            return ProviderError(
                "The conversation no longer fits the model's context window.",
                hint=(
                    "Start a new conversation with /new, or lower [model].summarize_after_tokens in hailer.toml so "
                    "older turns are summarised sooner.\n" + said
                ),
            )
        switches = []
        if prov.stream:
            switches.append("stream = false")
        if prov.stream and prov.stream_options:
            switches.append("stream_options = false")
        where = (
            f"the provider's [model_providers] table in hailer.toml ({' or '.join(switches)})" if switches else "the provider's [model_providers] table in hailer.toml"
        )
        return ProviderError(
            f"The endpoint at {base_url} rejected the request (HTTP {status}).",
            hint=(
                f"{_request_line(prov)} If the message names a request field the endpoint does not support, check "
                f'{where} and [model].reasoning_effort (reasoning_effort = "" stops sending it); if it names the '
                "model, check [model].name. Run `hailer --verbose` for the full error.\n" + said
            ),
        )
    if isinstance(exc, openai.APITimeoutError):
        return ProviderError(
            f"The model endpoint at {base_url} did not answer in time.",
            hint="Retry in a moment. Run `hailer doctor` to test reachability.",
        )
    if isinstance(exc, openai.APIConnectionError):
        return ProviderError(
            f"Could not reach the model endpoint at {base_url}.",
            hint=(
                "Check base_url in hailer.toml, your VPN/proxy, and that the endpoint is running. "
                "Run `hailer doctor` to test reachability."
            ),
        )
    return AgentError("The agent run failed.", hint=redact(f"{type(exc).__name__}: {exc}")[:600])


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #


def _message_text(message: Any) -> str:
    """The plain text of a model message (Responses replies arrive as a list of content blocks)."""
    text = getattr(message, "text", None)
    if callable(text):  # older langchain-core: a method
        text = text()
    if isinstance(text, str):
        return text
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else ""


def _preview(value: Any, limit: int = 80) -> str:
    text = " ".join(str(value if value is not None else "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


class HailerAgent:
    """Owns the chat model, the tool loop and one conversation thread at a time."""

    def __init__(
        self,
        config: HailerConfig,
        bundle: ContextBundle,
        *,
        model_factory: ModelFactory | None = None,
        tools: list[Any] | None = None,
        env: Mapping[str, str] | None = None,
        threads_path: Path | None = None,
    ) -> None:
        self.config = config
        self._bundle = bundle
        self._environ: Mapping[str, str] = env if env is not None else os.environ
        self._model_factory = model_factory
        self._tools = tools
        self._threads_path = threads_path or (Path(config.workspace) / SESSION_DIRNAME / THREADS_FILENAME)
        self._model = config.model.name
        self._provider_id = config.model.provider
        self.thread_id: str | None = None
        self._runner: asyncio.Runner | None = None
        self._conn: Any = None
        self._saver: Any = None
        self._graph: Any = None
        self._http: Any = None  # the HTTP client of the current chat model; lives on this agent's event loop
        provider = provider_by_id(config, self._provider_id)  # fail early on an undeclared provider
        self.key_source: str = _secrets.resolve_provider_key(provider, self._environ)[1]

    # ---- state the CLI reads or sets -----------------------------------------

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def started(self) -> bool:
        return self._runner is not None

    @property
    def bundle(self) -> ContextBundle:
        return self._bundle

    @bundle.setter
    def bundle(self, value: ContextBundle) -> None:
        """New project context (``/reload``): the next turn is sent with the new instructions."""
        self._bundle = value
        self._graph = None

    # ---- lifecycle ---------------------------------------------------------------

    def _run(self, coro: Any) -> Any:
        if self._runner is None:
            disable_tracing_unless_opted_in()
            self._runner = asyncio.Runner()
        return self._runner.run(coro)

    def _chat_model(self) -> Any:
        provider = provider_by_id(self.config, self._provider_id)
        key, source = _secrets.resolve_provider_key(provider, self._environ)
        self.key_source = source
        if not key:
            raise CredentialsError(
                f"No API key found for provider '{provider.id}' ({provider.env_key or 'env_key'} is not set).",
                hint=_login_hint(provider),
            )
        if self._model_factory is not None:
            return self._model_factory(provider, self._model, key)
        import openai

        self._http = openai.DefaultAsyncHttpxClient()  # honours HTTP(S)_PROXY / NO_PROXY and the OS trust store
        return build_model(
            provider,
            self._model,
            key,
            reasoning_effort=self.config.model.reasoning_effort,
            environ=self._environ,
            http_async_client=self._http,
        )

    async def _close_http(self) -> None:
        http, self._http = self._http, None
        if http is not None:
            try:
                await http.aclose()
            except Exception as exc:  # noqa: BLE001 - best effort
                log.debug("closing the HTTP client failed: %s", type(exc).__name__)

    async def _ensure_graph(self) -> Any:
        if self._graph is not None:
            return self._graph
        from langchain.agents import create_agent
        from langchain.agents.middleware import SummarizationMiddleware
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        await self._close_http()  # the model is being rebuilt (/model, /reload): drop the old client
        model = self._chat_model()
        if self._saver is None:
            import aiosqlite

            self._threads_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = await aiosqlite.connect(str(self._threads_path))
            self._saver = AsyncSqliteSaver(self._conn)
        if self._tools is None:
            from hailer.tools import hailer_tools

            self._tools = hailer_tools(self.config)
        middleware: list[Any] = []
        after = self.config.model.summarize_after_tokens
        if after > 0:
            # An absolute token count: the fractional trigger needs a model profile, which no
            # custom deployment name has. The summary is written by the configured model.
            middleware.append(
                SummarizationMiddleware(model=model, trigger=("tokens", after), keep=("messages", SUMMARY_KEEP_MESSAGES))
            )
        self._graph = create_agent(
            model,
            self._tools,
            system_prompt=system_prompt(self.config, self._bundle),
            middleware=middleware,
            checkpointer=self._saver,
        )
        return self._graph

    def _thread_config(self) -> dict[str, Any]:
        return {"configurable": {"thread_id": self.thread_id}}

    def start(self, *, resume_thread_id: str | None = None) -> str:
        """Open the conversation store and pick the thread: ``resume_thread_id`` when it exists, else a new one."""

        async def go() -> str:
            await self._ensure_graph()
            if resume_thread_id:
                saved = await self._saver.aget_tuple({"configurable": {"thread_id": resume_thread_id}})
                if saved is not None:
                    self.thread_id = resume_thread_id
                    log.debug("resumed thread %s", resume_thread_id)
                    return resume_thread_id
                log.debug("thread %s is not in %s; starting a new one", resume_thread_id, self._threads_path.name)
            return self._new_thread_id()

        return self._guarded(go())

    def _new_thread_id(self) -> str:
        self.thread_id = uuid.uuid4().hex
        log.debug("started thread %s", self.thread_id)
        return self.thread_id

    def new_thread(self) -> str:
        """Start a new conversation. The previous one is deleted: Hailer only ever resumes the latest."""
        previous = self.thread_id

        async def go() -> str:
            await self._ensure_graph()
            if previous:
                try:
                    await self._saver.adelete_thread(previous)
                except Exception as exc:  # noqa: BLE001 - housekeeping, never fatal
                    log.debug("could not delete thread %s: %s", previous, type(exc).__name__)
            return self._new_thread_id()

        return self._guarded(go())

    def set_model(self, name: str, provider: str | None = None) -> None:
        """Change the model (and optionally the provider) for the following turns."""
        if provider is not None:
            provider_by_id(self.config, provider)  # raises ConfigError for an undeclared one
            self._provider_id = provider
        self._model = name
        self._graph = None  # rebuilt with the new model on the next turn

    def close(self) -> None:
        runner, self._runner = self._runner, None
        conn, self._conn = self._conn, None
        self._saver = None
        self._graph = None
        if runner is None:
            return

        async def go() -> None:
            await self._close_http()
            if conn is not None:
                await conn.close()

        try:
            runner.run(go())
        except Exception as exc:  # noqa: BLE001 - best effort on the way out
            log.debug("close failed: %s", type(exc).__name__)
        finally:
            runner.close()

    def _guarded(self, coro: Any) -> Any:
        try:
            return self._run(coro)
        except HailerError:
            raise
        except Exception as exc:
            raise map_exception(exc, self.config, model=self._model, provider=self._provider_id) from exc

    # ---- turns ---------------------------------------------------------------------

    def _turn_text(self, text: str, skill: SkillInfo | None, preamble: str | None) -> str:
        """The user's message with Hailer's own notices above it, each starting with ``[Hailer]``."""
        parts: list[str] = []
        if skill is not None:
            from hailer.context import read_skill

            parts.append(
                f"[Hailer] The user attached the skill '{skill.name}' to this request. Follow its instructions:\n\n"
                + read_skill(self.config, skill.name)
            )
        if preamble and preamble.strip():
            parts.append(preamble.strip())
        parts.append(text)
        return "\n\n".join(parts)

    async def _settle_history(self, graph: Any, content: str) -> Any:
        """Make the stored conversation valid again after an interrupted or failed turn.

        Ctrl+C while a tool runs leaves the model's tool call without a result, which endpoints
        reject with a 400; each such call gets :data:`INTERRUPTED_TOOL_RESULT`. A turn that died
        before the model answered leaves the user's message last; the new text is appended to
        that message (same id, so it is replaced) instead of sending two user messages in a row,
        which strict chat templates refuse.
        """
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        messages = (await graph.aget_state(self._thread_config())).values.get("messages", [])
        answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
        dangling = [
            ToolMessage(content=INTERRUPTED_TOOL_RESULT, tool_call_id=call["id"], name=call.get("name"))
            for m in messages
            if isinstance(m, AIMessage)
            for call in (m.tool_calls or [])
            if call.get("id") and call["id"] not in answered
        ]
        if dangling:
            await graph.aupdate_state(self._thread_config(), {"messages": dangling}, as_node="tools")
            return HumanMessage(content)
        last = messages[-1] if messages else None
        if isinstance(last, HumanMessage) and isinstance(last.content, str) and last.id:
            return HumanMessage(content=f"{last.content}\n\n{content}", id=last.id)
        return HumanMessage(content)

    def run_turn(
        self,
        text: str,
        *,
        on_event: Callable[[AgentEvent], None] | None = None,
        skill: SkillInfo | None = None,
        preamble: str | None = None,
    ) -> TurnSummary:
        """Run one turn with ``text`` as the user's message.

        ``preamble`` is a notice from the CLI (for example ``[Hailer] The active notebook is now
        notebooks/q2_churn.py ...``) and ``skill`` a project skill the user attached; both are
        placed above the user's text in the same message (the system prompt explains ``[Hailer]``
        paragraphs). Ctrl+C cancels the turn and re-raises ``KeyboardInterrupt``.
        """
        if self.thread_id is None:
            self.start()
        emit = on_event or (lambda _event: None)
        content = self._turn_text(text, skill, preamble)
        started = time.monotonic()
        tool_calls: list[str] = []
        usage = {"input": 0, "output": 0, "seen": False}
        final: list[str] = [""]

        def on_update(update: Mapping[str, Any]) -> None:
            from langchain_core.messages import AIMessage

            for node, change in update.items():
                if node != "model" or not isinstance(change, Mapping):
                    continue
                for message in change.get("messages", []):
                    if not isinstance(message, AIMessage):
                        continue
                    for call in message.tool_calls or []:
                        tool_calls.append(call["name"])
                        emit(AgentEvent("tool_call", call["name"], {"arguments": _preview(call.get("args"))}))
                    counts = message.usage_metadata
                    if counts:
                        usage["seen"] = True
                        usage["input"] += counts.get("input_tokens", 0) or 0
                        usage["output"] += counts.get("output_tokens", 0) or 0
                    if not message.tool_calls:
                        final[0] = _message_text(message)

        async def go() -> None:
            graph = await self._ensure_graph()
            message = await self._settle_history(graph, content)
            stream = graph.astream({"messages": [message]}, self._thread_config(), stream_mode=["updates", "messages"])
            async for mode, data in stream:
                if mode == "updates":
                    on_update(data)
                    continue
                chunk, meta = data
                if meta.get("langgraph_node") == "model":  # not the summariser's own model call
                    delta = _message_text(chunk)
                    if delta:
                        emit(AgentEvent("message_delta", delta))

        self._guarded(go())  # KeyboardInterrupt passes through: Runner.run cancelled the turn already

        return TurnSummary(
            final_response=final[0],
            thread_id=self.thread_id or "",
            duration_ms=int((time.monotonic() - started) * 1000),
            tool_calls=tool_calls,
            input_tokens=usage["input"] if usage["seen"] else None,
            output_tokens=usage["output"] if usage["seen"] else None,
        )
