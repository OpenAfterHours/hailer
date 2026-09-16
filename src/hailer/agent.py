"""Codex lifecycle for Hailer: provider/tool configuration, threads, streamed turns.

Everything Codex needs to know about the model endpoint, the Hailer MCP server and
what to leave out of the request is expressed as ``codex -c key=value`` overrides
built by :func:`build_config_overrides` (a pure function, so it is unit-testable).
Secrets travel only through the Codex child-process environment
(:func:`build_child_env`); they are never placed on a command line or in a log.
"""

from __future__ import annotations

import importlib.resources
import logging
import os
import queue
import sys
import threading
import time
import tomllib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from hailer import secrets as _secrets
from hailer.errors import AgentError, ConfigError, CredentialsError, ProviderError
from hailer.models import (
    AgentEvent,
    ContextBundle,
    HailerConfig,
    ProviderConfig,
    SkillInfo,
    TurnSummary,
)
from hailer.wire import ChatBridge, ChatUpstream

try:  # log.py is written by another owner; fall back to plain logging if absent.
    from hailer.log import get_logger
except Exception:  # pragma: no cover - only when hailer.log is missing

    def get_logger(name: str = "hailer") -> logging.Logger:
        return logging.getLogger(name)


log = get_logger("hailer.agent")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Codex features that have no place in an analysis session and would otherwise add
#: tools/instructions to every request (and break strict non-OpenAI gateways).
TRIMMING_OVERRIDES: tuple[tuple[str, Any], ...] = (
    ("web_search", "disabled"),
    ("features.web_search_request", False),
    ("features.multi_agent", False),
    ("features.multi_agent_v2", False),
    ("features.plugins", False),
    ("features.apps", False),
    ("features.codex_apps", False),
    ("features.connectors", False),
    ("features.image_generation", False),
)

MCP_SERVER_NAME = "hailer"
MCP_TOOL_TIMEOUT_SEC = 600
MCP_STARTUP_TIMEOUT_SEC = 60
LOCAL_HOSTS = ("127.0.0.1", "localhost")

_FALLBACK_SYSTEM_PROMPT = (
    "You are Hailer, a conversational local data-analysis assistant. The terminal is for "
    "conversation; a live Marimo notebook is your visual workspace. Prefer Polars for "
    "DataFrames, use DuckDB for SQL over many files, keep raw data local and inspect compact "
    "summaries rather than dumping tables. Change the notebook through the marimo_execute tool "
    "and marimo._code_mode, never by editing the notebook file. One notebook is active at a time: "
    "marimo_status names it, and notebook_list / notebook_create / notebook_open switch to another "
    "notebook in the notebooks folder when the user asks for one. Explain briefly what you changed."
)

_PROVIDER_FIELDS = (
    "name",
    "base_url",
    "wire_api",
    "env_key",
    "requires_openai_auth",
    "http_headers",
    "env_http_headers",
    "query_params",
)

# --------------------------------------------------------------------------- #
# TOML rendering for -c overrides
# --------------------------------------------------------------------------- #


def _bare_key(key: str) -> bool:
    return bool(key) and all(ch.isalnum() or ch in "_-" for ch in key)


def toml_key(key: str) -> str:
    """Render one key segment, quoting when it is not a bare TOML key."""
    return key if _bare_key(key) else toml_value(key)


def toml_value(value: Any) -> str:
    """Render a Python value as a TOML inline value (strings escaped, Windows paths safe)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, Path):
        return toml_value(str(value))
    if isinstance(value, str):
        escaped = (
            value.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
        )
        return f'"{escaped}"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{toml_key(str(k))} = {toml_value(v)}" for k, v in value.items()) + "}"
    raise TypeError(f"cannot render {type(value).__name__} as TOML")


def _override(key: str, value: Any) -> str:
    return f"{key}={toml_value(value)}"


# --------------------------------------------------------------------------- #
# Configuration builders (pure)
# --------------------------------------------------------------------------- #


def read_user_codex_config(codex_home: Path | None = None) -> dict:
    """Parse ``<codex_home>/config.toml`` (default ``~/.codex``); ``{}`` when absent or invalid."""
    home = codex_home or Path(os.environ.get("CODEX_HOME") or (Path.home() / ".codex"))
    path = home / "config.toml"
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as exc:  # unparsable user config is not Hailer's problem to fix
        log.debug("could not parse %s: %s", path, type(exc).__name__)
        return {}


def _active_provider(config: HailerConfig) -> ProviderConfig:
    try:
        return config.provider
    except KeyError:
        raise ConfigError(
            f'Model provider "{config.model.provider}" is not declared.',
            hint=f'Add a [model_providers.{config.model.provider}] table to hailer.toml or set [model].provider = "openai".',
        ) from None


def _custom_providers(config: HailerConfig) -> list[ProviderConfig]:
    return [p for pid, p in sorted(config.providers.items()) if not p.is_builtin_openai]


def chat_completion_upstreams(config: HailerConfig) -> dict[str, ChatUpstream]:
    """Provider id -> bridge upstream (``base_url`` and ``stream``) for every ``wire_api = "chat"`` provider.

    ``stream`` is a bridge setting, not a Codex one: it is never passed through as a
    ``model_providers.<id>.*`` override (see ``_PROVIDER_FIELDS``).
    """
    return {p.id: ChatUpstream(p.base_url, stream=p.stream) for p in _custom_providers(config) if p.uses_chat_completions and p.base_url}


def build_config_overrides(
    config: HailerConfig, user_codex_config: dict, bridge_urls: Mapping[str, str] | None = None
) -> tuple[str, ...]:
    """Return the ``key=value`` overrides passed to ``codex app-server``.

    ``bridge_urls`` maps the id of each ``wire_api = "chat"`` provider to the loopback URL of
    the running :class:`~hailer.wire.ChatBridge`; those providers are handed to Codex as
    Responses-API providers at that URL (Codex refuses ``wire_api = "chat"`` itself). The
    credentials and custom headers stay on the provider, so Codex still attaches them and the
    bridge forwards them upstream.

    Pure: depends only on the arguments (plus ``sys.executable`` for the MCP server command).
    """
    _active_provider(config)  # validates that the active provider exists
    out: list[str] = [
        _override("model", config.model.name),
        _override("model_provider", config.model.provider),
    ]

    # Every declared custom provider is passed through, so /model can switch providers
    # without restarting the app-server.
    bridged = bridge_urls or {}
    for prov in _custom_providers(config):
        for field in _PROVIDER_FIELDS:
            value = getattr(prov, field)
            if prov.uses_chat_completions and prov.id in bridged:
                if field == "base_url":
                    value = bridged[prov.id]
                elif field == "wire_api":
                    value = "responses"
            if value is None or value == {} or value == "":
                continue
            out.append(_override(f"model_providers.{toml_key(prov.id)}.{field}", value))

    if config.model.reasoning_effort:
        out.append(_override("model_reasoning_effort", config.model.reasoning_effort))

    for key, value in TRIMMING_OVERRIDES:
        out.append(_override(key, value))

    plugins = user_codex_config.get("plugins") or {}
    if isinstance(plugins, Mapping):
        for plugin_id in plugins:
            out.append(_override(f"plugins.{toml_key(str(plugin_id))}.enabled", False))

    servers = user_codex_config.get("mcp_servers") or {}
    if isinstance(servers, Mapping):
        for name in servers:
            if name == MCP_SERVER_NAME:
                continue
            out.append(_override(f"mcp_servers.{toml_key(str(name))}.enabled", False))

    mcp_env: dict[str, str] = {}
    if config.config_path is not None:
        mcp_env["HAILER_CONFIG"] = str(config.config_path)
    mcp_env["HAILER_WORKSPACE"] = str(config.workspace)
    mcp_env["HAILER_LOG_LEVEL"] = config.log_level  # not secret; secrets never go in -c overrides
    out.extend(
        [
            _override(f"mcp_servers.{MCP_SERVER_NAME}.command", sys.executable),
            _override(f"mcp_servers.{MCP_SERVER_NAME}.args", ["-m", "hailer.mcp_server"]),
            _override(f"mcp_servers.{MCP_SERVER_NAME}.env", mcp_env),
            _override(f"mcp_servers.{MCP_SERVER_NAME}.tool_timeout_sec", MCP_TOOL_TIMEOUT_SEC),
            _override(f"mcp_servers.{MCP_SERVER_NAME}.startup_timeout_sec", MCP_STARTUP_TIMEOUT_SEC),
            _override(f"mcp_servers.{MCP_SERVER_NAME}.default_tools_approval_mode", "auto"),
        ]
    )

    if config.web.allow_shell_network:
        domains = {d: "allow" for d in config.web.allowed_domains}
        for host in LOCAL_HOSTS:
            domains.setdefault(host, "allow")
        out.extend(
            [
                _override("features.network_proxy.enabled", True),
                _override("features.network_proxy.domains", domains),
                _override("network.enabled", True),
                _override("network.mode", "limited"),
                _override("network.domains", domains),
                _override("sandbox_workspace_write.network_access", True),
            ]
        )

    return tuple(out)


#: Hosts the Codex child must reach directly (never through HTTP(S)_PROXY): the chat bridge.
BRIDGE_NO_PROXY_HOSTS = ("127.0.0.1", "localhost")


def _with_no_proxy(existing: str | None) -> str:
    """Append the bridge hosts to a NO_PROXY value, keeping whatever the user already listed."""
    entries = [e.strip() for e in (existing or "").split(",") if e.strip()]
    for host in BRIDGE_NO_PROXY_HOSTS:
        if host not in entries:
            entries.append(host)
    return ",".join(entries)


def build_child_env(
    config: HailerConfig,
    key: str | None,
    extra_keys: Mapping[str, str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Variables to *add* to the Codex child environment (the SDK copies ``os.environ`` itself).

    ``key`` is the active provider's API key (or ``None``); ``extra_keys`` maps other
    providers' ``env_key`` names to their values so ``/model <provider>:<name>`` works
    mid-session. ``environ`` is the parent environment (``os.environ`` by default); it is
    consulted so the loopback bridge for ``wire_api = "chat"`` providers is exempted from
    the user's HTTP(S)_PROXY without dropping their own NO_PROXY entries. Values are never
    logged.
    """
    env: dict[str, str] = {}
    parent = environ if environ is not None else os.environ
    provider = _active_provider(config)
    if key and provider.env_key:
        env[provider.env_key] = key
    for name, value in (extra_keys or {}).items():
        if value and name not in env:
            env[name] = value
    if config.config_path is not None:
        env["HAILER_CONFIG"] = str(config.config_path)
    env["HAILER_WORKSPACE"] = str(config.workspace)
    # HAILER_MARIMO_TOKEN is a secret: it goes into the process environment only, never into
    # the -c mcp_servers.hailer.env map (a command-line argument). MCP servers spawned by
    # Codex inherit the app-server's environment, so the Hailer MCP server sees it too.
    if config.marimo_token:
        env["HAILER_MARIMO_TOKEN"] = config.marimo_token
    if config.codex_home is not None:
        env["CODEX_HOME"] = str(config.codex_home)
    if chat_completion_upstreams(config):
        # Codex (reqwest) honours HTTP(S)_PROXY with no implicit loopback bypass; without this a
        # corporate proxy would be asked to reach the bridge on the user's own machine.
        for name in ("NO_PROXY", "no_proxy"):
            env[name] = _with_no_proxy(parent.get("NO_PROXY") or parent.get("no_proxy"))
    return env


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
    """Base instructions for the Codex thread: persona + project context + skills + web + workspace."""
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
    # while the thread lives, and the prompt fingerprint must stay stable across those switches.
    # The model learns the active notebook from marimo_status() and from CLI notices.
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


def _login_hint(provider: ProviderConfig) -> str:
    if provider.env_key:
        return (
            f"Run `hailer login {provider.id}` to store the key in the OS credential store, "
            f"or set the {provider.env_key} environment variable."
        )
    return f"Check the credentials for provider '{provider.id}'."


_MODEL_NOT_FOUND_SIGNALS = (
    "not found",
    "does not exist",
    "not exist",
    "unknown model",
    "invalid model",
    "unsupported model",
    "no such model",
)
_CONNECTION_SIGNALS = (
    "connection refused",
    "failed to connect",
    "could not reach",  # the chat bridge's own wording for an unreachable upstream
    "error sending request",
    "dns error",
    "name resolution",
    "connect error",
    "connection reset",
    "no such host",
    "timed out",
)
# Signals that the endpoint rejected the *shape* of the request (not the model, not auth).
_RESPONSES_API_SIGNALS = ("404", "unsupported", "unknown endpoint", "responses", "unknown parameter", "schema")


def _provider_by_id(config: HailerConfig, provider_id: str | None) -> ProviderConfig:
    """The provider the agent is currently using (``/model`` may have switched it)."""
    if provider_id is None or provider_id == config.model.provider:
        return _active_provider(config)
    if provider_id in config.providers:
        return config.providers[provider_id]
    if provider_id == "openai":
        return ProviderConfig(id="openai", env_key="OPENAI_API_KEY", requires_openai_auth=True)
    return _active_provider(config)


def map_exception(
    exc: BaseException, config: HailerConfig, *, model: str | None = None, provider: str | None = None
) -> Exception:
    """Translate SDK/runtime failures into actionable Hailer errors.

    ``model`` and ``provider`` are the model name and provider id in use (they default to
    the configured ones) so unknown-model and endpoint errors name the right thing after a
    ``/model`` switch.
    """
    text = str(exc)
    low = text.lower()
    provider = _provider_by_id(config, provider)
    base_url = provider.base_url or "https://api.openai.com/v1"
    model_name = model or config.model.name

    from openai_codex.errors import TransportClosedError  # local import: keep module import cheap

    if isinstance(exc, TransportClosedError):
        if "invalid transport" in low or "config error" in low or "error loading" in low:
            return ConfigError(
                "The Codex runtime refused to start because of a configuration error.",
                hint=(
                    "This usually comes from ~/.codex/config.toml combined with Hailer's overrides. "
                    "Run `hailer --verbose` to see the runtime's message, or set HAILER_CODEX_HOME "
                    "to an empty folder to isolate Hailer from your existing Codex config."
                ),
            )
        return AgentError("The Codex runtime stopped unexpectedly.", hint=text[:600])

    # An unknown model must not be mistaken for an incompatible endpoint.
    if "model" in low and any(s in low for s in _MODEL_NOT_FOUND_SIGNALS):
        return ProviderError(
            f"Unknown model '{model_name}' for provider '{provider.id}'.",
            hint=(
                "Check [model].name in hailer.toml (or the name given to /model). "
                "Run `hailer status` to see the active provider and endpoint."
            ),
        )
    if "401" in low or "unauthorized" in low or "invalid api key" in low or "incorrect api key" in low:
        return CredentialsError(
            f"The model endpoint rejected the API key for provider '{provider.id}'.",
            hint=_login_hint(provider),
        )
    # Tool timeouts come from Hailer's MCP server (notebook computation), not the endpoint.
    if "timed out" in low and ("tool" in low or "mcp" in low):
        return AgentError(
            "A tool call timed out before the notebook finished.",
            hint=(
                "Reduce the data touched per call (filter, aggregate, sample) or run the heavy "
                "step as a notebook cell and inspect its result in a later call."
            ),
        )
    if any(s in low for s in _CONNECTION_SIGNALS):
        return ProviderError(
            f"Could not reach the model endpoint at {base_url}.",
            hint=(
                "Check [model_providers].base_url in hailer.toml, your VPN/proxy, and that the "
                "endpoint is running. Run `hailer doctor` to test reachability."
            ),
        )
    if "chat completions bridge" in low:
        return ProviderError(
            f"Codex asked for a Responses API endpoint the Chat Completions bridge for provider '{provider.id}' does not offer.",
            hint=(
                "Only POST /responses is translated to /chat/completions (Codex features such as server-side "
                'compaction are not). Run `hailer --verbose` to see the endpoint, or set wire_api = "responses" '
                "if the gateway implements the Responses API."
            ),
        )
    if any(s in low for s in _RESPONSES_API_SIGNALS):
        if provider.uses_chat_completions and provider.stream:
            hint = (
                f"This provider uses wire_api = \"chat\": Hailer sends POST {base_url}/chat/completions "
                "(streaming, with function tools). Check that the gateway implements Chat Completions with "
                "function calling at that path; set stream = false in its [model_providers] table if it "
                'rejects stream = true, or set wire_api = "responses" if it implements the Responses API.'
            )
        elif provider.uses_chat_completions:
            hint = (
                f"This provider uses wire_api = \"chat\" with stream = false: Hailer sends POST {base_url}/chat/completions "
                "(one JSON reply, with function tools). Check that the gateway implements Chat Completions with "
                'function calling at that path, or set wire_api = "responses" if it implements the Responses API.'
            )
        else:
            hint = (
                f"This provider uses the OpenAI Responses API (POST {base_url}/responses, streaming). "
                'If the gateway only offers Chat Completions, set wire_api = "chat" in its '
                "[model_providers] table instead."
            )
        return ProviderError(f"The endpoint at {base_url} did not accept the request.", hint=hint)
    return AgentError("The agent run failed.", hint=text[:600])


# --------------------------------------------------------------------------- #
# Helpers for notification handling (duck-typed so tests can use stand-ins)
# --------------------------------------------------------------------------- #


def _unwrap(item: Any) -> Any:
    return getattr(item, "root", item)


def _item_type(item: Any) -> str:
    item = _unwrap(item)
    kind = getattr(item, "type", None)
    if isinstance(kind, str):
        return kind
    return type(item).__name__


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _preview(value: Any, limit: int = 80) -> str:
    text = str(value) if value is not None else ""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _final_response(items: list[Any]) -> str | None:
    unknown_phase: str | None = None
    for item in reversed(items):
        item = _unwrap(item)
        if _item_type(item) not in ("agentMessage", "AgentMessageThreadItem"):
            continue
        phase = _enum_value(getattr(item, "phase", None))
        text = getattr(item, "text", "") or ""
        if phase == "final_answer":
            return text
        if phase is None and unknown_phase is None:
            unknown_phase = text
    return unknown_phase


_STREAM_END = object()


class _StreamError:
    """Carries an exception raised by ``handle.stream()`` from the pump thread."""

    __slots__ = ("exc",)

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


def _pump_events(handle: Any, out: "queue.Queue[Any]") -> None:
    """Consume a turn's notification stream on a worker thread.

    The SDK's stream blocks in an untimed ``Condition.wait()``, which Windows cannot
    interrupt with Ctrl+C; the main thread instead waits on the queue with a timeout.
    """
    try:
        for event in handle.stream():
            out.put(event)
    except BaseException as exc:  # noqa: BLE001 - forwarded to the main thread
        out.put(_StreamError(exc))
    finally:
        out.put(_STREAM_END)


def _safe_interrupt(handle: Any) -> None:
    try:
        handle.interrupt()
    except Exception as exc:  # already finished, transport gone ...
        log.debug("interrupt failed: %s", type(exc).__name__)


def _default_codex_factory(cfg: Any) -> Any:
    from openai_codex import Codex

    return Codex(cfg)


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #


class HailerAgent:
    """Owns the Codex runtime and one conversation thread at a time."""

    def __init__(
        self,
        config: HailerConfig,
        bundle: ContextBundle,
        *,
        codex_factory: Callable[..., Any] | None = None,
        user_codex_config: dict | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config
        self.bundle = bundle
        self._codex_factory = codex_factory or _default_codex_factory
        self._codex: Any = None
        self._bridge: ChatBridge | None = None
        self._thread: Any = None
        self._handle: Any = None
        self._interrupt_requested = False
        self.thread_id: str | None = None
        self._model = config.model.name
        self._provider_id = config.model.provider
        self._thread_provider_id: str | None = None

        environ = env if env is not None else os.environ
        active = _active_provider(config)
        key, source = _secrets.resolve_provider_key(active, environ)
        self.key_source: str = source
        self.key_sources: dict[str, str] = {active.id: source}
        extra: dict[str, str] = {}
        for prov in _custom_providers(config):
            if prov.id == active.id or not prov.env_key:
                continue
            value, src = _secrets.resolve_provider_key(prov, environ)
            self.key_sources[prov.id] = src
            if value:
                extra[prov.env_key] = value
        self._child_env = build_child_env(config, key, extra, environ)
        del key, extra  # do not keep secrets on the instance

        self._user_codex_config = user_codex_config if user_codex_config is not None else read_user_codex_config(config.codex_home)
        build_config_overrides(config, self._user_codex_config)  # fail early on an undeclared provider

    # ---- lifecycle ---------------------------------------------------------

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def started(self) -> bool:
        return self._codex is not None

    @property
    def bridge(self) -> ChatBridge | None:
        """The Chat Completions bridge, once started (only when a provider has ``wire_api = "chat"``)."""
        return self._bridge

    @property
    def overrides(self) -> tuple[str, ...]:
        """The ``-c`` overrides Codex is (or will be) started with.

        Derived on demand so that, once the chat bridge is up, ``wire_api = "chat"`` providers
        are rendered at the bridge URL; there is no stale pre-bridge copy to read by mistake.
        """
        bridge_urls = self._bridge.urls if self._bridge is not None else None
        return build_config_overrides(self.config, self._user_codex_config, bridge_urls)

    def _ensure_bridge(self) -> None:
        """Start the loopback bridge for ``wire_api = "chat"`` providers (Codex is pointed at it)."""
        if self._bridge is not None:
            return
        upstreams = chat_completion_upstreams(self.config)
        if not upstreams:
            return
        bridge = ChatBridge(upstreams)
        try:
            bridge.start()
        except OSError as exc:
            raise ProviderError(
                "Could not start the local Chat Completions bridge on 127.0.0.1.",
                hint=(
                    f"Providers with wire_api = \"chat\" ({', '.join(sorted(upstreams))}) need a loopback "
                    f"port. The OS refused one: {exc}. Check local firewall or sandbox rules, or set "
                    'wire_api = "responses" if the endpoint implements the Responses API.'
                ),
            ) from exc
        self._bridge = bridge

    def _require_key(self) -> None:
        provider = self.config.providers.get(self._provider_id)
        if provider is None or provider.is_builtin_openai:
            return  # a Codex ChatGPT login may exist; the CLI preflight reports the situation
        if self.key_sources.get(provider.id, "missing") == "missing":
            raise CredentialsError(
                f"No API key found for provider '{provider.id}' ({provider.env_key} is not set).",
                hint=_login_hint(provider),
            )

    def _thread_kwargs(self) -> dict[str, Any]:
        from openai_codex import ApprovalMode, Sandbox

        return {
            "sandbox": Sandbox.workspace_write,
            # auto_review, not deny_all: with approval_policy "never" Codex rejects every MCP
            # tool call ("MCP tool call requires approval, but approval policy is never")
            # regardless of per-server/per-tool approval settings (verified live). With
            # auto_review, Hailer's tools (default_tools_approval_mode="auto") run without
            # prompting and the SDK auto-accepts command/file-change approvals.
            "approval_mode": ApprovalMode.auto_review,
            "base_instructions": system_prompt(self.config, self.bundle),
            "cwd": str(self.config.workspace),
            "model": self._model,
            "model_provider": self._provider_id,
        }

    def _ensure_codex(self) -> None:
        if self._codex is not None:
            return
        self._ensure_bridge()
        from openai_codex import CodexConfig

        cfg = CodexConfig(
            cwd=str(self.config.workspace),
            config_overrides=self.overrides,
            env=dict(self._child_env) or None,
        )
        log.debug("starting codex app-server with %d overrides", len(self.overrides))
        try:
            self._codex = self._codex_factory(cfg)
        except Exception as exc:
            raise map_exception(exc, self.config, model=self._model, provider=self._provider_id) from exc

    def start(self, *, resume_thread_id: str | None = None) -> str:
        """Start the runtime and a thread; returns the thread id."""
        self._require_key()
        self._ensure_codex()
        kwargs = self._thread_kwargs()
        if resume_thread_id:
            try:
                self._thread = self._codex.thread_resume(resume_thread_id, **kwargs)
                self.thread_id = self._thread.id
                self._thread_provider_id = self._provider_id
                log.debug("resumed thread %s", self.thread_id)
                return self.thread_id
            except Exception as exc:
                log.warning("could not resume thread %s (%s); starting a new one", resume_thread_id, type(exc).__name__)
        return self.new_thread()

    def new_thread(self) -> str:
        self._require_key()
        self._ensure_codex()
        try:
            self._thread = self._codex.thread_start(**self._thread_kwargs())
        except Exception as exc:
            raise map_exception(exc, self.config, model=self._model, provider=self._provider_id) from exc
        self.thread_id = self._thread.id
        self._thread_provider_id = self._provider_id
        log.debug("started thread %s", self.thread_id)
        return self.thread_id

    def set_model(self, name: str, provider: str | None = None) -> bool:
        """Change the model (and optionally provider) for subsequent turns.

        A provider change needs a new thread; one is started automatically when the
        runtime is already running. Returns ``True`` when that happened, so the caller
        must not start another thread itself.
        """
        if provider is not None and provider != "openai" and provider not in self.config.providers:
            raise ConfigError(
                f'Unknown provider "{provider}".',
                hint="Declare it as [model_providers.<id>] in hailer.toml (or use openai).",
            )
        self._model = name
        if provider is not None and provider != self._provider_id:
            self._provider_id = provider
            if self._codex is not None:
                self.new_thread()
                return True
        return False

    def interrupt(self) -> None:
        """Stop the active turn, if any. Idempotent; a no-op when no turn is running.

        After an interrupt the turn's remaining notifications are still collected (so the
        turn ends cleanly) but no longer surfaced through ``on_event``.
        """
        self._interrupt_requested = True
        handle = self._handle
        if handle is None:
            return
        _safe_interrupt(handle)

    def close(self) -> None:
        codex, self._codex = self._codex, None
        self._thread = None
        self._handle = None
        if codex is not None:
            try:
                codex.close()
            except Exception as exc:
                log.debug("close failed: %s", type(exc).__name__)
        bridge, self._bridge = self._bridge, None
        if bridge is not None:
            bridge.close()

    # ---- turns -------------------------------------------------------------

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
        notebooks/q2_churn.py ...``) sent as its own text item ahead of the user's message; the
        system prompt tells the model to treat such lines as facts, not as the user's words. A
        blank preamble is ignored, so a plain turn stays a plain string input.
        """
        if self._thread is None:
            self.start()
        assert self._thread is not None

        from openai_codex import SkillInput, TextInput

        turn_input: Any = text
        notice = preamble if preamble and preamble.strip() else None
        if skill is not None or notice is not None:
            items_in: list[Any] = []
            if skill is not None:
                items_in.append(SkillInput(name=skill.name, path=str(Path(skill.path) / "SKILL.md")))
            if notice is not None:
                items_in.append(TextInput(notice))
            items_in.append(TextInput(text))
            turn_input = items_in

        emit = on_event or (lambda _e: None)
        started = time.monotonic()
        try:
            handle = self._thread.turn(turn_input, model=self._model)
        except Exception as exc:
            raise map_exception(exc, self.config, model=self._model, provider=self._provider_id) from exc
        self._handle = handle
        self._interrupt_requested = False

        items: list[Any] = []
        commands: list[str] = []
        tool_calls: list[str] = []
        usage: Any = None
        completed_turn: Any = None
        partial: list[str] = []

        def handle_event(event: Any) -> None:
            nonlocal usage, completed_turn
            method = getattr(event, "method", "")
            payload = getattr(event, "payload", None)
            quiet = self._interrupt_requested  # after an interrupt: collect, do not display
            if method == "item/agentMessage/delta":
                delta = getattr(payload, "delta", "") or ""
                partial.append(delta)
                if not quiet:
                    emit(AgentEvent("message_delta", delta))
            elif method == "item/started":
                item = _unwrap(getattr(payload, "item", None))
                kind = _item_type(item)
                if kind in ("commandExecution", "CommandExecutionThreadItem"):
                    command = getattr(item, "command", "") or ""
                    commands.append(command)
                    if not quiet:
                        emit(AgentEvent("command", command, {"cwd": getattr(item, "cwd", None)}))
                elif kind in ("mcpToolCall", "McpToolCallThreadItem"):
                    name = f"{getattr(item, 'server', '')}.{getattr(item, 'tool', '')}".strip(".")
                    tool_calls.append(name)
                    if not quiet:
                        preview = _preview(getattr(item, "arguments", None))
                        emit(AgentEvent("tool_call", name, {"arguments": preview}))
            elif method == "item/commandExecution/outputDelta":
                if not quiet:
                    emit(AgentEvent("command_output", getattr(payload, "delta", "") or ""))
            elif method in ("item/reasoning/textDelta", "item/reasoning/summaryTextDelta"):
                if not quiet:
                    emit(AgentEvent("reasoning", getattr(payload, "delta", "") or ""))
            elif method == "item/completed":
                items.append(getattr(payload, "item", None))
            elif method == "thread/tokenUsage/updated":
                usage = getattr(payload, "token_usage", None)
            elif method == "error":
                if not quiet:
                    emit(AgentEvent("status", _preview(getattr(payload, "message", payload), 200)))
            elif method == "turn/completed":
                completed_turn = getattr(payload, "turn", None)

        events: queue.Queue[Any] = queue.Queue()
        pump = threading.Thread(
            target=_pump_events, args=(handle, events), name="hailer-turn-stream", daemon=True
        )
        pump.start()

        try:
            try:
                while True:
                    try:
                        msg = events.get(timeout=0.2)  # timed wait: interruptible on Windows
                    except queue.Empty:
                        if not pump.is_alive() and events.empty():
                            break
                        continue
                    if msg is _STREAM_END:
                        break
                    if isinstance(msg, _StreamError):
                        raise msg.exc
                    handle_event(msg)
            except KeyboardInterrupt:
                # Ctrl+C: stop the turn on the LOCAL handle (the finally below clears
                # self._handle), let the runtime acknowledge so the next message does not
                # join this turn, then hand the interrupt back to the CLI.
                self._interrupt_requested = True
                _safe_interrupt(handle)
                deadline = time.monotonic() + 10.0
                while completed_turn is None and time.monotonic() < deadline:
                    try:
                        msg = events.get(timeout=0.2)
                    except queue.Empty:
                        if not pump.is_alive():
                            break
                        continue
                    if msg is _STREAM_END or isinstance(msg, _StreamError):
                        break
                    handle_event(msg)
                raise
        except Exception as exc:
            raise map_exception(exc, self.config, model=self._model, provider=self._provider_id) from exc
        finally:
            self._handle = None

        if completed_turn is None:
            raise AgentError("The turn ended without a completion event.", hint="Retry the request.")

        status = _enum_value(getattr(completed_turn, "status", "completed")) or "completed"
        error = getattr(completed_turn, "error", None)
        if status == "failed":
            message = getattr(error, "message", None) or str(error) or "unknown error"
            raise map_exception(RuntimeError(message), self.config, model=self._model, provider=self._provider_id)

        final = _final_response(items)
        if final is None:
            final = "".join(partial).strip()

        duration = getattr(completed_turn, "duration_ms", None)
        if duration is None:
            duration = int((time.monotonic() - started) * 1000)

        last = getattr(usage, "last", None)
        return TurnSummary(
            final_response=final or "",
            thread_id=self.thread_id or "",
            turn_id=getattr(completed_turn, "id", "") or "",
            duration_ms=duration,
            commands=commands,
            tool_calls=tool_calls,
            input_tokens=getattr(last, "input_tokens", None),
            output_tokens=getattr(last, "output_tokens", None),
            status=str(status),
        )
