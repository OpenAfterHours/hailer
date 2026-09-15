"""Configuration loading for Hailer.

Sources, lowest to highest precedence:

1. built-in defaults
2. the project config file (``hailer.toml`` or ``.config/hailer/hailer.toml``)
3. environment variables (``HAILER_*``)

Secrets are never read from the file. ``HAILER_MARIMO_TOKEN`` is the only
secret this module touches and it is kept in memory on the returned
:class:`~hailer.models.HailerConfig` only.

``validate()`` returns human-readable problems. Strings that start with
``"Warning:"`` are non-fatal (the CLI shows them but continues); everything
else should stop startup.
"""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hailer.errors import ConfigError
from hailer.models import HailerConfig, ModelConfig, ProviderConfig, WebConfig

CONFIG_FILENAMES: tuple[str, ...] = ("hailer.toml", ".config/hailer/hailer.toml")

DEFAULT_NOTEBOOK = "notebooks/analysis.py"
DEFAULT_DATA_DIR = "data"
DEFAULT_CONTEXT_DIR = ".config/hailer/context"
DEFAULT_SKILLS_DIR = ".config/hailer/skills"
DEFAULT_PROMPTS_DIR = ".config/hailer/prompts"
DEFAULT_LOG_LEVEL = "WARNING"

VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
VALID_REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")

# Known keys per section; anything else is reported by validate() as a warning.
_KNOWN_TOP = {"hailer", "model", "model_providers", "web"}
_KNOWN_HAILER = {
    "notebook",
    "data_dir",
    "marimo_url",
    "context_dir",
    "skills_dir",
    "prompts_dir",
    "max_tool_output_chars",
    "max_context_bytes",
    "codex_home",
    "log_level",
}
_KNOWN_MODEL = {"name", "provider", "reasoning_effort"}
_KNOWN_PROVIDER = {
    "base_url",
    "wire_api",
    "env_key",
    "requires_openai_auth",
    "name",
    "http_headers",
    "env_http_headers",
    "query_params",
}
_KNOWN_WEB = {"allowed_domains", "max_page_bytes", "allow_shell_network"}

# A domain rule: optional "*." (subdomains only) or "**." (apex + subdomains)
# prefix, then DNS labels. "localhost" and IPv4 literals are also accepted.
_DOMAIN_RE = re.compile(
    r"^(?:\*\*\.|\*\.)?"
    r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)"
    r"(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)*$"
)
_IPV4_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")

DEFAULT_CONFIG_TEMPLATE = """\
# Hailer project configuration.
# Relative paths are resolved against the workspace (the folder containing this file).
# Secrets never go in this file: API keys come from the OS credential store
# (`uv run hailer login <provider>`) or an environment variable named by `env_key`.

[hailer]
notebook = "notebooks/analysis.py"   # live marimo notebook Hailer works in
data_dir = "data"                    # where the monthly parquet files live
# marimo_url = "http://127.0.0.1:2718"  # optional; auto-discovered from the marimo registry when omitted
# context_dir = ".config/hailer/context"  # always-on context (*.md) sent with every session
# skills_dir  = ".config/hailer/skills"   # on-demand skills (<name>/SKILL.md)
# prompts_dir = ".config/hailer/prompts"  # reusable prompts (/prompt <name>)
# max_tool_output_chars = 12000        # cap on any single tool result sent to the model
# max_context_bytes = 24000            # cap on the concatenated context files
# log_level = "WARNING"                # or set HAILER_LOG_LEVEL / --verbose

[model]
name = "gpt-5.5"
provider = "openai"                  # "openai" uses your existing Codex login or OPENAI_API_KEY
# reasoning_effort = "medium"        # minimal | low | medium | high | xhigh

# A bespoke / internal endpoint. It must implement the OpenAI Responses API
# (streaming). If it only offers Chat Completions, run a translating proxy and
# point base_url at that.
#
# [model_providers.internal]
# base_url             = "https://llm.example.internal/v1"
# wire_api             = "responses"
# env_key              = "INTERNAL_MODEL_API_KEY"   # env var name; value from `hailer login internal` or the shell
# requires_openai_auth = false
# name                 = "Internal"
# http_headers         = { "X-Team" = "risk-analytics" }
# env_http_headers     = { "X-Client-Id" = "INTERNAL_CLIENT_ID" }
# query_params         = { "api-version" = "2025-04-01-preview" }

# Websites the agent may read for context. Without this section the agent has
# no internet access at all. "*.d" = subdomains only, "**.d" = apex + subdomains.
#
# [web]
# allowed_domains = ["docs.pola.rs", "duckdb.org", "**.bankofengland.co.uk"]
# max_page_bytes = 200000
# allow_shell_network = false        # also let shell commands reach the same domains
"""


# --------------------------------------------------------------------------- #
# Locating files
# --------------------------------------------------------------------------- #


def _has_marker(directory: Path) -> bool:
    if (directory / "pyproject.toml").is_file():
        return True
    return any((directory / name).is_file() for name in CONFIG_FILENAMES)


def find_workspace(start: Path | None = None) -> Path:
    """Return the workspace root: ``start`` (or cwd) or the first parent holding a marker file."""
    origin = Path(start) if start is not None else Path.cwd()
    origin = origin.resolve()
    for candidate in (origin, *origin.parents):
        if _has_marker(candidate):
            return candidate
    return origin


def find_config_path(
    workspace: Path,
    explicit: Path | None = None,
    env: Mapping[str, str] = os.environ,
) -> Path | None:
    """Return the config file to load, or ``None`` when defaults should be used.

    An explicitly requested file (argument or ``HAILER_CONFIG``) must exist.
    """
    workspace = Path(workspace)
    requested: Path | None = None
    source = ""
    if explicit is not None:
        requested, source = Path(explicit), "--config"
    elif env.get("HAILER_CONFIG"):
        requested, source = Path(env["HAILER_CONFIG"]), "HAILER_CONFIG"
    if requested is not None:
        requested = requested.expanduser()
        if not requested.is_absolute():
            requested = workspace / requested
        if not requested.is_file():
            raise ConfigError(
                f"Config file not found: {requested} (from {source}).",
                hint="Check the path, or run `uv run hailer init` to create a config.",
            )
        return requested.resolve()
    for name in CONFIG_FILENAMES:
        candidate = workspace / name
        if candidate.is_file():
            return candidate.resolve()
    return None


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Could not read config file {path}: {exc.strerror or exc}", hint="Check the file permissions.") from exc
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        match = re.search(r"at line (\d+)", str(exc))
        where = f" line {match.group(1)}" if match else ""
        raise ConfigError(
            f"Invalid TOML in {path}{where}: {exc}",
            hint="Fix the syntax error shown above; every value needs quotes unless it is a number, boolean or table.",
        ) from exc


def _section(data: Mapping[str, Any], name: str, path: Path | None) -> dict[str, Any]:
    value = data.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(
            f"[{name}] in {path} must be a table, not {type(value).__name__}.",
            hint=f"Write it as a TOML table: [{name}] followed by key = value lines.",
        )
    return value


def _str(table: Mapping[str, Any], key: str, section: str, path: Path | None, default: str | None = None) -> str | None:
    value = table.get(key, default)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(
            f"[{section}].{key} in {path} must be a string, not {type(value).__name__}.",
            hint=f'Quote the value, e.g. {key} = "...".',
        )
    return value


def _int(table: Mapping[str, Any], key: str, section: str, path: Path | None, default: int) -> int:
    value = table.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(
            f"[{section}].{key} in {path} must be an integer, not {type(value).__name__}.",
            hint=f"Write a plain number, e.g. {key} = {default}.",
        )
    return value


def _bool(table: Mapping[str, Any], key: str, section: str, path: Path | None, default: bool) -> bool:
    value = table.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(
            f"[{section}].{key} in {path} must be true or false, not {type(value).__name__}.",
            hint=f"Write {key} = true or {key} = false (no quotes).",
        )
    return value


def _str_map(table: Mapping[str, Any], key: str, section: str, path: Path | None) -> dict[str, str]:
    value = table.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        raise ConfigError(
            f"[{section}].{key} in {path} must be a table of string values.",
            hint=f'Example: {key} = {{ "Header-Name" = "value" }}.',
        )
    return dict(value)


def _str_list(table: Mapping[str, Any], key: str, section: str, path: Path | None) -> list[str]:
    value = table.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(
            f"[{section}].{key} in {path} must be a list of strings.",
            hint=f'Example: {key} = ["docs.pola.rs", "duckdb.org"].',
        )
    return list(value)


def _resolve(workspace: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    return candidate.resolve()


def _clean_url(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value.rstrip("/") or None


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_config(
    workspace: Path | None = None,
    config_path: Path | None = None,
    env: Mapping[str, str] = os.environ,
) -> HailerConfig:
    """Build a fully resolved :class:`HailerConfig` from file + environment."""
    if workspace is not None:
        ws = Path(workspace).expanduser().resolve()
    elif env.get("HAILER_WORKSPACE"):
        ws = Path(env["HAILER_WORKSPACE"]).expanduser().resolve()
    else:
        ws = find_workspace()

    path = find_config_path(ws, config_path, env)
    data: dict[str, Any] = _read_toml(path) if path is not None else {}

    hailer_tbl = _section(data, "hailer", path)
    model_tbl = _section(data, "model", path)
    providers_tbl = _section(data, "model_providers", path)
    web_tbl = _section(data, "web", path)

    notebook = env.get("HAILER_NOTEBOOK") or _str(hailer_tbl, "notebook", "hailer", path, DEFAULT_NOTEBOOK)
    data_dir = env.get("HAILER_DATA_DIR") or _str(hailer_tbl, "data_dir", "hailer", path, DEFAULT_DATA_DIR)
    context_dir = _str(hailer_tbl, "context_dir", "hailer", path, DEFAULT_CONTEXT_DIR)
    skills_dir = _str(hailer_tbl, "skills_dir", "hailer", path, DEFAULT_SKILLS_DIR)
    prompts_dir = _str(hailer_tbl, "prompts_dir", "hailer", path, DEFAULT_PROMPTS_DIR)
    marimo_url = _clean_url(env.get("HAILER_MARIMO_URL") or _str(hailer_tbl, "marimo_url", "hailer", path))
    codex_home_raw = env.get("HAILER_CODEX_HOME") or _str(hailer_tbl, "codex_home", "hailer", path)
    log_level = (env.get("HAILER_LOG_LEVEL") or _str(hailer_tbl, "log_level", "hailer", path, DEFAULT_LOG_LEVEL) or DEFAULT_LOG_LEVEL).upper()
    max_tool_output_chars = _int(hailer_tbl, "max_tool_output_chars", "hailer", path, 12_000)
    max_context_bytes = _int(hailer_tbl, "max_context_bytes", "hailer", path, 24_000)

    model_name = env.get("HAILER_MODEL") or _str(model_tbl, "name", "model", path, ModelConfig().name) or ModelConfig().name
    model_provider = env.get("HAILER_MODEL_PROVIDER") or _str(model_tbl, "provider", "model", path, ModelConfig().provider) or ModelConfig().provider
    reasoning_effort = _str(model_tbl, "reasoning_effort", "model", path, ModelConfig().reasoning_effort)
    if reasoning_effort is not None and not reasoning_effort.strip():
        reasoning_effort = None

    providers: dict[str, ProviderConfig] = {}
    for provider_id, raw in providers_tbl.items():
        section = f"model_providers.{provider_id}"
        if not isinstance(raw, dict):
            raise ConfigError(
                f"[{section}] in {path} must be a table.",
                hint=f"Write it as [{section}] followed by base_url = ..., env_key = ... lines.",
            )
        providers[provider_id] = ProviderConfig(
            id=provider_id,
            base_url=_clean_url(_str(raw, "base_url", section, path)),
            wire_api=_str(raw, "wire_api", section, path, "responses") or "responses",
            env_key=_str(raw, "env_key", section, path),
            requires_openai_auth=_bool(raw, "requires_openai_auth", section, path, False),
            name=_str(raw, "name", section, path),
            http_headers=_str_map(raw, "http_headers", section, path),
            env_http_headers=_str_map(raw, "env_http_headers", section, path),
            query_params=_str_map(raw, "query_params", section, path),
        )

    web = WebConfig(
        allowed_domains=tuple(d.strip().lower() for d in _str_list(web_tbl, "allowed_domains", "web", path) if d.strip()),
        max_page_bytes=_int(web_tbl, "max_page_bytes", "web", path, WebConfig().max_page_bytes),
        allow_shell_network=_bool(web_tbl, "allow_shell_network", "web", path, False),
    )

    return HailerConfig(
        workspace=ws,
        notebook=_resolve(ws, notebook or DEFAULT_NOTEBOOK),
        data_dir=_resolve(ws, data_dir or DEFAULT_DATA_DIR),
        context_dir=_resolve(ws, context_dir or DEFAULT_CONTEXT_DIR),
        skills_dir=_resolve(ws, skills_dir or DEFAULT_SKILLS_DIR),
        prompts_dir=_resolve(ws, prompts_dir or DEFAULT_PROMPTS_DIR),
        model=ModelConfig(name=model_name, provider=model_provider, reasoning_effort=reasoning_effort),
        providers=providers,
        web=web,
        marimo_url=marimo_url,
        marimo_token=(env.get("HAILER_MARIMO_TOKEN") or None),
        codex_home=_resolve(ws, codex_home_raw) if codex_home_raw else None,
        log_level=log_level,
        config_path=path,
        max_tool_output_chars=max_tool_output_chars,
        max_context_bytes=max_context_bytes,
    )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _domain_ok(rule: str) -> bool:
    rule = rule.lower()
    bare = rule.removeprefix("**.").removeprefix("*.")
    if bare == "localhost":
        return not rule.startswith("*")
    if _IPV4_RE.match(bare):
        return not rule.startswith("*")
    return bool(_DOMAIN_RE.match(rule))


def _unknown_key_warnings(path: Path) -> list[str]:
    try:
        data = _read_toml(path)
    except ConfigError:
        return []
    warnings: list[str] = []

    def check(table: Mapping[str, Any], known: set[str], section: str) -> None:
        for key in table:
            if key not in known:
                warnings.append(f"Warning: unknown key [{section}].{key} in {path} is ignored.")

    for key in data:
        if key not in _KNOWN_TOP:
            warnings.append(f"Warning: unknown section [{key}] in {path} is ignored.")
    hailer_tbl = data.get("hailer") or {}
    if isinstance(hailer_tbl, dict):
        if "marimo_token" in hailer_tbl:
            warnings.append(
                f"Warning: [hailer].marimo_token in {path} is ignored; secrets do not belong in the file. "
                "Set the HAILER_MARIMO_TOKEN environment variable instead."
            )
        check({k: v for k, v in hailer_tbl.items() if k != "marimo_token"}, _KNOWN_HAILER, "hailer")
    model_tbl = data.get("model") or {}
    if isinstance(model_tbl, dict):
        check(model_tbl, _KNOWN_MODEL, "model")
    providers_tbl = data.get("model_providers") or {}
    if isinstance(providers_tbl, dict):
        for pid, raw in providers_tbl.items():
            if isinstance(raw, dict):
                if any(k in raw for k in ("api_key", "key", "token", "secret", "bearer_token")):
                    warnings.append(
                        f"Warning: [model_providers.{pid}] in {path} contains a secret-looking key; it is ignored. "
                        f"Use env_key plus `hailer login {pid}` instead."
                    )
                check({k: v for k, v in raw.items() if k not in ("api_key", "key", "token", "secret", "bearer_token")}, _KNOWN_PROVIDER, f"model_providers.{pid}")
    web_tbl = data.get("web") or {}
    if isinstance(web_tbl, dict):
        check(web_tbl, _KNOWN_WEB, "web")
    return warnings


def validate(config: HailerConfig) -> list[str]:
    """Return problems with the configuration. ``"Warning:"``-prefixed entries are non-fatal."""
    problems: list[str] = []

    if not config.notebook.is_file():
        problems.append(
            f"Notebook not found: {config.notebook}. "
            "Set [hailer].notebook (or HAILER_NOTEBOOK) to an existing marimo notebook, "
            "or run `uv run hailer init` to create the default one."
        )
    if not config.data_dir.is_dir():
        problems.append(
            f"Warning: data directory not found: {config.data_dir}. "
            "Create it (or set [hailer].data_dir) before loading parquet files."
        )

    if config.log_level not in VALID_LOG_LEVELS:
        problems.append(f"Invalid log level {config.log_level!r}; use one of {', '.join(VALID_LOG_LEVELS)}.")

    effort = config.model.reasoning_effort
    if effort is not None and effort not in VALID_REASONING_EFFORTS:
        problems.append(
            f"Invalid [model].reasoning_effort {effort!r}; use one of {', '.join(VALID_REASONING_EFFORTS)}."
        )
    if not config.model.name.strip():
        problems.append("[model].name is empty; set the model id your provider expects.")

    pid = config.model.provider
    if pid not in config.providers and pid != "openai":
        problems.append(
            f"Model provider {pid!r} is not declared. Add a [model_providers.{pid}] table with base_url, "
            'wire_api = "responses" and env_key, or set [model].provider = "openai".'
        )

    for provider in config.providers.values():
        section = f"[model_providers.{provider.id}]"
        if provider.wire_api != "responses":
            problems.append(
                f'{section}.wire_api must be "responses" (got {provider.wire_api!r}); '
                "Codex only speaks the OpenAI Responses API. If the endpoint offers Chat Completions only, "
                "run a translating proxy and point base_url at it."
            )
        if provider.is_builtin_openai:
            continue
        if not provider.base_url:
            problems.append(f'{section}.base_url is missing; set it to the endpoint, e.g. "https://llm.example.internal/v1".')
        elif not provider.base_url.lower().startswith(("http://", "https://")):
            problems.append(f"{section}.base_url must start with http:// or https:// (got {provider.base_url!r}).")
        if not provider.requires_openai_auth and not provider.env_key:
            problems.append(
                f'{section}.env_key is missing; name the environment variable that holds the API key, e.g. env_key = "{provider.id.upper()}_API_KEY". '
                f"Store the value with `uv run hailer login {provider.id}`."
            )
        for key in ("http_headers", "env_http_headers", "query_params"):
            for header_name in getattr(provider, key):
                if not header_name.strip():
                    problems.append(f"{section}.{key} contains an empty name.")

    for rule in config.web.allowed_domains:
        if rule == "*" or rule == "**":
            problems.append(
                f"[web].allowed_domains contains {rule!r}, which would allow every site; list explicit domains instead."
            )
        elif not _domain_ok(rule):
            problems.append(
                f"[web].allowed_domains entry {rule!r} is not a valid domain rule. "
                'Use "example.com", "*.example.com" (subdomains only) or "**.example.com" (apex plus subdomains).'
            )
    if config.web.max_page_bytes <= 0:
        problems.append("[web].max_page_bytes must be a positive integer.")
    if config.max_tool_output_chars <= 0:
        problems.append("[hailer].max_tool_output_chars must be a positive integer.")
    if config.max_context_bytes <= 0:
        problems.append("[hailer].max_context_bytes must be a positive integer.")

    if config.config_path is not None and config.config_path.is_file():
        problems.extend(_unknown_key_warnings(config.config_path))

    return problems


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def write_default_config(path: Path, *, overwrite: bool = False) -> None:
    """Write :data:`DEFAULT_CONFIG_TEMPLATE` to ``path`` (parents created)."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise ConfigError(
            f"Config file already exists: {path}",
            hint="Edit it in place, or pass --force to replace it with the default template.",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEFAULT_CONFIG_TEMPLATE, encoding="utf-8", newline="\n")
