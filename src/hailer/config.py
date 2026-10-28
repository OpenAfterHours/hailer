"""Configuration loading for Hailer.

Sources, lowest to highest precedence:

1. built-in defaults (in the project layout, folders inside ``.hailer/``)
2. the project config file (``hailer.toml``, ``.hailer/hailer.toml`` or ``.config/hailer/hailer.toml``)
3. this machine's ``.hailer/local.toml`` (never committed: data folder paths, a personal provider)
4. environment variables (``HAILER_*``)

Secrets are never read from the file (API keys come from the environment or the OS credential
store, see :mod:`hailer.secrets`).

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
from hailer.statedir import PROJECT_CONFIG_NAME, STATE_DIRNAME, is_project_layout, local_config_path, state_dir
from hailer.models import (
    KERNEL_RUNTIME_DOCKER,
    VALID_KERNEL_RUNTIMES,
    VALID_WIRE_APIS,
    WIRE_API_RESPONSES,
    CodeChecksConfig,
    HailerConfig,
    KernelConfig,
    ModelConfig,
    ProviderConfig,
    WebConfig,
)

#: The config file, looked up in this order in the workspace. ``.hailer/hailer.toml`` is the project
#: layout (Hailer inside another project: its notebooks, context, skills and prompts in ``.hailer/``).
PROJECT_CONFIG_FILENAME = f"{STATE_DIRNAME}/{PROJECT_CONFIG_NAME}"
CONFIG_FILENAMES: tuple[str, ...] = ("hailer.toml", PROJECT_CONFIG_FILENAME, ".config/hailer/hailer.toml")

DEFAULT_NOTEBOOK = "notebooks/analysis.py"
DEFAULT_DATA_DIR = "data"
DEFAULT_CONTEXT_DIR = ".config/hailer/context"
DEFAULT_SKILLS_DIR = ".config/hailer/skills"
DEFAULT_PROMPTS_DIR = ".config/hailer/prompts"
#: The same defaults in the project layout: everything of Hailer's in one folder.
PROJECT_DEFAULTS: dict[str, str] = {
    "notebook": ".hailer/notebooks/analysis.py",
    "data_dir": ".hailer/data",
    "context_dir": ".hailer/context",
    "skills_dir": ".hailer/skills",
    "prompts_dir": ".hailer/prompts",
}
DEFAULT_LOG_LEVEL = "WARNING"

VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
VALID_REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")

# Known keys per section; anything else is reported by validate() as a warning.
_KNOWN_TOP = {"hailer", "model", "model_providers", "web", "kernel", "checks"}
_KNOWN_HAILER = {
    "notebook",
    "notebooks_dir",
    "data_dir",
    "context_dir",
    "skills_dir",
    "prompts_dir",
    "max_tool_output_chars",
    "max_context_bytes",
    "log_level",
}
_KNOWN_MODEL = {"name", "provider", "reasoning_effort", "summarize_after_tokens"}
_KNOWN_PROVIDER = {
    "base_url",
    "wire_api",
    "stream",
    "stream_options",
    "env_key",
    "name",
    "http_headers",
    "env_http_headers",
    "query_params",
}
_KNOWN_WEB = {"allowed_domains", "max_page_bytes"}
_KNOWN_KERNEL = {"runtime", "image", "memory", "cpus", "network"}
_KNOWN_CHECKS = {"lint", "typecheck", "format"}

# docker --memory: a number with an optional b/k/m/g unit ("4g", "512m", "1.5g").
_MEMORY_RE = re.compile(r"^(\d+(?:\.\d+)?)[bkmg]?$", re.IGNORECASE)

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
# (`uvx hailer login <provider>`) or an environment variable named by `env_key`.

[hailer]
notebook = "notebooks/analysis.py"   # default notebook; the chat can create and open others
# notebooks_dir = "notebooks"        # folder for notebooks created or opened from the chat (default: the notebook's folder)
data_dir = "data"                    # the data files to analyse (CSV, Parquet, JSON, ...)
# context_dir = ".config/hailer/context"  # always-on context (*.md) sent with every session
# skills_dir  = ".config/hailer/skills"   # on-demand skills (<name>/SKILL.md)
# prompts_dir = ".config/hailer/prompts"  # reusable prompts (/prompt <name>)
# max_tool_output_chars = 12000        # cap on any single tool result sent to the model
# max_context_bytes = 24000            # cap on the concatenated context files
# log_level = "WARNING"                # or set HAILER_LOG_LEVEL / --verbose

[model]
name = "gpt-5.5"
provider = "openai"                  # "openai" = api.openai.com with OPENAI_API_KEY (`uvx hailer login openai`)
# reasoning_effort = "medium"        # minimal | low | medium | high | xhigh; "" sends no reasoning
#                                      effort at all (for endpoints that reject the field)
# summarize_after_tokens = 100000    # summarise older turns past this size; lower it for small
#                                      context windows, 0 turns it off

# A bespoke / internal endpoint. wire_api picks the protocol the endpoint speaks:
#   "responses" - the OpenAI Responses API (POST {base_url}/responses)
#   "chat"      - Chat Completions (POST {base_url}/chat/completions)
#
# [model_providers.internal]
# base_url             = "https://llm.example.internal/v1"
# wire_api             = "responses"                      # or "chat"
# stream               = true                             # false if the endpoint rejects stream = true
# stream_options       = true                             # false omits stream_options (token counts may be lost)
# env_key              = "INTERNAL_MODEL_API_KEY"   # env var name; value from `hailer login internal` or the shell
# name                 = "Internal"
# http_headers         = { "X-Team" = "data-analytics" }
# env_http_headers     = { "X-Client-Id" = "INTERNAL_CLIENT_ID" }
# query_params         = { "api-version" = "2025-04-01-preview" }

# Websites the agent may read for context. Without this section the agent has
# no internet access at all. "*.d" = subdomains only, "**.d" = apex + subdomains.
#
# [web]
# allowed_domains = ["docs.pola.rs", "duckdb.org", "**.marimo.io"]
# max_page_bytes = 200000

# Where notebook code runs. "docker" (the default): marimo runs in a Linux container with a
# notebooks folder of its own (your notebooks are copied in and back) that sees only the data
# folder, read-only, with no network; it needs Docker Desktop or Docker Engine, running.
# "unsafe-local": marimo runs in Hailer's own Python, as you, with your files and network (not
# isolated); write it only if you accept that. HAILER_KERNEL overrides runtime for one run.

[kernel]
runtime = "docker"                   # "docker" or "unsafe-local"
# image   = ""                       # docker: default ghcr.io/openafterhours/hailer-kernel:<kernel contract>
# memory  = "4g"                     # docker: container memory limit (no swap on top)
# cpus    = 2                        # docker: container CPU limit
# network = false                    # docker: true lets notebook code reach the internet and this machine

# The cells the agent writes. After each notebook change Hailer checks the changed cells with ruff
# (lint) and ty (types) and hands the findings back to the agent; code mode formats new and edited
# cells with ruff before marimo applies them. All three are on by default.
#
# [checks]
# lint      = true                   # ruff: undefined names, unused variables, likely bugs
# typecheck = true                   # ty: wrong attributes, arguments and types across cells
# format    = true                   # ruff format, with marimo's line length
"""


# --------------------------------------------------------------------------- #
# Locating files
# --------------------------------------------------------------------------- #


def _has_config(directory: Path) -> bool:
    return any((directory / name).is_file() for name in CONFIG_FILENAMES)


def _has_marker(directory: Path) -> bool:
    return (directory / "pyproject.toml").is_file() or _has_config(directory)


def _declares_uv_workspace(pyproject: Path) -> bool:
    """Whether ``pyproject`` has a ``[tool.uv.workspace]`` table (the root of a uv workspace)."""
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return False
    tool = data.get("tool")
    uv = tool.get("uv") if isinstance(tool, dict) else None
    return isinstance(uv, dict) and isinstance(uv.get("workspace"), dict)


def _repository_root(origin: Path) -> Path | None:
    """The nearest folder at or above ``origin`` holding ``.git`` (a folder, or a file in a worktree)."""
    return next((candidate for candidate in (origin, *origin.parents) if (candidate / ".git").exists()), None)


def find_workspace(start: Path | None = None) -> Path:
    """Return the workspace root for ``start`` (or cwd).

    Inside a git repository, searched up to the repository's root: the nearest folder holding a
    Hailer config (:data:`CONFIG_FILENAMES`); else the nearest uv workspace root (a
    ``pyproject.toml`` with ``[tool.uv.workspace]``), so ``uv run hailer`` in a workspace member
    belongs to the whole repository; else the nearest folder with a ``pyproject.toml``. Outside a
    repository: the first folder at or above ``start`` holding a ``pyproject.toml`` or a config.
    Either way ``start`` itself when nothing matches.
    """
    origin = Path(start) if start is not None else Path.cwd()
    origin = origin.resolve()
    repository = _repository_root(origin)
    if repository is None:
        return next((candidate for candidate in (origin, *origin.parents) if _has_marker(candidate)), origin)
    first_project: Path | None = None
    uv_root: Path | None = None
    for candidate in (origin, *origin.parents):
        if _has_config(candidate):
            return candidate
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file():
            first_project = first_project or candidate
            if uv_root is None and _declares_uv_workspace(pyproject):
                uv_root = candidate
        if candidate == repository:
            break
    return uv_root or first_project or origin


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
                hint="Check the path, or run `uvx hailer init` to create a config.",
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


def _section(data: Mapping[str, Any], name: str, path: Path | str | None) -> dict[str, Any]:
    value = data.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(
            f"[{name}] in {path} must be a table, not {type(value).__name__}.",
            hint=f"Write it as a TOML table: [{name}] followed by key = value lines.",
        )
    return value


def _written(value: object) -> str:
    """How a value looks in the TOML file, for an error message ("2" for the text 2, 2 for the number)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, (int, float)):
        return repr(value)
    return f"a {type(value).__name__}"


def _str(table: Mapping[str, Any], key: str, section: str, path: Path | str | None, default: str | None = None) -> str | None:
    value = table.get(key, default)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(
            f"[{section}].{key} in {path} must be a string, not {_written(value)} ({type(value).__name__}).",
            hint=f'Quote the value, e.g. {key} = "...".',
        )
    return value


def _int(table: Mapping[str, Any], key: str, section: str, path: Path | str | None, default: int) -> int:
    value = table.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(
            f"[{section}].{key} in {path} must be an integer, not {_written(value)} ({type(value).__name__}).",
            hint=f"Write a plain number, e.g. {key} = {default}.",
        )
    return value


def _number(table: Mapping[str, Any], key: str, section: str, path: Path | str | None, default: float) -> float:
    value = table.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            f"[{section}].{key} in {path} must be a number, not {_written(value)} ({type(value).__name__}).",
            hint=f"Write a plain number without quotes, e.g. {key} = {default:g}.",
        )
    return float(value)


def _bool(table: Mapping[str, Any], key: str, section: str, path: Path | str | None, default: bool) -> bool:
    value = table.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(
            f"[{section}].{key} in {path} must be true or false, not {_written(value)} ({type(value).__name__}).",
            hint=f"Write {key} = true or {key} = false (no quotes).",
        )
    return value


def _str_map(table: Mapping[str, Any], key: str, section: str, path: Path | str | None) -> dict[str, str]:
    value = table.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        raise ConfigError(
            f"[{section}].{key} in {path} must be a table of string values.",
            hint=f'Example: {key} = {{ "Header-Name" = "value" }}.',
        )
    return dict(value)


def _str_list(
    table: Mapping[str, Any], key: str, section: str, path: Path | str | None, example: str = '["docs.pola.rs", "duckdb.org"]'
) -> list[str]:
    value = table.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        got = "a list with unquoted items" if isinstance(value, list) else _written(value)
        raise ConfigError(
            f"[{section}].{key} in {path} must be a list of strings, not {got}.",
            hint=f"Example: {key} = {example}.",
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


def _merge_tables(base: Mapping[str, Any], top: Mapping[str, Any]) -> dict[str, Any]:
    """``base`` with ``top`` on top: tables merge key by key, any other value replaces."""
    merged = dict(base)
    for key, value in top.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_tables(merged[key], value)
        else:
            merged[key] = value
    return merged


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
    local = local_config_path(ws)
    local_path = local.resolve() if local.is_file() and local.resolve() != path else None
    where: Path | str | None = path
    if local_path is not None:
        data = _merge_tables(data, _read_toml(local_path))
        where = f"{path} or {local_path}" if path is not None else local_path
    defaults = PROJECT_DEFAULTS if is_project_layout(ws) else {}

    hailer_tbl = _section(data, "hailer", where)
    model_tbl = _section(data, "model", where)
    providers_tbl = _section(data, "model_providers", where)
    web_tbl = _section(data, "web", where)
    kernel_tbl = _section(data, "kernel", where)
    checks_tbl = _section(data, "checks", where)

    notebook = _str(hailer_tbl, "notebook", "hailer", where, defaults.get("notebook", DEFAULT_NOTEBOOK))
    notebooks_dir = _str(hailer_tbl, "notebooks_dir", "hailer", where)
    data_dir = _str(hailer_tbl, "data_dir", "hailer", where, defaults.get("data_dir", DEFAULT_DATA_DIR))
    context_dir = _str(hailer_tbl, "context_dir", "hailer", where, defaults.get("context_dir", DEFAULT_CONTEXT_DIR))
    skills_dir = _str(hailer_tbl, "skills_dir", "hailer", where, defaults.get("skills_dir", DEFAULT_SKILLS_DIR))
    prompts_dir = _str(hailer_tbl, "prompts_dir", "hailer", where, defaults.get("prompts_dir", DEFAULT_PROMPTS_DIR))
    log_level = (env.get("HAILER_LOG_LEVEL") or _str(hailer_tbl, "log_level", "hailer", where, DEFAULT_LOG_LEVEL) or DEFAULT_LOG_LEVEL).upper()
    max_tool_output_chars = _int(hailer_tbl, "max_tool_output_chars", "hailer", where, 12_000)
    max_context_bytes = _int(hailer_tbl, "max_context_bytes", "hailer", where, 24_000)

    model_name = env.get("HAILER_MODEL") or _str(model_tbl, "name", "model", where, ModelConfig().name) or ModelConfig().name
    model_provider = env.get("HAILER_MODEL_PROVIDER") or _str(model_tbl, "provider", "model", where, ModelConfig().provider) or ModelConfig().provider
    reasoning_effort = _str(model_tbl, "reasoning_effort", "model", where, ModelConfig().reasoning_effort)
    if reasoning_effort is not None and not reasoning_effort.strip():
        reasoning_effort = None
    summarize_after_tokens = _int(model_tbl, "summarize_after_tokens", "model", where, ModelConfig().summarize_after_tokens)

    providers: dict[str, ProviderConfig] = {}
    for provider_id, raw in providers_tbl.items():
        section = f"model_providers.{provider_id}"
        if not isinstance(raw, dict):
            raise ConfigError(
                f"[{section}] in {where} must be a table.",
                hint=f"Write it as [{section}] followed by base_url = ..., env_key = ... lines.",
            )
        providers[provider_id] = ProviderConfig(
            id=provider_id,
            base_url=_clean_url(_str(raw, "base_url", section, where)),
            wire_api=_str(raw, "wire_api", section, where, WIRE_API_RESPONSES) or WIRE_API_RESPONSES,
            stream=_bool(raw, "stream", section, where, True),
            stream_options=_bool(raw, "stream_options", section, where, True),
            env_key=_str(raw, "env_key", section, where) or ("OPENAI_API_KEY" if provider_id == "openai" else None),
            name=_str(raw, "name", section, where),
            http_headers=_str_map(raw, "http_headers", section, where),
            env_http_headers=_str_map(raw, "env_http_headers", section, where),
            query_params=_str_map(raw, "query_params", section, where),
        )

    web = WebConfig(
        allowed_domains=tuple(d.strip().lower() for d in _str_list(web_tbl, "allowed_domains", "web", where) if d.strip()),
        max_page_bytes=_int(web_tbl, "max_page_bytes", "web", where, WebConfig().max_page_bytes),
    )

    kernel_defaults = KernelConfig()
    runtime = env.get("HAILER_KERNEL") or _str(kernel_tbl, "runtime", "kernel", where, kernel_defaults.runtime)
    image = env.get("HAILER_KERNEL_IMAGE") or _str(kernel_tbl, "image", "kernel", where)
    kernel = KernelConfig(
        runtime=(runtime or "").strip().lower() or kernel_defaults.runtime,  # validate() reports unknown values
        image=(image or "").strip() or None,
        memory=(_str(kernel_tbl, "memory", "kernel", where, kernel_defaults.memory) or "").strip(),  # as written: messages quote it
        cpus=_number(kernel_tbl, "cpus", "kernel", where, kernel_defaults.cpus),
        network=_bool(kernel_tbl, "network", "kernel", where, kernel_defaults.network),
    )

    checks_defaults = CodeChecksConfig()
    checks = CodeChecksConfig(
        lint=_bool(checks_tbl, "lint", "checks", where, checks_defaults.lint),
        typecheck=_bool(checks_tbl, "typecheck", "checks", where, checks_defaults.typecheck),
        format=_bool(checks_tbl, "format", "checks", where, checks_defaults.format),
    )

    resolved_notebook = _resolve(ws, notebook or defaults.get("notebook", DEFAULT_NOTEBOOK))
    return HailerConfig(
        workspace=ws,
        notebook=resolved_notebook,
        notebooks_dir=_resolve(ws, notebooks_dir) if notebooks_dir else resolved_notebook.parent,
        data_dir=_resolve(ws, data_dir or defaults.get("data_dir", DEFAULT_DATA_DIR)),
        context_dir=_resolve(ws, context_dir or defaults.get("context_dir", DEFAULT_CONTEXT_DIR)),
        skills_dir=_resolve(ws, skills_dir or defaults.get("skills_dir", DEFAULT_SKILLS_DIR)),
        prompts_dir=_resolve(ws, prompts_dir or defaults.get("prompts_dir", DEFAULT_PROMPTS_DIR)),
        model=ModelConfig(
            name=model_name,
            provider=model_provider,
            reasoning_effort=reasoning_effort,
            summarize_after_tokens=summarize_after_tokens,
        ),
        providers=providers,
        web=web,
        kernel=kernel,
        checks=checks,
        log_level=log_level,
        config_path=path,
        local_config_path=local_path,
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

    def misplaced_kernel_keys(table: Mapping[str, Any], section: str) -> dict[str, Any]:
        """Warn about [kernel] settings that landed in ``section`` (a ``runtime`` line without a
        ``[kernel]`` line above it); the rest of ``table``."""
        for key in table:
            if key in _KNOWN_KERNEL and key not in _KNOWN_HAILER and key not in _KNOWN_MODEL:
                warnings.append(
                    f"Warning: [{section}].{key} in {path} is ignored: {key} belongs under [kernel]. Add a "
                    "[kernel] line above it (uvx hailer notebook --kernel <runtime> picks a runtime for one run "
                    "without editing the file)."
                )
        return {k: v for k, v in table.items() if not (k in _KNOWN_KERNEL and k not in _KNOWN_HAILER and k not in _KNOWN_MODEL)}

    for key in data:
        if key not in _KNOWN_TOP:
            warnings.append(f"Warning: unknown section [{key}] in {path} is ignored.")
    hailer_tbl = data.get("hailer") or {}
    if isinstance(hailer_tbl, dict):
        check(misplaced_kernel_keys(hailer_tbl, "hailer"), _KNOWN_HAILER, "hailer")
    model_tbl = data.get("model") or {}
    if isinstance(model_tbl, dict):
        check(misplaced_kernel_keys(model_tbl, "model"), _KNOWN_MODEL, "model")
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
    kernel_tbl = data.get("kernel") or {}
    if isinstance(kernel_tbl, dict):
        check(kernel_tbl, _KNOWN_KERNEL, "kernel")
    checks_tbl = data.get("checks") or {}
    if isinstance(checks_tbl, dict):
        check(checks_tbl, _KNOWN_CHECKS, "checks")
    return warnings


def _inside_or_equal(path: Path, root: Path) -> bool:
    """True when ``path`` is ``root`` or below it: compared resolved, and by file identity for the
    folders that exist (case-insensitive file systems, links)."""
    try:
        path, root = Path(path).resolve(), Path(root).resolve()
    except OSError:
        return False
    try:
        path.relative_to(root)
        return True
    except ValueError:
        pass
    if not root.is_dir():
        return False
    for candidate in (path, *path.parents):
        try:
            if candidate.is_dir() and os.path.samefile(candidate, root):
                return True
        except OSError:
            continue
    return False


def _same_folder(a: Path, b: Path) -> bool:
    return _inside_or_equal(a, b) and _inside_or_equal(b, a)


def _is_drive_root(folder: Path) -> bool:
    try:
        resolved = Path(folder).resolve()
    except OSError:
        return False
    return resolved.parent == resolved


#: Folders under the home folder that hold credentials (keys, cloud and cluster logins, other
#: tools' tokens): a docker kernel never mounts them, a folder inside them or a folder around them.
CREDENTIAL_FOLDERS: tuple[str, ...] = (".config", ".ssh", ".aws", ".azure", ".gnupg", ".docker", ".kube")
#: On Windows also these (applications keep their logins and browser profiles there).
CREDENTIAL_FOLDER_VARS: tuple[str, ...] = ("APPDATA", "LOCALAPPDATA")


def _on_windows() -> bool:
    return os.name == "nt"


def is_unc_path(path: Path | str) -> bool:
    """True for a Windows network-share path (``\\\\server\\share\\...``, also ``\\\\?\\UNC\\...``)."""
    import ntpath

    drive = ntpath.splitdrive(str(path))[0].replace("/", "\\")
    if drive.upper().startswith(("\\\\?\\", "\\\\.\\")):
        return drive[4:].upper().startswith("UNC\\")
    return drive.startswith("\\\\")


def _credential_folders(windows: bool) -> list[Path]:
    try:
        home = Path.home()
    except RuntimeError:  # no home folder to protect
        home = None
    folders = [home / name for name in CREDENTIAL_FOLDERS] if home is not None else []
    if windows:
        folders += [Path(os.environ[var]) for var in CREDENTIAL_FOLDER_VARS if os.environ.get(var)]
    return folders


def _in_temp(folder: Path) -> bool:
    """Inside the system's temporary folder (on Windows that is inside %LOCALAPPDATA%)."""
    import tempfile

    return _inside_or_equal(folder, Path(tempfile.gettempdir()))


def docker_mount_problems(config: HailerConfig, *, windows: bool | None = None) -> list[str]:
    """Every reason, in docker mode, not to mount the data folder (at most one; empty when it is
    fine). The one place for these rules: ``validate()`` reports them and
    :class:`~hailer.kernel_docker.DockerRuntime` refuses to start on them, on every start path
    (``--foreground`` never runs ``validate()``). The data folder is the only folder of this machine a docker kernel
    sees (read-only); its notebooks folder is the container's own, copied to and from the
    workspace's (:mod:`hailer.notebook_sync`).

    - Windows: a UNC path (a network share): Docker cannot mount it.
    - It may not be a whole drive, or be or contain the home folder, the workspace folder, Hailer's
      ``.hailer`` folder (the conversations), the config file or the context, skills and prompts
      folders: notebook code would read Hailer's own files.
    - It may not be, contain or sit inside a folder that holds credentials
      (:data:`CREDENTIAL_FOLDERS` under home; on Windows %APPDATA% and %LOCALAPPDATA%, outside the
      temporary folder).
    """
    windows = _on_windows() if windows is None else windows
    folder = config.data_dir
    workspace = Path(config.workspace)
    guarded: list[tuple[Path, str]] = [
        (workspace, "the workspace folder"),
        (workspace / STATE_DIRNAME, "Hailer's .hailer folder"),
        (state_dir(workspace), "Hailer's state folder"),
    ]
    try:
        guarded.insert(0, (Path.home(), "your home folder"))
    except RuntimeError:  # no home folder to protect
        pass
    for path in (config.config_path, config.local_config_path):
        if path is not None:
            guarded.append((path, f"the config file {path.name}"))
    guarded += [(config.context_dir, "the context folder"), (config.skills_dir, "the skills folder"), (config.prompts_dir, "the prompts folder")]
    docker = '[kernel] runtime = "docker"'
    fix = "Keep the data folder a folder of its own, such as data/, and point [hailer].data_dir at it."
    if windows and is_unc_path(folder):  # first: resolving a share's path can wait on the network
        return [
            f"The data folder {folder} is on a network share (UNC path), which Docker cannot mount. "
            "Copy it to a folder on a local disk and point [hailer].data_dir at it."
        ]
    if _is_drive_root(folder):
        return [f"The data folder ({folder}) is a whole drive. With {docker} it is mounted into the container. {fix}"]
    found = next(((target, name) for target, name in guarded if _inside_or_equal(target, folder)), None)
    if found is not None:
        target, name = found
        relation = "is" if _same_folder(target, folder) else "contains"
        return [
            f"The data folder ({folder}) {relation} {name}. With {docker} it is mounted into the container, "
            f"so notebook code could read Hailer's own files. {fix}"
        ]
    credentials = _credential_folders(windows)
    secret = next((target for target in credentials if _inside_or_equal(target, folder)), None)
    if secret is None:
        appdata = [Path(os.environ[var]) for var in CREDENTIAL_FOLDER_VARS if windows and os.environ.get(var)]
        secret = next((target for target in credentials if _inside_or_equal(folder, target)
                       and not (_in_temp(folder) and any(_same_folder(target, app) for app in appdata))), None)
    if secret is not None:
        relation = "is" if _same_folder(secret, folder) else ("contains" if _inside_or_equal(secret, folder) else "is inside")
        return [
            f"The data folder ({folder}) {relation} {secret}, a folder that holds credentials. With {docker} it is "
            f"mounted into the container, so notebook code could read them. {fix}"
        ]
    return []


def runtime_problem(runtime: str, where: str = "[kernel].runtime") -> str:
    """Why ``runtime``, read from ``where``, is not a kernel runtime. The retired ``local`` gets its
    own message: the unisolated runtime is only ever chosen by writing ``unsafe-local``."""
    value = f'{where} "{runtime}"' + (" (or HAILER_KERNEL)" if where == "[kernel].runtime" else "")
    if runtime == "local":  # the runtime's name before 2026-09-22: refused, never mapped
        verb = "Use" if where.startswith("--") else "Write"
        return (
            f"Invalid {value}: the runtime that runs notebook code on this machine is now called "
            f'"unsafe-local", because notebook code then runs as you, with your files and network. {verb} '
            '"unsafe-local" only if you accept that; otherwise use "docker", the isolated default.'
        )
    return (
        f'Invalid {value}; use "docker" (notebook code runs in an isolated container, the default) '
        'or "unsafe-local" (it runs as you, with your files and network: not isolated).'
    )


def _kernel_problems(config: HailerConfig) -> list[str]:
    kernel = config.kernel
    problems: list[str] = []
    if kernel.runtime not in VALID_KERNEL_RUNTIMES:
        problems.append(runtime_problem(kernel.runtime))
    memory = _MEMORY_RE.match(kernel.memory)
    if memory is None or float(memory.group(1)) <= 0:
        problems.append(
            f'Invalid [kernel].memory "{kernel.memory}"; write a size such as "4g" or "512m" '
            "(a number with an optional b, k, m or g unit, no space)."
        )
    if not kernel.cpus > 0:
        problems.append(f"Invalid [kernel].cpus {kernel.cpus:g}; use a number above 0, e.g. cpus = 2.")
    if kernel.runtime == KERNEL_RUNTIME_DOCKER:
        problems.extend(docker_mount_problems(config))
    return problems


def validate(config: HailerConfig) -> list[str]:
    """Return problems with the configuration. ``"Warning:"``-prefixed entries are non-fatal."""
    problems: list[str] = []

    if not config.notebook.is_file():
        problems.append(
            f"Notebook not found: {config.notebook}. "
            "Set [hailer].notebook to an existing marimo notebook, "
            "or run `uvx hailer init` to create the default one."
        )
    root = config.notebooks_root
    try:
        config.notebook.resolve().relative_to(root.resolve())
    except ValueError:
        problems.append(
            f"[hailer].notebook ({config.notebook}) is not inside [hailer].notebooks_dir ({root}). "
            "Point notebooks_dir at the folder that holds the notebook, or move the notebook into it."
        )
    if not root.is_dir():
        problems.append(
            f"Warning: notebooks folder not found: {root}. "
            "Create it (or set [hailer].notebooks_dir) so notebooks can be listed, created and opened from the chat."
        )
    try:
        folder_is_workspace = root.resolve() == Path(config.workspace).resolve()
    except OSError:
        folder_is_workspace = False
    if folder_is_workspace:
        problems.append(
            f"Warning: the notebooks folder is the workspace itself ({root}); marimo would scan the whole "
            "workspace (including .venv) for notebooks. Keep notebooks in a subfolder such as notebooks/ and "
            "point [hailer].notebook and [hailer].notebooks_dir there."
        )
    if not config.data_dir.is_dir():
        problems.append(
            f"Warning: data directory not found: {config.data_dir}. "
            "Create it (or set [hailer].data_dir) and put the files to analyse there."
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
    if config.model.summarize_after_tokens < 0:
        problems.append("[model].summarize_after_tokens must be 0 (off) or a positive number of tokens.")

    pid = config.model.provider
    if pid not in config.providers and pid != "openai":
        problems.append(
            f"Model provider {pid!r} is not declared. Add a [model_providers.{pid}] table with base_url, "
            'wire_api = "responses" or "chat", and env_key; or set [model].provider = "openai".'
        )

    for provider in config.providers.values():
        section = f"[model_providers.{provider.id}]"
        if provider.wire_api not in VALID_WIRE_APIS:
            problems.append(
                f'{section}.wire_api must be "responses" or "chat" (got {provider.wire_api!r}): '
                '"responses" when the endpoint implements the OpenAI Responses API, '
                '"chat" when it implements Chat Completions (POST <base_url>/chat/completions).'
            )
        if provider.is_builtin_openai:  # [model_providers.openai] without base_url: api.openai.com, OPENAI_API_KEY
            continue
        if not provider.base_url:
            problems.append(f'{section}.base_url is missing; set it to the endpoint, e.g. "https://llm.example.internal/v1".')
        elif not provider.base_url.lower().startswith(("http://", "https://")):
            problems.append(f"{section}.base_url must start with http:// or https:// (got {provider.base_url!r}).")
        if not provider.env_key:
            problems.append(
                f'{section}.env_key is missing; name the environment variable that holds the API key, e.g. env_key = "{provider.id.upper()}_API_KEY". '
                f"Store the value with `uvx hailer login {provider.id}`."
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

    problems.extend(_kernel_problems(config))

    for path in (config.config_path, config.local_config_path):
        if path is not None and path.is_file():
            problems.extend(_unknown_key_warnings(path))
    configs = [name for name in CONFIG_FILENAMES if (Path(config.workspace) / name).is_file()]
    if len(configs) > 1:
        problems.append(
            f"Warning: {config.workspace} has more than one Hailer config ({', '.join(configs)}); "
            f"{configs[0]} is used. Keep one of them."
        )

    return problems


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


_KERNEL_RUNTIME_LINE = f'[kernel]\nruntime = "{KERNEL_RUNTIME_DOCKER}"'

#: ``init``'s two layouts: a folder of its own (``hailer.toml``, ``notebooks/``, ``data/``,
#: ``.config/hailer/``) or Hailer inside another project (all of it in ``.hailer/``).
LAYOUT_STANDALONE = "standalone"
LAYOUT_PROJECT = "project"
VALID_LAYOUTS = (LAYOUT_STANDALONE, LAYOUT_PROJECT)

#: The template's lines that differ in the project layout: (start of the line up to its comment, new value).
_PROJECT_TEMPLATE_LINES: tuple[tuple[str, str], ...] = (
    ('notebook = "notebooks/analysis.py"   ', f'notebook = "{PROJECT_DEFAULTS["notebook"]}"'),
    ('# notebooks_dir = "notebooks"        ', '# notebooks_dir = ".hailer/notebooks"'),
    ('data_dir = "data"                    ', f'data_dir = "{PROJECT_DEFAULTS["data_dir"]}"'),
    ('# context_dir = ".config/hailer/context"  ', f'# context_dir = "{PROJECT_DEFAULTS["context_dir"]}"'),
    ('# skills_dir  = ".config/hailer/skills"   ', f'# skills_dir  = "{PROJECT_DEFAULTS["skills_dir"]}"'),
    ('# prompts_dir = ".config/hailer/prompts"  ', f'# prompts_dir = "{PROJECT_DEFAULTS["prompts_dir"]}"'),
)
_TEMPLATE_WORKSPACE_LINE = "# Relative paths are resolved against the workspace (the folder containing this file)."
_PROJECT_WORKSPACE_LINE = "# Relative paths are resolved against the workspace (the folder that holds .hailer/)."


def config_template(kernel: str | None = None, layout: str = LAYOUT_STANDALONE) -> str:
    """:data:`DEFAULT_CONFIG_TEMPLATE` (``runtime = "docker"``); with ``kernel`` ("docker" or
    "unsafe-local") its ``[kernel]`` section names that runtime instead. In the project layout the
    folders are the ones inside ``.hailer/``."""
    text = DEFAULT_CONFIG_TEMPLATE
    if layout not in VALID_LAYOUTS:
        raise ConfigError(f"Unknown layout {layout!r}; use {' or '.join(repr(name) for name in VALID_LAYOUTS)}.")
    if layout == LAYOUT_PROJECT:
        text = text.replace(_TEMPLATE_WORKSPACE_LINE, _PROJECT_WORKSPACE_LINE, 1)
        for start, value in _PROJECT_TEMPLATE_LINES:  # the comments stay in their column
            text = text.replace(start + "#", value.ljust(len(start) - 1) + " #", 1)
    if kernel is None:
        return text
    if kernel not in VALID_KERNEL_RUNTIMES:
        raise ConfigError(runtime_problem(kernel, "kernel runtime"))
    return text.replace(_KERNEL_RUNTIME_LINE, f'[kernel]\nruntime = "{kernel}"', 1)


def write_default_config(path: Path, *, overwrite: bool = False, kernel: str | None = None, layout: str = LAYOUT_STANDALONE) -> None:
    """Write :func:`config_template` (``kernel`` sets ``[kernel] runtime``) to ``path`` (parents created)."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise ConfigError(
            f"Config file already exists: {path}",
            hint="Edit it in place, or pass --force to replace it with the default template.",
        )
    text = config_template(kernel, layout)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
