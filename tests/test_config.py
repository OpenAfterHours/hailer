"""Tests for hailer.config: defaults, file parsing, env overrides, validation, template."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hailer.config import (
    CONFIG_FILENAMES,
    DEFAULT_CONFIG_TEMPLATE,
    config_template,
    docker_mount_problems,
    find_config_path,
    find_workspace,
    load_config,
    validate,
    write_default_config,
)
from hailer.errors import ConfigError
from hailer.models import HailerConfig, KernelConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _make_workspace(tmp_path: Path, config_text: str | None = None, *, notebook: bool = True, data: bool = True) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    if notebook:
        _write(ws / "notebooks" / "analysis.py", "import marimo\n")
    if data:
        (ws / "data").mkdir()
    if config_text is not None:
        _write(ws / "hailer.toml", config_text)
    return ws


def _errors(problems: list[str]) -> list[str]:
    return [p for p in problems if not p.startswith("Warning:")]


# --------------------------------------------------------------------------- #
# Defaults and discovery
# --------------------------------------------------------------------------- #


def test_defaults_without_config_file(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    cfg = load_config(workspace=ws, env={})
    assert isinstance(cfg, HailerConfig)
    assert cfg.config_path is None
    assert cfg.workspace == ws.resolve()
    assert cfg.notebook == (ws / "notebooks" / "analysis.py").resolve()
    assert cfg.notebooks_dir == (ws / "notebooks").resolve(), "defaults to the notebook's folder"
    assert cfg.notebooks_root == cfg.notebooks_dir
    assert cfg.data_dir == (ws / "data").resolve()
    assert cfg.context_dir == (ws / ".config" / "hailer" / "context").resolve()
    assert cfg.skills_dir == (ws / ".config" / "hailer" / "skills").resolve()
    assert cfg.prompts_dir == (ws / ".config" / "hailer" / "prompts").resolve()
    assert cfg.model.name == "gpt-5.5"
    assert cfg.model.provider == "openai"
    assert cfg.model.reasoning_effort == "medium"
    assert cfg.model.summarize_after_tokens == 100_000
    assert cfg.providers == {}
    assert cfg.web.allowed_domains == ()
    assert cfg.marimo_url is None
    assert cfg.marimo_token is None
    assert cfg.log_level == "WARNING"
    assert cfg.max_tool_output_chars == 12_000
    assert cfg.max_context_bytes == 24_000
    assert validate(cfg) == []


def test_builtin_openai_provider_is_synthesised(tmp_path: Path) -> None:
    cfg = load_config(workspace=_make_workspace(tmp_path), env={})
    provider = cfg.provider
    assert provider.id == "openai"
    assert provider.env_key == "OPENAI_API_KEY"
    assert provider.base_url is None and provider.wire_api == "responses"
    assert provider.is_builtin_openai


def test_declared_openai_provider_keeps_the_default_key_name(tmp_path: Path) -> None:
    """[model_providers.openai] without base_url only tunes the built-in endpoint (here: Chat Completions)."""
    ws = _make_workspace(tmp_path, '[model_providers.openai]\nwire_api = "chat"\nstream = false\n')
    cfg = load_config(workspace=ws, env={})
    provider = cfg.provider
    assert provider.is_builtin_openai and provider.env_key == "OPENAI_API_KEY"
    assert (provider.wire_api, provider.stream) == ("chat", False)
    assert not [p for p in validate(cfg) if not p.startswith("Warning:")]


def test_find_config_path_prefers_root_then_dot_config(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    assert find_config_path(ws, env={}) is None
    dot = _write(ws / ".config" / "hailer" / "hailer.toml", "[hailer]\n")
    assert find_config_path(ws, env={}) == dot.resolve()
    root = _write(ws / "hailer.toml", "[hailer]\n")
    assert find_config_path(ws, env={}) == root.resolve()
    assert CONFIG_FILENAMES[0] == "hailer.toml"


def test_find_config_path_explicit_and_env(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    custom = _write(ws / "custom.toml", "[hailer]\n")
    assert find_config_path(ws, explicit=Path("custom.toml"), env={}) == custom.resolve()
    assert find_config_path(ws, env={"HAILER_CONFIG": str(custom)}) == custom.resolve()
    with pytest.raises(ConfigError) as excinfo:
        find_config_path(ws, explicit=Path("missing.toml"), env={})
    assert "missing.toml" in str(excinfo.value)
    assert "--config" in str(excinfo.value)
    with pytest.raises(ConfigError) as excinfo:
        find_config_path(ws, env={"HAILER_CONFIG": str(ws / "nope.toml")})
    assert "HAILER_CONFIG" in str(excinfo.value)


def test_find_workspace_walks_up_to_marker(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path, "[hailer]\n")
    nested = ws / "a" / "b"
    nested.mkdir(parents=True)
    assert find_workspace(nested) == ws.resolve()
    lonely = tmp_path / "lonely"
    lonely.mkdir()
    assert find_workspace(lonely) == lonely.resolve()


def test_workspace_from_env(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    cfg = load_config(env={"HAILER_WORKSPACE": str(ws)})
    assert cfg.workspace == ws.resolve()


# --------------------------------------------------------------------------- #
# File parsing
# --------------------------------------------------------------------------- #


FULL_CONFIG = """
[hailer]
notebook = "nb/main.py"
notebooks_dir = "nb"
data_dir = "parquet"
marimo_url = "http://127.0.0.1:2718/"
context_dir = "ctx"
skills_dir = "sk"
prompts_dir = "pr"
max_tool_output_chars = 5000
max_context_bytes = 1000
log_level = "info"

[model]
name = "risk-analyst-v3"
provider = "internal"
reasoning_effort = "high"
summarize_after_tokens = 24000

[model_providers.internal]
base_url = "https://llm.example.internal/v1/"
wire_api = "responses"
env_key = "INTERNAL_MODEL_API_KEY"
name = "Internal"
http_headers = { "X-Team" = "risk" }
env_http_headers = { "X-Client-Id" = "INTERNAL_CLIENT_ID" }
query_params = { "api-version" = "2025-04-01-preview" }

[model_providers.azure]
base_url = "https://x.openai.azure.com/openai/v1"
env_key = "AZURE_OPENAI_API_KEY"

[web]
allowed_domains = ["Docs.pola.rs", "**.bankofengland.co.uk", " duckdb.org "]
max_page_bytes = 50000
"""


def test_full_file_is_mapped(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path, FULL_CONFIG)
    cfg = load_config(workspace=ws, env={})
    assert cfg.config_path == (ws / "hailer.toml").resolve()
    assert cfg.notebook == (ws / "nb" / "main.py").resolve()
    assert cfg.notebooks_dir == (ws / "nb").resolve()
    assert cfg.data_dir == (ws / "parquet").resolve()
    assert cfg.marimo_url == "http://127.0.0.1:2718"  # trailing slash stripped
    assert cfg.context_dir == (ws / "ctx").resolve()
    assert cfg.skills_dir == (ws / "sk").resolve()
    assert cfg.prompts_dir == (ws / "pr").resolve()
    assert cfg.max_tool_output_chars == 5000
    assert cfg.max_context_bytes == 1000
    assert cfg.log_level == "INFO"

    assert cfg.model.name == "risk-analyst-v3"
    assert cfg.model.provider == "internal"
    assert cfg.model.reasoning_effort == "high"
    assert cfg.model.summarize_after_tokens == 24_000

    internal = cfg.providers["internal"]
    assert internal.id == "internal"
    assert internal.base_url == "https://llm.example.internal/v1"
    assert internal.wire_api == "responses"
    assert internal.env_key == "INTERNAL_MODEL_API_KEY"
    assert internal.name == "Internal"
    assert internal.http_headers == {"X-Team": "risk"}
    assert internal.env_http_headers == {"X-Client-Id": "INTERNAL_CLIENT_ID"}
    assert internal.query_params == {"api-version": "2025-04-01-preview"}
    assert not internal.is_builtin_openai
    assert cfg.provider is internal

    assert internal.stream is True  # default

    azure = cfg.providers["azure"]
    assert azure.wire_api == "responses"  # default
    assert azure.http_headers == {}

    assert cfg.web.allowed_domains == ("docs.pola.rs", "**.bankofengland.co.uk", "duckdb.org")
    assert cfg.web.max_page_bytes == 50000


def test_absolute_paths_are_kept(tmp_path: Path) -> None:
    nb = _write(tmp_path / "elsewhere" / "nb.py", "import marimo\n")
    ws = _make_workspace(tmp_path, f'[hailer]\nnotebook = {str(nb)!r}\n'.replace("'", '"').replace("\\", "\\\\"))
    cfg = load_config(workspace=ws, env={})
    assert cfg.notebook == nb.resolve()


def test_invalid_toml_reports_file_and_line(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path, '[hailer]\nnotebook = "ok"\n[model]\nname = not quoted\n')
    with pytest.raises(ConfigError) as excinfo:
        load_config(workspace=ws, env={})
    message = str(excinfo.value)
    assert "hailer.toml" in message
    assert "line 4" in message
    assert excinfo.value.hint


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ('[hailer]\nnotebook = 3\n', "[hailer].notebook"),
        ('[hailer]\nmax_context_bytes = "big"\n', "[hailer].max_context_bytes"),
        ('[web]\nallowed_domains = "docs.pola.rs"\n', "[web].allowed_domains"),
        ('[model]\nsummarize_after_tokens = "lots"\n', "[model].summarize_after_tokens"),
        ('[model_providers.x]\nstream = "no"\n', "[model_providers.x].stream"),
        ('[model_providers.x]\nhttp_headers = { a = 1 }\n', "[model_providers.x].http_headers"),
        ('[model_providers]\nx = "nope"\n', "[model_providers.x]"),
        ('hailer = "nope"\n', "[hailer]"),
    ],
)
def test_wrong_types_raise_config_error(tmp_path: Path, text: str, fragment: str) -> None:
    ws = _make_workspace(tmp_path, text)
    with pytest.raises(ConfigError) as excinfo:
        load_config(workspace=ws, env={})
    assert fragment in str(excinfo.value)
    assert excinfo.value.hint


# --------------------------------------------------------------------------- #
# Environment overrides
# --------------------------------------------------------------------------- #


def test_every_env_override(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path, FULL_CONFIG)
    other_nb = _write(tmp_path / "other.py", "import marimo\n")
    env = {
        "HAILER_NOTEBOOK": str(other_nb),
        "HAILER_NOTEBOOKS_DIR": "envnbs",
        "HAILER_DATA_DIR": "envdata",
        "HAILER_MARIMO_URL": "http://localhost:9999/",
        "HAILER_MARIMO_TOKEN": "supersecrettoken",
        "HAILER_MODEL": "gpt-5.5",
        "HAILER_MODEL_PROVIDER": "openai",
        "HAILER_LOG_LEVEL": "debug",
    }
    cfg = load_config(workspace=ws, env=env)
    assert cfg.notebook == other_nb.resolve()
    assert cfg.notebooks_dir == (ws / "envnbs").resolve()
    assert cfg.data_dir == (ws / "envdata").resolve()
    assert cfg.marimo_url == "http://localhost:9999"
    assert cfg.marimo_token == "supersecrettoken"
    assert cfg.model.name == "gpt-5.5"
    assert cfg.model.provider == "openai"
    assert cfg.model.reasoning_effort == "high"  # not overridden by env
    assert cfg.log_level == "DEBUG"


def test_env_config_and_workspace_override(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    custom = _write(tmp_path / "somewhere" / "cfg.toml", '[model]\nname = "from-env-file"\n')
    cfg = load_config(env={"HAILER_WORKSPACE": str(ws), "HAILER_CONFIG": str(custom)})
    assert cfg.workspace == ws.resolve()
    assert cfg.config_path == custom.resolve()
    assert cfg.model.name == "from-env-file"


def test_token_never_read_from_file_and_never_in_validate_output(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path, '[hailer]\nmarimo_token = "filesecret123"\n')
    cfg = load_config(workspace=ws, env={"HAILER_MARIMO_TOKEN": "envsecret456"})
    assert cfg.marimo_token == "envsecret456"
    problems = validate(cfg)
    joined = "\n".join(problems)
    assert "filesecret123" not in joined
    assert "envsecret456" not in joined
    assert any("marimo_token" in p and p.startswith("Warning:") for p in problems)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def test_validate_notebook_missing_is_fatal(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path, notebook=False)
    problems = validate(load_config(workspace=ws, env={}))
    errors = _errors(problems)
    assert len(errors) == 1
    assert "Notebook not found" in errors[0]
    assert "analysis.py" in errors[0]


def test_validate_notebook_outside_notebooks_dir_is_fatal(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path, '[hailer]\nnotebooks_dir = "other"\n')
    (ws / "other").mkdir()
    problems = validate(load_config(workspace=ws, env={}))
    errors = _errors(problems)
    assert len(errors) == 1
    assert "[hailer].notebook" in errors[0] and "[hailer].notebooks_dir" in errors[0]
    assert "analysis.py" in errors[0]


def test_validate_notebooks_dir_missing_is_warning(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path, notebook=False)
    problems = validate(load_config(workspace=ws, env={}))
    assert any(p.startswith("Warning:") and "notebooks folder not found" in p for p in problems)
    # an env override may point at a folder that does not exist yet: warning, not fatal
    (tmp_path / "second").mkdir()
    ws2 = _make_workspace(tmp_path / "second")
    problems = validate(load_config(workspace=ws2, env={"HAILER_NOTEBOOKS_DIR": "nbs"}))
    assert any("notebooks folder not found" in p for p in problems)
    assert any("not inside [hailer].notebooks_dir" in p for p in _errors(problems))


def test_validate_notebooks_dir_equal_to_workspace_is_warning(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    (ws / "root_nb.py").write_text("import marimo\napp = marimo.App()\n", encoding="utf-8")
    problems = validate(load_config(workspace=ws, env={"HAILER_NOTEBOOK": "root_nb.py", "HAILER_NOTEBOOKS_DIR": "."}))
    assert _errors(problems) == []
    assert any(p.startswith("Warning:") and "workspace itself" in p and ".venv" in p for p in problems)
    # the normal layout does not warn
    assert not any("workspace itself" in p for p in validate(load_config(workspace=ws, env={})))


def test_validate_data_dir_missing_is_warning(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path, data=False)
    problems = validate(load_config(workspace=ws, env={}))
    assert _errors(problems) == []
    assert any(p.startswith("Warning:") and "data directory" in p for p in problems)


def test_validate_undeclared_provider(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path, '[model]\nprovider = "internal"\n')
    problems = _errors(validate(load_config(workspace=ws, env={})))
    assert any("'internal' is not declared" in p and "[model_providers.internal]" in p for p in problems)


def test_validate_custom_provider_requirements(tmp_path: Path) -> None:
    ws = _make_workspace(
        tmp_path,
        '[model]\nprovider = "internal"\n[model_providers.internal]\nwire_api = "grpc"\n',
    )
    problems = _errors(validate(load_config(workspace=ws, env={})))
    joined = "\n".join(problems)
    assert "[model_providers.internal].base_url is missing" in joined
    assert "[model_providers.internal].env_key is missing" in joined
    assert 'wire_api must be "responses" or "chat"' in joined
    assert "chat/completions" in joined


def test_validate_accepts_chat_wire_api(tmp_path: Path) -> None:
    ws = _make_workspace(
        tmp_path,
        '[model]\nprovider = "internal"\n[model_providers.internal]\n'
        'base_url = "https://llm.example.internal/v1"\nwire_api = "chat"\nenv_key = "K"\n',
    )
    cfg = load_config(workspace=ws, env={})
    assert cfg.providers["internal"].wire_api == "chat"
    assert not any("wire_api" in p for p in validate(cfg))


def test_chat_provider_can_turn_streaming_off(tmp_path: Path) -> None:
    ws = _make_workspace(
        tmp_path,
        '[model]\nprovider = "internal"\n[model_providers.internal]\n'
        'base_url = "https://llm.example.internal/v1"\nwire_api = "chat"\nstream = false\nenv_key = "K"\n',
    )
    cfg = load_config(workspace=ws, env={})
    assert cfg.providers["internal"].stream is False
    assert not any("stream" in p for p in validate(cfg))


def test_stream_switches_apply_to_either_wire_api(tmp_path: Path) -> None:
    """Hailer talks to the endpoint itself, so stream / stream_options work for the Responses API too."""
    ws = _make_workspace(
        tmp_path,
        '[model]\nprovider = "internal"\n[model_providers.internal]\n'
        'base_url = "https://llm.example.internal/v1"\nstream = false\nstream_options = false\nenv_key = "K"\n',
    )
    cfg = load_config(workspace=ws, env={})
    provider = cfg.providers["internal"]
    assert (provider.wire_api, provider.stream, provider.stream_options) == ("responses", False, False)
    assert _errors(validate(cfg)) == []


def test_validate_negative_summarize_after_tokens(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path, "[model]\nsummarize_after_tokens = -1\n")
    assert any("summarize_after_tokens" in p for p in _errors(validate(load_config(workspace=ws, env={}))))
    (tmp_path / "off").mkdir()
    off = _make_workspace(tmp_path / "off", "[model]\nsummarize_after_tokens = 0\n")
    assert _errors(validate(load_config(workspace=off, env={}))) == []


@pytest.mark.parametrize(
    "key",
    ["requires_openai_auth = true", "merge_messages = false", "parallel_tool_calls = false"],
)
def test_keys_from_the_codex_releases_are_ignored_with_a_warning(tmp_path: Path, key: str) -> None:
    """A hailer.toml written for 0.2 still loads: keys that no longer mean anything are reported, not fatal."""
    ws = _make_workspace(
        tmp_path,
        f'[hailer]\ncodex_home = ".codex-home"\n[model]\nprovider = "gw"\n[model_providers.gw]\n'
        f'base_url = "https://gw.example/v1"\nenv_key = "GW_KEY"\n{key}\n[web]\nallow_shell_network = true\n',
    )
    problems = validate(load_config(workspace=ws, env={}))
    assert _errors(problems) == []
    name = key.split(" ")[0]
    assert any(f"[model_providers.gw].{name}" in p and "ignored" in p for p in problems)
    assert any("[hailer].codex_home" in p for p in problems) and any("[web].allow_shell_network" in p for p in problems)


def test_validate_base_url_scheme(tmp_path: Path) -> None:
    ws = _make_workspace(
        tmp_path,
        '[model_providers.gw]\nbase_url = "llm.example/v1"\nenv_key = "GW_KEY"\n',
    )
    problems = _errors(validate(load_config(workspace=ws, env={})))
    assert any("must start with http:// or https://" in p for p in problems)


@pytest.mark.parametrize(
    ("domains", "bad"),
    [
        ('["*"]', "'*'"),
        ('["**"]', "'**'"),
        ('["bad domain.com"]', "'bad domain.com'"),
        ('["-example.com"]', "'-example.com'"),
        ('["http://example.com"]', "'http://example.com'"),
        ('["*.localhost"]', "'*.localhost'"),
    ],
)
def test_validate_bad_domain_rules(tmp_path: Path, domains: str, bad: str) -> None:
    ws = _make_workspace(tmp_path, f"[web]\nallowed_domains = {domains}\n")
    problems = _errors(validate(load_config(workspace=ws, env={})))
    assert len(problems) == 1
    assert bad in problems[0]


def test_validate_good_domain_rules(tmp_path: Path) -> None:
    ws = _make_workspace(
        tmp_path,
        '[web]\nallowed_domains = ["example.com", "*.example.com", "**.example.co.uk", "localhost", "127.0.0.1", "a-b.c1.d"]\n',
    )
    assert _errors(validate(load_config(workspace=ws, env={}))) == []


def test_validate_numbers_levels_and_effort(tmp_path: Path) -> None:
    ws = _make_workspace(
        tmp_path,
        '[hailer]\nlog_level = "LOUD"\nmax_tool_output_chars = 0\n[model]\nreasoning_effort = "extreme"\n[web]\nmax_page_bytes = -1\n',
    )
    problems = _errors(validate(load_config(workspace=ws, env={})))
    joined = "\n".join(problems)
    assert "Invalid log level 'LOUD'" in joined
    assert "max_tool_output_chars must be a positive integer" in joined
    assert "reasoning_effort 'extreme'" in joined
    assert "max_page_bytes must be a positive integer" in joined


def test_validate_unknown_keys_are_warnings(tmp_path: Path) -> None:
    ws = _make_workspace(
        tmp_path,
        '[hailer]\ncolour = "blue"\n[model]\ntemperature = 0.1\n[model_providers.gw]\nbase_url = "https://gw/v1"\nenv_key = "GW_KEY"\napi_key = "sk-abcdefghijklmnop"\nextra = 1\n[web]\nfoo = 1\n[other]\nx = 1\n',
    )
    problems = validate(load_config(workspace=ws, env={}))
    assert _errors(problems) == []
    joined = "\n".join(problems)
    assert "[hailer].colour" in joined
    assert "[model].temperature" in joined
    assert "[model_providers.gw].extra" in joined
    assert "[web].foo" in joined
    assert "unknown section [other]" in joined
    assert "secret-looking key" in joined
    assert "sk-abcdefghijklmnop" not in joined


# --------------------------------------------------------------------------- #
# [kernel]
# --------------------------------------------------------------------------- #


def test_kernel_defaults_to_local(tmp_path: Path) -> None:
    cfg = load_config(workspace=_make_workspace(tmp_path), env={})
    assert cfg.kernel == KernelConfig(runtime="local", image=None, memory="4g", cpus=2.0, network=False)


def test_kernel_section_is_parsed(tmp_path: Path) -> None:
    ws = _make_workspace(
        tmp_path,
        '[kernel]\nruntime = "Docker"\nimage = "registry.example/hailer-kernel:dev"\nmemory = "8G"\ncpus = 1.5\nnetwork = true\n',
    )
    cfg = load_config(workspace=ws, env={})
    assert cfg.kernel == KernelConfig(runtime="docker", image="registry.example/hailer-kernel:dev", memory="8G", cpus=1.5, network=True)
    assert _errors(validate(cfg)) == []
    (tmp_path / "ints").mkdir()
    cfg2 = load_config(workspace=_make_workspace(tmp_path / "ints", '[kernel]\ncpus = 4\nimage = ""\n'), env={})
    assert cfg2.kernel.cpus == 4.0 and cfg2.kernel.image is None, "an empty image means the default one"


def test_kernel_env_overrides_the_file(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path, '[kernel]\nruntime = "docker"\nimage = "from-file"\n')
    cfg = load_config(workspace=ws, env={"HAILER_KERNEL": "local", "HAILER_KERNEL_IMAGE": "from-env"})
    assert (cfg.kernel.runtime, cfg.kernel.image) == ("local", "from-env")
    assert load_config(workspace=ws, env={}).kernel.runtime == "docker"


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ('[kernel]\ncpus = "2"\n', "[kernel].cpus"),
        ("[kernel]\ncpus = true\n", "[kernel].cpus"),
        ("[kernel]\nmemory = 4\n", "[kernel].memory"),
        ('[kernel]\nnetwork = "no"\n', "[kernel].network"),
        ("[kernel]\nruntime = 1\n", "[kernel].runtime"),
        ('kernel = "docker"\n', "[kernel]"),
    ],
)
def test_kernel_wrong_types_raise_config_error(tmp_path: Path, text: str, fragment: str) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_config(workspace=_make_workspace(tmp_path, text), env={})
    assert fragment in str(excinfo.value) and excinfo.value.hint


@pytest.mark.parametrize(
    ("text", "env", "fragment"),
    [
        ('[kernel]\nruntime = "podman"\n', {}, '[kernel].runtime "podman"'),
        ("", {"HAILER_KERNEL": "vm"}, '[kernel].runtime "vm"'),
        ('[kernel]\nmemory = "lots"\n', {}, '[kernel].memory "lots"'),
        ('[kernel]\nmemory = "4 GB"\n', {}, '[kernel].memory "4 GB"'),  # quoted as written, not lower-cased
        ('[kernel]\npass_env = [""]\n', {}, "[kernel].pass_env"),
        ('[kernel]\nmemory = "4tb"\n', {}, "[kernel].memory"),
        ('[kernel]\nmemory = "0g"\n', {}, "[kernel].memory"),
        ("[kernel]\ncpus = 0\n", {}, "[kernel].cpus"),
        ("[kernel]\ncpus = -1.5\n", {}, "[kernel].cpus"),
    ],
)
def test_kernel_bad_values_are_fatal(tmp_path: Path, text: str, env: dict, fragment: str) -> None:
    errors = _errors(validate(load_config(workspace=_make_workspace(tmp_path, text), env=env)))
    assert len(errors) == 1 and fragment in errors[0], errors


def test_docker_refuses_marimo_url(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path, '[hailer]\nmarimo_url = "http://127.0.0.1:2718"\n[kernel]\nruntime = "docker"\n')
    errors = _errors(validate(load_config(workspace=ws, env={})))
    assert len(errors) == 1 and "marimo_url" in errors[0] and "its own container" in errors[0]
    assert _errors(validate(load_config(workspace=ws, env={"HAILER_KERNEL": "local"}))) == [], "fine for local"
    (tmp_path / "env").mkdir()
    plain = _make_workspace(tmp_path / "env", '[kernel]\nruntime = "docker"\n')
    errors = _errors(validate(load_config(workspace=plain, env={"HAILER_MARIMO_URL": "http://127.0.0.1:2718"})))
    assert len(errors) == 1 and "HAILER_MARIMO_URL" in errors[0]


@pytest.mark.parametrize("data_dir", ["notebooks", "notebooks/data", "NOTEBOOKS/Data"])
def test_docker_refuses_data_inside_the_notebooks_folder(tmp_path: Path, data_dir: str) -> None:
    if data_dir != data_dir.lower() and not Path(str(tmp_path).upper()).exists():
        pytest.skip("case-insensitive file system only")
    ws = _make_workspace(tmp_path, f'[hailer]\ndata_dir = "{data_dir}"\n[kernel]\nruntime = "docker"\n')
    (ws / "notebooks" / "data").mkdir(exist_ok=True)
    errors = _errors(validate(load_config(workspace=ws, env={})))
    assert len(errors) == 1 and "[hailer].data_dir" in errors[0] and "writable" in errors[0]
    assert _errors(validate(load_config(workspace=ws, env={"HAILER_KERNEL": "local"}))) == [], "local mounts nothing"


def test_docker_allows_notebooks_inside_the_data_folder(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path, notebook=False)
    _write(ws / "data" / "notebooks" / "analysis.py", "import marimo\n")
    _write(
        ws / "hailer.toml",
        '[hailer]\nnotebook = "data/notebooks/analysis.py"\ndata_dir = "data"\n[kernel]\nruntime = "docker"\n',
    )
    assert _errors(validate(load_config(workspace=ws, env={}))) == []


def test_kernel_unknown_keys_are_warnings(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path, '[kernel]\nruntime = "local"\nmounts = ["x"]\n')
    problems = validate(load_config(workspace=ws, env={}))
    assert _errors(problems) == []
    assert any(p.startswith("Warning:") and "[kernel].mounts" in p for p in problems)
    assert not any("unknown section [kernel]" in p for p in problems)


def test_template_documents_the_kernel_section_commented_out() -> None:
    assert "# [kernel]\n# runtime = \"local\"" in DEFAULT_CONFIG_TEMPLATE
    assert "HAILER_KERNEL" in DEFAULT_CONFIG_TEMPLATE
    for key in ("image", "memory", "cpus", "network"):
        assert f"\n# {key} " in DEFAULT_CONFIG_TEMPLATE
    uncommented = DEFAULT_CONFIG_TEMPLATE.replace("# [kernel]", "[kernel]").replace('# runtime = "local"', 'runtime = "docker"')
    import tomllib

    assert tomllib.loads(uncommented)["kernel"] == {"runtime": "docker"}, "`init --kernel docker` can switch it on in place"


@pytest.mark.parametrize("runtime", ["docker", "local"])
def test_write_default_config_can_switch_the_kernel_section_on(tmp_path: Path, runtime: str) -> None:
    ws = _make_workspace(tmp_path)
    write_default_config(ws / "hailer.toml", kernel=runtime)
    text = (ws / "hailer.toml").read_text(encoding="utf-8")
    assert f'\n[kernel]\nruntime = "{runtime}"' in text and "\n# image " in text, "the other keys stay commented out"
    config = load_config(workspace=ws, env={})
    assert config.kernel == KernelConfig(runtime=runtime)
    assert _errors(validate(config)) == [] and not any("[kernel]" in p for p in validate(config))
    assert config_template() == DEFAULT_CONFIG_TEMPLATE
    with pytest.raises(ConfigError):
        config_template("podman")


# --------------------------------------------------------------------------- #
# Template and repo example
# --------------------------------------------------------------------------- #


def test_template_round_trips(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path, notebook=False, data=False)
    target = ws / "hailer.toml"
    write_default_config(target)
    assert target.read_text(encoding="utf-8") == DEFAULT_CONFIG_TEMPLATE
    cfg = load_config(workspace=ws, env={})
    assert cfg.config_path == target.resolve()
    assert cfg.model.name == "gpt-5.5"
    assert cfg.model.provider == "openai"
    assert cfg.providers == {}
    problems = validate(cfg)
    errors = _errors(problems)
    assert len(errors) == 1 and "Notebook not found" in errors[0]
    assert not any("unknown" in p for p in problems)
    # with the notebook and data dir present the template is clean
    _write(ws / "notebooks" / "analysis.py", "import marimo\n")
    (ws / "data").mkdir()
    assert validate(load_config(workspace=ws, env={})) == []


def test_write_default_config_refuses_to_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "hailer.toml"
    write_default_config(target)
    with pytest.raises(ConfigError) as excinfo:
        write_default_config(target)
    assert "already exists" in str(excinfo.value)
    assert "--force" in excinfo.value.hint
    target.write_text("[hailer]\n", encoding="utf-8")
    write_default_config(target, overwrite=True)
    assert target.read_text(encoding="utf-8") == DEFAULT_CONFIG_TEMPLATE


def test_repo_example_matches_template() -> None:
    example = REPO_ROOT / "hailer.toml"
    assert example.is_file()
    assert example.read_text(encoding="utf-8").replace("\r\n", "\n") == DEFAULT_CONFIG_TEMPLATE


def test_provider_stream_switches_default_to_on(tmp_path: Path) -> None:
    ws = _make_workspace(
        tmp_path,
        '[model]\nprovider = "internal"\n[model_providers.internal]\n'
        'base_url = "https://llm.example.internal/v1"\nwire_api = "chat"\nenv_key = "K"\n',
    )
    p = load_config(workspace=ws, env={}).providers["internal"]
    assert (p.stream, p.stream_options) == (True, True)


def test_validate_provider_key_typo_is_an_unknown_key_warning(tmp_path: Path) -> None:
    ws = _make_workspace(
        tmp_path,
        '[model]\nprovider = "internal"\n[model_providers.internal]\n'
        'base_url = "https://llm.example.internal/v1"\nwire_api = "chat"\nenv_key = "K"\nstream_option = false\n',
    )
    problems = validate(load_config(workspace=ws, env={}))
    assert any("stream_option" in p and "unknown" in p.lower() for p in problems)


# --------------------------------------------------------------------------- #
# [kernel]: mounts that would expose Hailer's own files, pass_env, misplaced keys, wording
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ('[hailer]\nnotebook = "analysis.py"\n', "The notebooks folder ({ws}) is the workspace folder"),
        ('[hailer]\nnotebooks_dir = "."\n', "The notebooks folder ({ws}) is the workspace folder"),
        ('[hailer]\ndata_dir = "."\n', "The data folder ({ws}) is the workspace folder"),
        ('[hailer]\ndata_dir = ".hailer"\n', "The data folder ({ws}{sep}.hailer) is Hailer's .hailer folder"),
        ('[hailer]\nnotebooks_dir = ".config/hailer/context"\nnotebook = ".config/hailer/context/a.py"\n', "is the context folder"),
        ('[hailer]\nnotebooks_dir = ".hailer/nb"\nnotebook = ".hailer/nb/a.py"\n', "is inside Hailer's .hailer folder"),
    ],
)
def test_docker_refuses_mounts_that_expose_hailers_own_files(tmp_path: Path, text: str, fragment: str) -> None:
    """The kernel could rewrite hailer.toml (switch to local), read .hailer/ (the kernel's token,
    conversations) or plant context. Fatal in docker mode only: the local runtime mounts nothing."""
    ws = _make_workspace(tmp_path, text + '[kernel]\nruntime = "docker"\n')
    for name in ("analysis.py", ".config/hailer/context/a.py", ".hailer/nb/a.py"):
        _write(ws / name, "import marimo\n")
    errors = _errors(validate(load_config(workspace=ws, env={})))
    expected = fragment.format(ws=ws.resolve(), sep=os.sep)
    assert any(expected in e for e in errors), errors
    local = _errors(validate(load_config(workspace=ws, env={"HAILER_KERNEL": "local"})))
    assert not any("With [kernel] runtime" in e for e in local), "the local runtime mounts nothing"


def test_docker_refuses_a_data_folder_moved_by_the_environment(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path, '[kernel]\nruntime = "docker"\n')
    errors = _errors(validate(load_config(workspace=ws, env={"HAILER_DATA_DIR": str(ws)})))
    assert any("The data folder" in e and "is the workspace folder" in e for e in errors), errors


def test_docker_refuses_the_home_folder_and_a_whole_drive(tmp_path: Path, monkeypatch) -> None:
    ws = _make_workspace(tmp_path, '[kernel]\nruntime = "docker"\n')
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    errors = _errors(validate(load_config(workspace=ws, env={"HAILER_DATA_DIR": str(tmp_path)})))
    assert any("The data folder" in e and "is your home folder" in e for e in errors), errors
    errors = _errors(validate(load_config(workspace=ws, env={"HAILER_DATA_DIR": ws.anchor})))
    assert any("is a whole drive" in e for e in errors), errors


def test_the_default_layout_is_fine_for_docker(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path, '[kernel]\nruntime = "docker"\n')
    config = load_config(workspace=ws, env={})
    assert docker_mount_problems(config) == [] and _errors(validate(config)) == []


def test_pass_env_is_a_list_of_names_for_the_local_runtime(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path, '[kernel]\npass_env = ["DB_PASSWORD", " AWS_SECRET_ACCESS_KEY "]\n')
    config = load_config(workspace=ws, env={})
    assert config.kernel.pass_env == ("DB_PASSWORD", "AWS_SECRET_ACCESS_KEY")
    assert not any("pass_env" in p for p in validate(config))
    (tmp_path / "docker").mkdir()
    docker = _make_workspace(tmp_path / "docker", '[kernel]\nruntime = "docker"\npass_env = ["DB_PASSWORD"]\n')
    problems = validate(load_config(workspace=docker, env={}))
    assert any(p.startswith("Warning: [kernel].pass_env applies to the local runtime only") for p in problems)
    (tmp_path / "bad").mkdir()
    with pytest.raises(ConfigError) as excinfo:
        load_config(workspace=_make_workspace(tmp_path / "bad", '[kernel]\npass_env = "DB_PASSWORD"\n'), env={})
    assert 'must be a list of strings, not "DB_PASSWORD"' in str(excinfo.value) and 'pass_env = ["DB_PASSWORD"]' in excinfo.value.hint


@pytest.mark.parametrize("section", ["model", "hailer"])
def test_a_kernel_setting_in_another_table_is_named(tmp_path: Path, section: str) -> None:
    """Uncommenting only the runtime line of the template lands it under [model]: silently local."""
    text = '[hailer]\nruntime = "docker"\n' if section == "hailer" else '[model]\nname = "gpt-5.5"\nruntime = "docker"\n'
    ws = _make_workspace(tmp_path, text)
    config = load_config(workspace=ws, env={})
    assert config.kernel.runtime == "local"
    warnings = [p for p in validate(config) if p.startswith("Warning:")]
    assert any(f"[{section}].runtime" in w and "belongs under [kernel]" in w and "Uncomment the [kernel] line" in w for w in warnings), warnings
    assert not any(f"unknown key [{section}].runtime" in w for w in warnings), "one warning, the specific one"


def test_type_errors_quote_what_was_written(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_config(workspace=_make_workspace(tmp_path, '[kernel]\ncpus = "2"\n'), env={})
    assert 'must be a number, not "2" (str)' in str(excinfo.value) and "without quotes" in excinfo.value.hint


@pytest.mark.parametrize("name", [".ssh", ".aws", ".config", ".kube"])
def test_docker_refuses_credential_folders_under_home(tmp_path: Path, monkeypatch, name: str) -> None:
    import hailer.config as config_module

    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(config_module, "_in_temp", lambda folder: False)  # tmp_path stands in for a real home
    (home / name / "sub").mkdir(parents=True)
    ws = _make_workspace(tmp_path, '[kernel]\nruntime = "docker"\n')
    for data, relation in ((home / name, "is"), (home / name / "sub", "is inside")):
        errors = _errors(validate(load_config(workspace=ws, env={"HAILER_DATA_DIR": str(data)})))
        expected = f"The data folder ({data.resolve()}) {relation} {(home / name).resolve()}, a folder that holds credentials."
        assert any(e.startswith(expected) for e in errors), errors


def test_docker_refuses_appdata_on_windows_but_not_the_temporary_folder(tmp_path: Path, monkeypatch) -> None:
    import tempfile

    import hailer.config as config_module

    roaming = tmp_path / "Roaming"
    (roaming / "Tool").mkdir(parents=True)
    monkeypatch.setenv("APPDATA", str(roaming))
    ws = _make_workspace(tmp_path, '[kernel]\nruntime = "docker"\n')
    config = load_config(workspace=ws, env={"HAILER_DATA_DIR": str(roaming / "Tool")})
    in_temp = config_module._in_temp
    monkeypatch.setattr(config_module, "_in_temp", lambda folder: False)  # tmp_path stands in for %APPDATA%
    problems = docker_mount_problems(config, windows=True)
    assert any("is inside" in p and "a folder that holds credentials" in p for p in problems), problems
    monkeypatch.setattr(config_module, "_in_temp", in_temp)
    monkeypatch.setenv("APPDATA", str(tmp_path / "elsewhere"))
    monkeypatch.setenv("LOCALAPPDATA", str(Path(tempfile.gettempdir()).parent))
    assert docker_mount_problems(load_config(workspace=ws, env={}), windows=True) == [], "a workspace in the temporary folder is fine"
    assert not any("credentials" in p for p in docker_mount_problems(config, windows=False)), "Windows only"


def test_docker_refuses_a_notebooks_folder_that_is_a_git_repository(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path, '[kernel]\nruntime = "docker"\n')
    (ws / "notebooks" / ".git").mkdir()
    errors = _errors(validate(load_config(workspace=ws, env={})))
    assert any(e.startswith(f"The notebooks folder ({(ws / 'notebooks').resolve()}) is a git repository") for e in errors), errors
    assert not _errors(validate(load_config(workspace=ws, env={"HAILER_KERNEL": "local"}))), "local mode mounts nothing"


def test_docker_refuses_unc_folders_as_errors(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path, '[kernel]\nruntime = "docker"\n')
    config = load_config(workspace=ws, env={})
    from dataclasses import replace

    share = replace(config, data_dir=Path(r"\\fileserver\team\sales"))
    assert docker_mount_problems(share, windows=True) == [
        "The data folder \\\\fileserver\\team\\sales is on a network share (UNC path), which Docker cannot mount. "
        "Copy it to a folder on a local disk and point [hailer].data_dir at it."
    ]


def test_a_notebooks_folder_that_is_the_workspace_is_reported_once_in_docker_mode(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path, '[hailer]\nnotebooks_dir = "."\nnotebook = "analysis.py"\n[kernel]\nruntime = "docker"\n')
    _write(ws / "analysis.py", "import marimo\n")
    problems = validate(load_config(workspace=ws, env={}))
    assert sum("is the workspace" in p for p in problems) == 1, problems
    local = validate(load_config(workspace=ws, env={"HAILER_KERNEL": "local"}))
    assert any(p.startswith("Warning: the notebooks folder is the workspace itself") for p in local), "the local warning stays"


def test_pass_env_naming_hailers_own_secrets_is_a_warning(tmp_path: Path) -> None:
    text = (
        '[model]\nprovider = "corp"\n[model_providers.corp]\nbase_url = "https://llm.example.internal/v1"\n'
        'env_key = "CORP_API_KEY"\nenv_http_headers = { "X-Client-Id" = "CORP_CLIENT_ID" }\n'
        '[kernel]\npass_env = ["CORP_API_KEY", "CORP_CLIENT_ID", "HAILER_MARIMO_TOKEN", "DB_PASSWORD"]\n'
    )
    warnings = [p for p in validate(load_config(workspace=_make_workspace(tmp_path, text), env={})) if "pass_env" in p]
    assert len(warnings) == 3 and all(w.startswith("Warning: [kernel].pass_env lets notebook code read ") for w in warnings)
    assert 'CORP_API_KEY (the API key of provider "corp")' in warnings[0]
    assert 'CORP_CLIENT_ID (the X-Client-Id header of provider "corp")' in warnings[1]
    assert "HAILER_MARIMO_TOKEN (the marimo server token)" in warnings[2]


def test_kernel_runtime_errors_quote_like_the_rest(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        config_template("podman")
    assert str(excinfo.value) == 'Unknown kernel runtime "podman".'
