"""Tests for hailer.config: defaults, file parsing, env overrides, validation, template."""

from __future__ import annotations

from pathlib import Path

import pytest

from hailer.config import (
    CONFIG_FILENAMES,
    DEFAULT_CONFIG_TEMPLATE,
    find_config_path,
    find_workspace,
    load_config,
    validate,
    write_default_config,
)
from hailer.errors import ConfigError
from hailer.models import HailerConfig

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
    assert cfg.providers == {}
    assert cfg.web.allowed_domains == ()
    assert cfg.web.allow_shell_network is False
    assert cfg.marimo_url is None
    assert cfg.marimo_token is None
    assert cfg.codex_home is None
    assert cfg.log_level == "WARNING"
    assert cfg.max_tool_output_chars == 12_000
    assert cfg.max_context_bytes == 24_000
    assert validate(cfg) == []


def test_builtin_openai_provider_is_synthesised(tmp_path: Path) -> None:
    cfg = load_config(workspace=_make_workspace(tmp_path), env={})
    provider = cfg.provider
    assert provider.id == "openai"
    assert provider.env_key == "OPENAI_API_KEY"
    assert provider.requires_openai_auth is True
    assert provider.is_builtin_openai


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
codex_home = ".codex-home"
log_level = "info"

[model]
name = "risk-analyst-v3"
provider = "internal"
reasoning_effort = "high"

[model_providers.internal]
base_url = "https://llm.example.internal/v1/"
wire_api = "responses"
env_key = "INTERNAL_MODEL_API_KEY"
requires_openai_auth = false
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
allow_shell_network = true
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
    assert cfg.codex_home == (ws / ".codex-home").resolve()
    assert cfg.log_level == "INFO"

    assert cfg.model.name == "risk-analyst-v3"
    assert cfg.model.provider == "internal"
    assert cfg.model.reasoning_effort == "high"

    internal = cfg.providers["internal"]
    assert internal.id == "internal"
    assert internal.base_url == "https://llm.example.internal/v1"
    assert internal.wire_api == "responses"
    assert internal.env_key == "INTERNAL_MODEL_API_KEY"
    assert internal.requires_openai_auth is False
    assert internal.name == "Internal"
    assert internal.http_headers == {"X-Team": "risk"}
    assert internal.env_http_headers == {"X-Client-Id": "INTERNAL_CLIENT_ID"}
    assert internal.query_params == {"api-version": "2025-04-01-preview"}
    assert not internal.is_builtin_openai
    assert cfg.provider is internal

    azure = cfg.providers["azure"]
    assert azure.wire_api == "responses"  # default
    assert azure.http_headers == {}

    assert cfg.web.allowed_domains == ("docs.pola.rs", "**.bankofengland.co.uk", "duckdb.org")
    assert cfg.web.max_page_bytes == 50000
    assert cfg.web.allow_shell_network is True


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
        ('[web]\nallow_shell_network = "yes"\n', "[web].allow_shell_network"),
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
        "HAILER_CODEX_HOME": str(tmp_path / "ch"),
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
    assert cfg.codex_home == (tmp_path / "ch").resolve()


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
    assert cfg.providers["internal"].uses_chat_completions
    assert not any("wire_api" in p for p in validate(cfg))


def test_validate_chat_wire_api_needs_a_base_url_for_openai(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path, '[model_providers.openai]\nwire_api = "chat"\n')
    problems = _errors(validate(load_config(workspace=ws, env={})))
    assert any('wire_api = "chat" needs a base_url' in p for p in problems)


def test_validate_requires_openai_auth_provider_needs_no_env_key(tmp_path: Path) -> None:
    ws = _make_workspace(
        tmp_path,
        '[model]\nprovider = "gw"\n[model_providers.gw]\nbase_url = "https://gw.example/v1"\nrequires_openai_auth = true\n',
    )
    assert _errors(validate(load_config(workspace=ws, env={}))) == []


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
