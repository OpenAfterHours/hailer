"""The project's own files for the agent (hailer.project and the project_* tools).

Which files the agent may read, what a search and a read return, and that nothing that looks like a
credential, Hailer's state, the data or the notebooks reaches a tool result.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from hailer import project
from hailer.agent import system_prompt
from hailer.config import load_config, validate
from hailer.errors import HailerError
from hailer.models import ContextBundle, HailerConfig, ProjectConfig
from hailer.project import WITHHELD, ProjectFiles, glob_matcher, secret_named
from hailer.tools import HailerTools

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _write(path: Path, text: str = "x = 1\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def _config(ws: Path, **project: object) -> HailerConfig:
    return HailerConfig(
        workspace=ws,
        notebook=ws / ".hailer" / "notebooks" / "analysis.py",
        notebooks_dir=ws / ".hailer" / "notebooks",
        data_dir=ws / ".hailer" / "data",
        context_dir=ws / ".hailer" / "context",
        skills_dir=ws / ".hailer" / "skills",
        prompts_dir=ws / ".hailer" / "prompts",
        project=ProjectConfig(root=ws, **project),  # type: ignore[arg-type]
    )


def _monorepo(tmp_path: Path) -> Path:
    """A project in the project layout, with everything the agent must and must not see."""
    ws = tmp_path / "repo"
    _write(ws / ".hailer" / "hailer.toml", "[project]\n")
    _write(ws / "README.md", "# Shop\nRevenue is net of refunds.\n")
    _write(ws / "packages" / "billing" / "revenue.py", "def net_revenue(orders):\n    return orders.gross - orders.refunds\n")
    _write(ws / "packages" / "billing" / "sql" / "orders.sql", "select * from orders where status <> 'test'\n")
    _write(ws / "packages" / "billing" / "data" / "seed.csv", "a,b\n1,2\n")  # not code or docs
    _write(ws / ".hailer" / "skills" / "s" / "SKILL.md", "---\nname: s\n---\n")
    _write(ws / ".hailer" / "notebooks" / "analysis.py", "import marimo\n")
    _write(ws / ".hailer" / "data" / "sales.json", "{}\n")
    _write(ws / ".hailer" / "state" / "session.json", "{}\n")
    _write(ws / ".hailer" / "local.toml", "[model]\n")
    _write(ws / ".env", "OPENAI_API_KEY=sk-live\n")
    _write(ws / ".env.example", "OPENAI_API_KEY=\n")
    _write(ws / "deploy" / "server.pem", "-----BEGIN PRIVATE KEY-----\n")
    _write(ws / "node_modules" / "left" / "index.js", "module.exports = 1\n")
    _write(ws / ".venv" / "lib" / "site.py", "x = 1\n")
    _write(ws / "huge.md", "x" * (project.MAX_FILE_BYTES + 1))
    return ws


EXPECTED = {
    ".env.example",
    ".hailer/hailer.toml",
    ".hailer/skills/s/SKILL.md",
    "README.md",
    "packages/billing/revenue.py",
    "packages/billing/sql/orders.sql",
}


# --------------------------------------------------------------------------- #
# Which files
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("pattern", "yes", "no"),
    [
        ("*.sql", ["a.sql", "x/y/b.SQL"], ["a.sql.txt", "sql/a.py"]),
        ("packages/billing/**", ["packages/billing/a.py", "packages/billing/x/y.md"], ["packages/bill/a.py", "x/packages/billing/a.py"]),
        ("src/**/models/*.py", ["src/models/a.py", "src/a/b/models/c.py"], ["src/models/x/c.py", "lib/models/a.py"]),
        ("fixtures", ["fixtures/a.json", "tests/fixtures/b.py"], ["fixturesx/a.py"]),
        ("docs/", ["docs/index.md"], ["src/docs.py"]),
        ("./README.md", ["README.md"], ["docs/README.md"]),
        ("", ["anything/at/all.py"], []),
    ],
)
def test_globs_match_like_gitignore(pattern: str, yes: list[str], no: list[str]) -> None:
    matcher = glob_matcher(pattern)
    for path in yes:
        assert matcher.match(path), (pattern, path)
    for path in no:
        assert not matcher.match(path), (pattern, path)


@pytest.mark.parametrize(
    "name", [".env", ".env.production", "server.pem", "tls.key", "id_rsa", "id_ed25519.pub", ".netrc", ".npmrc",
             "credentials", "credentials.json", "secrets.yaml", "my-service-account-123.json", "prod.tfstate", "x.tfvars"]
)  # fmt: skip
def test_credential_files_are_never_project_files(name: str) -> None:
    assert secret_named(name)


@pytest.mark.parametrize("name", [".env.example", ".env.sample", "keys.py", "secrets.py", "credentials_test.py", "tokenizer.json"])
def test_ordinary_names_are_not_credentials(name: str) -> None:
    assert not secret_named(name)


@needs_git
def test_in_a_git_repository_ignored_files_hailers_files_and_credentials_are_left_out(tmp_path: Path) -> None:
    ws = _monorepo(tmp_path)
    _git(ws, "init", "-q")
    _write(ws / ".gitignore", "build/\n")
    _write(ws / "build" / "generated.py", "x = 1\n")
    _git(ws, "add", "README.md")  # tracked; the rest is new and still listed unless ignored
    files = ProjectFiles.for_config(_config(ws)).files()
    assert set(files) == EXPECTED, set(files) ^ EXPECTED
    assert files["README.md"] == len("# Shop\nRevenue is net of refunds.\n")


def test_without_git_the_folder_is_walked_with_the_same_exclusions(tmp_path: Path) -> None:
    ws = _monorepo(tmp_path)
    _write(ws / "env" / "pyvenv.cfg", "home = /usr\n")
    _write(ws / "env" / "lib" / "x.py", "x = 1\n")  # a virtual environment under another name
    files = ProjectFiles.for_config(_config(ws)).files()
    assert set(files) == EXPECTED, set(files) ^ EXPECTED


def test_project_files_uses_git_when_it_can_and_walks_otherwise(tmp_path: Path) -> None:
    ws = _monorepo(tmp_path)
    missing = ProjectFiles(ws, git=str(tmp_path / "no-such-git"))
    assert "README.md" in missing.files() and missing._current().method == "walk"


def test_include_replaces_the_defaults_and_exclude_removes(tmp_path: Path) -> None:
    ws = _monorepo(tmp_path)
    files = ProjectFiles.for_config(_config(ws, include=("*.csv", "*.sql"), exclude=("packages/billing/sql",))).files()
    assert set(files) == {"packages/billing/data/seed.csv"}, "a seed CSV the user asked for; never .env or the data folder"


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_a_link_out_of_the_root_is_never_followed(tmp_path: Path) -> None:
    ws = _monorepo(tmp_path)
    outside = _write(tmp_path / "elsewhere" / "private.md", "secret plans\n")
    (ws / "notes.md").symlink_to(outside)
    (ws / "inside.md").symlink_to(ws / "README.md")
    files = ProjectFiles.for_config(_config(ws)).files()
    assert "notes.md" not in files and "inside.md" in files


def test_a_listing_is_reused_for_a_while_then_refreshed(tmp_path: Path) -> None:
    ws = _monorepo(tmp_path)
    now = [0.0]
    files = ProjectFiles(ws, clock=lambda: now[0])
    assert "new.md" not in files.files()
    _write(ws / "new.md", "hi\n")
    assert "new.md" not in files.files()
    now[0] += project.REFRESH_SEC + 1
    assert "new.md" in files.files()


def test_without_a_project_section_the_tools_say_how_to_turn_it_on(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config = HailerConfig(**{**config.__dict__, "project": ProjectConfig()})
    with pytest.raises(HailerError) as excinfo:
        ProjectFiles.for_config(config)
    assert "[project]" in str(excinfo.value) and "[project]" in (excinfo.value.hint or "")
    text = HailerTools(config).project_search("revenue")
    assert text.startswith("ERROR: Project files are not available")


# --------------------------------------------------------------------------- #
# Listing, search, read
# --------------------------------------------------------------------------- #


def test_listing_names_files_or_summarises_a_large_project_by_folder(tmp_path: Path, monkeypatch) -> None:
    ws = _monorepo(tmp_path)
    files = ProjectFiles.for_config(_config(ws))
    text = files.listing()
    assert text.startswith("6 project file(s) under") and "packages/billing/revenue.py  (" in text
    assert "packages/billing/sql/orders.sql" in files.listing("*.sql") and "revenue.py" not in files.listing("*.sql")
    assert "No project files" in files.listing("*.rs")
    for i in range(30):
        _write(ws / "packages" / f"pkg{i % 3}" / f"m{i}.py")
    monkeypatch.setattr(project, "MAX_LISTED", 10)
    summary = ProjectFiles.for_config(_config(ws)).listing()
    assert "too many to list, so by folder" in summary
    assert "packages/pkg0/  10" in summary and "Narrow it with a pattern" in summary


def test_search_finds_lines_with_their_place(tmp_path: Path) -> None:
    ws = _monorepo(tmp_path)
    files = ProjectFiles.for_config(_config(ws))
    text = files.search("refunds")
    assert "packages/billing/revenue.py:2:     return orders.gross - orders.refunds" in text
    assert "README.md:2: Revenue is net of refunds." in text
    assert text.startswith("2 matching line(s) in 2 file(s):")
    assert "README.md" not in files.search("refunds", "*.py")
    assert "No matches for 'nothing-like-this'" in files.search("nothing-like-this")
    literal = files.search("net_revenue(orders")
    assert "not a valid regular expression" in literal and "revenue.py:1: def net_revenue(orders):" in literal
    with pytest.raises(HailerError):
        files.search("  ")


def test_search_stops_at_the_match_cap_and_at_its_time_budget(tmp_path: Path) -> None:
    ws = tmp_path / "p"
    _write(ws / "many.py", "".join(f"hit_{i} = {i}\n" for i in range(project.MAX_MATCHES + 50)))
    _write(ws / "z.py", "hit_last = 1\n")
    files = ProjectFiles(ws)
    text = files.search("hit_")
    assert text.count("many.py:") == project.MAX_MATCHES and f"Stopped at {project.MAX_MATCHES}" in text
    now = iter([0.0, 0.0, 100.0, 100.0, 100.0])
    slow = ProjectFiles(ws, clock=lambda: next(now, 100.0))
    assert "Stopped after 5 s" in slow.search("hit_last", budget_sec=5)


def test_read_numbers_lines_and_says_how_to_continue(tmp_path: Path) -> None:
    ws = tmp_path / "p"
    _write(ws / "long.py", "".join(f"line_{i} = {i}\n" for i in range(1, 1001)))
    files = ProjectFiles(ws)
    text = files.read("long.py")
    assert text.startswith("long.py, lines 1-400 of 1000:")
    assert "\n  1  line_1 = 1\n" in text and "line_400 = 400" in text and "line_401" not in text
    assert text.endswith("(more: project_read('long.py', start_line=401))")
    window = files.read("long.py", 990, 2000)
    assert window.startswith("long.py, lines 990-1000 of 1000:") and "more:" not in window
    small = files.read("long.py", 1, 0, max_chars=1500)
    assert len(small) < 1500 + 200 and "start_line=" in small, "the output budget cuts the window"
    with pytest.raises(HailerError):
        files.read("long.py", 5000)
    (ws / "logo.py").write_bytes(b"\x00\x01binary")
    with pytest.raises(HailerError) as excinfo:
        ProjectFiles(ws).read("logo.py")
    assert "binary" in str(excinfo.value)
    assert "No matches" in ProjectFiles(ws).search("binary", "logo.py")


@pytest.mark.parametrize(
    "path", ["../outside.md", "/etc/passwd", ".env", ".hailer/local.toml", ".hailer/state/session.json",
             ".hailer/data/sales.json", ".hailer/notebooks/analysis.py", "node_modules/left/index.js", "packages"]
)  # fmt: skip
def test_read_refuses_anything_that_is_not_a_project_file(tmp_path: Path, path: str) -> None:
    ws = _monorepo(tmp_path)
    _write(tmp_path / "outside.md", "no\n")
    with pytest.raises(HailerError) as excinfo:
        ProjectFiles.for_config(_config(ws)).read(path)
    assert "not a project file you can read" in str(excinfo.value)


def test_read_accepts_another_spelling_of_a_listed_name(tmp_path: Path) -> None:
    ws = _monorepo(tmp_path)
    files = ProjectFiles.for_config(_config(ws))
    for path in ("./README.md", "readme.md", str(ws.resolve() / "README.md"), "packages\\billing\\revenue.py"):
        assert "lines 1-" in files.read(path), path


# --------------------------------------------------------------------------- #
# Credentials never reach a tool result
# --------------------------------------------------------------------------- #


SETTINGS = """\
import os
token = request.headers.get("Authorization")
password = settings.database_password
API_KEY = "sk-live-0123456789abcdefghij"
db_password = "correct-horse-battery"
KEY = '''
-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEAu1SU1LfVLPHCozMxH2Mo4lgOEePzNm0tRgeLezV6ffAt0gunVTLw
-----END RSA PRIVATE KEY-----
'''
done = True
"""


def test_lines_that_look_like_credentials_are_withheld_in_reads_and_searches(tmp_path: Path) -> None:
    ws = tmp_path / "p"
    _write(ws / "settings.py", SETTINGS)
    _write(ws / "config.yaml", "database:\n  password: hunter2hunter2hunter2\n  host: db.internal\n")
    files = ProjectFiles(ws)
    text = files.read("settings.py")
    assert 'token = request.headers.get("Authorization")' in text, "an expression in code, not a credential"
    assert "password = settings.database_password" in text
    for secret in ("sk-live", "correct-horse", "MIIEow", "BEGIN RSA", "END RSA"):
        assert secret not in text, secret
    assert text.count(WITHHELD) == 5 and "done = True" in text
    assert "MIIEow" not in files.read("settings.py", 8, 9), "a key block that starts above the window"
    yaml = files.read("config.yaml")
    assert "hunter2" not in yaml and "host: db.internal" in yaml, "outside code, an unquoted value counts"
    _write(ws / "check.py", 'PEM = "-----BEGIN PRIVATE KEY-----"\n\ndef is_key(text):\n    return PEM in text\n')
    check = ProjectFiles(ws).read("check.py")
    assert "def is_key(text):" in check and "return PEM in text" in check, "a key block ends where key material does"
    search = files.search("password|BEGIN|MII")
    assert "correct-horse" not in search and "hunter2" not in search and "MIIEow" not in search
    assert "settings.py:3: password = settings.database_password" in search
    assert "matching line(s) withheld: they look like credentials" in search


# --------------------------------------------------------------------------- #
# Configuration, prompt, tools
# --------------------------------------------------------------------------- #


def test_the_project_section_turns_project_files_on(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    _write(ws / "hailer.toml", "[hailer]\n")
    assert load_config(workspace=ws, env={}).project.root is None
    _write(ws / "hailer.toml", '[project]\ninclude = ["*.py"]\nexclude = ["tests/**"]\n')
    config = load_config(workspace=ws, env={})
    assert config.project == ProjectConfig(root=ws.resolve(), include=("*.py",), exclude=("tests/**",))
    _write(ws / "hailer.toml", '[project]\nroot = "app"\n')
    assert load_config(workspace=ws, env={}).project.root == (ws / "app").resolve()
    _write(ws / ".hailer" / "local.toml", "[project]\nenabled = false\n")
    assert load_config(workspace=ws, env={}).project.root is None, "one machine can turn it off"


def test_the_project_folder_may_not_be_home_a_drive_or_hold_credentials(tmp_path: Path, monkeypatch) -> None:
    ws = tmp_path / "home" / "me" / "repo"
    _write(ws / "notebooks" / "analysis.py", "import marimo\n")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home" / "me"))
    for root, fragment in (("..", "is your home folder"), ("../..", "contains your home folder"), (ws.anchor, "is a whole drive"),
                           ("../.ssh", "holds credentials")):  # fmt: skip
        (tmp_path / "home" / "me" / ".ssh").mkdir(parents=True, exist_ok=True)
        _write(ws / "hailer.toml", f"[project]\nroot = '{root}'\n")
        errors = [p for p in validate(load_config(workspace=ws, env={})) if not p.startswith("Warning:")]
        assert any(fragment in e for e in errors), (root, errors)
    _write(ws / "hailer.toml", "[project]\nroot = 'missing'\n")
    problems = validate(load_config(workspace=ws, env={}))
    assert any(p.startswith("Warning: the project folder") for p in problems)
    _write(ws / "hailer.toml", "[project]\n")
    assert not any("project folder" in p for p in validate(load_config(workspace=ws, env={})))


def test_the_prompt_says_whether_project_files_can_be_read(tmp_path: Path) -> None:
    on = system_prompt(_config(tmp_path), ContextBundle())
    assert "## Project files" in on and "(the workspace)" in on and "cannot see these files" in on
    assert "## Questions about the project" in on, "the guidance is part of the instructions"
    off = system_prompt(HailerConfig(**{**_config(tmp_path).__dict__, "project": ProjectConfig()}), ContextBundle())
    assert "No project files are available" in off


def test_the_tools_return_project_text_capped_like_any_tool_result(tmp_path: Path) -> None:
    ws = _monorepo(tmp_path)
    config = HailerConfig(**{**_config(ws).__dict__, "max_tool_output_chars": 2000})
    tools = HailerTools(config)
    assert "packages/billing/revenue.py" in tools.project_files("packages/**")
    assert "revenue.py:1: def net_revenue(orders):" in tools.project_search("def net_revenue")
    assert tools.project_read("packages/billing/revenue.py").startswith("packages/billing/revenue.py, lines 1-2 of 2:")
    assert tools.project_read(".env").startswith("ERROR:")
    _write(ws / "big.py", "".join(f"value_{i} = {'x' * 60}\n" for i in range(400)))
    assert len(tools.project_read("big.py")) <= 2000


def test_the_startup_panel_says_which_project_files_the_agent_reads(tmp_path: Path) -> None:
    from hailer.cli.common import _project_line

    assert _project_line(_config(tmp_path)).startswith("the workspace (read-only;")
    assert _project_line(HailerConfig(**{**_config(tmp_path).__dict__, "project": ProjectConfig()})) == "none"
