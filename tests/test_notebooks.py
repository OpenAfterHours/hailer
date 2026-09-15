"""Tests for hailer.notebooks: active-notebook state, listing, name resolution and creation.

Everything runs against tmp_path; no marimo, no network, no symlinks.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from hailer import notebooks as nbs
from hailer.errors import NotebookExistsError, NotebookNotFoundError, NotebookPathError
from hailer.models import HailerConfig

NB_TEXT = "import marimo\n\n__generated_with = \"0.24.2\"\napp = marimo.App()\n"


def _write(path: Path, text: str = NB_TEXT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def make_config(tmp_path: Path, *, notebooks_dir: Path | None = None) -> HailerConfig:
    ws = tmp_path / "ws"
    nb = _write(ws / "notebooks" / "analysis.py")
    cfg_dir = ws / ".config" / "hailer"
    return HailerConfig(
        workspace=ws,
        notebook=nb,
        data_dir=ws / "data",
        context_dir=cfg_dir / "context",
        skills_dir=cfg_dir / "skills",
        prompts_dir=cfg_dir / "prompts",
        notebooks_dir=notebooks_dir,
    )


# --------------------------------------------------------------------------- #
# Active-notebook state
# --------------------------------------------------------------------------- #


def test_state_path_and_default_active(tmp_path):
    cfg = make_config(tmp_path)
    assert nbs.state_path(cfg.workspace) == cfg.workspace / ".hailer" / "notebook.json"
    assert cfg.notebooks_root == cfg.workspace / "notebooks"
    assert nbs.load_active_notebook(cfg) == cfg.notebook
    assert nbs.load_recent(cfg) == []


def test_save_and_load_active_round_trip(tmp_path):
    cfg = make_config(tmp_path)
    other = _write(cfg.notebooks_root / "q2_churn.py")
    nbs.save_active_notebook(cfg, other)
    assert nbs.load_active_notebook(cfg) == other.resolve()
    payload = json.loads(nbs.state_path(cfg.workspace).read_text(encoding="utf-8"))
    assert payload["active"] == "notebooks/q2_churn.py", "stored workspace-relative with forward slashes"
    assert payload["recent"] == ["notebooks/q2_churn.py"]
    assert not nbs.state_path(cfg.workspace).with_suffix(".json.tmp").exists()


def test_active_falls_back_when_state_is_corrupt_or_stale(tmp_path):
    cfg = make_config(tmp_path)
    state = nbs.state_path(cfg.workspace)
    state.parent.mkdir(parents=True)
    state.write_text("{not json", encoding="utf-8")
    assert nbs.load_active_notebook(cfg) == cfg.notebook
    state.write_text(json.dumps(["a list"]), encoding="utf-8")
    assert nbs.load_active_notebook(cfg) == cfg.notebook
    # deleted notebook
    gone = _write(cfg.notebooks_root / "gone.py")
    nbs.save_active_notebook(cfg, gone)
    gone.unlink()
    assert nbs.load_active_notebook(cfg) == cfg.notebook
    # a path outside the notebooks folder is ignored even when it exists
    outside = _write(cfg.workspace / "elsewhere" / "x.py")
    state.write_text(json.dumps({"active": "elsewhere/x.py"}), encoding="utf-8")
    assert outside.exists()
    assert nbs.load_active_notebook(cfg) == cfg.notebook


def test_environment_never_bypasses_the_state_file(tmp_path, monkeypatch):
    """HAILER_NOTEBOOK is inherited by the MCP server; if it won there, tool calls would ignore
    every switch. The CLI persists the override into the state file instead."""
    cfg = make_config(tmp_path)
    other = _write(cfg.notebooks_root / "other.py")
    nbs.save_active_notebook(cfg, other)
    monkeypatch.setenv("HAILER_NOTEBOOK", "anything")
    assert nbs.load_active_notebook(cfg) == other.resolve()
    assert "env" not in inspect.signature(nbs.load_active_notebook).parameters


def test_save_active_notebook_tolerates_a_corrupt_state_file(tmp_path):
    cfg = make_config(tmp_path)
    state = nbs.state_path(cfg.workspace)
    state.parent.mkdir(parents=True)
    state.write_text("{not json", encoding="utf-8")
    other = _write(cfg.notebooks_root / "other.py")
    nbs.save_active_notebook(cfg, other)
    payload = json.loads(state.read_text(encoding="utf-8"))
    assert payload == {"active": "notebooks/other.py", "recent": ["notebooks/other.py"]}
    state.write_text(json.dumps({"active": 3, "recent": "nope"}), encoding="utf-8")
    nbs.save_active_notebook(cfg, other)
    assert nbs.load_active_notebook(cfg) == other.resolve()
    assert nbs.load_recent(cfg) == [other.resolve()]


def test_recent_list_is_bounded_unique_and_filtered(tmp_path):
    cfg = make_config(tmp_path)
    files = [_write(cfg.notebooks_root / f"n{i:02d}.py") for i in range(12)]
    for f in files:
        nbs.save_active_notebook(cfg, f)
    nbs.save_active_notebook(cfg, files[3])  # re-activating moves it to the front, no duplicate
    payload = json.loads(nbs.state_path(cfg.workspace).read_text(encoding="utf-8"))
    assert len(payload["recent"]) == nbs.RECENT_LIMIT == 10
    assert payload["recent"][0] == "notebooks/n03.py"
    assert payload["recent"].count("notebooks/n03.py") == 1
    files[11].unlink()
    recent = nbs.load_recent(cfg)
    assert recent[0] == files[3].resolve()
    assert files[11].resolve() not in recent
    assert all(p.exists() for p in recent)


# --------------------------------------------------------------------------- #
# Finding notebooks
# --------------------------------------------------------------------------- #


def test_is_marimo_notebook(tmp_path):
    assert nbs.is_marimo_notebook(_write(tmp_path / "a.py"))
    assert not nbs.is_marimo_notebook(_write(tmp_path / "b.py", "import polars\n"))
    assert not nbs.is_marimo_notebook(_write(tmp_path / "c.py", "import marimo\n"))  # no marimo.App
    assert not nbs.is_marimo_notebook(_write(tmp_path / "d.txt"))
    assert not nbs.is_marimo_notebook(tmp_path / "missing.py")


def test_list_notebooks_recursive_sorted_and_filtered(tmp_path):
    cfg = make_config(tmp_path)
    root = cfg.notebooks_root
    _write(root / "zeta.py")
    _write(root / "sub" / "x.py")
    _write(root / "helper.py", "def f():\n    return 1\n")
    _write(root / "__marimo__" / "cache.py")
    _write(root / ".hidden" / "h.py")
    _write(root / "__pycache__" / "p.py")
    found = nbs.list_notebooks(cfg)
    assert [nbs.notebook_display_name(cfg, i.path) for i in found] == [
        "notebooks/analysis.py",
        "notebooks/sub/x.py",
        "notebooks/zeta.py",
    ]
    assert found[0].name == "analysis" and found[0].size > 0 and found[0].modified > 0
    assert found[1].name == "x"


def test_list_notebooks_missing_root(tmp_path):
    cfg = make_config(tmp_path, notebooks_dir=tmp_path / "nowhere")
    assert nbs.list_notebooks(cfg) == []


@pytest.mark.parametrize(
    ("name", "slug"),
    [
        ("Q2 Churn (draft).py", "q2_churn_draft"),
        ("  2025 review ", "nb_2025_review"),
        ("a", "a"),
        ("Hello--World__", "hello_world"),
        ("Résumé.PY", "r_sum"),
        # Windows reserved device names get a prefix, whatever the case or extension
        ("CON", "nb_con"),
        ("nul.py", "nb_nul"),
        ("Aux", "nb_aux"),
        ("com1", "nb_com1"),
        ("LPT9.PY", "nb_lpt9"),
        ("con.txt", "con_txt"),  # the slug is not a bare device name, so it is fine
        ("console", "console"),
    ],
)
def test_slugify(name, slug):
    assert nbs.slugify(name) == slug


def test_slugify_caps_the_length():
    slug = nbs.slugify("x" * 300)
    assert slug == "x" * nbs.SLUG_MAX_LEN == "x" * 64
    # the cap never leaves a trailing underscore or exceeds the limit after prefixing
    assert nbs.slugify("9" + "a" * 100) == ("nb_9" + "a" * 100)[:64]
    assert not nbs.slugify("a" * 63 + " b").endswith("_")


def test_slugify_rejects_empty():
    with pytest.raises(ValueError) as info:
        nbs.slugify("___")
    assert "ASCII letter or digit" in str(info.value)
    with pytest.raises(ValueError):
        nbs.slugify(".py")
    with pytest.raises(ValueError):
        nbs.slugify("日本語")


def test_resolve_notebook_accepted_forms(tmp_path):
    cfg = make_config(tmp_path)
    root = cfg.notebooks_root
    q2 = _write(root / "q2_churn.py").resolve()
    sub = _write(root / "sub" / "x.py").resolve()
    assert nbs.resolve_notebook(cfg, "q2 churn") == q2
    assert nbs.resolve_notebook(cfg, "Q2_CHURN") == q2
    assert nbs.resolve_notebook(cfg, "q2_churn.py") == q2
    assert nbs.resolve_notebook(cfg, "notebooks/q2_churn.py") == q2
    assert nbs.resolve_notebook(cfg, "sub/x.py") == sub
    assert nbs.resolve_notebook(cfg, "notebooks/sub/x.py") == sub
    assert nbs.resolve_notebook(cfg, str(sub)) == sub
    assert nbs.resolve_notebook(cfg, " 'analysis' ") == cfg.notebook.resolve()
    assert nbs.resolve_notebook(cfg, "X") == sub, "case-insensitive unique name match, also in subfolders"


def test_resolve_notebook_rejects_paths_outside_the_folder(tmp_path):
    cfg = make_config(tmp_path)
    outside = _write(cfg.workspace / "outside.py")
    with pytest.raises(NotebookPathError) as info:
        nbs.resolve_notebook(cfg, "../outside.py")  # relative to the notebooks folder
    assert "outside the notebooks folder" in str(info.value)
    with pytest.raises(NotebookPathError):
        nbs.resolve_notebook(cfg, "outside.py")  # workspace-relative, exists, but outside
    with pytest.raises(NotebookPathError):
        nbs.resolve_notebook(cfg, str(outside))
    with pytest.raises(NotebookPathError):
        nbs.resolve_notebook(cfg, "../does_not_exist.py")
    elsewhere = _write(tmp_path / "elsewhere" / "nb.py")
    with pytest.raises(NotebookPathError):
        nbs.resolve_notebook(cfg, str(elsewhere))


def test_resolve_notebook_rejects_non_notebooks_and_reports_missing(tmp_path):
    cfg = make_config(tmp_path)
    _write(cfg.notebooks_root / "helper.py", "x = 1\n")
    with pytest.raises(NotebookPathError) as info:
        nbs.resolve_notebook(cfg, "helper.py")
    assert "not a marimo notebook" in str(info.value)
    with pytest.raises(NotebookNotFoundError) as info:
        nbs.resolve_notebook(cfg, "nope")
    assert "No notebook named 'nope'" in str(info.value)
    assert "analysis" in info.value.hint
    with pytest.raises(NotebookNotFoundError):
        nbs.resolve_notebook(cfg, "   ")


def test_resolve_notebook_ambiguous_name(tmp_path):
    cfg = make_config(tmp_path)
    _write(cfg.notebooks_root / "a" / "dup.py")
    _write(cfg.notebooks_root / "b" / "dup.py")
    with pytest.raises(NotebookNotFoundError) as info:
        nbs.resolve_notebook(cfg, "dup")
    assert "Several notebooks match" in str(info.value)
    assert "notebooks/a/dup.py" in str(info.value) and "notebooks/b/dup.py" in str(info.value)
    assert nbs.resolve_notebook(cfg, "a/dup.py") == (cfg.notebooks_root / "a" / "dup.py").resolve()


def test_resolve_notebook_prefers_the_folder_over_the_workspace(tmp_path):
    """A helper script in the workspace root must never shadow a notebook of the same name."""
    cfg = make_config(tmp_path)
    _write(cfg.workspace / "main.py", "print('helper')\n")
    inside = _write(cfg.notebooks_root / "main.py").resolve()
    assert nbs.resolve_notebook(cfg, "main.py") == inside
    assert nbs.resolve_notebook(cfg, "main") == inside
    assert nbs.resolve_notebook(cfg, "notebooks/main.py") == inside


def test_resolve_notebook_folder_relative_without_extension(tmp_path):
    cfg = make_config(tmp_path)
    report = _write(cfg.notebooks_root / "sub" / "report.py").resolve()
    assert nbs.resolve_notebook(cfg, "sub/report") == report
    assert nbs.resolve_notebook(cfg, "sub\\report.py") == report
    with pytest.raises(NotebookNotFoundError) as info:
        nbs.resolve_notebook(cfg, "sub/missing")
    assert "No notebook named" in str(info.value)
    with pytest.raises(NotebookPathError):
        nbs.resolve_notebook(cfg, "../sub/report")


def test_resolve_notebook_survives_odd_input(tmp_path):
    cfg = make_config(tmp_path)
    for odd in ("a\x00b", "q2: churn?", "x" * 400, "..", ".", "\\\\?\\C:\\nope.py"):
        with pytest.raises((NotebookNotFoundError, NotebookPathError)):
            nbs.resolve_notebook(cfg, odd)


def test_notebook_display_name(tmp_path):
    cfg = make_config(tmp_path)
    assert nbs.notebook_display_name(cfg, cfg.notebook) == "notebooks/analysis.py"
    elsewhere = tmp_path / "x" / "y.py"
    assert nbs.notebook_display_name(cfg, elsewhere) == elsewhere.resolve().as_posix()


# --------------------------------------------------------------------------- #
# Templates and creation
# --------------------------------------------------------------------------- #


def _compiles(text: str) -> bool:
    compile(text, "nb.py", "exec")
    return True


def test_marimo_version_from_metadata():
    version = nbs.marimo_version()
    assert version and version[0].isdigit()


@pytest.mark.parametrize("kind", ["starter", "empty"])
def test_render_template_is_valid_python_with_version(kind):
    text = nbs.render_template(kind, title="Q2 churn", version="9.9.9")
    assert _compiles(text)
    assert '__generated_with = "9.9.9"' in text
    assert "import marimo" in text and "marimo.App" in text
    assert "__HAILER" not in text
    if kind == "starter":
        assert "# Q2 churn" in text
        assert "hailer.periods" in text and "WORKSPACE" in text and "period_files" in text
    else:
        assert "Q2 churn" not in text


def test_render_template_uses_installed_version_and_safe_titles():
    text = nbs.render_template("starter", title=' Weird {title} "quoted" \\ name ')
    assert f'__generated_with = "{nbs.marimo_version()}"' in text
    assert _compiles(text)
    assert "{{title}}" in text, "braces are doubled inside the f-string markdown"
    with pytest.raises(ValueError):
        nbs.render_template("fancy", title="x")  # type: ignore[arg-type]


def test_create_notebook_starter_and_empty(tmp_path):
    cfg = make_config(tmp_path)
    created = nbs.create_notebook(cfg, "Q2 churn")
    assert created == (cfg.notebooks_root / "q2_churn.py").resolve()
    text = created.read_text(encoding="utf-8")
    assert _compiles(text) and nbs.is_marimo_notebook(created)
    assert "# Q2 churn" in text and f'__generated_with = "{nbs.marimo_version()}"' in text
    assert "\r\n" not in created.read_bytes().decode("utf-8")
    empty = nbs.create_notebook(cfg, "scratch", kind="empty")
    assert nbs.is_marimo_notebook(empty) and "hailer.periods" not in empty.read_text(encoding="utf-8")
    assert not nbs.state_path(cfg.workspace).exists(), "creation does not touch the active-notebook state"
    assert nbs.load_active_notebook(cfg) == cfg.notebook


def test_create_notebook_refuses_existing_and_bad_names(tmp_path):
    cfg = make_config(tmp_path)
    with pytest.raises(NotebookExistsError) as info:
        nbs.create_notebook(cfg, "Analysis")
    assert "analysis.py" in str(info.value) and info.value.hint
    with pytest.raises(NotebookPathError):
        nbs.create_notebook(cfg, "!!!")


def test_create_notebook_makes_the_folder(tmp_path):
    cfg = make_config(tmp_path, notebooks_dir=tmp_path / "ws" / "deep" / "nbs")
    created = nbs.create_notebook(cfg, "first")
    assert created.parent == (tmp_path / "ws" / "deep" / "nbs").resolve()
    assert nbs.resolve_notebook(cfg, "first") == created


def test_create_notebook_reserved_and_long_names(tmp_path):
    cfg = make_config(tmp_path)
    assert nbs.create_notebook(cfg, "CON").name == "nb_con.py"
    assert nbs.create_notebook(cfg, "nul.py").name == "nb_nul.py"
    created = nbs.create_notebook(cfg, "very " * 100)
    assert len(created.stem) <= nbs.SLUG_MAX_LEN and created.is_file()
    with pytest.raises(NotebookPathError) as info:
        nbs.create_notebook(cfg, "!!! ???")
    assert "ASCII letter or digit" in info.value.hint


def test_create_notebook_maps_os_errors_to_actionable_errors(tmp_path, monkeypatch):
    cfg = make_config(tmp_path)

    def boom(path, text):
        raise OSError(36, "File name too long")

    monkeypatch.setattr(nbs, "_write_new_file", boom)
    with pytest.raises(NotebookPathError) as info:
        nbs.create_notebook(cfg, "whatever")
    assert "Could not create whatever.py" in str(info.value) and "File name too long" in str(info.value)
    assert "shorter or simpler name" in info.value.hint


def test_create_notebook_never_overwrites_a_file_that_appears_late(tmp_path, monkeypatch):
    """The exists() pre-check can be raced; the exclusive-create open is the real guard."""
    cfg = make_config(tmp_path)
    original = cfg.notebook.read_text(encoding="utf-8")
    monkeypatch.setattr(Path, "exists", lambda self: False)
    with pytest.raises(NotebookExistsError):
        nbs.create_notebook(cfg, "analysis")
    assert cfg.notebook.read_text(encoding="utf-8") == original
