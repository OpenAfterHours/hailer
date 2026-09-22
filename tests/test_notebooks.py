"""Tests for hailer.notebooks: notebook names, active-notebook state (and the old format's
conversion), name resolution and templates.

Everything runs against tmp_path; no marimo, no network, no symlinks. The files themselves are
the sandbox's (tests/test_sandbox.py).
"""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import pytest

from hailer import notebooks as nbs
from hailer.errors import NotebookNotFoundError, NotebookPathError
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


def _state(cfg: HailerConfig) -> dict:
    return json.loads(nbs.state_path(cfg.workspace).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["sales.py", "q3/review.py", "a/b/c/d.py", "Q2 Churn.py", "résumé.py", "x.PY", "con_notes.py"])
def test_check_notebook_name_accepts_names_inside_the_folder(name):
    assert nbs.check_notebook_name(name) == name


def test_check_notebook_name_takes_either_separator():
    assert nbs.check_notebook_name("q3\\review.py") == "q3/review.py"


@pytest.mark.parametrize(
    "name",
    [
        "", "/work/notebooks/x.py", "\\\\server\\share\\x.py", "C:/x.py", "C:x.py", "../x.py", "a/../../x.py",
        "a/./x.py", "a//x.py", "./x.py", ".git/hooks/x.py", ".x.py", "__marimo__/x.py", "__init__.py",
        "x.txt", "x", "a:b.py", "x.py:stream", "con.py", "NUL.py", "q3/com1.py", "aux/x.py", "x.py.", "x.py ",
        "trailing./x.py", "a\x00b.py", "a\nb.py", "a?.py", "a*.py", 'a".py', "a|b.py", "a<b.py",
        "CONIN$.py", "conout$.py", "COM¹.py", "lpt³/x.py", "con .py", "a‮.py", "x" * 256 + ".py",
    ],
)
def test_check_notebook_name_refuses_escapes_and_unsafe_names(name):
    """Security: a name never leaves the notebooks folder, never names a dot-folder, a Windows
    device, a data stream or a path Windows would alias (trailing dot or space)."""
    with pytest.raises(NotebookPathError):
        nbs.check_notebook_name(name)


def test_notebook_name_is_path_arithmetic_on_the_notebooks_folder(tmp_path):
    cfg = make_config(tmp_path)
    root = cfg.notebooks_root
    assert nbs.notebook_name(cfg, root / "sub" / "x.py") == "sub/x.py"
    assert nbs.notebook_name(cfg, "notebooks/analysis.py") == "analysis.py", "relative to the workspace"
    assert nbs.notebook_name(cfg, str(root / "missing.py")) == "missing.py", "nothing is read"
    assert nbs.notebook_name(cfg, cfg.workspace / "elsewhere.py") is None
    assert nbs.notebook_name(cfg, root / ".." / "notebooks2" / "x.py") is None
    assert nbs.notebook_name(cfg, root / ".git" / "x.py") is None, "not a valid name"
    assert nbs.notebook_name(cfg, root) is None
    if Path("C:/").is_absolute():  # Windows: the folder matches whatever the case
        assert nbs.notebook_name(cfg, str(root / "A.py").upper()) == "A.PY"


def test_default_notebook(tmp_path):
    cfg = make_config(tmp_path)
    assert nbs.default_notebook(cfg) == "analysis.py"
    nested = make_config(tmp_path, notebooks_dir=tmp_path / "ws")
    assert nbs.default_notebook(nested) == "notebooks/analysis.py"
    outside = make_config(tmp_path, notebooks_dir=tmp_path / "ws" / "other")
    assert nbs.default_notebook(outside) == "analysis.py", "outside the folder: its file name (doctor reports it)"


# --------------------------------------------------------------------------- #
# Active-notebook state
# --------------------------------------------------------------------------- #


def test_state_path_and_default_active(tmp_path):
    cfg = make_config(tmp_path)
    assert nbs.state_path(cfg.workspace) == cfg.workspace / ".hailer" / "notebook.json"
    assert cfg.notebooks_root == cfg.workspace / "notebooks"
    assert nbs.load_active_notebook(cfg) == "analysis.py"
    assert nbs.load_recent(cfg) == []


def test_save_and_load_active_round_trip(tmp_path):
    cfg = make_config(tmp_path)
    nbs.save_active_notebook(cfg, "q2_churn.py")
    assert nbs.load_active_notebook(cfg) == "q2_churn.py"
    assert _state(cfg) == {"version": 2, "active": "q2_churn.py", "recent": ["q2_churn.py"]}, "names, not paths"
    assert not nbs.state_path(cfg.workspace).with_suffix(".json.tmp").exists()
    nbs.save_active_notebook(cfg, "q3\\review.py")
    assert nbs.load_active_notebook(cfg) == "q3/review.py"
    with pytest.raises(NotebookPathError):
        nbs.save_active_notebook(cfg, "../escape.py")


def test_the_state_file_is_read_without_touching_the_notebooks_folder(tmp_path):
    """Whether the active notebook still exists is the sandbox's to say (the tools never read the
    folder); a name the state file holds is returned as it is."""
    cfg = make_config(tmp_path)
    nbs.save_active_notebook(cfg, "gone.py")
    assert nbs.load_active_notebook(cfg) == "gone.py"


def test_active_falls_back_when_state_is_corrupt_or_invalid(tmp_path):
    cfg = make_config(tmp_path)
    state = nbs.state_path(cfg.workspace)
    state.parent.mkdir(parents=True)
    state.write_text("{not json", encoding="utf-8")
    assert nbs.load_active_notebook(cfg) == "analysis.py"
    state.write_text(json.dumps(["a list"]), encoding="utf-8")
    assert nbs.load_active_notebook(cfg) == "analysis.py"
    for bad in ("../x.py", "/work/notebooks/x.py", ".git/x.py", "con.py", 3, None):
        state.write_text(json.dumps({"version": 2, "active": bad, "recent": [bad, "ok.py"]}), encoding="utf-8")
        assert nbs.load_active_notebook(cfg) == "analysis.py", bad
        assert nbs.load_recent(cfg) == ["ok.py"]


def test_an_old_state_file_is_converted_to_names(tmp_path):
    """Before names, notebook.json held host paths (workspace-relative, or absolute outside the
    workspace). Entries inside the notebooks folder become names; anything else is dropped; the
    next save writes the new format."""
    cfg = make_config(tmp_path)
    state = nbs.state_path(cfg.workspace)
    state.parent.mkdir(parents=True)
    old = {
        "active": "notebooks/q3/review.py",
        "recent": [
            "notebooks/q3/review.py",
            str(cfg.notebooks_root / "sales.py"),  # absolute, inside
            "elsewhere/x.py",  # the workspace, outside the folder
            str(tmp_path / "other" / "y.py"),  # absolute, outside
            "notebooks/../secrets.py",
            "notebooks/.git/hooks.py",
            42,
        ],
    }
    state.write_text(json.dumps(old), encoding="utf-8")
    assert nbs.load_active_notebook(cfg) == "q3/review.py"
    assert nbs.load_recent(cfg) == ["q3/review.py", "sales.py"]
    nbs.save_active_notebook(cfg, "sales.py")
    assert _state(cfg) == {"version": 2, "active": "sales.py", "recent": ["sales.py", "q3/review.py"]}


@pytest.mark.parametrize(
    "payload",
    [{"recent": 5}, {"active": 5, "recent": [1, None]}, [1], "x", {"version": 2, "recent": {"a": 1}}, {"version": "2"}, {"version": 3, "active": "a.py"}],
)
def test_a_state_file_of_the_wrong_shape_or_a_newer_version_is_ignored(tmp_path, payload):
    """Startup must never crash on notebook.json; a version this Hailer does not know starts fresh."""
    cfg = make_config(tmp_path)
    state = nbs.state_path(cfg.workspace)
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps(payload), encoding="utf-8")
    assert nbs.load_active_notebook(cfg) == "analysis.py"
    assert nbs.load_recent(cfg) == []
    nbs.save_active_notebook(cfg, "b.py")
    assert _state(cfg) == {"version": 2, "active": "b.py", "recent": ["b.py"]}


def test_saving_leaves_no_temporary_file(tmp_path):
    cfg = make_config(tmp_path)
    for name in ("a.py", "b.py"):
        nbs.save_active_notebook(cfg, name)
    assert [p.name for p in nbs.state_path(cfg.workspace).parent.iterdir() if "notebook" in p.name] == ["notebook.json"]


@pytest.mark.skipif(os.name != "nt", reason="names ignore case on Windows only")
def test_case_variants_are_one_notebook_on_windows(tmp_path):
    cfg = make_config(tmp_path)
    nbs.save_active_notebook(cfg, "q3/review.py")
    nbs.save_active_notebook(cfg, "Q3/Review.py")
    assert nbs.load_recent(cfg) == ["Q3/Review.py"]
    state = nbs.state_path(cfg.workspace)
    state.write_text(json.dumps({"version": 2, "active": "a.py", "recent": ["a.py", "A.py"]}), encoding="utf-8")
    assert nbs.load_recent(cfg) == ["a.py"]
    assert nbs.same_name("A.py", "a.py")


def test_an_old_active_entry_outside_the_folder_is_dropped(tmp_path):
    cfg = make_config(tmp_path)
    state = nbs.state_path(cfg.workspace)
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({"active": str(tmp_path / "elsewhere" / "x.py")}), encoding="utf-8")
    assert nbs.load_active_notebook(cfg) == "analysis.py"
    # a new-format file with a path in it is not a name either
    state.write_text(json.dumps({"version": 2, "active": "notebooks/analysis.py"}), encoding="utf-8")
    assert nbs.load_active_notebook(cfg) == "notebooks/analysis.py", "a valid name (a subfolder called notebooks)"


def test_environment_never_bypasses_the_state_file(tmp_path, monkeypatch):
    """HAILER_NOTEBOOK stays set for the whole session; if it won in the tools, tool calls would
    ignore every switch. The CLI persists the override into the state file instead."""
    cfg = make_config(tmp_path)
    nbs.save_active_notebook(cfg, "other.py")
    monkeypatch.setenv("HAILER_NOTEBOOK", "anything")
    assert nbs.load_active_notebook(cfg) == "other.py"
    assert "env" not in inspect.signature(nbs.load_active_notebook).parameters


def test_save_active_notebook_tolerates_a_corrupt_state_file(tmp_path):
    cfg = make_config(tmp_path)
    state = nbs.state_path(cfg.workspace)
    state.parent.mkdir(parents=True)
    state.write_text("{not json", encoding="utf-8")
    nbs.save_active_notebook(cfg, "other.py")
    assert _state(cfg) == {"version": 2, "active": "other.py", "recent": ["other.py"]}
    state.write_text(json.dumps({"version": 2, "active": 3, "recent": "nope"}), encoding="utf-8")
    nbs.save_active_notebook(cfg, "other.py")
    assert nbs.load_active_notebook(cfg) == "other.py"
    assert nbs.load_recent(cfg) == ["other.py"]


def test_recent_list_is_bounded_and_unique(tmp_path):
    cfg = make_config(tmp_path)
    names = [f"n{i:02d}.py" for i in range(12)]
    for name in names:
        nbs.save_active_notebook(cfg, name)
    nbs.save_active_notebook(cfg, names[3])  # re-activating moves it to the front, no duplicate
    recent = _state(cfg)["recent"]
    assert len(recent) == nbs.RECENT_LIMIT == 10
    assert recent[0] == "n03.py" and recent.count("n03.py") == 1
    assert nbs.load_recent(cfg) == recent


# --------------------------------------------------------------------------- #
# Finding notebooks
# --------------------------------------------------------------------------- #


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
    assert nbs.check_notebook_name(f"{slug}.py"), "every slug is a valid name"


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


NAMES = ["analysis.py", "q2_churn.py", "sub/x.py", "sub/report.py"]


def test_resolve_notebook_accepted_forms(tmp_path):
    cfg = make_config(tmp_path)
    folders = nbs.reference_folders(cfg, "/work/notebooks")
    assert nbs.resolve_notebook("q2 churn", NAMES) == "q2_churn.py"
    assert nbs.resolve_notebook("Q2_CHURN", NAMES) == "q2_churn.py"
    assert nbs.resolve_notebook("q2_churn.py", NAMES) == "q2_churn.py"
    assert nbs.resolve_notebook("sub/x.py", NAMES) == "sub/x.py"
    assert nbs.resolve_notebook("sub\\x.py", NAMES) == "sub/x.py"
    assert nbs.resolve_notebook("sub/report", NAMES) == "sub/report.py"
    assert nbs.resolve_notebook(" 'analysis' ", NAMES) == "analysis.py"
    assert nbs.resolve_notebook("X", NAMES) == "sub/x.py", "case-insensitive unique name match, also in subfolders"
    # after one of the folders: the kernel's path, the workspace-relative folder, the host path
    assert nbs.resolve_notebook("/work/notebooks/sub/x.py", NAMES, folders=folders) == "sub/x.py"
    assert nbs.resolve_notebook("notebooks/q2_churn.py", NAMES, folders=folders) == "q2_churn.py"
    assert nbs.resolve_notebook(str(cfg.notebooks_root / "sub" / "x.py"), NAMES, folders=folders) == "sub/x.py"


def test_resolve_notebook_rejects_paths_outside_the_folder(tmp_path):
    cfg = make_config(tmp_path)
    folders = nbs.reference_folders(cfg, "/work/notebooks")
    for ref in ("../outside.py", "../does_not_exist.py", "/work/data/x.py", "/etc/passwd", str(tmp_path / "elsewhere" / "nb.py"), "C:/x.py", "~/x.py"):
        with pytest.raises(NotebookPathError) as info:
            nbs.resolve_notebook(ref, NAMES, folders=folders)
        assert "outside the notebooks folder" in str(info.value), ref


def test_resolve_notebook_reports_missing(tmp_path):
    with pytest.raises(NotebookNotFoundError) as info:
        nbs.resolve_notebook("nope", NAMES)
    assert "No notebook named 'nope'" in str(info.value)
    assert "analysis.py" in info.value.hint
    with pytest.raises(NotebookNotFoundError):
        nbs.resolve_notebook("   ", NAMES)
    with pytest.raises(NotebookNotFoundError) as info:
        nbs.resolve_notebook("x", [])
    assert "no notebooks" in info.value.hint


def test_resolve_notebook_ambiguous_name():
    names = ["a/dup.py", "b/dup.py"]
    with pytest.raises(NotebookNotFoundError) as info:
        nbs.resolve_notebook("dup", names)
    assert "Several notebooks match" in str(info.value)
    assert "a/dup.py" in str(info.value) and "b/dup.py" in str(info.value)
    assert nbs.resolve_notebook("a/dup.py", names) == "a/dup.py"


def test_resolve_notebook_refuses_an_explicit_escape_even_with_an_inside_twin():
    with pytest.raises(NotebookPathError) as info:
        nbs.resolve_notebook("../analysis.py", NAMES)
    assert "points outside the notebooks folder" in str(info.value)
    with pytest.raises(NotebookPathError):
        nbs.resolve_notebook("sub/../../analysis.py", NAMES)


def test_resolve_notebook_survives_odd_input():
    for odd in ("a\x00b", "q2: churn?", "x" * 400, "..", ".", "\\\\?\\C:\\nope.py"):
        with pytest.raises((NotebookNotFoundError, NotebookPathError)):
            nbs.resolve_notebook(odd, ["analysis.py"])


def test_notebook_display_name(tmp_path):
    cfg = make_config(tmp_path)
    assert nbs.notebook_display_name(cfg, cfg.notebook) == "notebooks/analysis.py"
    elsewhere = tmp_path / "x" / "y.py"
    assert nbs.notebook_display_name(cfg, elsewhere) == elsewhere.resolve().as_posix()


# --------------------------------------------------------------------------- #
# Templates
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
        assert "data_files = list_data_files(DATA_DIR)" in text and "pra101" not in text
    else:
        assert "Q2 churn" not in text


def test_render_template_uses_installed_version_and_safe_titles():
    text = nbs.render_template("starter", title=' Weird {title} "quoted" \\ name ')
    assert f'__generated_with = "{nbs.marimo_version()}"' in text
    assert _compiles(text)
    assert "{{title}}" in text, "braces are doubled inside the f-string markdown"
    with pytest.raises(ValueError):
        nbs.render_template("fancy", title="x")  # type: ignore[arg-type]



def test_new_notebook_renders_the_template_under_its_slug():
    filename, text = nbs.new_notebook("Q2 churn")
    assert filename == "q2_churn.py"
    assert _compiles(text) and "import marimo" in text and "marimo.App" in text
    assert "# Q2 churn" in text and f'__generated_with = "{nbs.marimo_version()}"' in text
    filename, empty = nbs.new_notebook("scratch", kind="empty")
    assert filename == "scratch.py" and "hailer.periods" not in empty


def test_new_notebook_reserved_long_and_bad_names():
    assert nbs.new_notebook("CON")[0] == "nb_con.py"
    assert nbs.new_notebook("nul.py")[0] == "nb_nul.py"
    assert len(nbs.new_notebook("very " * 100)[0]) <= nbs.SLUG_MAX_LEN + 3
    with pytest.raises(NotebookPathError) as info:
        nbs.new_notebook("!!! ???")
    assert "ASCII letter or digit" in info.value.hint


def test_ensure_notebook_writes_once_on_the_host(tmp_path):
    """Workspace setup (uvx hailer init, the first run) stays host-side."""
    target = tmp_path / "deep" / "nbs" / "analysis.py"
    assert nbs.ensure_notebook(target, title="Analysis")
    assert "\r\n" not in target.read_bytes().decode("utf-8")
    target.write_text("mine", encoding="utf-8")
    assert not nbs.ensure_notebook(target, title="Analysis")
    assert target.read_text(encoding="utf-8") == "mine", "an existing file is never touched"
