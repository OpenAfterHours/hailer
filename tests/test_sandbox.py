"""Tests for hailer.sandbox: the notebook operations both runtimes share, over marimo's file API
(tests/fake_marimo.py serves a tmp folder under the kernel path), and the data folder. Offline."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from fake_marimo import FakeMarimo
from hailer import sandbox as sb
from hailer.errors import (
    HailerError,
    MalformedParquetError,
    MarimoUnavailableError,
    NotebookExistsError,
    NotebookNotFoundError,
    NotebookPathError,
)
from hailer.marimo_client import notebook_file_key
from hailer.models import MarimoServer

TOKEN = "sandbox-token-0123"
NB = 'import marimo\n\napp = marimo.App()\n\n\n@app.cell\ndef _():\n    return\n'


def _write(path: Path, text: str = NB) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


@pytest.fixture
def fake(tmp_path):
    srv = FakeMarimo().start()
    srv.token = TOKEN
    (tmp_path / "notebooks").mkdir()
    srv.serve_files(tmp_path / "notebooks", "/work/notebooks")
    try:
        yield srv
    finally:
        srv.stop()


def docker_box(fake, tmp_path) -> sb.MarimoSandbox:
    """The sandbox as the docker runtime builds it (the kernel knows /work/notebooks)."""
    return sb.MarimoSandbox(
        MarimoServer(url=fake.url, token=TOKEN, runtime="docker"),
        notebooks_path="/work/notebooks", data_path="/work/data", data_dir=tmp_path / "data",
    )  # fmt: skip


def local_box(fake, tmp_path) -> sb.MarimoSandbox:
    """The sandbox as the local runtime builds it (the kernel knows the host folder)."""
    root = notebook_file_key(tmp_path / "notebooks")
    fake.serve_files(tmp_path / "notebooks", root)
    return sb.MarimoSandbox(
        MarimoServer(url=fake.url, token=TOKEN), notebooks_path=root, data_path=str(tmp_path / "data"),
        data_dir=tmp_path / "data", native_paths=True,
    )  # fmt: skip


@pytest.fixture(params=["docker", "local"])
def box(request, fake, tmp_path):
    return docker_box(fake, tmp_path) if request.param == "docker" else local_box(fake, tmp_path)


# --------------------------------------------------------------------------- #
# Notebooks
# --------------------------------------------------------------------------- #


def test_list_notebooks_recursive_sorted_and_filtered(box, tmp_path):
    root = tmp_path / "notebooks"
    _write(root / "zeta.py")
    _write(root / "analysis.py")
    _write(root / "sub" / "x.py")
    _write(root / "a" / "b" / "c" / "d" / "deep.py")  # four folders down: listed
    _write(root / "a" / "b" / "c" / "d" / "e" / "too_deep.py")
    _write(root / "helper.py", "def f():\n    return 1\n")
    _write(root / "notes.md", "marimo-version: 0.24.2\n")
    _write(root / "__marimo__" / "cache.py")
    _write(root / ".hidden" / "h.py")
    _write(root / ".git" / "hooks" / "x.py")
    _write(root / "__pycache__" / "p.py")
    found = box.list_notebooks()
    assert [f.name for f in found] == ["a/b/c/d/deep.py", "analysis.py", "sub/x.py", "zeta.py"]
    assert found[1].stem == "analysis" and found[1].size == len(NB.encode()) and found[1].modified > 0


def test_list_notebooks_of_a_missing_folder_is_empty(fake, tmp_path):
    fake.serve_files(tmp_path / "nowhere", "/work/notebooks")
    assert docker_box(fake, tmp_path).list_notebooks() == []


def test_read_notebook(box, tmp_path, monkeypatch):
    _write(tmp_path / "notebooks" / "q3" / "review.py")
    assert box.read_notebook("q3/review.py") == NB
    assert box.read_notebook("q3\\review.py") == NB
    with pytest.raises(NotebookNotFoundError):
        box.read_notebook("missing.py")
    (tmp_path / "notebooks" / "binary.py").write_bytes(b"\xff\xfe\x00import marimo")
    with pytest.raises(NotebookPathError) as info:
        box.read_notebook("binary.py")
    assert "not a UTF-8 text file" in str(info.value)
    monkeypatch.setattr(sb, "MAX_NOTEBOOK_BYTES", 10)
    with pytest.raises(NotebookPathError) as info:
        box.read_notebook("q3/review.py")
    assert "larger than" in str(info.value)


def test_write_notebook_creates_with_its_folders_and_replaces(box, tmp_path):
    created = box.write_notebook("q3/review.py", NB, replace=False)
    path = tmp_path / "notebooks" / "q3" / "review.py"
    assert created.name == "q3/review.py" and created.size == len(NB.encode())
    assert path.read_text(encoding="utf-8") == NB
    assert "\r\n" not in path.read_bytes().decode("utf-8"), "created from the bytes as given"
    box.write_notebook("q3/review.py", NB + "# changed\n")
    assert path.read_text(encoding="utf-8").endswith("# changed\n")
    assert [f.name for f in box.list_notebooks()] == ["q3/review.py"]


def test_write_notebook_never_overwrites_when_asked_not_to(box, tmp_path):
    original = _write(tmp_path / "notebooks" / "analysis.py", NB + "# mine\n")
    with pytest.raises(NotebookExistsError) as info:
        box.write_notebook("analysis.py", NB, replace=False)
    assert "already exists" in str(info.value) and info.value.hint
    assert original.read_text(encoding="utf-8").endswith("# mine\n")
    (tmp_path / "notebooks" / "folder.py").mkdir()
    with pytest.raises(NotebookExistsError):
        box.write_notebook("folder.py", NB)


def test_a_file_that_appears_during_a_create_is_never_overwritten(fake, tmp_path, monkeypatch):
    """The listing can be raced: marimo then writes <stem>_1.py (it never overwrites); the stray copy
    is deleted and the existing file stays as it was."""
    box = docker_box(fake, tmp_path)
    late = tmp_path / "notebooks" / "analysis.py"
    real_entry = sb.MarimoSandbox._entry

    def racing_entry(self, client, name):
        found = real_entry(self, client, name)
        _write(late, NB + "# theirs\n")  # appears right after the check
        return found

    monkeypatch.setattr(sb.MarimoSandbox, "_entry", racing_entry)
    with pytest.raises(NotebookExistsError):
        box.write_notebook("analysis.py", NB, replace=False)
    assert late.read_text(encoding="utf-8").endswith("# theirs\n")
    assert sorted(p.name for p in (tmp_path / "notebooks").iterdir()) == ["analysis.py"], "the _1 copy was removed"
    box.write_notebook("analysis.py", NB)  # replace=True: the text is then replaced
    assert late.read_text(encoding="utf-8") == NB


@pytest.mark.parametrize("name", ["../escape.py", "/work/data/x.py", "C:/x.py", "a:b.py", ".git/hooks/post.py", "__marimo__/x.py", "con.py", "q3/nul.py", "x.txt", "x.py."])
def test_names_are_checked_before_any_request(fake, tmp_path, name):
    """Security: a name that could leave the notebooks folder, or name a dot-folder, a device or
    a data stream, is refused before anything reaches marimo."""
    box = docker_box(fake, tmp_path)
    before = len(fake.requests)
    for call in (lambda: box.write_notebook(name, NB), lambda: box.read_notebook(name), lambda: box.notebook_url(name)):
        with pytest.raises(NotebookPathError):
            call()
    assert len(fake.requests) == before
    assert not (tmp_path / "escape.py").exists()


def test_every_file_request_carries_the_bearer_and_the_server_token(fake, tmp_path):
    box = docker_box(fake, tmp_path)
    box.write_notebook("a.py", NB)
    box.read_notebook("a.py")
    box.list_notebooks()
    posts = [r for r in fake.requests if r["path"].startswith("/api/files/")]
    assert {r["path"] for r in posts} == {"/api/files/list_files", "/api/files/create", "/api/files/file_details"}
    assert all(r["headers"]["Authorization"] == f"Bearer {TOKEN}" and r["headers"]["Marimo-Server-Token"] == fake.server_token for r in posts)


def test_urls_and_describe(fake, tmp_path):
    box = docker_box(fake, tmp_path)
    assert box.home_url(with_token=True) == f"{fake.url}/?access_token={TOKEN}" and box.home_url() == f"{fake.url}/"
    assert TOKEN not in repr(box)
    assert box.notebook_url("q3/r.py") == f"{fake.url}/?file=/work/notebooks/q3/r.py&view-as=present"
    assert box.notebook_url("q3/r.py", with_token=True).endswith(f"&access_token={TOKEN}")
    assert box.describe().startswith("docker (hailer-kernel ")
    assert box.sync_soon() is None, "the shared part copies nothing"
    box.stop()  # nothing to stop in the shared part; runtimes override it


def test_list_tree_lists_everything_but_never_looks_into_dot_or_dunder_folders(box, tmp_path):
    root = tmp_path / "notebooks"
    _write(root / "sales.py")
    _write(root / "q3" / "review.py")
    _write(root / "out.csv", "a\n")
    _write(root / ".git" / "config", "[core]\n")
    _write(root / "__marimo__" / "session" / "sales.py.json", "{}\n")
    _write(root / "a" / "b" / "c" / "d" / "e" / "deep.py")
    tree = box.list_tree()
    assert [(e.name, e.folder) for e in tree] == [
        (".git", True), ("__marimo__", True), ("a", True), ("a/b", True), ("a/b/c", True), ("a/b/c/d", True),
        ("a/b/c/d/e", True), ("out.csv", False), ("q3", True), ("q3/review.py", False), ("sales.py", False),
    ]  # fmt: skip
    assert next(e for e in tree if e.name == "sales.py").size == len(NB.encode())
    notices: list[str] = []
    box.notice = notices.append
    box.notify("careful")
    assert notices == ["careful"]


def _inside_root(box: sb.MarimoSandbox, paths: list[str]) -> bool:
    """Every path a file endpoint was asked about is the notebooks folder or inside it."""
    root = box.notebooks_path.replace("\\", "/").rstrip("/").casefold()
    return all(p.replace("\\", "/").casefold() == root or p.replace("\\", "/").casefold().startswith(root + "/") for p in paths)


def test_hailer_never_sends_a_path_outside_the_notebooks_folder(box, fake, tmp_path):
    """marimo confines nothing (it reads and writes any path it is given), so Hailer must."""
    _write(tmp_path / "notebooks" / "q3" / "review.py")
    box.list_notebooks()
    box.write_notebook("new.py", NB)
    box.read_notebook("q3/review.py")
    box.has_notebook("missing/x.py")
    for bad in ("../escape.py", "C:/x.py", "/etc/x.py"):
        with pytest.raises(NotebookPathError):
            box.write_notebook(bad, NB)
    assert fake.file_paths and _inside_root(box, fake.file_paths), fake.file_paths


def _link(link: Path, target: Path) -> None:
    """A directory link: a junction on Windows (no privilege needed), a symlink elsewhere."""
    target.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, text=True)
        if result.returncode != 0:
            pytest.skip(f"cannot create a junction here: {result.stdout} {result.stderr}")
    else:
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError as err:
            pytest.skip(f"cannot create a symlink here: {err}")


def test_a_link_out_of_the_notebooks_folder_is_never_followed(fake, tmp_path):
    """Security (live probe on marimo 0.24.2: jn/escape.py landed outside): a local kernel's
    writes, reads and listings stop at a link or junction that leaves the notebooks folder."""
    box = local_box(fake, tmp_path)
    outside = tmp_path / "outside"
    _link(tmp_path / "notebooks" / "jn", outside)
    _write(outside / "secret.py")
    with pytest.raises(NotebookPathError) as info:
        box.write_notebook("jn/escape.py", NB)
    assert "outside the notebooks folder" in str(info.value)
    with pytest.raises(NotebookPathError):
        box.read_notebook("jn/secret.py")
    assert not box.has_notebook("jn/secret.py")
    assert sorted(p.name for p in outside.iterdir()) == ["secret.py"], "nothing was written through the link"
    assert box.list_notebooks() == [], "nothing behind the link is listed"


@pytest.mark.skipif(os.name != "nt", reason="names ignore case on a Windows kernel only")
def test_a_case_variant_is_the_same_notebook_on_windows(fake, tmp_path):
    """Live probe: Q3/Review.py overwrote q3/review.py and came back spelt Q3/Review.py."""
    box = local_box(fake, tmp_path)
    _write(tmp_path / "notebooks" / "q3" / "review.py", NB + "# mine\n")
    with pytest.raises(NotebookExistsError) as info:
        box.write_notebook("Q3/REVIEW.py", NB, replace=False)
    assert "q3/review.py" in str(info.value), "the listing's spelling"
    assert (tmp_path / "notebooks" / "q3" / "review.py").read_text(encoding="utf-8").endswith("# mine\n")
    assert box.write_notebook("Q3/Review.py", NB).name == "q3/review.py"
    assert [f.name for f in box.list_notebooks()] == ["q3/review.py"]
    assert box.has_notebook("Q3/REVIEW.PY")


def test_listing_is_bounded_and_skips_tool_folders(box, tmp_path, monkeypatch):
    root = tmp_path / "notebooks"
    for folder in ("node_modules", "venv", ".venv", "__pycache__", "__marimo__", ".git"):
        _write(root / folder / "x.py")
    for i in range(6):
        _write(root / f"n{i}.py")
    assert [f.name for f in box.list_notebooks()] == [f"n{i}.py" for i in range(6)]
    monkeypatch.setattr(sb, "MAX_NOTEBOOKS", 3)
    assert len(box.list_notebooks()) == 3
    monkeypatch.setattr(sb, "MAX_NOTEBOOKS", 500)
    for i in range(5):
        _write(root / f"f{i}" / "y.py")
    monkeypatch.setattr(sb, "MAX_NOTEBOOK_FOLDERS", 2)
    listed = [f.name for f in box.list_notebooks()]
    assert sum("/" in name for name in listed) == 1, "the root and one more folder were looked into"


def test_writes_stop_at_the_listing_depth(box, tmp_path):
    assert box.write_notebook("a/b/c/d/four.py", NB).name == "a/b/c/d/four.py"
    with pytest.raises(NotebookPathError) as info:
        box.write_notebook("a/b/c/d/e/five.py", NB)
    assert "more than 4 folders deep" in str(info.value)
    assert "a/b/c/d/four.py" in [f.name for f in box.list_notebooks()], "what can be written can be listed"


def test_has_notebook_is_one_request(box, fake, tmp_path):
    _write(tmp_path / "notebooks" / "q3" / "review.py")
    before = len(fake.file_paths)
    assert box.has_notebook("q3/review.py") and not box.has_notebook("q3/other.py") and not box.has_notebook("q3.py")
    assert len(fake.file_paths) - before == 3


def test_marimo_errors_are_hailer_errors(box, fake):
    """An HTTP error from a file endpoint reaches the CLI's handlers as a HailerError."""
    fake.mode = "files_500"
    for call in (box.list_notebooks, lambda: box.write_notebook("x.py", NB), lambda: box.has_notebook("x.py")):
        with pytest.raises(MarimoUnavailableError) as info:
            call()
        assert "HTTP 500" in str(info.value)


def test_notebook_digest_ignores_line_endings(box, tmp_path):
    """marimo's update writes the kernel's line endings, create the bytes as given."""
    assert sb.notebook_digest("a\r\nb\n") == sb.notebook_digest("a\nb\n") != sb.notebook_digest("a\nc\n")
    box.write_notebook("x.py", NB)
    box.write_notebook("x.py", NB)  # replaced by marimo's update
    assert sb.notebook_digest(box.read_notebook("x.py")) == sb.notebook_digest(NB)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def test_list_data_is_the_host_data_folder_flat(fake, tmp_path):
    box = docker_box(fake, tmp_path)
    with pytest.raises(HailerError) as info:
        box.list_data()
    assert str(info.value) == "Data directory does not exist: /work/data"
    data = tmp_path / "data"
    (data / "sub").mkdir(parents=True)
    (data / "sub" / "inner.csv").write_text("x", encoding="utf-8")
    (data / ".secret.csv").write_text("x", encoding="utf-8")
    (data / "b.csv").write_text("x", encoding="utf-8")
    (data / "A.parquet").write_bytes(b"x")
    assert [(f.name, f.size) for f in box.list_data()] == [("A.parquet", 1), ("b.csv", 1)]


def test_data_schema(fake, tmp_path):
    pl = pytest.importorskip("polars")
    box = docker_box(fake, tmp_path)
    (tmp_path / "data").mkdir()
    pl.DataFrame({"region": ["N"], "revenue": [1.0]}).write_parquet(tmp_path / "data" / "25-01 sales.parquet")
    assert box.data_schema("25-01 sales.parquet") == {"region": "String", "revenue": "Float64"}
    (tmp_path / "data" / "broken.parquet").write_bytes(b"not parquet")
    with pytest.raises(MalformedParquetError):
        box.data_schema("broken.parquet")
    _write(tmp_path / "secret.parquet", "x")
    for bad in ("../secret.parquet", "sub/x.parquet", "..\\secret.parquet", ".hidden.parquet", "C:x.parquet"):
        with pytest.raises(HailerError) as info:
            box.data_schema(bad)
        assert "not a file name in the data folder" in str(info.value), bad
