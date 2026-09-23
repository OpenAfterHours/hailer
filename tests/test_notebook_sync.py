"""Tests for hailer.notebook_sync: the notebook copies between the workspace's notebooks folder and a
sandbox with its own (the docker kernel's tmpfs). tests/fake_marimo.py serves a separate "container"
folder under /work/notebooks, so every copy goes through marimo's file API as it does in Docker.
The allow-list is a security boundary: host tools run what lands here. Offline."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

from fake_docker import FakeDocker
from fake_kernel import TOKEN, make_config
from fake_marimo import FakeMarimo
from hailer import kernel_docker as kd
from hailer import notebook_sync as ns
from hailer.models import KernelConfig, MarimoServer
from hailer.sandbox import MAX_NOTEBOOK_BYTES, MarimoSandbox, SandboxEntry

NB = 'import marimo\n\napp = marimo.App()\n\n\n@app.cell\ndef _():\n    return\n'


def notebook(marker: str = "") -> str:
    return NB + (f"# {marker}\n" if marker else "")


def _write(path: Path, text: str = NB) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _link(link: Path, target: Path, *, folder: bool = True) -> None:
    """A symlink, or on Windows without the privilege for one, a junction (folders only)."""
    try:
        link.symlink_to(target, target_is_directory=folder)
    except OSError:
        if os.name != "nt" or not folder:
            pytest.skip("cannot create links here")
        import _winapi

        try:
            _winapi.CreateJunction(str(target), str(link))
        except OSError:
            pytest.skip("cannot create links here")


@pytest.fixture
def fake(tmp_path):
    srv = FakeMarimo().start()
    srv.token = TOKEN
    (tmp_path / "container").mkdir()
    srv.serve_files(tmp_path / "container", "/work/notebooks")
    try:
        yield srv
    finally:
        srv.stop()


@pytest.fixture
def said():
    return []


@pytest.fixture
def box(fake, tmp_path, said):
    """The sandbox as the docker runtime builds it: its notebooks are /work/notebooks (the fake
    serves ``container``), and its warnings are recorded."""
    return MarimoSandbox(
        MarimoServer(url=fake.url, token=TOKEN, runtime="docker"),
        notebooks_path="/work/notebooks", data_path="/work/data", data_dir=tmp_path / "data", notice=said.append,
    )  # fmt: skip


@pytest.fixture
def host(tmp_path):
    folder = tmp_path / "workspace" / "notebooks"
    folder.mkdir(parents=True)
    return folder


@pytest.fixture
def sync(box, host):
    return ns.NotebookSync(box, host, host.parent)


def container(tmp_path) -> Path:
    return tmp_path / "container"


# --------------------------------------------------------------------------- #
# The allow-list (both directions)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["sales.py", "q3/review.py", "a/b/c/d/deep.py", "Sales 2025.py", "tests.py", "contest.py"])
def test_the_allow_list_accepts_plain_notebook_names(name):
    assert ns.sync_refusal(name) is None
    assert ns.sync_refusal(name, NB) is None


@pytest.mark.parametrize(
    "name",
    [
        "", "/abs.py", "C:/x.py", "../x.py", "a/../x.py", "q3\\review.py", "a//b.py", "./x.py",
        ".git/hooks/post-checkout.py", ".vscode/x.py", ".hidden.py", "q3/.x.py",
        "__marimo__/x.py", "__pycache__/x.py", "q3/__pycache__/x.py",
        "notes.txt", "x.PY", "x.pyc", "x.pth", "q3", "a/b/c/d/e/too_deep.py",
        "conftest.py", "q3/conftest.py", "CONFTEST.py", "setup.py", "sitecustomize.py", "usercustomize.py",
        "__init__.py", "q3/__init__.py", "__main__.py", "noxfile.py",
        "test_sales.py", "q3/Test_review.py", "sales_test.py", "Sales_TEST.py",
        "conf.py", "tasks.py", "fabfile.py", "dodo.py", "manage.py", "gunicorn.conf.py", "ipython_config.py",
        "jupyter_server_config.py", "Jupyter_Notebook_Config.py",
        "pandas.py", "json.py", "polars.py", "Json.py", "json/sales.py", "q3/polars/review.py", "hailer.py", "marimo.py",
        "PROGRA~1/x.py", "SALES~1.py", "a~9b.py",
        "con.py", "a:b.py", "trailing./x.py", "space /x.py", "x\u202e.py",
    ],
)  # fmt: skip
def test_the_allow_list_refuses_everything_else(name):
    """Names other tools run, import instead of a module (a json.py next to a script shadows json),
    or that Windows may resolve to another file (8.3 short names), besides everything not a plain name."""
    assert ns.sync_refusal(name) is not None
    assert ns.sync_refusal(name, NB) is not None, "a notebook's text never excuses its name"


def test_a_modules_name_is_named_in_the_warning(sync, host, said):
    _write(host / "pandas.py")
    _write(host / "sales.py")
    assert sync.sync_in() == ["sales.py"]
    assert said == ["Not copied into the kernel (only marimo notebooks with plain names are): pandas.py (the name of the Python module pandas)."]


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("print('hello')\n", "not a marimo notebook"),
        ("import marimo\n", "not a marimo notebook"),
        ("app = marimo.App()\n", "not a marimo notebook"),
        (NB + "\x00", "not text"),
        (NB + "\ud800", "not UTF-8 text"),
        (NB + "#" * MAX_NOTEBOOK_BYTES, "larger than 5 MiB"),
    ],
    ids=["plain-python", "no-app", "no-import", "nul", "surrogate", "too-large"],
)
def test_the_allow_list_checks_the_text_too(text, reason):
    assert ns.sync_refusal("sales.py", text) == reason
    assert ns.sync_refusal("sales.py", NB) is None


# --------------------------------------------------------------------------- #
# In: when the kernel starts
# --------------------------------------------------------------------------- #


def test_sync_in_copies_the_notebooks_and_names_what_it_left_once(sync, host, tmp_path, said):
    _write(host / "sales.py", notebook("sales"))
    _write(host / "q3" / "review.py", notebook("review"))
    _write(host / "helpers.py", "def f():\n    return 1\n")  # a module, not a notebook
    _write(host / "test_sales.py", NB)
    _write(host / "conftest.py", NB)
    _write(host / "big.py", NB + "#" * MAX_NOTEBOOK_BYTES)
    _write(host / "latin1.py", "")
    (host / "latin1.py").write_bytes(NB.encode() + b"# \xe9\n")
    _write(host / "a" / "b" / "c" / "d" / "e" / "deep.py")
    _write(host / ".ipynb_checkpoints" / "x.py")  # hidden, marimo's and packaging folders are skipped quietly
    _write(host / "__marimo__" / "x.py")
    _write(host / "node_modules" / "x.py")
    _write(host / "sales.csv", "region,revenue\n")  # not a .py file: not a word
    assert sync.sync_in() == ["q3/review.py", "sales.py"]
    box = container(tmp_path)
    assert sorted(p.relative_to(box).as_posix() for p in box.rglob("*") if p.is_file()) == ["q3/review.py", "sales.py"]
    assert (box / "q3" / "review.py").read_text(encoding="utf-8") == notebook("review")
    assert said == ["Not copied into the kernel (only marimo notebooks with plain names are): big.py, conftest.py, helpers.py, latin1.py, test_sales.py."]


def test_sync_in_of_a_missing_or_empty_folder_copies_nothing(box, tmp_path, said):
    assert ns.NotebookSync(box, tmp_path / "nowhere", tmp_path).sync_in() == []
    assert said == []


def test_sync_in_never_follows_a_link_out_of_the_folder(sync, host, tmp_path, said):
    elsewhere = tmp_path / "elsewhere"
    _write(elsewhere / "secret.py", notebook("outside"))
    _link(host / "shared", elsewhere)
    _write(host / "sales.py")
    assert sync.sync_in() == ["sales.py"]
    assert not (container(tmp_path) / "shared").exists()


def test_a_kernel_that_does_not_answer_is_a_warning_on_the_way_in(host, tmp_path, said, monkeypatch):
    monkeypatch.setattr(ns, "SYNC_DEADLINE_SEC", 0.3)  # a refused connection takes about 2 s on Windows
    monkeypatch.setattr(ns, "SYNC_CYCLE_SEC", 0.3)
    _write(host / "sales.py")
    dead = MarimoSandbox(MarimoServer(url="http://127.0.0.1:9", token=TOKEN, runtime="docker"), notebooks_path="/work/notebooks", notice=said.append)
    assert ns.NotebookSync(dead, host, host.parent).sync_in() == []
    assert len(said) == 1 and said[0].startswith(f"Warning: could not copy the notebooks in {host} into the kernel")
    assert said[0].endswith("0 copied. The files there are unchanged.")


# --------------------------------------------------------------------------- #
# Out: after turns, periodically, at stop
# --------------------------------------------------------------------------- #


def test_sync_out_copies_notebook_edits_back_and_nothing_else(sync, host, tmp_path, said):
    """What notebook code can plant in its notebooks folder never reaches this machine: git and
    editor folders, files test runners and Python run, modules, outputs."""
    _write(host / "sales.py", notebook("v1"))
    sync.sync_in()
    box = container(tmp_path)
    _write(box / "sales.py", notebook("v2"))  # the agent or the browser changed it
    _write(box / "new" / "fresh.py", notebook("fresh"))
    _write(box / ".git" / "hooks" / "post-checkout", "#!/bin/sh\necho owned\n")
    _write(box / ".git" / "config", "[core]\n\tfsmonitor = echo owned\n")
    _write(box / ".vscode" / "settings.json", '{"python.defaultInterpreterPath": "/tmp/x"}\n')
    _write(box / ".devcontainer" / "devcontainer.json", "{}\n")
    _write(box / "conftest.py", NB)
    _write(box / "q3" / "test_review.py", NB)
    _write(box / "sitecustomize.py", NB)
    _write(box / "helper.py", "import os\nos.system('echo owned')\n")
    _write(box / "hailer.toml", '[kernel]\nruntime = "local"\n')
    _write(box / "out.csv", "a,b\n")
    _write(box / "__marimo__" / "session" / "sales.py.json", "{}\n")
    assert sync.sync_out() == ["new/fresh.py", "sales.py"]
    assert (host / "sales.py").read_text(encoding="utf-8") == notebook("v2")
    assert (host / "new" / "fresh.py").read_text(encoding="utf-8") == notebook("fresh")
    assert sorted(p.relative_to(host).as_posix() for p in host.rglob("*")) == ["new", "new/fresh.py", "sales.py"]
    assert said == [
        "Not copied back from the kernel (only marimo notebooks with plain names are; everything else stays in the "
        "kernel and is gone when it stops): .devcontainer/, .git/, .vscode/, conftest.py, hailer.toml and 4 more."
    ]
    assert sync.sync_out() == [] and len(said) == 1, "nothing new: no copy, no repeated warning"


def test_an_unchanged_notebook_is_not_read_again(sync, host, fake, tmp_path):
    _write(host / "sales.py")
    sync.sync_in()
    sync.sync_out()
    before = len(fake.file_paths)
    sync.sync_out()
    sync.sync_out()
    assert not any(path.endswith("sales.py") for path in fake.file_paths[before:]), "listed, never read: time and size match"


def test_line_endings_alone_are_no_change(sync, host, tmp_path):
    (host / "sales.py").write_bytes(NB.replace("\n", "\r\n").encode())
    sync.sync_in()
    _write(container(tmp_path) / "sales.py", NB)  # marimo rewrote it with the kernel's line endings
    assert sync.sync_out() == []
    assert (host / "sales.py").read_bytes() == NB.replace("\n", "\r\n").encode()


def test_a_notebook_deleted_in_the_kernel_stays_here(sync, host, tmp_path):
    _write(host / "sales.py")
    sync.sync_in()
    (container(tmp_path) / "sales.py").unlink()
    assert sync.sync_out() == []
    assert (host / "sales.py").read_text(encoding="utf-8") == NB


def test_a_file_changed_here_and_in_the_kernel_is_backed_up_first(sync, host, tmp_path, said):
    _write(host / "q3" / "review.py", notebook("v1"))
    sync.sync_in()
    _write(host / "q3" / "review.py", notebook("edited here"))
    assert sync.sync_out() == [], "only changed here: left alone"
    assert (host / "q3" / "review.py").read_text(encoding="utf-8") == notebook("edited here")
    _write(container(tmp_path) / "q3" / "review.py", notebook("edited in the kernel"))
    assert sync.sync_out() == ["q3/review.py"]
    assert (host / "q3" / "review.py").read_text(encoding="utf-8") == notebook("edited in the kernel")
    backups = list((host.parent / ".hailer" / ns.BACKUP_FOLDER / "q3").glob("review.*.py"))
    assert len(backups) == 1 and backups[0].read_text(encoding="utf-8") == notebook("edited here")
    assert said == [
        f"Warning: q3/review.py changed both here and in the kernel. The kernel's version replaced it; the copy that "
        f"was here is saved as {backups[0]}."
    ]
    assert (host.parent / ".hailer" / ".gitignore").is_file(), "backups live in Hailer's ignored state folder"


def test_a_file_that_appeared_here_meanwhile_is_backed_up_too(sync, host, tmp_path):
    sync.sync_in()
    _write(host / "fresh.py", notebook("mine"))
    _write(container(tmp_path) / "fresh.py", notebook("the kernel's"))
    assert sync.sync_out() == ["fresh.py"]
    [backup] = (host.parent / ".hailer" / ns.BACKUP_FOLDER).glob("fresh.*.py")
    assert backup.read_text(encoding="utf-8") == notebook("mine")


def test_a_failed_backup_leaves_the_file_here_as_it_was(sync, host, tmp_path, said, monkeypatch):
    _write(host / "sales.py", notebook("v1"))
    sync.sync_in()
    _write(host / "sales.py", notebook("mine"))
    _write(container(tmp_path) / "sales.py", notebook("theirs"))

    def full(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(ns.NotebookSync, "_backup", full)
    assert sync.sync_out() == []
    assert (host / "sales.py").read_text(encoding="utf-8") == notebook("mine")
    assert said == [f"Warning: could not copy back sales.py (disk full); the files in {host} are unchanged."]


def test_writes_here_are_atomic_and_leave_nothing_behind(sync, host, tmp_path, said, monkeypatch):
    _write(host / "sales.py", notebook("v1"))
    sync.sync_in()
    _write(container(tmp_path) / "sales.py", notebook("v2"))

    def interrupted(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(ns.os, "replace", interrupted)
    assert sync.sync_out() == []
    assert (host / "sales.py").read_text(encoding="utf-8") == notebook("v1"), "never half a file"
    assert sorted(p.name for p in host.iterdir()) == ["sales.py"], "the temporary file is gone"
    monkeypatch.undo()
    assert sync.sync_out() == ["sales.py"], "the next copy tries again"
    assert sorted(p.name for p in host.iterdir()) == ["sales.py"]


def test_a_write_never_goes_through_a_link_or_junction(sync, host, tmp_path, said):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _link(host / "q3", elsewhere)
    sync.sync_in()
    _write(container(tmp_path) / "q3" / "review.py", notebook("through the link"))
    assert sync.sync_out() == []
    assert list(elsewhere.iterdir()) == [], "nothing written outside the notebooks folder"
    assert len(said) == 1 and said[0].startswith("Warning: could not copy back q3/review.py (") and "link or junction" in said[0]


def test_a_write_never_replaces_a_file_link(sync, host, tmp_path, said):
    target = tmp_path / "elsewhere.py"
    _write(target, notebook("outside"))
    _link(host / "sales.py", target, folder=False)
    sync.sync_in()
    _write(container(tmp_path) / "sales.py", notebook("kernel"))
    assert sync.sync_out() == []
    assert target.read_text(encoding="utf-8") == notebook("outside")


def test_names_that_differ_only_in_case_are_copied_once(sync, host, tmp_path, said, monkeypatch):
    entries = [SandboxEntry("Sales.py", False, 1.0, 10), SandboxEntry("sales.py", False, 1.0, 10)]
    monkeypatch.setattr(sync.sandbox, "list_tree", lambda: entries)
    monkeypatch.setattr(sync.sandbox, "read_notebook", lambda name: notebook(name))
    assert sync.sync_out() == ["Sales.py"]
    assert "sales.py" in said[-1], "the second spelling would overwrite the first on Windows and macOS"


def test_at_most_500_notebooks_are_copied(sync, monkeypatch, said):
    monkeypatch.setattr(ns, "SESSION_NEW_FILES", 10_000)
    entries = [SandboxEntry(f"n{i:03}.py", False, 1.0, 10) for i in range(ns.MAX_NOTEBOOKS + 2)]
    monkeypatch.setattr(sync.sandbox, "list_tree", lambda: entries)
    monkeypatch.setattr(sync.sandbox, "read_notebook", lambda name: NB)
    assert len(sync.sync_out()) == ns.MAX_NOTEBOOKS
    assert said[-1].endswith("n500.py, n501.py.")


def test_a_kernel_that_stopped_answering_is_one_warning_and_changes_nothing(host, tmp_path, said, monkeypatch):
    monkeypatch.setattr(ns, "SYNC_DEADLINE_SEC", 0.3)  # a refused connection takes about 2 s on Windows
    monkeypatch.setattr(ns, "SYNC_CYCLE_SEC", 0.3)
    _write(host / "sales.py")
    dead = MarimoSandbox(MarimoServer(url="http://127.0.0.1:9", token=TOKEN, runtime="docker"), notebooks_path="/work/notebooks", notice=said.append)
    sync = ns.NotebookSync(dead, host, host.parent)
    assert sync.sync_out() == [] and sync.sync_out() == []
    assert len(said) == 1 and said[0].startswith("Warning: could not copy the notebooks back from the kernel (")
    assert said[0].endswith(f"The files in {host} are unchanged.")
    assert (host / "sales.py").read_text(encoding="utf-8") == NB


def test_the_periodic_copy_runs_on_a_daemon_thread_until_closed(sync, host, tmp_path):
    _write(host / "sales.py", notebook("v1"))
    sync.sync_in()
    sync.start(interval=3600)
    thread = sync._thread
    assert thread is not None and thread.daemon and thread.name == "hailer-notebook-sync"
    _write(container(tmp_path) / "sales.py", notebook("after a turn"))
    sync.soon()  # a turn ended: no waiting for the interval
    deadline = time.monotonic() + 10
    while (host / "sales.py").read_text(encoding="utf-8") != notebook("after a turn"):
        assert time.monotonic() < deadline, "soon() did not wake the copy"
        time.sleep(0.02)
    _write(container(tmp_path) / "sales.py", notebook("last edit"))
    sync.close()
    assert not thread.is_alive()
    assert (host / "sales.py").read_text(encoding="utf-8") == notebook("last edit"), "the final copy"


def test_copies_never_overlap(sync, monkeypatch):
    inside = threading.Event()
    release = threading.Event()
    active: list[int] = []
    overlapped: list[bool] = []

    def slow_tree():
        active.append(1)
        overlapped.append(len(active) > 1)
        inside.set()
        release.wait(5)
        active.pop()
        return []

    monkeypatch.setattr(sync.sandbox, "list_tree", slow_tree)
    workers = [threading.Thread(target=sync.sync_out) for _ in range(2)]
    workers[0].start()
    assert inside.wait(5)
    workers[1].start()
    time.sleep(0.1)
    release.set()
    for worker in workers:
        worker.join(5)
    assert overlapped == [False, False], "the second copy waited for the first"


def test_a_pass_stops_at_its_deadline_or_byte_budget_and_leaves_the_rest(sync, host, monkeypatch, said):
    """A forged server cannot make one pass run long or rewrite gigabytes: the rest waits."""
    entries = [SandboxEntry(f"n{i}.py", False, 1.0, 10) for i in range(5)]
    monkeypatch.setattr(sync.sandbox, "list_tree", lambda: entries)

    def slow(name):
        time.sleep(0.2)
        return notebook(name)

    monkeypatch.setattr(sync.sandbox, "read_notebook", slow)
    copied = sync.sync_out(seconds=0.3)
    assert 1 <= len(copied) < 5
    assert said[-1] == f"Warning: copying the notebooks back from the kernel stopped after 0.3 s or 50 MiB; the files in {host} not copied yet are unchanged."
    monkeypatch.setattr(sync.sandbox, "read_notebook", lambda name: notebook(name) + "#" * 1000)
    assert len(sync.sync_out(budget=1500)) == 2, "two reads, then over budget"


def test_the_last_copy_has_one_deadline_whatever_the_server_does(sync, host, tmp_path, monkeypatch, said):
    """A server that trickles its reply cannot hold the exit: the whole last copy ends at its deadline."""
    _write(host / "sales.py", notebook("v1"))
    sync.sync_in()
    monkeypatch.setattr(ns, "SYNC_DEADLINE_SEC", 0.5)
    real = sync.sandbox.list_tree

    def trickle():
        time.sleep(0.6)  # past the deadline: the next request must not be sent
        return real()

    monkeypatch.setattr(sync.sandbox, "list_tree", trickle)
    _write(container(tmp_path) / "sales.py", notebook("v2"))
    started = time.monotonic()
    assert sync.close() == []
    assert time.monotonic() - started < 5
    assert (host / "sales.py").read_text(encoding="utf-8") == notebook("v1"), "the copy here is kept"
    assert "out of time" in said[-1] or "stopped after 0.5 s" in said[-1], said


# --------------------------------------------------------------------------- #
# The docker kernel: copies in at start, the last copy before its containers go
# --------------------------------------------------------------------------- #


def _docker_start(tmp_path, fake, docker):
    workspace = tmp_path / "workspace"
    config = make_config(workspace, kernel=KernelConfig(runtime="docker"))
    runtime = kd.DockerRuntime(
        config, runner=docker, token_factory=lambda: TOKEN, health=lambda *a, **k: True, user=lambda: None, suffix="a1b2c3"
    )  # fmt: skip
    return config, runtime.start(fake.server_address[1])


class RecordingDocker(FakeDocker):
    """Records what the host notebook held when the first container removal ran."""

    def __init__(self, host: Path) -> None:
        super().__init__()
        self.host = host
        self.at_removal: str | None = None

    def run(self, args, *, timeout=None, check=False):
        if list(args[:2]) == ["rm", "-f"] and self.at_removal is None:
            self.at_removal = (self.host / "sales.py").read_text(encoding="utf-8")
        return super().run(args, timeout=timeout, check=check)


def test_a_docker_start_copies_in_before_it_returns_and_its_stop_copies_back_before_removal(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(kd, "_sleep", lambda seconds: None)
    host = tmp_path / "workspace" / "notebooks"
    _write(host / "sales.py", notebook("v1"))
    _write(host / "conftest.py", NB)
    docker = RecordingDocker(host)
    monkeypatch.setattr(ns, "SYNC_INTERVAL_SEC", 3600.0)
    config, kernel = _docker_start(tmp_path, fake, docker)
    said: list[str] = []
    kernel.notice = said.append
    assert (container(tmp_path) / "sales.py").read_text(encoding="utf-8") == notebook("v1"), "in before start returned"
    assert not (container(tmp_path) / "conftest.py").exists()
    assert kernel.sync is not None and kernel.sync._thread is not None
    _write(container(tmp_path) / "sales.py", notebook("v2"))
    _write(container(tmp_path) / ".git" / "config", "[core]\n\tfsmonitor = echo owned\n")
    thread = kernel.sync._thread
    kernel.stop()
    assert docker.at_removal == notebook("v2"), "the last copy ran before the containers were removed"
    assert not thread.is_alive() and kernel.sync is None
    assert not (host / ".git").exists()
    assert said == [
        "Not copied back from the kernel (only marimo notebooks with plain names are; everything else stays in the "
        "kernel and is gone when it stops): .git/."
    ]
    kernel.stop()  # idempotent: no second copy
    assert docker.containers == {}


def test_ctrl_c_during_the_last_copy_still_removes_the_containers(fake, tmp_path, monkeypatch):
    import threading as real_threading

    monkeypatch.setattr(kd, "_sleep", lambda seconds: None)
    docker = FakeDocker()
    monkeypatch.setattr(ns, "SYNC_INTERVAL_SEC", 3600.0)
    _config, kernel = _docker_start(tmp_path, fake, docker)
    said: list[str] = []
    kernel.notice = said.append

    class Interrupted(real_threading.Thread):
        def join(self, timeout=None):
            raise KeyboardInterrupt

    monkeypatch.setattr(kd.threading, "Thread", Interrupted)
    kernel.stop()
    assert docker.containers == {} and docker.networks == {}
    assert said == ["Warning: stopped before the notebooks were copied back; the kernel's changes since the last copy are lost."]


def test_a_last_copy_that_never_ends_cannot_keep_the_containers(fake, tmp_path, monkeypatch):
    """Whatever the kernel's server does (a forged one can stall a request the deadline does not
    cover, or hold the lock), the stop removes the containers after a bounded wait."""
    monkeypatch.setattr(kd, "_sleep", lambda seconds: None)
    monkeypatch.setattr(ns, "SYNC_INTERVAL_SEC", 3600.0)
    monkeypatch.setattr(ns, "SYNC_STOP_SEC", 0.3)
    docker = FakeDocker()
    _config, kernel = _docker_start(tmp_path, fake, docker)
    said: list[str] = []
    kernel.notice = said.append
    stuck = threading.Event()
    monkeypatch.setattr(kernel.sync, "close", lambda final=True: stuck.wait(30))
    started = time.monotonic()
    kernel.stop()
    stuck.set()
    assert time.monotonic() - started < 5
    assert docker.containers == {} and docker.networks == {}
    assert said == ["Warning: the last copy of the notebooks back from the kernel did not finish within 0.3 s; the kernel's latest changes may be missing here."]


def test_a_server_that_trickles_its_page_cannot_hold_the_last_copy(host, tmp_path, monkeypatch, said):
    """The server token's page used to be read without the deadline: a trickling page held the
    background pass, its lock, and so the stop, for minutes."""
    import socket

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    done = threading.Event()

    def serve():
        listener.settimeout(0.2)
        while not done.is_set():
            try:
                conn, _ = listener.accept()
            except OSError:
                continue
            with conn:
                try:
                    conn.recv(65536)
                    conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: 100000\r\n\r\n")
                    while not done.is_set():
                        conn.sendall(b" ")
                        time.sleep(0.05)
                except OSError:
                    pass
        listener.close()

    threading.Thread(target=serve, daemon=True).start()
    try:
        forged = MarimoSandbox(
            MarimoServer(url=f"http://127.0.0.1:{listener.getsockname()[1]}", token=TOKEN, runtime="docker"),
            notebooks_path="/work/notebooks", notice=said.append,
        )  # fmt: skip
        sync = ns.NotebookSync(forged, host, host.parent)
        monkeypatch.setattr(ns, "SYNC_DEADLINE_SEC", 0.5)
        monkeypatch.setattr(ns, "SYNC_CYCLE_SEC", 0.5)
        sync.start(interval=0.01)
        time.sleep(0.2)  # a background pass is now reading the page
        started = time.monotonic()
        sync.close()
        assert time.monotonic() - started < 5, "the page read ends at the deadline"
        assert any("out of time" in line or "stopped after" in line for line in said), said
    finally:
        done.set()


def test_a_session_stops_copying_new_notebooks_back_after_its_cap(sync, host, monkeypatch, said):
    monkeypatch.setattr(ns, "SESSION_NEW_FILES", 2)
    entries = [SandboxEntry(f"n{i}.py", False, 1.0, 10) for i in range(4)]
    monkeypatch.setattr(sync.sandbox, "list_tree", lambda: entries)
    monkeypatch.setattr(sync.sandbox, "read_notebook", lambda name: notebook(name))
    assert sync.sync_out() == ["n0.py", "n1.py"]
    assert sorted(p.name for p in host.iterdir()) == ["n0.py", "n1.py"]
    assert said == ["Warning: this session already copied back 2 new notebooks (0 MiB); new notebooks from the kernel are no longer copied back."]
    entries[0] = SandboxEntry("n0.py", False, 2.0, 11)
    monkeypatch.setattr(sync.sandbox, "read_notebook", lambda name: notebook(name + " edited"))
    assert sync.sync_out() == ["n0.py"], "a notebook it already has still comes back"


def test_names_in_warnings_are_printable(sync, monkeypatch, said):
    entries = [SandboxEntry("evil\x1b[2Jname.txt", False, 1.0, 10)]
    monkeypatch.setattr(sync.sandbox, "list_tree", lambda: entries)
    sync.sync_out()
    assert "\x1b" not in said[-1] and "evil?[2Jname.txt" in said[-1]


def test_ctrl_c_while_copying_in_stops_the_kernel(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(kd, "_sleep", lambda seconds: None)
    docker = FakeDocker()

    def interrupted(self):
        raise KeyboardInterrupt

    monkeypatch.setattr(ns.NotebookSync, "sync_in", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _docker_start(tmp_path, fake, docker)
    assert docker.containers == {} and docker.networks == {}, "the caller never got the kernel: nothing left running"
    assert not list((tmp_path / "workspace" / ".hailer").glob("owner-*.lock"))


def test_a_docker_kernel_refuses_to_create_a_notebook_that_would_never_come_back(fake, tmp_path):
    """The agent or /notebook new naming one test_sales.py would work until the kernel stops, then lose it."""
    from hailer.errors import NotebookPathError

    kernel = kd.DockerKernel(server=MarimoServer(url=fake.url, token=TOKEN, runtime="docker"), notebooks_path="/work/notebooks")
    for name in ("test_sales.py", "q3/conftest.py", "sales_test.py", "json.py"):
        with pytest.raises(NotebookPathError) as info:
            kernel.write_notebook(name, NB, replace=False)
        assert "would stay in the docker kernel and be lost when it stops" in str(info.value)
    assert list(container(tmp_path).iterdir()) == []
    assert kernel.write_notebook("q3\\sales.py", NB, replace=False).name == "q3/sales.py"


def test_a_notebook_marimo_refuses_does_not_stop_the_others_going_in(sync, host, monkeypatch, said):
    from hailer.errors import NotebookPathError

    _write(host / "a.py")
    _write(host / "b.py")
    real = sync.sandbox.write_notebook

    def picky(name, source, *, replace=True):
        if name == "a.py":
            raise NotebookPathError("Could not create a.py: marimo refused.")
        return real(name, source, replace=replace)

    monkeypatch.setattr(sync.sandbox, "write_notebook", picky)
    assert sync.sync_in() == ["b.py"]
    assert said == ["Not copied into the kernel (only marimo notebooks with plain names are): a.py."]


def test_the_local_kernel_copies_nothing(tmp_path):
    from hailer.kernel import LocalKernel

    local = LocalKernel(server=MarimoServer(url="http://127.0.0.1:9"), notebooks_path=str(tmp_path), native_paths=True)
    assert local.sync_soon() is None and not hasattr(local, "sync")
