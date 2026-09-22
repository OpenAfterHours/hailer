"""``.hailer/`` ignores itself: every writer creates it through statedir.ensure_state_dir.

The agent's conversation store (the fourth writer) is covered in test_agent.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hailer import kernel, notebooks, session, statedir
from hailer.models import HailerConfig, SessionState

NOTEBOOK_SOURCE = "import marimo\n\napp = marimo.App()\n"
CUSTOM = "# mine\n*.log\n"


def make_config(ws: Path) -> HailerConfig:
    nb = ws / "notebooks" / "analysis.py"
    nb.parent.mkdir(parents=True)
    nb.write_text(NOTEBOOK_SOURCE, encoding="utf-8")
    return HailerConfig(workspace=ws, notebook=nb, data_dir=ws / "data", context_dir=ws / "c", skills_dir=ws / "s", prompts_dir=ws / "p")


class FakeProc:
    pid = 1
    returncode = None

    def poll(self):
        return None


def save_session(config: HailerConfig, monkeypatch) -> Path:
    session.save_session(config.workspace, SessionState(thread_id="t"))
    return session.session_path(config.workspace)


def save_active_notebook(config: HailerConfig, monkeypatch) -> Path:
    notebooks.save_active_notebook(config, "analysis.py")
    return notebooks.state_path(config.workspace)


def start_marimo(config: HailerConfig, monkeypatch) -> Path:
    """The marimo log: `hailer` / `hailer notebook` create it when they start a server."""
    from fake_kernel import Procs

    def spawn(cmd, cwd, log_path, **kwargs):
        with kernel._fresh_log(log_path) as log:
            log.write(b"started\n")
        return FakeProc()

    procs = Procs().local_processes()
    procs.spawn = spawn
    running = kernel.LocalRuntime(config, procs=procs).start(2718)
    return running.log_path


WRITERS = [save_session, save_active_notebook, start_marimo]


@pytest.mark.parametrize("write", WRITERS, ids=lambda f: f.__name__)
def test_every_writer_leaves_a_gitignore_that_ignores_everything(tmp_path, monkeypatch, write):
    config = make_config(tmp_path)
    written = write(config, monkeypatch)
    folder = tmp_path / ".hailer"
    assert written.parent == folder and written.is_file()
    ignore = (folder / ".gitignore").read_text(encoding="utf-8")
    assert [line for line in ignore.splitlines() if not line.startswith("#")] == ["*"]
    assert not (tmp_path / ".gitignore").exists(), "the user's own .gitignore is never created"


@pytest.mark.parametrize("write", WRITERS, ids=lambda f: f.__name__)
def test_an_existing_gitignore_in_hailer_is_preserved(tmp_path, monkeypatch, write):
    config = make_config(tmp_path)
    (tmp_path / ".hailer").mkdir()
    (tmp_path / ".hailer" / ".gitignore").write_text(CUSTOM, encoding="utf-8")
    write(config, monkeypatch)
    assert (tmp_path / ".hailer" / ".gitignore").read_text(encoding="utf-8") == CUSTOM


def test_the_users_gitignore_is_never_touched(tmp_path):
    (tmp_path / ".gitignore").write_text("node_modules/\n", encoding="utf-8")
    statedir.ensure_state_dir(tmp_path)
    statedir.ensure_state_dir(tmp_path)  # idempotent
    assert (tmp_path / ".gitignore").read_text(encoding="utf-8") == "node_modules/\n"
    assert sorted(p.name for p in (tmp_path / ".hailer").iterdir()) == [".gitignore"]


def test_a_gitignore_that_cannot_be_written_does_not_stop_hailer(tmp_path):
    (tmp_path / ".hailer" / ".gitignore").mkdir(parents=True)  # something odd in its place
    assert statedir.ensure_state_dir(tmp_path) == tmp_path / ".hailer"
    session.save_session(tmp_path, SessionState(thread_id="t"))
    assert session.load_session(tmp_path).thread_id == "t"
