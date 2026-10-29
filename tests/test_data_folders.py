"""More data folders (``[data_folders]``): the config, the mounts, what notebooks and the agent see,
and ``uvx hailer data`` / ``/data`` editing this machine's ``.hailer/local.toml``.

No Docker needed: the docker runtime runs against tests/fake_docker.py.
"""

from __future__ import annotations

import importlib.util
import json
import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fake_docker import FakeDocker
from fake_kernel import TOKEN, Procs, make_config
from fake_sandbox import folder_sandbox
from hailer import cli
from hailer import kernel as k
from hailer import kernel_docker as kd
from hailer.agent import system_prompt
from hailer.config import available_data_folders, docker_mount_problems, load_config, validate
from hailer.datafolders import add_data_folder, remove_data_folder
from hailer.errors import ConfigError, HailerError
from hailer.models import ContextBundle, DataFolder, KernelConfig
from hailer.notebooks import render_template
from hailer.sandbox import SandboxFolder
from hailer.tools import HailerTools

runner = CliRunner()


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _workspace(tmp_path: Path, config: str = "") -> Path:
    ws = tmp_path / "ws"
    _write(ws / "notebooks" / "analysis.py", "import marimo\n")
    (ws / "data").mkdir()
    _write(ws / "hailer.toml", config)
    return ws


def _errors(problems: list[str]) -> list[str]:
    return [p for p in problems if not p.startswith("Warning:")]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_a_folder_is_a_path_or_a_table_with_a_description(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    ws = _workspace(
        tmp_path,
        f'[data_folders]\nfinance = "{shared.as_posix()}"\nwarehouse = {{ path = "exports", description = "Nightly Parquet" }}\n'
        '[data_folders.crm]\ndescription = "CRM extracts"\n',
    )
    folders = load_config(workspace=ws, env={}).data_folders
    assert folders == (
        DataFolder("finance", shared.resolve()),
        DataFolder("warehouse", (ws / "exports").resolve(), "Nightly Parquet"),
        DataFolder("crm", None, "CRM extracts"),
    ), "in the order written; relative to the workspace; crm has no path on this machine"


def test_this_machines_path_merges_into_the_folder_the_project_declares(tmp_path: Path) -> None:
    ws = _workspace(tmp_path, '[data_folders]\ncrm = { description = "CRM extracts" }\n')
    mine = tmp_path / "mine" / "crm"
    mine.mkdir(parents=True)
    _write(ws / ".hailer" / "local.toml", f'[data_folders]\ncrm = "{mine.as_posix()}"\n')
    config = load_config(workspace=ws, env={})
    assert config.data_folders == (DataFolder("crm", mine.resolve(), "CRM extracts"),)
    assert available_data_folders(config) == list(config.data_folders)


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ('[data_folders]\nFinance = "x"\n', "is not a data folder name"),
        ('[data_folders]\n"a/b" = "x"\n', "is not a data folder name"),
        ("[data_folders]\nfinance = 3\n", "must be a folder path or a table"),
        ("[data_folders]\nfinance = { path = 3 }\n", "must be a string"),
        ('data_folders = "x"\n', "must be a table"),
    ],
)
def test_bad_data_folders_are_config_errors(tmp_path: Path, text: str, fragment: str) -> None:
    ws = _workspace(tmp_path, text)
    with pytest.raises(ConfigError) as excinfo:
        load_config(workspace=ws, env={})
    assert fragment in str(excinfo.value)


def test_a_folder_without_a_path_or_folder_is_a_warning_and_left_out(tmp_path: Path) -> None:
    ws = _workspace(tmp_path, '[data_folders]\ncrm = { description = "x" }\ngone = "missing"\n[data_folders.finance]\npath = "data"\nnotes = 1\n')
    config = load_config(workspace=ws, env={})
    problems = validate(config)
    assert _errors(problems) == [], problems
    assert any("'crm' has no path on this machine" in p and "hailer data add crm" in p for p in problems)
    assert any("'gone'" in p and "does not exist" in p for p in problems)
    assert any("unknown key [data_folders.finance].notes" in p for p in problems)
    assert [f.name for f in available_data_folders(config)] == ["finance"]


def test_docker_refuses_a_data_folder_that_would_expose_more_than_data(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    ws = _workspace(tmp_path, f'[data_folders]\nall = "{home.as_posix()}"\nkeys = "{(home / ".ssh").as_posix()}"\nme = "."\n[kernel]\nruntime = "docker"\n')
    errors = _errors(validate(load_config(workspace=ws, env={})))
    assert any("The data folder 'all'" in e and "is your home folder" in e and "hailer data add all" in e for e in errors), errors
    assert any("The data folder 'keys'" in e and "holds credentials" in e for e in errors), errors
    assert any("The data folder 'me'" in e and "is the workspace folder" in e for e in errors), errors


def test_a_network_share_is_refused_without_looking_at_it(tmp_path: Path) -> None:
    config = make_config(tmp_path, data_folders=(DataFolder("share", Path(r"\\server\exports")),))
    (tmp_path / "data").mkdir()
    problems = docker_mount_problems(config, windows=True)
    assert any("The data folder 'share'" in p and "network share" in p for p in problems), problems


# --------------------------------------------------------------------------- #
# The kernels
# --------------------------------------------------------------------------- #


def _healthy(url, timeout, token=None, should_stop=None):
    return True


class _NoSync:
    def __init__(self, *args, **kwargs):
        self.complete = None  # NotebookSync's record of its latest copy back

    def sync_in(self):
        return []

    def start(self, interval=None):
        pass

    def soon(self):
        pass

    def close(self, *, final=True):
        return []


def test_the_docker_kernel_mounts_each_folder_read_only_and_says_nothing_in_its_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(kd, "NotebookSync", _NoSync)
    finance = tmp_path / "elsewhere" / "finance"
    finance.mkdir(parents=True)
    config = make_config(
        tmp_path,
        kernel=KernelConfig(runtime="docker"),
        data_folders=(DataFolder("finance", finance, "Monthly extracts"), DataFolder("crm", None)),
    )
    fake = FakeDocker()
    runtime = kd.DockerRuntime(config, runner=fake, token_factory=lambda: TOKEN, health=_healthy, user=lambda: None, suffix="a1b2c3")
    box = runtime.start(2731)
    [run] = fake.commands("run")
    assert f"type=bind,source={finance},target=/work/folders/finance,readonly" in run
    assert not {"-e", "--env", "--env-file"} & set(run), "the starter notebook lists /work/folders instead"
    assert box.data_folders == {"finance": SandboxFolder("/work/folders/finance", finance)}
    rows = {row.name: row for row in runtime.check()}
    assert rows["data:finance"].ok and "/work/folders/finance" in rows["data:finance"].summary
    box.stop()


def test_the_unsafe_local_kernel_is_told_where_its_folders_are(tmp_path: Path) -> None:
    finance = tmp_path / "finance"
    finance.mkdir()
    config = make_config(tmp_path, kernel=KernelConfig(runtime="unsafe-local"), data_folders=(DataFolder("finance", finance),))
    procs = Procs()
    runtime = k.LocalRuntime(config, procs=procs.local_processes(), environ={"PATH": "p"}, token_factory=lambda: TOKEN)
    box = runtime.start(2718)
    env = procs.calls[0][4]
    assert env["HAILER_DATA_DIR"] == str(tmp_path / "data")
    assert json.loads(env["HAILER_DATA_FOLDERS"]) == {"finance": str(finance)}
    assert box.data_folders == {"finance": SandboxFolder(str(finance), finance)}
    box.stop()


def _run_starter(path: Path) -> dict:
    spec = importlib.util.spec_from_file_location("starter_notebook", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.app.run()[1]


def test_the_starter_notebook_finds_the_folders_either_way(tmp_path: Path, monkeypatch) -> None:
    """Hailer's variables (unsafe-local), else <workspace>/folders/<name> (a docker kernel's /work)."""
    ws = tmp_path / "work"
    _write(ws / "hailer.toml", "")
    (ws / "data").mkdir()
    (ws / "folders" / "finance").mkdir(parents=True)
    notebook = _write(ws / "notebooks" / "analysis.py", render_template("starter", title="T"))
    monkeypatch.delenv("HAILER_DATA_DIR", raising=False)
    monkeypatch.delenv("HAILER_DATA_FOLDERS", raising=False)
    defs = _run_starter(notebook)
    assert defs["DATA_DIR"] == ws.resolve() / "data"
    assert defs["DATA_FOLDERS"] == {"finance": ws.resolve() / "folders" / "finance"}
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setenv("HAILER_DATA_DIR", str(elsewhere))
    monkeypatch.setenv("HAILER_DATA_FOLDERS", json.dumps({"crm": str(elsewhere)}))
    defs = _run_starter(notebook)
    assert defs["DATA_DIR"] == elsewhere and defs["DATA_FOLDERS"] == {"crm": elsewhere}


# --------------------------------------------------------------------------- #
# What the agent sees
# --------------------------------------------------------------------------- #


def test_list_periods_names_the_folders_and_describes_one(tmp_path: Path) -> None:
    finance = tmp_path / "finance"
    finance.mkdir()
    (finance / "budget.csv").write_text("a\n1\n", encoding="utf-8")
    (finance / "notes.txt").write_text("x", encoding="utf-8")
    config = make_config(tmp_path)
    config.data_dir.mkdir()
    box = folder_sandbox(config, docker=True, data_folders={"finance": SandboxFolder("/work/folders/finance", finance)})
    tools = HailerTools(config, box)
    text = tools.list_periods()
    assert "More data folders (describe one with list_periods(folder='<name>')):" in text
    assert "- finance: /work/folders/finance (1 data file(s))" in text
    one = tools.list_periods(folder="finance")
    assert one.startswith("Data folder 'finance' (/work/folders/finance, DATA_FOLDERS['finance'] in notebooks):")
    assert "budget.csv" in one and "More data folders" not in one
    assert tools.list_periods(folder="nope").startswith("ERROR: There is no data folder named 'nope' (data folders: finance)")


def test_the_prompt_lists_the_folders_as_the_kernel_sees_them(tmp_path: Path) -> None:
    finance = tmp_path / "finance"
    finance.mkdir()
    folders = (DataFolder("finance", finance, "Monthly extracts"), DataFolder("crm", None))
    docker = system_prompt(make_config(tmp_path, data_folders=folders), ContextBundle())
    assert '- Data folder "finance": /work/folders/finance (read-only; DATA_FOLDERS["finance"] in notebooks): Monthly extracts' in docker
    assert "not on this machine (the user can add them with `hailer data add`): crm" in docker
    local = system_prompt(make_config(tmp_path, data_folders=folders, kernel=KernelConfig(runtime="unsafe-local")), ContextBundle())
    assert f'- Data folder "finance": {finance} (read-only;' in local


# --------------------------------------------------------------------------- #
# Editing this machine's .hailer/local.toml
# --------------------------------------------------------------------------- #


def test_add_writes_local_toml_and_keeps_what_else_is_there(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    folder = tmp_path / "Shared Drive" / "finance"
    folder.mkdir(parents=True)
    local = _write(ws / ".hailer" / "local.toml", '# my settings\n[model]\nname = "mine"  # personal\n\n[web]\nallowed_domains = []\n')
    config = load_config(workspace=ws, env={})
    assert add_data_folder(config, "finance", folder) == folder.resolve()
    text = local.read_text(encoding="utf-8")
    assert text.startswith('# my settings\n[model]\nname = "mine"  # personal\n'), "comments and other settings stay"
    assert tomllib.loads(text)["data_folders"] == {"finance": str(folder.resolve())}
    other = tmp_path / "other"
    other.mkdir()
    add_data_folder(load_config(workspace=ws, env={}), "finance", other, description='Q3 "final" numbers')
    add_data_folder(load_config(workspace=ws, env={}), "crm", other)
    data = tomllib.loads(local.read_text(encoding="utf-8"))
    assert data["data_folders"] == {"finance": {"path": str(other.resolve()), "description": 'Q3 "final" numbers'}, "crm": str(other.resolve())}
    assert data["model"] == {"name": "mine"}
    config = load_config(workspace=ws, env={})
    assert [f.name for f in config.data_folders] == ["finance", "crm"]
    assert remove_data_folder(config, "finance") and not remove_data_folder(config, "finance")
    assert [f.name for f in load_config(workspace=ws, env={}).data_folders] == ["crm"]


def test_local_toml_is_ignored_by_git_in_both_layouts(tmp_path: Path) -> None:
    folder = tmp_path / "f"
    folder.mkdir()
    ws = _workspace(tmp_path)
    add_data_folder(load_config(workspace=ws, env={}), "finance", folder)
    assert (ws / ".hailer" / ".gitignore").read_text(encoding="utf-8").endswith("*\n"), "standalone: the state folder ignores itself"
    project = tmp_path / "project"
    _write(project / ".hailer" / "hailer.toml", "")
    _write(project / ".hailer" / "notebooks" / "analysis.py", "import marimo\n")
    add_data_folder(load_config(workspace=project, env={}), "finance", folder)
    assert "local.toml" in (project / ".hailer" / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert (project / ".hailer" / "local.toml").is_file() and not (project / ".hailer" / "state" / "local.toml").exists()


@pytest.mark.parametrize(
    ("name", "folder", "fragment"),
    [
        ("Finance", "f", "is not a data folder name"),
        ("finance", "missing", "is not a folder"),
        ("finance", ".", "is the workspace folder"),
    ],
)
def test_add_refuses_bad_names_missing_folders_and_folders_a_kernel_may_not_mount(tmp_path, monkeypatch, name, folder, fragment) -> None:
    ws = _workspace(tmp_path)
    (ws / "f").mkdir()
    monkeypatch.chdir(ws)
    with pytest.raises(HailerError) as excinfo:
        add_data_folder(load_config(workspace=ws, env={}), name, folder)
    assert fragment in str(excinfo.value)
    assert not (ws / ".hailer" / "local.toml").exists()


def test_a_form_the_editor_does_not_handle_is_refused_not_mangled(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    folder = tmp_path / "f"
    folder.mkdir()
    before = '[data_folders.finance]\npath = "/old"\n'
    local = _write(ws / ".hailer" / "local.toml", before)
    with pytest.raises(ConfigError) as excinfo:
        add_data_folder(load_config(workspace=ws, env={}), "finance", folder)
    assert "by hand" in (excinfo.value.hint or "") or "by hand" in str(excinfo.value)
    assert local.read_text(encoding="utf-8") == before


def test_hailer_data_commands(tmp_path: Path) -> None:
    ws = _workspace(tmp_path, '[data_folders]\ncrm = { description = "CRM extracts" }\n[kernel]\nruntime = "docker"\n')
    folder = tmp_path / "crm"
    folder.mkdir()

    def run(*args: str):
        return runner.invoke(cli.app, ["--workspace", str(ws), "data", *args], catch_exceptions=False)

    listed = run()
    assert listed.exit_code == 0 and "crm" in listed.output and "no path on this machine" in listed.output
    added = run("add", "crm", str(folder))
    assert added.exit_code == 0, added.output
    assert "read-only in the kernel at /work/folders/crm" in added.output and "DATA_FOLDERS['crm']" in added.output
    listed = run("list")
    assert str(folder.resolve()) in listed.output and "CRM extracts" in listed.output
    removed = run("remove", "crm")
    assert "Removed data folder 'crm'" in removed.output and "still declared in hailer.toml" in removed.output
    bad = run("add", "crm", str(tmp_path / "nope"))
    assert bad.exit_code == 1 and "is not a folder" in bad.output


def test_slash_data_lists_and_adds(tmp_path: Path) -> None:
    from rich.console import Console

    from hailer.chat import ChatController

    ws = _workspace(tmp_path)
    folder = tmp_path / "With Space"
    folder.mkdir()
    config = load_config(workspace=ws, env={})
    console = Console(record=True, width=200)
    controller = ChatController.__new__(ChatController)  # only what /data needs
    controller.config, controller.console = config, console
    from hailer.cli import common

    controller._cli = common
    controller._data(f'add finance "{folder}"')
    controller._data("")
    controller._data("remove finance")
    controller._data("bogus")
    out = console.export_text()
    assert "Data folder 'finance' is" in out and "next start" in out
    assert "DATA_FOLDERS['finance']" in out
    assert "Removed data folder 'finance'" in out and "Usage: /data" in out
