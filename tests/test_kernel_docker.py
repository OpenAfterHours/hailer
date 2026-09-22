"""Tests for hailer.kernel_docker: the docker kernel runtime against a scripted docker CLI with
state (tests/fake_docker.py: nothing is run), ``uvx hailer kernel stop``, the folder checks and
the subprocess runner. No Docker needed; liveness probes are stubs."""

from __future__ import annotations

import hashlib
import io
import os
import re
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from fake_docker import CONTRACT, FakeDocker
from fake_kernel import LIVE, TOKEN, Procs, answers, make_config
from hailer import __version__
from hailer import kernel as k
from hailer import kernel_docker as kd
from hailer.errors import KernelRuntimeError
from hailer.models import KernelConfig, MarimoServer

IMAGE = f"ghcr.io/openafterhours/hailer-kernel:{CONTRACT}"


def healthy(url, timeout, should_stop=None):
    return True


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch):
    """Retries of `network rm` do not sleep in tests."""
    monkeypatch.setattr(kd, "_sleep", lambda seconds: None)


def docker_runtime(tmp_path: Path, fake: FakeDocker, *, kernel: dict | None = None, config=None, **kw) -> kd.DockerRuntime:
    config = config or make_config(tmp_path, kernel=KernelConfig(runtime="docker", **(kernel or {})))
    kw.setdefault("health", healthy)
    kw.setdefault("user", lambda: None)
    kw.setdefault("probe", answers())
    return kd.DockerRuntime(config, runner=fake, token_factory=lambda: TOKEN, **kw)


def labels(tmp_path: Path, role: str) -> list[str]:
    return [
        "--label", f"org.openafterhours.hailer.workspace={tmp_path}",
        "--label", f"org.openafterhours.hailer.version={__version__}",
        "--label", f"org.openafterhours.hailer.role={role}",
    ]  # fmt: skip


def follows(seq: list[str], part: list[str]) -> bool:
    return any(seq[i : i + len(part)] == part for i in range(len(seq) - len(part) + 1))


def started(tmp_path: Path, fake: FakeDocker, **kw) -> tuple[kd.DockerRuntime, kd.DockerKernel]:
    rt = docker_runtime(tmp_path, fake, **kw)
    return rt, rt.start(2731)


# --------------------------------------------------------------------------- #
# Names, mounts, users
# --------------------------------------------------------------------------- #


def test_workspace_id_and_names(tmp_path):
    ident = kd.workspace_id(tmp_path)
    assert re.fullmatch(r"[0-9a-f]{10}", ident)
    assert ident == hashlib.sha256(os.path.normcase(str(tmp_path.resolve())).encode("utf-8")).hexdigest()[:10]
    assert kd.docker_names(tmp_path) == kd.DockerNames(f"hailer-kernel-{ident}", f"hailer-fwd-{ident}", f"hailer-net-{ident}")
    assert kd.workspace_id(tmp_path / "other") != ident, "two workspaces run side by side"
    if os.name == "nt":
        assert kd.workspace_id(Path(str(tmp_path).upper())) == ident, "Windows paths compare case-insensitively"


def test_mount_arg_quotes_like_csv():
    assert kd.mount_arg("/srv/notebooks", k.KERNEL_NOTEBOOKS_DIR) == "type=bind,source=/srv/notebooks,target=/work/notebooks"
    assert kd.mount_arg("/srv/data", k.KERNEL_DATA_DIR, readonly=True) == "type=bind,source=/srv/data,target=/work/data,readonly"
    assert kd.mount_arg(r"C:\Users\Ann Lee\data", "/work/data") == r"type=bind,source=C:\Users\Ann Lee\data,target=/work/data"
    assert kd.mount_arg("/srv/q1,q2", "/work/data") == 'type=bind,"source=/srv/q1,q2",target=/work/data'
    assert kd.mount_arg('/srv/"best" data', "/work/data") == 'type=bind,"source=/srv/""best"" data",target=/work/data'


def test_linux_host_user(monkeypatch):
    monkeypatch.setattr(kd.os, "getuid", lambda: 1234, raising=False)
    monkeypatch.setattr(kd.os, "getgid", lambda: 5678, raising=False)
    monkeypatch.setattr(kd.sys, "platform", "linux")
    assert kd.linux_host_user() == (1234, 5678)
    monkeypatch.setattr(kd.os, "getuid", lambda: 0, raising=False)
    assert kd.linux_host_user() is None, "never run the kernel as root"
    for platform in ("win32", "darwin"):
        monkeypatch.setattr(kd.sys, "platform", platform)
        assert kd.linux_host_user() is None, "Docker Desktop maps file ownership itself"


def test_docker_runtime_describes_itself(tmp_path):
    rt = kd.DockerRuntime(make_config(tmp_path), runner=FakeDocker())  # built for `kernel stop` from a local config
    assert rt.name == "docker" and rt.describe() == f"docker (hailer-kernel {CONTRACT}; no network; data read-only)"
    assert rt.paths.to_kernel(tmp_path / "data" / "a.csv") == "/work/data/a.csv"
    assert not hasattr(rt, "prompt_notes"), "one prompt path: runtime_prompt_notes"


# --------------------------------------------------------------------------- #
# Checks: fail closed
# --------------------------------------------------------------------------- #


def test_check_rows_when_docker_is_ready(tmp_path):
    rows = docker_runtime(tmp_path, FakeDocker()).check()
    assert [(r.name, r.ok) for r in rows] == [("kernel", True), ("docker", True), ("image", True), ("data", True)]
    assert rows[1].summary == "Docker 29.4.3 (Linux engine)"
    assert rows[2].summary == f"{IMAGE} (kernel contract {CONTRACT})"


@pytest.mark.parametrize(
    ("fake", "message", "hint"),
    [
        (FakeDocker(installed=False), "Docker is not installed.", 'runtime = "local"'),
        (FakeDocker(engine=None), "Docker is not running.", "Start Docker Desktop"),
        (FakeDocker(engine="29.4.3 windows"), "Docker runs windows containers; Hailer's kernel image needs Linux containers.", "Switch to Linux containers"),
    ],
)
def test_docker_that_cannot_run_the_kernel_fails_closed(tmp_path, fake, message, hint):
    rt = docker_runtime(tmp_path, fake)
    rows = rt.check()
    docker = next(r for r in rows if r.name == "docker")
    assert not docker.ok and docker.fatal and docker.summary == message and hint in docker.hint
    assert "image" not in [r.name for r in rows]
    with pytest.raises(KernelRuntimeError) as exc:
        rt.start(2731)
    assert str(exc.value) == message
    assert not fake.commands("run") and not fake.commands("create") and k.read_kernel_state(tmp_path) is None


def test_engine_down_hint_quotes_docker_and_a_hung_engine_counts_as_down(tmp_path):
    rt = docker_runtime(tmp_path, FakeDocker(engine=None))
    with pytest.raises(KernelRuntimeError) as exc:
        rt.prepare()
    assert "docker said: error during connect" in exc.value.hint
    hung = FakeDocker()
    hung.timeout.add(("version",))
    with pytest.raises(KernelRuntimeError) as exc:
        docker_runtime(tmp_path, hung).prepare()
    assert str(exc.value) == "Docker is not running." and "did not answer within 20 s" in exc.value.hint


def test_missing_image_is_a_warning_and_is_pulled_with_progress_before_anything_starts(tmp_path):
    fake = FakeDocker(image_contract=None)
    rt = docker_runtime(tmp_path, fake)
    image_row = next(r for r in rt.check() if r.name == "image")
    assert not image_row.ok and not image_row.fatal and "uvx hailer kernel pull" in image_row.hint
    said: list[str] = []
    rt.prepare(say=said.append)
    assert fake.streams == [["pull", IMAGE]] and said == [f"Downloading the kernel image {IMAGE} (first use; this can take a few minutes) ..."]
    assert not fake.commands("run")
    inspected = len(fake.commands("image", "inspect"))
    rt.start(2731)
    assert fake.streams == [["pull", IMAGE]], "no second pull: the image is here now"
    assert len(fake.commands("image", "inspect")) == inspected, "prepare() runs once per start"


def test_a_failed_pull_names_the_build_command_and_the_image_setting(tmp_path):
    fake = FakeDocker(image_contract=None, pull_code=1, pull_error="Error response from daemon: Get https://ghcr.io/v2/: dial tcp: lookup ghcr.io: no such host")
    with pytest.raises(KernelRuntimeError) as exc:
        docker_runtime(tmp_path, fake).start(2731)
    assert str(exc.value) == f"Could not download the kernel image {IMAGE}."
    assert "uvx hailer kernel build" in exc.value.hint and "[kernel].image" in exc.value.hint
    assert not fake.commands("run")


@pytest.mark.parametrize("said", ["Error response from daemon: error from registry: denied\ndenied", "manifest unknown", "unauthorized: authentication required"])
def test_an_unpublished_image_says_so_and_how_to_build_it(tmp_path, said):
    """The first docker start before the release that publishes the image."""
    fake = FakeDocker(image_contract=None, pull_code=1, pull_error=said)
    with pytest.raises(KernelRuntimeError) as exc:
        docker_runtime(tmp_path, fake).start(2731)
    assert str(exc.value) == f"The kernel image {CONTRACT} is not published (or not visible to you): {IMAGE}."
    assert exc.value.hint.startswith("Build it on this machine with: uvx hailer kernel build") and "[kernel].image" in exc.value.hint


@pytest.mark.parametrize(("present", "pulled"), [("marimo0.24.2-000000000000", CONTRACT), (None, "marimo0.23.0-000000000000"), ("", CONTRACT)])
def test_an_image_of_another_kernel_contract_is_refused(tmp_path, present, pulled):
    """A different marimo would break code mode; "" is an image without the label (from before contracts)."""
    fake = FakeDocker(image_contract=present, pulled_contract=pulled)
    rt = docker_runtime(tmp_path, fake)
    with pytest.raises(KernelRuntimeError) as exc:
        rt.start(2731)
    shown = present if present is not None else pulled
    assert f"has kernel contract {shown or '(no contract label)'}; this Hailer needs {CONTRACT}." in str(exc.value)
    assert "uvx hailer kernel pull" in exc.value.hint and "uvx hailer kernel build" in exc.value.hint
    assert not fake.commands("run")
    if present is not None:
        image_row = next(r for r in rt.check() if r.name == "image")
        assert not image_row.ok and image_row.fatal


def test_cpus_above_what_docker_has_are_lowered_with_a_note(tmp_path):
    fake = FakeDocker(ncpu=4)
    rt = docker_runtime(tmp_path, fake, kernel={"cpus": 8})
    said: list[str] = []
    rt.prepare(say=said.append)
    assert said == ["Note: [kernel].cpus = 8 is more than the 4 CPUs Docker has; the kernel gets 4."]
    rt.start(2731)
    assert follows(fake.commands("run")[0], ["--cpus", "4"])
    assert k.read_kernel_state(tmp_path).cpus == 8.0, "the setting as written: a later run compares it with hailer.toml"


# --------------------------------------------------------------------------- #
# Folders that must never be mounted (S1, S2) and folders Docker cannot mount
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("notebooks", "data", "fragment"),
    [
        ("", "data", "The notebooks folder ({ws}) is the workspace folder"),  # notebook = "analysis.py"
        ("notebooks", "", "The data folder ({ws}) is the workspace folder"),
        (".hailer/nb", "data", "is inside Hailer's .hailer folder"),
        (".config/hailer", "data", "contains the context folder"),
        ("notebooks", ".hailer", "is Hailer's .hailer folder"),
        ("notebooks", "notebooks/data", "[hailer].data_dir ({ws}{sep}notebooks{sep}data) is inside the notebooks folder"),
        ("notebooks", "notebooks", "[hailer].data_dir ({ws}{sep}notebooks) is inside the notebooks folder"),
    ],
)
def test_start_refuses_mounts_that_expose_hailers_own_files(tmp_path, notebooks, data, fragment):
    """Checked on every start path, from the one rule set (config.docker_mount_problems): --foreground
    never validates hailer.toml, and HAILER_NOTEBOOKS_DIR / HAILER_DATA_DIR can move the folders."""
    config = make_config(
        tmp_path,
        notebooks_dir=tmp_path / notebooks if notebooks else tmp_path,
        notebook=(tmp_path / notebooks if notebooks else tmp_path) / "analysis.py",
        data_dir=tmp_path / data if data else tmp_path,
        kernel=KernelConfig(runtime="docker"),
        config_path=tmp_path / "hailer.toml",
    )
    fake = FakeDocker(image_contract=None)
    rt = docker_runtime(tmp_path, fake, config=config)
    for step in (rt.prepare, lambda: rt.start(2731)):
        with pytest.raises(KernelRuntimeError) as exc:
            step()
        assert fragment.format(ws=tmp_path, sep=os.sep) in f"{exc.value}\n{exc.value.hint}", str(exc.value)
    assert not fake.streams and not fake.commands("run") and not fake.commands("network", "create"), "refused before any pull or container"


def test_start_refuses_a_notebooks_folder_that_is_a_git_repository(tmp_path):
    """Notebook code could add git hooks or core.fsmonitor there, which run on this machine the next
    time git or an editor (VS Code scans nested repositories) touches it."""
    (tmp_path / "notebooks" / ".git").mkdir(parents=True)
    fake = FakeDocker()
    with pytest.raises(KernelRuntimeError) as exc:
        docker_runtime(tmp_path, fake).start(2731)
    assert str(exc.value).startswith(f"The notebooks folder ({tmp_path / 'notebooks'}) is a git repository (it has .git at its top level).")
    assert "core.fsmonitor" in str(exc.value) and "plain subfolder of your repository" in str(exc.value)
    (tmp_path / "notebooks" / ".git").rmdir()
    (tmp_path / "notebooks" / ".git").write_text("gitdir: ../elsewhere\n", encoding="utf-8")  # a worktree's .git file
    with pytest.raises(KernelRuntimeError):
        docker_runtime(tmp_path, fake).start(2731)
    assert not fake.commands("run")


def test_unc_folders_are_refused_before_docker_run(tmp_path, monkeypatch):
    """Docker Desktop answers a UNC mount with "%!(EXTRA string=is not a valid Windows path)"."""
    monkeypatch.setattr(kd, "_on_windows", lambda: True)
    share = Path(r"\\fileserver\team\notebooks")
    config = make_config(tmp_path, notebooks_dir=share, notebook=share / "analysis.py", kernel=KernelConfig(runtime="docker"))
    fake = FakeDocker()
    with pytest.raises(KernelRuntimeError) as exc:
        docker_runtime(tmp_path, fake, config=config).start(2731)
    assert str(exc.value).startswith(f"The notebooks folder {share} is on a network share (UNC path), which Docker cannot mount.")
    assert "[hailer].notebooks_dir" in str(exc.value) and not fake.commands("run")


@pytest.mark.parametrize("path", [r"\\fileserver\team\sales", r"\\?\UNC\fileserver\team\sales", "//fileserver/team/sales"])
def test_a_unc_data_folder_is_fatal_in_the_config_row_and_nothing_else_says_otherwise(tmp_path, path, monkeypatch):
    """One rule, one wording, FAIL: doctor used to show WARN while a start refused."""
    from hailer.config import docker_mount_problems, is_unc_path

    scanned: list[Path] = []
    monkeypatch.setattr(kd, "_links_outside", lambda folder, limit=0: (scanned.append(folder) or []))  # never touch a share
    monkeypatch.setattr(kd, "_on_windows", lambda: True)
    assert is_unc_path(path)
    config = make_config(tmp_path, data_dir=Path(path), kernel=KernelConfig(runtime="docker"))
    problems = docker_mount_problems(config, windows=True)
    assert problems == [
        f"The data folder {Path(path)} is on a network share (UNC path), which Docker cannot mount. Copy it to a "
        "folder on a local disk and point [hailer].data_dir at it."
    ]
    assert kd.data_path_problems(Path(path), windows=True, drive_type=lambda root: 3) == [], "no second, softer wording"
    rows = docker_runtime(tmp_path, FakeDocker(), config=config).check()
    assert "data" not in [r.name for r in rows], "no OK row for a folder that cannot be mounted"
    assert scanned == [], "a share is not walked for links"
    assert docker_mount_problems(config, windows=False) == [] or "UNC" not in " ".join(docker_mount_problems(config, windows=False))


def test_mapped_network_drives_are_flagged_and_local_drives_are_not(monkeypatch):
    monkeypatch.setattr(kd, "_links_outside", lambda folder, limit=0: [])
    drives = {"Z:\\": kd.DRIVE_REMOTE, "C:\\": 3}
    problems = kd.data_path_problems(Path("Z:/sales"), windows=True, drive_type=lambda root: drives.get(root, 0))
    assert len(problems) == 1 and "a mapped network drive (Z:)" in problems[0]
    assert kd.data_path_problems(Path("C:/sales"), windows=True, drive_type=lambda root: drives.get(root, 0)) == []
    assert kd.data_path_problems(Path(r"\\?\C:\sales"), windows=True, drive_type=lambda root: drives.get(root, 0)) == []


def test_a_notebooks_folder_on_a_mapped_drive_gets_the_same_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(kd, "_on_windows", lambda: True)
    monkeypatch.setattr(kd, "_drive_type", lambda root: kd.DRIVE_REMOTE if root == "Z:\\" else 3)
    rt = docker_runtime(tmp_path, FakeDocker(), config=make_config(tmp_path, notebooks_dir=Path("Z:/team/notebooks"), kernel=KernelConfig(runtime="docker")))
    row = next(r for r in rt.check() if r.name == "notebooks")
    assert not row.ok and not row.fatal and "a mapped network drive (Z:)" in row.hint and "[hailer].notebooks_dir" in row.hint


def test_doctor_warns_about_files_other_programs_run_code_from(tmp_path):
    notebooks = tmp_path / "notebooks"
    (notebooks / ".vscode").mkdir(parents=True)
    (notebooks / "hailer.toml").write_text("", encoding="utf-8")
    row = next(r for r in docker_runtime(tmp_path, FakeDocker()).check() if r.name == "notebooks")
    assert not row.ok and not row.fatal and row.summary == ".vscode, hailer.toml in the notebooks folder"
    assert row.hint.startswith(f"WARNING: the notebooks folder {notebooks} contains .vscode, hailer.toml.")
    assert "delete them before you run git in that folder, open it in an editor" in row.hint
    assert kd.planted_files(notebooks) == [".vscode", "hailer.toml"] and kd.planted_files(None) == []


def _link(link: Path, target: Path) -> None:
    """A symlink, or on Windows without the privilege for one, a junction."""
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        import _winapi

        try:
            _winapi.CreateJunction(str(target), str(link))
        except OSError:
            pytest.skip("cannot create links here")


def test_links_that_leave_the_data_folder_are_flagged(tmp_path):
    data = tmp_path / "data"
    (data / "archive").mkdir(parents=True)
    (data / "sales.csv").write_text("region,revenue\nNorth,1\n", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "private.csv").write_text("x\n", encoding="utf-8")
    _link(data / "shared", elsewhere)
    _link(data / "recent", data / "archive")
    problems = kd.data_path_problems(data, windows=False)
    assert len(problems) == 1 and problems[0].startswith("1 link(s) in the data folder point outside it (shared)")
    assert "recent" not in problems[0], "a link inside the folder resolves in the container too"
    assert kd.data_path_problems(tmp_path / "missing", windows=False) == []
    row = next(r for r in docker_runtime(tmp_path, FakeDocker()).check() if r.name == "data")
    assert not row.ok and not row.fatal and "(shared)" in row.hint


# --------------------------------------------------------------------------- #
# Starting and stopping
# --------------------------------------------------------------------------- #


def test_start_runs_the_hardened_kernel_offline_behind_a_forwarder(tmp_path):
    fake = FakeDocker()
    waits = []

    def health(url, timeout, should_stop=None):
        waits.append((url, timeout, should_stop()))
        return True

    names = kd.docker_names(tmp_path)
    rt, running = started(tmp_path, fake, health=health)
    token_file = next(Path(p.split("source=")[1].split(",")[0]) for p in fake.commands("run")[0] if "hailer-token" in p)

    assert [c[:2] for c in fake.calls if c[0] != "inspect"] == [
        ["version", "--format"], ["image", "inspect"], ["info", "--format"], ["ps", "-a"], ["network", "ls"],
        ["network", "create"], ["run", "-d"], ["create", "--pull"], ["network", "connect"], ["start", running.container_ids[1]],
    ]  # fmt: skip
    assert fake.commands("network", "create") == [["network", "create", "--internal", *labels(tmp_path, "network"), names.network]]
    assert fake.commands("run") == [[
        "run", "-d", "--pull", "never", "--name", names.kernel, "--network", names.network,
        "--init", "--read-only", "--tmpfs", "/tmp", "--tmpfs", "/home/analyst:uid=1000,gid=1000",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "256",
        "--memory", "4g", "--memory-swap", "4g", "--cpus", "2", "-w", "/work", *labels(tmp_path, "kernel"),
        "--mount", f"type=bind,source={tmp_path / 'notebooks'},target=/work/notebooks",
        "--mount", f"type=bind,source={tmp_path / 'data'},target=/work/data,readonly",
        "--mount", f"type=bind,source={token_file},target=/run/secrets/hailer-token,readonly",
        IMAGE, "marimo", "edit", "notebooks", "--host", "0.0.0.0", "--port", "2718",
        "--headless", "--skip-update-check", "--token-password-file", "/run/secrets/hailer-token",
    ]]  # fmt: skip
    assert fake.commands("create") == [[
        "create", "--pull", "never", "--name", names.forwarder, "-p", "127.0.0.1:2731:2718",
        "--init", "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--pids-limit", "64", "--memory", "64m", *labels(tmp_path, "forwarder"),
        IMAGE, "python", "-m", "hailer._forward", names.kernel, "2718", "2718",
    ]]  # fmt: skip
    kernel_c, forwarder_c = fake.container(names.kernel), fake.container(names.forwarder)
    network = fake.network(names.network)
    assert fake.commands("network", "connect") == [["network", "connect", network.id, forwarder_c.id]], "by id"

    run = fake.commands("run")[0]
    assert "-p" not in run and "--publish" not in run, "no port on the kernel: the forwarder publishes it"
    assert run.count("--mount") == 3 and not {"-v", "--volume", "--privileged", "--user"} & set(run), "nothing else is mounted"
    assert "docker.sock" not in " ".join(run) and f"source={tmp_path}," not in " ".join(run), "not the workspace, not the Docker socket"
    assert all(TOKEN not in " ".join(c) for c in fake.calls), "the token is in no docker argument (docker inspect shows them)"
    assert waits == [("http://127.0.0.1:2731", k.START_TIMEOUT_SEC, False)], "should_stop: both containers run"
    assert (tmp_path / "data").is_dir() and (tmp_path / "notebooks").is_dir(), "created before Docker could create them"

    assert isinstance(running, kd.DockerKernel)
    assert running.server == MarimoServer(
        url="http://127.0.0.1:2731", server_id="127.0.0.1:2731", source="kernel", token=TOKEN, runtime="docker",
        paths=k.docker_paths(make_config(tmp_path)),
    )  # fmt: skip
    assert running.log_hint == f"docker logs {names.kernel}" and "uvx hailer kernel stop" in running.stop_hint
    state = k.read_kernel_state(tmp_path)
    assert (state.runtime, state.url, state.port, state.token, state.image) == ("docker", "http://127.0.0.1:2731", 2731, TOKEN, IMAGE)
    assert state.containers == (names.kernel, names.forwarder) and state.container_ids == (kernel_c.id, forwarder_c.id)
    assert state.network == names.network and state.network_id == network.id and state.network_access is False
    assert state.mounts == ((str(tmp_path / "notebooks"), "/work/notebooks"), (str(tmp_path / "data"), "/work/data"))
    assert (state.memory, state.cpus) == ("4g", 2.0)

    running.stop()
    assert fake.calls[-3:] == [["rm", "-f", kernel_c.id], ["rm", "-f", forwarder_c.id], ["network", "rm", network.id]], "by id"
    assert fake.containers == {} and fake.networks == {} and k.read_kernel_state(tmp_path) is None
    running.stop()  # idempotent: "No such container" is not an error


def test_start_on_a_linux_host_runs_the_kernel_as_the_user(tmp_path):
    fake = FakeDocker()
    started(tmp_path, fake, user=lambda: (1234, 5678))
    run = fake.commands("run")[0]
    assert follows(run, ["--tmpfs", "/home/analyst:uid=1234,gid=5678", "--user", "1234:5678", "-e", "HOME=/home/analyst"])
    assert "--user" not in fake.commands("create")[0], "the forwarder touches no files"


def test_the_token_reaches_marimo_in_a_file_that_is_gone_once_it_started(tmp_path):
    """F7: never on the command line or in the environment, so ``docker inspect`` (Args, Env,
    Cmd) and the container's process list do not show it."""
    fake = FakeDocker()
    seen: dict = {}

    def health(url, timeout, should_stop=None):
        run = fake.commands("run")[0]
        seen["file"] = Path(next(m for m in run if "hailer-token" in m).split("source=")[1].split(",")[0])
        seen["exists"] = seen["file"].is_file()
        return True

    rt, running = started(tmp_path, fake, health=health)
    kernel_c = fake.container(running.container_ids[0])
    assert kernel_c.files["/run/secrets/hailer-token"] == TOKEN, "marimo reads it with --token-password-file"
    assert follows(kernel_c.argv, ["--token-password-file", "/run/secrets/hailer-token"])
    assert "--token-password" not in kernel_c.argv and not {"-e", "--env", "--env-file"} & set(kernel_c.argv)
    assert all(TOKEN not in arg for call in fake.calls for arg in call), "not in any docker argument"
    token_file = seen["file"]
    assert seen["exists"] and token_file.parent.parent == tmp_path / ".hailer", "there while marimo starts, in .hailer"
    assert token_file.parent.name.startswith("kernel-token-") and token_file.name == "token"
    assert not token_file.parent.exists(), "deleted once marimo answered"
    assert follows(kernel_c.argv, ["--mount", f"type=bind,source={token_file},target=/run/secrets/hailer-token,readonly"])


@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
def test_the_token_file_is_in_a_folder_only_this_user_can_open(tmp_path):
    old = os.umask(0o077)
    try:
        path = kd.write_token_file(tmp_path, TOKEN)
    finally:
        os.umask(old)
    assert path.read_text(encoding="utf-8") == TOKEN
    assert path.parent.stat().st_mode & 0o777 == 0o700, "other users cannot reach the file"
    assert path.stat().st_mode & 0o777 == 0o644, "the kernel's user may be another uid (root host, rootless Docker)"
    assert kd.remove_token_folder(path.parent) is None and not path.parent.exists()
    assert kd.remove_token_folder(path.parent) is None, "quietly: already gone"


def test_a_token_folder_that_cannot_be_deleted_is_a_warning(tmp_path, monkeypatch, capsys):
    def refuse(path):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(kd.shutil, "rmtree", refuse)
    folder = tmp_path / ".hailer" / "kernel-token-x"
    assert kd.remove_token_folder(folder) == f"Warning: could not delete {folder} ([Errno 13] Access is denied). It holds a kernel token: delete it yourself."
    folder.mkdir(parents=True)
    docker_runtime(tmp_path, FakeDocker()).start(2731)
    assert f"could not delete {folder}" in capsys.readouterr().out, "said at start, never silently ignored"


def test_stale_token_folders_are_swept_at_start_and_by_kernel_stop(tmp_path):
    """Left by a Hailer killed while its kernel was starting."""
    stale = tmp_path / ".hailer" / "kernel-token-stale"
    stale.mkdir(parents=True)
    (stale / "token").write_text("old-token", encoding="utf-8")
    fake = FakeDocker()
    docker_runtime(tmp_path, fake).start(2731)
    assert not stale.exists()
    stale.mkdir()
    report = kd.stop_workspace_kernels(make_config(tmp_path, kernel=KernelConfig(runtime="docker")), runner=fake, probe=answers())
    assert not stale.exists() and not report.warnings


@pytest.mark.parametrize("failure", ["docker", "unhealthy", "interrupt"])
def test_the_token_file_is_removed_when_a_start_fails(tmp_path, failure):
    fake = FakeDocker()
    if failure == "docker":
        fake.fail[("run",)] = (125, "docker: Error response from daemon: oops")

    def health(url, timeout, should_stop=None):
        if failure == "interrupt":
            raise KeyboardInterrupt
        return failure != "unhealthy"

    with pytest.raises((KernelRuntimeError, KeyboardInterrupt)):
        docker_runtime(tmp_path, fake, health=health).start(2731)
    assert not list((tmp_path / ".hailer").glob("kernel-token-*")), "no token left behind"


def test_start_with_network_access_publishes_the_kernel_directly(tmp_path):
    fake = FakeDocker()
    names = kd.docker_names(tmp_path)
    rt = docker_runtime(tmp_path, fake, kernel={"network": True, "memory": "8G", "cpus": 1.5})
    assert rt.describe() == f"docker (hailer-kernel {CONTRACT}; network on: the internet and this machine; data read-only)"
    running = rt.start(2731)
    run = fake.commands("run")[0]
    assert follows(run, ["--name", names.kernel, "-p", "127.0.0.1:2731:2718", "--init"])
    assert "--network" not in run and follows(run, ["--memory", "8G", "--memory-swap", "8G", "--cpus", "1.5"])
    assert not fake.commands("create") and not fake.commands("network", "create") and not fake.commands("start")
    state = k.read_kernel_state(tmp_path)
    assert state.network_access is True and state.containers == (names.kernel,) and state.network is None
    assert running.server.network_access is True
    running.stop()
    assert fake.calls[-1] == ["rm", "-f", running.container_ids[0]] and not fake.commands("network", "rm")


def test_start_refuses_to_orphan_a_live_local_server(tmp_path):
    k.write_kernel_state(tmp_path, k.KernelState(runtime="local", url=LIVE, port=2718, token=TOKEN, pid=77))
    fake = FakeDocker(image_contract=None)
    with pytest.raises(KernelRuntimeError) as exc:
        docker_runtime(tmp_path, fake, probe=answers((LIVE, TOKEN))).start(2731)
    assert str(exc.value) == f"A local marimo server Hailer started for this workspace is still running at {LIVE}."
    assert "uvx hailer kernel stop" in exc.value.hint
    assert not fake.streams, "refused before any download"
    assert not fake.commands("run") and not fake.commands("rm") and k.read_kernel_state(tmp_path).runtime == "local"


def test_start_refuses_to_start_over_a_live_docker_kernel(tmp_path):
    k.write_kernel_state(tmp_path, k.KernelState(runtime="docker", url=LIVE, port=2718, token=TOKEN))
    fake = FakeDocker()
    with pytest.raises(KernelRuntimeError) as exc:
        docker_runtime(tmp_path, fake, probe=answers((LIVE, TOKEN))).start(2731)
    assert "already running" in str(exc.value) and "uvx hailer kernel stop" in exc.value.hint
    assert not fake.commands("run") and not fake.commands("rm")


def test_start_removes_stopped_leftovers_by_id(tmp_path):
    names = kd.docker_names(tmp_path)
    fake = FakeDocker()
    workspace = str(tmp_path)
    old_kernel = fake.add_container(names.kernel, workspace=workspace, state="exited")
    old_forwarder = fake.add_container(names.forwarder, workspace=workspace, role="forwarder", state="created")
    old_network = fake.add_network(names.network, workspace=workspace)
    other = fake.add_container("hailer-kernel-0123456789", workspace=str(tmp_path / "another"), state="exited")
    k.write_kernel_state(tmp_path, k.KernelState(runtime="docker", url="http://127.0.0.1:9", port=9, token="old"))
    docker_runtime(tmp_path, fake).start(2731)
    label = f"label=org.openafterhours.hailer.workspace={tmp_path}"
    assert fake.commands("ps")[0][:4] == ["ps", "-a", "--filter", label]
    order = [c[:2] for c in fake.calls]
    assert order.index(["rm", "-f"]) < order.index(["network", "rm"]) < order.index(["network", "create"])
    assert ["rm", "-f", old_kernel.id[:12]] in fake.calls and ["rm", "-f", old_forwarder.id[:12]] in fake.calls
    assert ["network", "rm", old_network.id[:12]] in fake.calls
    assert fake.container(other.name) is other, "another workspace's container is not touched (the filter)"
    assert k.read_kernel_state(tmp_path).token == TOKEN


def test_start_never_removes_a_running_kernel_it_has_no_record_of(tmp_path):
    """A liveness probe that got it wrong (a busy server, a slow engine) must not destroy a kernel in use."""
    names = kd.docker_names(tmp_path)
    fake = FakeDocker()
    running_kernel = fake.add_container(names.kernel, workspace=str(tmp_path))
    fake.add_container(names.forwarder, workspace=str(tmp_path), role="forwarder")
    with pytest.raises(KernelRuntimeError) as exc:
        docker_runtime(tmp_path, fake).start(2731)
    assert str(exc.value) == f"A kernel container for this workspace is still running, but Hailer cannot reach it: {names.kernel}."
    assert "uvx hailer kernel stop" in exc.value.hint and "never removes a running kernel" in exc.value.hint
    assert not fake.commands("rm") and fake.container(names.kernel) is running_kernel


def test_a_running_forwarder_whose_kernel_stopped_is_removed(tmp_path):
    names = kd.docker_names(tmp_path)
    fake = FakeDocker()
    fake.add_container(names.kernel, workspace=str(tmp_path), state="exited")
    fake.add_container(names.forwarder, workspace=str(tmp_path), role="forwarder")
    docker_runtime(tmp_path, fake).start(2731)
    assert fake.container(names.kernel).state == "running" and k.read_kernel_state(tmp_path).token == TOKEN


def test_a_kernel_that_exits_early_is_reported_with_its_log_and_removed(tmp_path):
    fake = FakeDocker()
    names = kd.docker_names(tmp_path)

    def health(url, timeout, should_stop=None):
        kernel = fake.container(names.kernel)
        kernel.state, kernel.exit_code = "exited", 3
        assert should_stop() is True
        return False

    with pytest.raises(KernelRuntimeError) as exc:
        docker_runtime(tmp_path, fake, health=health).start(2731)
    assert str(exc.value) == "The kernel container exited early (code 3)."
    hint = exc.value.hint
    assert hint.startswith(f"Last lines of docker logs {names.kernel}:") and "    marimo is starting" in hint
    assert TOKEN not in hint and "access_token=<token>" in hint, "the token is masked in the log tail"
    assert fake.commands("logs") and fake.containers == {} and fake.networks == {}
    assert k.read_kernel_state(tmp_path) is None


def test_a_forwarder_that_exits_is_reported_with_its_own_log(tmp_path):
    fake = FakeDocker()
    names = kd.docker_names(tmp_path)

    def health(url, timeout, should_stop=None):
        forwarder = fake.container(names.forwarder)
        forwarder.state, forwarder.exit_code = "exited", 1
        return False

    with pytest.raises(KernelRuntimeError) as exc:
        docker_runtime(tmp_path, fake, health=health).start(2731)
    assert str(exc.value) == "The forwarder container exited early (code 1)."
    assert exc.value.hint.startswith(f"Last lines of docker logs {names.forwarder}:")


def test_a_kernel_that_never_answers_times_out_and_is_removed(tmp_path):
    fake = FakeDocker()
    with pytest.raises(KernelRuntimeError) as exc:
        docker_runtime(tmp_path, fake, health=lambda url, timeout, should_stop=None: False, start_timeout=7).start(2731)
    assert str(exc.value) == "Marimo did not answer on http://127.0.0.1:2731 within 7 s."
    assert fake.containers == {} and fake.networks == {}


def test_a_docker_error_mid_start_removes_what_was_created(tmp_path):
    fake = FakeDocker()
    fake.fail[("start",)] = (1, "Error response from daemon: ports are not available: exposing port TCP 127.0.0.1:2731")
    with pytest.raises(KernelRuntimeError) as exc:
        docker_runtime(tmp_path, fake).start(2731)
    assert str(exc.value) == "Docker could not start the forwarder container."
    assert exc.value.hint == "docker said: Error response from daemon: ports are not available: exposing port TCP 127.0.0.1:2731"
    assert fake.containers == {} and fake.networks == {} and k.read_kernel_state(tmp_path) is None


def test_a_docker_command_that_hangs_is_an_error_and_cleaned_up(tmp_path):
    fake = FakeDocker()
    fake.timeout.add(("run",))
    with pytest.raises(KernelRuntimeError) as exc:
        docker_runtime(tmp_path, fake).start(2731)
    assert str(exc.value) == "Docker did not start the kernel container within 60 s."
    assert fake.networks == {}, "the network it had created is removed"


def test_ctrl_c_while_waiting_removes_the_containers(tmp_path):
    fake = FakeDocker()

    def interrupted(url, timeout, should_stop=None):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        docker_runtime(tmp_path, fake, health=interrupted).start(2731)
    assert fake.containers == {} and fake.networks == {} and k.read_kernel_state(tmp_path) is None


def test_an_old_handle_never_removes_a_newer_kernel(tmp_path):
    """A chat started before `kernel stop` + a new start still holds a handle to the old kernel; when
    it ends, it must not remove the new kernel that reuses the names."""
    fake = FakeDocker()
    _rt, old = started(tmp_path, fake)
    fake.containers.clear()  # uvx hailer kernel stop in another terminal
    fake.networks.clear()
    k.delete_kernel_state(tmp_path)
    fake2_token = "a-newer-token-0123456789"
    newer_rt = kd.DockerRuntime(
        make_config(tmp_path, kernel=KernelConfig(runtime="docker")), runner=fake, token_factory=lambda: fake2_token,
        health=healthy, user=lambda: None, probe=answers(),
    )  # fmt: skip
    newer = newer_rt.start(2732)
    names = kd.docker_names(tmp_path)
    assert newer.containers == old.containers and newer.container_ids != old.container_ids, "same names, new ids"
    old.stop()
    assert fake.container(names.kernel) is not None and fake.container(names.forwarder) is not None and fake.network(names.network)
    assert k.read_kernel_state(tmp_path).token == fake2_token, "the newer record stays"
    newer.stop()
    assert fake.containers == {} and fake.networks == {}


def test_find_running_needs_the_token_and_this_workspaces_label(tmp_path):
    fake = FakeDocker()
    names = kd.docker_names(tmp_path)
    rt = docker_runtime(tmp_path, fake, probe=answers((LIVE, TOKEN)))
    assert rt.find_running() is None, "no record"
    kernel_c = fake.add_container(names.kernel, workspace=str(tmp_path / "elsewhere"))
    state = k.KernelState(
        runtime="docker", url=LIVE, port=2718, token=TOKEN, containers=(names.kernel,), container_ids=(kernel_c.id,),
        mounts=((str(tmp_path / "notebooks"), "/work/notebooks"), (str(tmp_path / "data"), "/work/data")),
    )  # fmt: skip
    k.write_kernel_state(tmp_path, state)
    assert rt.find_running() is None, "another workspace's container"
    kernel_c.labels[kd.LABEL_WORKSPACE] = str(tmp_path)
    running = rt.find_running()
    assert isinstance(running, kd.DockerKernel) and running.server.token == TOKEN and running.server.runtime == "docker"
    assert running.container_ids == (kernel_c.id,)
    assert running.server.paths.to_kernel(tmp_path / "notebooks" / "a.py") == "/work/notebooks/a.py"
    k.write_kernel_state(tmp_path, replace(state, token="not-its-token"))
    assert rt.find_running() is None
    k.write_kernel_state(tmp_path, replace(state, runtime="local"))
    assert rt.find_running() is None, "a local server is not the docker runtime's"


def test_docker_kernel_stop_never_raises(tmp_path):
    class Broken:
        def run(self, *args, **kwargs):
            raise OSError("docker vanished")

    k.write_kernel_state(tmp_path, k.KernelState(runtime="docker", url="http://127.0.0.1:2731", port=2731, token=TOKEN))
    kernel = kd.DockerKernel(
        server=MarimoServer(url="http://127.0.0.1:2731", token=TOKEN, runtime="docker"),
        workspace=tmp_path, runner=Broken(), containers=("hailer-kernel-x",), container_ids=("a" * 64,), network_id="b" * 64,
    )  # fmt: skip
    kernel.stop()
    assert k.read_kernel_state(tmp_path) is not None, "kept so another stop can retry"
    assert "docker vanished" in kernel.stop_error
    assert kernel.log_tail() == []


def test_removal_reports_what_it_did(tmp_path):
    fake = FakeDocker()
    _rt, running = started(tmp_path, fake)
    fake.containers.pop(running.container_ids[1])  # the forwarder went away by itself
    fake.fail[("network", "rm")] = (1, "Error response from daemon: error while removing network: network has active endpoints")
    outcome = running.remove()
    names = kd.docker_names(tmp_path)
    assert outcome.removed == [names.kernel] and outcome.gone == [names.forwarder]
    assert outcome.failed == [f"{names.network}: docker said: Error response from daemon: error while removing network: network has active endpoints"]


def test_removal_racing_another_cleanup_counts_as_gone(tmp_path):
    """`kernel stop` while the --foreground terminal removes the same kernel: "removal ... already in
    progress" is gone, and a network whose containers are still detaching is retried."""
    fake = FakeDocker()
    _rt, running = started(tmp_path, fake)
    fake.container(running.container_ids[0]).state = "removing"  # the other terminal is removing it
    fake.network_busy = 3
    outcome = running.remove()
    names = kd.docker_names(tmp_path)
    assert outcome.gone == [names.kernel] and outcome.removed == [names.forwarder, names.network] and outcome.failed == []
    assert len(fake.commands("network", "rm")) == 4, "three busy answers, then removed"
    assert fake.containers == {} and fake.networks == {}


def test_a_failure_is_only_reported_when_the_object_is_still_there(tmp_path):
    fake = FakeDocker()
    _rt, running = started(tmp_path, fake)
    kernel_id = running.container_ids[0]
    fake.fail[("rm", "-f", kernel_id)] = (1, "Error response from daemon: could not kill: tried to kill container, but did not receive an exit event")
    fake.containers.pop(kernel_id)  # ... and yet it is gone by the time docker is asked again
    outcome = running.remove()
    assert kd.docker_names(tmp_path).kernel in outcome.gone and outcome.failed == []
    assert ["inspect", "--type", "container", "--format", "{{.Id}}", kernel_id] in fake.calls


def test_a_broken_docker_context_is_never_taken_for_a_missing_object(tmp_path):
    """"context not found" once matched a bare "not found": records were dropped and removals reported
    as done. It is Docker that cannot be reached."""
    said = 'Failed to initialize: unable to resolve docker endpoint: context "nope": context not found: open C:\\x\\meta.json'
    broken = subprocess.CompletedProcess(["docker"], 1, "", said)
    assert not kd._missing(broken)
    assert kd._missing(subprocess.CompletedProcess(["docker"], 1, "", "Error response from daemon: network hailer-net-x not found"))
    assert kd._missing(subprocess.CompletedProcess(["docker"], 1, "", "Error response from daemon: No such container: x"))
    fake = FakeDocker()
    _rt, running = started(tmp_path, fake)
    fake.fail[("rm",)] = (1, said)
    fake.fail[("network", "rm")] = (1, said)
    fake.fail[("inspect",)] = (1, said)
    outcome = running.remove()
    assert outcome.removed == [] and outcome.gone == [] and len(outcome.failed) == 3
    state = k.KernelState(runtime="docker", url=LIVE, port=2718, token=TOKEN, container_ids=(running.container_ids[0],))
    assert not kd.containers_gone(state, fake), "a record is not dropped because the context is broken"
    fake.fail[("version",)] = (1, said)
    with pytest.raises(KernelRuntimeError) as exc:
        docker_runtime(tmp_path, fake).engine_version()
    assert str(exc.value) == "Docker is not running." and "DOCKER_CONTEXT" in exc.value.hint and "docker context use default" in exc.value.hint


def test_a_refused_start_keeps_the_record_of_a_kernel_that_does_not_answer(tmp_path):
    """Its containers are still there: the record is how kernel stop finds them by id."""
    fake = FakeDocker()
    names = kd.docker_names(tmp_path)
    kernel_c = fake.add_container(names.kernel, workspace=str(tmp_path))
    state = k.KernelState(runtime="docker", url=LIVE, port=2718, token="theirs", containers=(names.kernel,), container_ids=(kernel_c.id,))
    k.write_kernel_state(tmp_path, state)
    with pytest.raises(KernelRuntimeError) as exc:
        docker_runtime(tmp_path, fake).start(2731)  # the probe says it does not answer
    assert str(exc.value) == f"A docker kernel Hailer started for this workspace ({names.kernel}) may still be running, but it does not answer at {LIVE}."
    assert "uvx hailer kernel stop" in exc.value.hint
    assert k.read_kernel_state(tmp_path) == state and fake.container(names.kernel) is kernel_c, "nothing forgotten, nothing removed"
    kernel_c.state = "exited"  # now provably gone: the start goes ahead, removes it and replaces the record
    docker_runtime(tmp_path, fake).start(2731)
    assert k.read_kernel_state(tmp_path).token == TOKEN and kernel_c.id not in fake.containers



# --------------------------------------------------------------------------- #
# --foreground: following the log, and saying why it ended
# --------------------------------------------------------------------------- #


def test_foreground_follows_the_kernel_log_until_it_stops_or_ctrl_c(tmp_path):
    fake = FakeDocker()
    _rt, running = started(tmp_path, fake)
    kernel_id = running.container_ids[0]
    assert fake.commands("run")[0][:2] == ["run", "-d"], "detached either way"

    fake.container(kernel_id).state = "exited"
    assert running.wait() == 0 and running.ended == "The kernel container exited (code 0)."
    assert fake.streams[-1] == ["logs", "-f", "--tail", "0", kernel_id], "new lines only: marimo's banner shows the in-container URL"
    fake.container(kernel_id).exit_code = 137
    assert running.wait() == 137 and "stopped from outside this terminal" in running.ended
    fake.container(kernel_id).oom = True
    running.wait()
    assert running.ended.startswith("The kernel ran out of memory") and "[kernel].memory" in running.ended
    fake.container(kernel_id).state = "removing"
    running.wait()
    assert running.ended == "The kernel container was removed from outside this terminal (uvx hailer kernel stop, or docker rm)."
    fake.containers.pop(kernel_id)
    assert running.wait() == 1 and "removed from outside this terminal" in running.ended

    def ctrl_c(args):
        raise KeyboardInterrupt

    fake.stream_hook = ctrl_c
    running.ended = ""
    assert running.wait() == 0 and running.ended == "", "Ctrl+C ends the wait quietly; the caller removes the containers"


# --------------------------------------------------------------------------- #
# Reusing a kernel started with other settings
# --------------------------------------------------------------------------- #


def test_settings_mismatch_names_every_difference(tmp_path):
    config = make_config(tmp_path, kernel=KernelConfig(runtime="docker"))
    state = k.KernelState(
        runtime="docker", url=LIVE, port=2718, token=TOKEN, image=IMAGE, network_access=False, memory="4G", cpus=2.0,
        mounts=((str(tmp_path / "notebooks"), "/work/notebooks"), (str(tmp_path / "data"), "/work/data")),
    )  # fmt: skip
    assert kd.settings_mismatch(state, config) == [], "memory compares case-insensitively"
    other = replace(
        state, network_access=True, image="registry.example/hailer-kernel:dev", memory="8g", cpus=4.0,
        mounts=((str(tmp_path / "old-notebooks"), "/work/notebooks"), (str(tmp_path / "data"), "/work/data")),
    )  # fmt: skip
    diffs = kd.settings_mismatch(other, config)
    assert diffs == [
        f"notebooks folder {tmp_path / 'old-notebooks'}, hailer.toml: {tmp_path / 'notebooks'}",
        "network on, hailer.toml: off",
        f"image registry.example/hailer-kernel:dev, hailer.toml: {IMAGE}",
        "memory 8g, hailer.toml: 4g",
        "cpus 4, hailer.toml: 2",
    ]
    local = make_config(tmp_path)  # asking for local may attach to a docker kernel: only its folders must match
    assert kd.settings_mismatch(other, local) == diffs[:1]
    err = kd.mismatch_error(diffs[1:2])
    assert str(err) == "The running kernel was started with other settings (network on, hailer.toml: off)."
    assert "uvx hailer kernel stop" in err.hint


# --------------------------------------------------------------------------- #
# uvx hailer kernel stop
# --------------------------------------------------------------------------- #


def test_kernel_stop_removes_a_docker_kernel_and_strays(tmp_path):
    fake = FakeDocker()
    _rt, running = started(tmp_path, fake)
    stray = fake.add_container("hailer-kernel-0123456789", workspace=str(tmp_path))  # running: kernel stop removes it too
    names = kd.docker_names(tmp_path)
    report = kd.stop_workspace_kernels(make_config(tmp_path), runner=fake, probe=answers())
    assert report.done == [
        f"Stopped the docker kernel at http://127.0.0.1:2731 (removed {names.kernel}, {names.forwarder}, {names.network}).",
        f"Removed {stray.name}.",
    ]
    assert report.failed == [] and fake.containers == {} and fake.networks == {}
    assert k.read_kernel_state(tmp_path) is None
    assert kd.stop_workspace_kernels(make_config(tmp_path), runner=fake, probe=answers()).done == [], "nothing to stop"


def test_kernel_stop_is_honest_about_what_was_already_gone(tmp_path):
    fake = FakeDocker()
    _rt, running = started(tmp_path, fake)
    fake.containers.clear()
    fake.networks.clear()
    report = kd.stop_workspace_kernels(make_config(tmp_path), runner=fake, probe=answers())
    assert report.done == ["Removed the record of a docker kernel whose containers were already gone (http://127.0.0.1:2731)."]
    assert report.failed == [] and k.read_kernel_state(tmp_path) is None


def test_kernel_stop_reports_what_it_could_not_remove(tmp_path):
    fake = FakeDocker()
    _rt, running = started(tmp_path, fake)
    fake.fail[("network", "rm")] = (1, "Error response from daemon: network has active endpoints")
    report = kd.stop_workspace_kernels(make_config(tmp_path), runner=fake, probe=answers())
    names = kd.docker_names(tmp_path)
    assert "its record was kept" in report.done[0]
    assert f"Removed {names.kernel}, {names.forwarder}." in report.done[0]
    assert report.failed and all(line.startswith(f"Could not remove {names.network}: docker said:") for line in report.failed)
    assert k.read_kernel_state(tmp_path) is not None, "kept: something is still there"


def test_kernel_stop_stops_a_live_local_server_and_never_kills_a_stale_pid(tmp_path, monkeypatch):
    # These are fake PIDs; a CI runner may have real processes with the same numbers.
    monkeypatch.setattr(k, "process_running", lambda pid: False)
    monkeypatch.setattr(kd, "process_running", lambda pid: False)
    procs = Procs()
    k.write_kernel_state(tmp_path, k.KernelState(runtime="local", url=LIVE, port=2718, token=TOKEN, pid=77))
    report = kd.stop_workspace_kernels(make_config(tmp_path), runner=FakeDocker(installed=False), procs=procs.local_processes(), probe=answers((LIVE, TOKEN)))
    assert report.done == [f"Stopped the local marimo server at {LIVE} (process 77)."]
    assert procs.killed == [77] and k.read_kernel_state(tmp_path) is None
    k.write_kernel_state(tmp_path, k.KernelState(runtime="local", url="http://127.0.0.1:9", port=9, token=TOKEN, pid=78))
    report = kd.stop_workspace_kernels(make_config(tmp_path), runner=FakeDocker(), procs=procs.local_processes(), probe=answers())
    assert report.done == ["Removed the record of a marimo server that no longer answers (http://127.0.0.1:9)."]
    assert procs.killed == [77], "a pid from a record that does not answer may belong to anything now"


def test_kernel_stop_says_when_the_process_of_a_silent_server_still_runs(tmp_path, monkeypatch):
    """The start guard refuses such a record; kernel stop clears it, and says what it cannot know."""
    monkeypatch.setattr(kd, "process_running", lambda pid: True)
    k.write_kernel_state(tmp_path, k.KernelState(runtime="local", url=LIVE, port=2718, token=TOKEN, pid=4321))
    procs = Procs()
    report = kd.stop_workspace_kernels(make_config(tmp_path), runner=FakeDocker(), procs=procs.local_processes(), probe=answers())
    assert report.done[0].startswith(f"Removed the record of a marimo server that no longer answers ({LIVE}). Process 4321 is still running")
    assert procs.killed == [] and k.read_kernel_state(tmp_path) is None


def test_kernel_stop_warns_about_files_planted_in_the_notebooks_folder(tmp_path):
    fake = FakeDocker()
    _rt, running = started(tmp_path, fake)
    (tmp_path / "notebooks" / ".git").mkdir()  # written by notebook code during the session
    (tmp_path / "notebooks" / ".vscode").mkdir()
    report = kd.stop_workspace_kernels(make_config(tmp_path), runner=fake, probe=answers())
    assert report.done[0].startswith("Stopped the docker kernel") and report.failed == []
    assert report.warnings == [kd.planted_warning(tmp_path / "notebooks", [".git", ".vscode"])]
    assert running.planted() == [".git", ".vscode"] and running.notebooks_folder == tmp_path / "notebooks"
    assert kd.stop_workspace_kernels(make_config(tmp_path), runner=fake, probe=answers()).warnings == [], "nothing was stopped"


def test_kernel_stop_with_docker_down_or_missing(tmp_path):
    down = FakeDocker(engine=None)
    report = kd.stop_workspace_kernels(make_config(tmp_path), runner=down, probe=answers())
    assert report.done == [
        "Docker is not running, so containers an earlier docker kernel may have left were not checked "
        "(start Docker Desktop and run uvx hailer kernel stop again to remove them)."
    ]
    assert kd.stop_workspace_kernels(make_config(tmp_path), runner=FakeDocker(installed=False), probe=answers()).done == [], (
        "no Docker at all: nothing to say"
    )
    k.write_kernel_state(tmp_path, k.KernelState(runtime="docker", url="http://127.0.0.1:9", port=9, token=TOKEN))
    with pytest.raises(KernelRuntimeError, match="Docker is not running"):
        kd.stop_workspace_kernels(make_config(tmp_path), runner=down, probe=answers())
    assert k.read_kernel_state(tmp_path) is not None, "an unreachable engine cannot prove the kernel is gone"
    k.write_kernel_state(tmp_path, k.KernelState(runtime="docker", url=LIVE, port=2718, token=TOKEN))
    with pytest.raises(KernelRuntimeError) as exc:
        kd.stop_workspace_kernels(make_config(tmp_path), runner=down, probe=answers((LIVE, TOKEN)))
    assert str(exc.value) == "Docker is not running."
    assert k.read_kernel_state(tmp_path) is not None, "kept: the containers still need removing"


# --------------------------------------------------------------------------- #
# SubprocessDockerRunner
# --------------------------------------------------------------------------- #


def test_subprocess_runner_without_docker_says_so(monkeypatch):
    monkeypatch.setattr(kd.shutil, "which", lambda name: None)
    runner = kd.SubprocessDockerRunner()
    for call in (lambda: runner.run(["version"]), lambda: runner.stream(["pull", IMAGE])):
        with pytest.raises(KernelRuntimeError) as exc:
            call()
        assert str(exc.value) == "Docker is not installed."


def test_subprocess_runner_runs_docker_in_text_mode_with_empty_stdin(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[1] == "rm":
            return subprocess.CompletedProcess(argv, 1, "", "Error response from daemon: boom")
        return subprocess.CompletedProcess(argv, 0, "29.4.3 linux\n", "")

    monkeypatch.setattr(kd.subprocess, "run", fake_run)
    runner = kd.SubprocessDockerRunner("/usr/bin/docker")
    assert runner.run(["version"], timeout=5).stdout == "29.4.3 linux\n"
    argv, kwargs = calls[0]
    assert argv == ["/usr/bin/docker", "version"]
    assert kwargs["text"] and kwargs["encoding"] == "utf-8" and kwargs["capture_output"] and kwargs["timeout"] == 5
    assert kwargs["check"] is False and kwargs["stdin"] == subprocess.DEVNULL, "docker never reads the chat's terminal"
    assert runner.run(["rm", "-f", "x"]).returncode == 1, "check is off by default"
    with pytest.raises(KernelRuntimeError) as exc:  # never a raw CalledProcessError
        runner.run(["rm", "-f", "x"], check=True)
    assert str(exc.value) == "docker rm -f failed (exit code 1)." and exc.value.hint == "docker said: Error response from daemon: boom"


class _Proc:
    interrupt = False
    last = None
    stderr_lines: list[bytes] = []

    def __init__(self, argv, **kwargs):
        self.argv, self.kwargs = argv, kwargs
        self.waits = 0
        self.terminated = False
        self.stderr = io.BytesIO(b"".join(_Proc.stderr_lines)) if kwargs.get("stderr") == subprocess.PIPE else None
        _Proc.last = self

    def wait(self, timeout=None):
        self.waits += 1
        if _Proc.interrupt and self.waits == 2:
            raise KeyboardInterrupt
        if self.waits < 3:
            raise subprocess.TimeoutExpired("docker", timeout)
        return 1 if self.stderr is not None else 0

    def terminate(self):
        self.terminated = True


def test_subprocess_runner_stream_waits_in_short_steps_and_stops_docker_on_ctrl_c(monkeypatch):
    monkeypatch.setattr(kd.subprocess, "Popen", _Proc)
    runner = kd.SubprocessDockerRunner("docker")
    result = runner.stream(["build", "."])
    assert result.returncode == 0 and _Proc.last.argv == ["docker", "build", "."] and _Proc.last.waits == 3
    assert _Proc.last.kwargs["stdin"] == subprocess.DEVNULL and _Proc.last.kwargs["stderr"] is None, "build keeps its console progress"
    _Proc.interrupt = True
    try:
        with pytest.raises(KeyboardInterrupt):
            runner.stream(["logs", "-f", "hailer-kernel-x"])
        assert _Proc.last.terminated
    finally:
        _Proc.interrupt = False


def test_subprocess_runner_stream_keeps_dockers_errors_and_still_shows_them(monkeypatch, capsys):
    monkeypatch.setattr(kd.subprocess, "Popen", _Proc)
    monkeypatch.setattr(_Proc, "stderr_lines", [b"Error response from daemon: error from registry: denied\n", b"denied\n"])
    result = kd.SubprocessDockerRunner("docker").stream(["pull", IMAGE], keep_errors=True)
    assert result.returncode == 1 and result.stderr == "Error response from daemon: error from registry: denied\ndenied"
    assert "error from registry: denied" in capsys.readouterr().err, "shown as it comes"
