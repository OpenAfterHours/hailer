"""Tests for hailer.kernel_docker: the docker kernel runtime against a scripted docker CLI with
state (tests/fake_docker.py: nothing is run), owners and leftovers, ``uvx hailer status`` and
``uvx hailer kernel stop``, the folder checks and the subprocess runner. No Docker needed;
liveness probes are stubs."""

from __future__ import annotations

import hashlib
import io
import os
import re
import subprocess
from pathlib import Path

import pytest

from fake_docker import CONTRACT, FakeDocker
from fake_kernel import TOKEN, make_config
from hailer import kernel as k
from hailer import kernel_docker as kd
from hailer.errors import KernelRuntimeError
from hailer.models import KernelConfig, MarimoServer

IMAGE = f"ghcr.io/openafterhours/hailer-kernel:{CONTRACT}"
SUFFIX = "a1b2c3"
#: An owner id no lock file belongs to: its Hailer is gone.
GONE = "0123456789abcdef"


def owned(owner: str) -> dict[str, str]:
    """The owner label of an object a Hailer with lock ``owner`` started."""
    return {kd.LABEL_OWNER: owner}


def healthy(url, timeout, token=None, should_stop=None):
    return True


class NoSync:
    """No marimo answers behind these starts: the notebook copies are tests/test_notebook_sync.py's."""

    def __init__(self, *args, **kwargs):
        self.closed = []

    def sync_in(self):
        return []

    def start(self, interval=None):
        pass

    def soon(self):
        pass

    def close(self, *, final=True):
        self.closed.append(final)
        return []


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch):
    """Retries of `network rm` do not sleep in tests, and nothing is copied."""
    monkeypatch.setattr(kd, "_sleep", lambda seconds: None)
    monkeypatch.setattr(kd, "NotebookSync", NoSync)


def docker_runtime(tmp_path: Path, fake: FakeDocker, *, kernel: dict | None = None, config=None, **kw) -> kd.DockerRuntime:
    config = config or make_config(tmp_path, kernel=KernelConfig(runtime="docker", **(kernel or {})))
    kw.setdefault("health", healthy)
    kw.setdefault("user", lambda: None)
    kw.setdefault("suffix", SUFFIX)
    return kd.DockerRuntime(config, runner=fake, token_factory=lambda: TOKEN, **kw)


def labels(tmp_path: Path, role: str, owner: str) -> list[str]:
    return [
        "--label", f"org.openafterhours.hailer.workspace={tmp_path}",
        "--label", f"org.openafterhours.hailer.contract={CONTRACT}",
        "--label", f"org.openafterhours.hailer.role={role}",
        "--label", f"org.openafterhours.hailer.owner={owner}",
    ]  # fmt: skip


@pytest.fixture
def live_owner(tmp_path):
    """Another Hailer session of this workspace, still running (its owner lock is held)."""
    lock = kd.acquire_owner_lock(tmp_path)
    yield lock.id
    lock.release()


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
    assert kd.docker_names(tmp_path, SUFFIX) == kd.DockerNames(
        f"hailer-kernel-{ident}-{SUFFIX}", f"hailer-fwd-{ident}-{SUFFIX}", f"hailer-net-{ident}-{SUFFIX}"
    )
    fresh = kd.docker_names(tmp_path)
    assert re.fullmatch(rf"hailer-kernel-{ident}-[0-9a-f]{{6}}", fresh.kernel), "a random suffix per start"
    assert fresh != kd.docker_names(tmp_path), "two sessions in one workspace never share names"
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
    assert not hasattr(rt, "paths"), "names in the notebooks folder replaced the host <-> kernel path map"
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
        (FakeDocker(installed=False), "Docker is not installed.", "https://docs.docker.com/desktop/"),
        (FakeDocker(engine=None), "Docker is not running.", "Start Docker Desktop"),
        (FakeDocker(engine="29.4.3 windows"), "Docker runs windows containers; Hailer's kernel image needs Linux containers.", "Switch to Linux containers"),
    ],
)
def test_docker_that_cannot_run_the_kernel_fails_closed(tmp_path, fake, message, hint):
    rt = docker_runtime(tmp_path, fake)
    rows = rt.check()
    docker = next(r for r in rows if r.name == "docker")
    assert not docker.ok and docker.fatal and docker.summary == message and hint in docker.hint
    assert docker.hint.endswith(
        'Or, only if you accept that notebook code then runs as you, with your files and network (not isolated): '
        'set [kernel] runtime = "unsafe-local" in hailer.toml (or HAILER_KERNEL=unsafe-local).'
    ), "both ways on, the opt-in last"
    assert "image" not in [r.name for r in rows]
    with pytest.raises(KernelRuntimeError) as exc:
        rt.start(2731)
    assert str(exc.value) == message
    assert not fake.commands("run") and not fake.commands("create")


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


# --------------------------------------------------------------------------- #
# The data folder: never one that exposes Hailer's files; folders Docker cannot mount
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("notebooks", "data", "fragment"),
    [
        ("notebooks", "", "The data folder ({ws}) is the workspace folder"),
        ("notebooks", ".hailer", "is Hailer's .hailer folder"),
        ("notebooks", ".config/hailer", "contains the context folder"),
    ],
)
def test_start_refuses_mounts_that_expose_hailers_own_files(tmp_path, notebooks, data, fragment):
    """Checked on every start path, from the one rule set (config.docker_mount_problems): --foreground
    never validates hailer.toml, and HAILER_DATA_DIR can move the folder."""
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


@pytest.mark.parametrize("notebooks", ["", "notebooks/.git", ".hailer/nb", ".config/hailer/context", "unc"])
def test_the_notebooks_folder_is_never_mounted_so_no_layout_of_it_is_refused(tmp_path, monkeypatch, notebooks):
    """The kernel's notebooks folder is its own tmpfs: a notebooks folder that is the workspace, a
    git repository, inside .hailer or the context folder, or on a network share starts, and no
    part of it reaches the container (only allow-listed notebooks are copied back)."""
    monkeypatch.setattr(kd, "_on_windows", lambda: notebooks == "unc")
    if notebooks == "unc":
        folder = Path(r"\\fileserver\team\notebooks")
    else:
        folder = (tmp_path / notebooks).parent if notebooks.endswith(".git") else tmp_path / notebooks
        (tmp_path / notebooks).mkdir(parents=True, exist_ok=True)
    config = make_config(tmp_path, notebooks_dir=folder, notebook=folder / "analysis.py", kernel=KernelConfig(runtime="docker"))
    fake = FakeDocker()
    docker_runtime(tmp_path, fake, config=config).start(2731)
    run = " ".join(fake.commands("run")[0])
    assert f"source={folder}," not in run and "target=/work/notebooks" not in run
    assert "--tmpfs /work/notebooks:uid=1000,gid=1000,mode=0700,size=512m" in run


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


def test_doctor_has_no_notebooks_row_in_docker_mode(tmp_path, monkeypatch):
    """Nothing about the notebooks folder can stop or weaken a docker kernel: it is never mounted."""
    monkeypatch.setattr(kd, "_on_windows", lambda: True)
    monkeypatch.setattr(kd, "_drive_type", lambda root: kd.DRIVE_REMOTE if root == "Z:\\" else 3)
    (tmp_path / "notebooks" / ".vscode").mkdir(parents=True)
    config = make_config(tmp_path, notebooks_dir=Path("Z:/team/notebooks"), kernel=KernelConfig(runtime="docker"))
    assert "notebooks" not in [r.name for r in docker_runtime(tmp_path, FakeDocker(), config=config).check()]


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

    def health(url, timeout, token=None, should_stop=None):
        waits.append((url, timeout, should_stop()))
        return True

    names = kd.docker_names(tmp_path, SUFFIX)
    rt, running = started(tmp_path, fake, health=health)
    token_file = next(Path(p.split("source=")[1].split(",")[0]) for p in fake.commands("run")[0] if "hailer-token" in p)

    assert [c[:2] for c in fake.calls if c[0] != "inspect"] == [
        ["version", "--format"], ["image", "inspect"], ["info", "--format"], ["ps", "-a"], ["network", "ls"],
        ["network", "create"], ["run", "-d"], ["create", "--pull"], ["network", "connect"], ["start", running.container_ids[1]],
    ]  # fmt: skip
    assert fake.commands("network", "create") == [["network", "create", "--internal", *labels(tmp_path, "network", rt.owner_id), names.network]]
    assert fake.commands("run") == [[
        "run", "-d", "--pull", "never", "--name", names.kernel, "--network", names.network,
        "--init", "--read-only", "--tmpfs", "/tmp", "--tmpfs", "/home/analyst:uid=1000,gid=1000",
        "--tmpfs", "/work/notebooks:uid=1000,gid=1000,mode=0700,size=512m",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "256",
        "--memory", "4g", "--memory-swap", "4g", "--cpus", "2", "-w", "/work", *labels(tmp_path, "kernel", rt.owner_id),
        "--mount", f"type=bind,source={tmp_path / 'data'},target=/work/data,readonly",
        "--mount", f"type=bind,source={token_file},target=/run/secrets/hailer-token,readonly",
        IMAGE, "marimo", "edit", "notebooks", "--host", "0.0.0.0", "--port", "2718",
        "--headless", "--skip-update-check", "--token-password-file", "/run/secrets/hailer-token",
    ]]  # fmt: skip
    assert fake.commands("create") == [[
        "create", "--pull", "never", "--name", names.forwarder, "-p", "127.0.0.1:2731:2718",
        "--init", "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--pids-limit", "64", "--memory", "64m", *labels(tmp_path, "forwarder", rt.owner_id),
        IMAGE, "python", "-m", "hailer._forward", names.kernel, "2718", "2718",
    ]]  # fmt: skip
    kernel_c, forwarder_c = fake.container(names.kernel), fake.container(names.forwarder)
    network = fake.network(names.network)
    assert fake.commands("network", "connect") == [["network", "connect", network.id, forwarder_c.id]], "by id"

    run = fake.commands("run")[0]
    assert "-p" not in run and "--publish" not in run, "no port on the kernel: the forwarder publishes it"
    assert run.count("--mount") == 2 and not {"-v", "--volume", "--privileged", "--user"} & set(run), "nothing else is mounted"
    assert "docker.sock" not in " ".join(run) and f"source={tmp_path}," not in " ".join(run), "not the workspace, not the Docker socket"
    assert f"source={tmp_path / 'notebooks'}" not in " ".join(run), "the notebooks folder is copied, never mounted"
    assert all(TOKEN not in " ".join(c) for c in fake.calls), "the token is in no docker argument (docker inspect shows them)"
    assert waits == [("http://127.0.0.1:2731", k.START_TIMEOUT_SEC, False)], "should_stop: both containers run"
    assert (tmp_path / "data").is_dir(), "created before Docker could create it"

    assert isinstance(running, kd.DockerKernel)
    assert running.server == MarimoServer(url="http://127.0.0.1:2731", token=TOKEN, runtime="docker")
    # the sandbox: the kernel knows the folders by its own paths; data is listed on the host
    assert running.notebooks_path == "/work/notebooks" and running.data_path == "/work/data" and not running.native_paths
    assert running.data_dir == tmp_path / "data" and isinstance(running.sync, NoSync), "the start set up the copies"
    assert running.notebook_url("q3/r.py") == "http://127.0.0.1:2731/?file=/work/notebooks/q3/r.py&view-as=present"
    assert running.describe() == f"docker (hailer-kernel {CONTRACT}; no network; data read-only)"
    assert running.log_hint == f"docker logs {names.kernel}"
    assert running.containers == (names.kernel, names.forwarder) and running.container_ids == (kernel_c.id, forwarder_c.id)
    assert running.network == names.network and running.network_id == network.id
    assert [p.name for p in (tmp_path / ".hailer").glob("*.json")] == ["last-kernel.json"], "no kernel record: the chat holds it in memory"

    sync = running.sync
    running.stop()
    assert sync.closed == [True] and running.sync is None, "the last copy back, before removal"
    assert fake.calls[-3:] == [["rm", "-f", kernel_c.id], ["rm", "-f", forwarder_c.id], ["network", "rm", network.id]], "by id"
    assert fake.containers == {} and fake.networks == {}
    running.stop()  # idempotent: "No such container" is not an error


def test_start_on_a_linux_host_runs_the_kernel_as_the_user(tmp_path):
    fake = FakeDocker()
    started(tmp_path, fake, user=lambda: (1234, 5678))
    run = fake.commands("run")[0]
    assert follows(run, [
        "--tmpfs", "/home/analyst:uid=1234,gid=5678", "--tmpfs", "/work/notebooks:uid=1234,gid=5678,mode=0700,size=512m",
        "--user", "1234:5678", "-e", "HOME=/home/analyst",
    ])  # fmt: skip
    assert "--user" not in fake.commands("create")[0], "the forwarder touches no files"


def test_the_token_reaches_marimo_in_a_file_that_is_gone_once_it_started(tmp_path):
    """F7: never on the command line or in the environment, so ``docker inspect`` (Args, Env,
    Cmd) and the container's process list do not show it."""
    fake = FakeDocker()
    seen: dict = {}

    def health(url, timeout, token=None, should_stop=None):
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
    docker_runtime(tmp_path, FakeDocker()).start(2731)
    out = capsys.readouterr().out
    assert "Warning: could not delete" in out and "kernel-token-" in out, "said at start, never silently ignored"


@pytest.mark.parametrize("failure", ["docker", "unhealthy", "interrupt"])
def test_the_token_file_is_removed_when_a_start_fails(tmp_path, failure):
    fake = FakeDocker()
    if failure == "docker":
        fake.fail[("run",)] = (125, "docker: Error response from daemon: oops")

    def health(url, timeout, token=None, should_stop=None):
        if failure == "interrupt":
            raise KeyboardInterrupt
        return failure != "unhealthy"

    with pytest.raises((KernelRuntimeError, KeyboardInterrupt)):
        docker_runtime(tmp_path, fake, health=health).start(2731)
    assert not list((tmp_path / ".hailer").glob("kernel-token-*")), "no token left behind"
    assert not list((tmp_path / ".hailer").glob("owner-*.lock")), "the owner lock is released with what was created"


def test_start_with_network_access_publishes_the_kernel_directly(tmp_path):
    fake = FakeDocker()
    names = kd.docker_names(tmp_path, SUFFIX)
    rt = docker_runtime(tmp_path, fake, kernel={"network": True, "memory": "8G", "cpus": 1.5})
    assert rt.describe() == f"docker (hailer-kernel {CONTRACT}; network on: the internet and this machine; data read-only)"
    running = rt.start(2731)
    run = fake.commands("run")[0]
    assert follows(run, ["--name", names.kernel, "-p", "127.0.0.1:2731:2718", "--init"])
    assert "--network" not in run and follows(run, ["--memory", "8G", "--memory-swap", "8G", "--cpus", "1.5"])
    assert not fake.commands("create") and not fake.commands("network", "create") and not fake.commands("start")
    assert running.containers == (names.kernel,) and running.network is None
    assert running.server.network_access is True
    running.stop()
    assert fake.calls[-1] == ["rm", "-f", running.container_ids[0]] and not fake.commands("network", "rm")


def test_start_removes_what_owners_that_are_gone_left_by_id(tmp_path):
    """A killed Hailer's kernel (even a running one), an older Hailer's container (no owner label)
    and a network whose owner is gone all go; another workspace's container stays."""
    fake = FakeDocker()
    workspace = str(tmp_path)
    stale = tmp_path / ".hailer" / f"owner-{GONE}.lock"  # its Hailer was killed: nobody holds the lock
    stale.parent.mkdir(parents=True)
    stale.write_text("", encoding="utf-8")
    os.utime(stale, (1, 1))
    killed_kernel = fake.add_container("hailer-kernel-x-1", workspace=workspace, **owned(GONE))
    killed_forwarder = fake.add_container("hailer-fwd-x-1", workspace=workspace, role="forwarder", state="created", **owned(GONE))
    killed_network = fake.add_network("hailer-net-x-1", workspace=workspace, **owned(GONE))
    old_kernel = fake.add_container("hailer-kernel-0123456789", workspace=workspace)
    other = fake.add_container("hailer-kernel-y-1", workspace=str(tmp_path / "another"), state="exited")
    docker_runtime(tmp_path, fake).start(2731)
    label = f"label=org.openafterhours.hailer.workspace={tmp_path}"
    assert fake.commands("ps")[0][:4] == ["ps", "-a", "--filter", label]
    order = [c[:2] for c in fake.calls]
    assert order.index(["rm", "-f"]) < order.index(["network", "rm"]) < order.index(["network", "create"])
    for gone in (killed_kernel, killed_forwarder, old_kernel):
        assert ["rm", "-f", gone.id[:12]] in fake.calls
    assert ["network", "rm", killed_network.id[:12]] in fake.calls
    assert fake.container(other.name) is other, "another workspace's container is not touched (the filter)"
    assert not stale.exists(), "a lock file nobody holds is swept"


def test_a_start_never_removes_a_live_sessions_objects(tmp_path, live_owner):
    """Another session in this workspace (it holds its owner lock): its running kernel, the
    forwarder it created but has not started yet, and its network all stay."""
    fake = FakeDocker()
    workspace = str(tmp_path)
    live = [
        fake.add_container("hailer-kernel-live-1", workspace=workspace, **owned(live_owner)),
        fake.add_container("hailer-fwd-live-1", workspace=workspace, role="forwarder", state="created", **owned(live_owner)),
    ]
    live_network = fake.add_network("hailer-net-live-1", workspace=workspace, **owned(live_owner))
    rt, running = started(tmp_path, fake)
    assert not fake.commands("rm") and not fake.commands("network", "rm")
    assert all(fake.container(c.id) is c for c in live) and fake.network(live_network.id) is live_network
    assert running.containers == (rt.names.kernel, rt.names.forwarder), "its own names, next to the others"
    assert (tmp_path / ".hailer" / f"owner-{live_owner}.lock").exists(), "a held lock is never swept"
    running.stop()
    assert all(fake.container(c.id) is c for c in live), "a session removes only what it started"


def test_owner_locks_say_whether_their_hailer_still_runs(tmp_path):
    """The OS holds the lock for the owner's lifetime: a lock file anyone can take is a dead owner,
    whatever pid the process had (a reused pid cannot make it look alive)."""
    lock = kd.acquire_owner_lock(tmp_path)
    assert lock.path == tmp_path / ".hailer" / f"owner-{lock.id}.lock"
    assert kd.owner_alive(tmp_path, lock.id), "held by this process, through another handle"
    assert not kd.owner_alive(tmp_path, GONE) and not kd.owner_alive(tmp_path, "") and not kd.owner_alive(tmp_path, "../x")
    kd.sweep_owner_locks(tmp_path, grace=0)
    assert lock.path.exists(), "a held lock is never swept"
    lock.release()
    assert not lock.path.exists() and not kd.owner_alive(tmp_path, lock.id)
    lock.release()  # idempotent
    fresh = tmp_path / ".hailer" / f"owner-{GONE}.lock"
    fresh.write_text("", encoding="utf-8")
    kd.sweep_owner_locks(tmp_path)
    assert fresh.exists(), "too young: its owner may be about to lock it"
    kd.sweep_owner_locks(tmp_path, grace=0)
    assert not fresh.exists()


def test_a_start_waits_for_its_own_token_after_checking_its_containers(tmp_path):
    """Two sessions can pick the same port: the wait gets this start's token (only a server holding
    it counts) and a should_stop that is true once one of its own containers has stopped."""
    fake = FakeDocker()
    seen: dict = {}

    def health(url, timeout, token=None, should_stop=None):
        seen["token"], seen["stopped"] = token, should_stop()
        return True

    started(tmp_path, fake, health=health)
    assert seen == {"token": TOKEN, "stopped": False}


def test_a_kernel_that_exits_early_is_reported_with_its_log_and_removed(tmp_path):
    fake = FakeDocker()
    names = kd.docker_names(tmp_path, SUFFIX)

    def health(url, timeout, token=None, should_stop=None):
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


def test_a_forwarder_that_exits_is_reported_with_its_own_log(tmp_path):
    fake = FakeDocker()
    names = kd.docker_names(tmp_path, SUFFIX)

    def health(url, timeout, token=None, should_stop=None):
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
        docker_runtime(tmp_path, fake, health=lambda url, timeout, token=None, should_stop=None: False, start_timeout=7).start(2731)
    assert str(exc.value) == "Marimo did not answer on http://127.0.0.1:2731 within 7 s."
    assert fake.containers == {} and fake.networks == {}


def test_a_docker_error_mid_start_removes_what_was_created(tmp_path):
    fake = FakeDocker()
    fake.fail[("start",)] = (1, "Error response from daemon: ports are not available: exposing port TCP 127.0.0.1:2731")
    with pytest.raises(KernelRuntimeError) as exc:
        docker_runtime(tmp_path, fake).start(2731)
    assert str(exc.value) == "Docker could not start the forwarder container."
    assert exc.value.hint == "docker said: Error response from daemon: ports are not available: exposing port TCP 127.0.0.1:2731"
    assert fake.containers == {} and fake.networks == {}


def test_a_docker_command_that_hangs_is_an_error_and_cleaned_up(tmp_path):
    fake = FakeDocker()
    fake.timeout.add(("run",))
    with pytest.raises(KernelRuntimeError) as exc:
        docker_runtime(tmp_path, fake).start(2731)
    assert str(exc.value) == "Docker did not start the kernel container within 60 s."
    assert fake.networks == {}, "the network it had created is removed"


def test_ctrl_c_while_waiting_removes_the_containers(tmp_path):
    fake = FakeDocker()

    def interrupted(url, timeout, token=None, should_stop=None):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        docker_runtime(tmp_path, fake, health=interrupted).start(2731)
    assert fake.containers == {} and fake.networks == {}


def test_two_sessions_run_side_by_side_and_each_removes_only_its_own(tmp_path):
    fake = FakeDocker()
    first_rt = kd.DockerRuntime(
        make_config(tmp_path, kernel=KernelConfig(runtime="docker")), runner=fake, token_factory=lambda: TOKEN,
        health=healthy, user=lambda: None,
    )  # fmt: skip
    first = first_rt.start(2731)
    second_rt = kd.DockerRuntime(
        make_config(tmp_path, kernel=KernelConfig(runtime="docker")), runner=fake, token_factory=lambda: "another-token-0123456789",
        health=healthy, user=lambda: None,
    )  # fmt: skip
    second = second_rt.start(2732)
    assert set(first.containers).isdisjoint(second.containers) and first.network != second.network, "unique names"
    assert first_rt.owner_id != second_rt.owner_id and all(fake.container(i) for i in first.container_ids), "each its own owner"
    first.stop()
    assert all(fake.container(ident) for ident in second.container_ids) and fake.network(second.network_id)
    assert not first.owner.path.exists() and second.owner.path.exists(), "a stop releases only its own lock"
    second.stop()
    assert fake.containers == {} and fake.networks == {}


def test_docker_kernel_stop_never_raises(tmp_path):
    class Broken:
        def run(self, *args, **kwargs):
            raise OSError("docker vanished")

    kernel = kd.DockerKernel(
        server=MarimoServer(url="http://127.0.0.1:2731", token=TOKEN, runtime="docker"),
        runner=Broken(), containers=("hailer-kernel-x",), container_ids=("a" * 64,), network_id="b" * 64,
    )  # fmt: skip
    kernel.stop()
    assert "docker vanished" in kernel.stop_error and "uvx hailer kernel stop" in kernel.stop_error
    assert kernel.log_tail() == []


def test_removal_reports_what_it_did(tmp_path):
    fake = FakeDocker()
    _rt, running = started(tmp_path, fake)
    fake.containers.pop(running.container_ids[1])  # the forwarder went away by itself
    fake.fail[("network", "rm")] = (1, "Error response from daemon: error while removing network: network has active endpoints")
    outcome = running.remove()
    names = kd.docker_names(tmp_path, SUFFIX)
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
    names = kd.docker_names(tmp_path, SUFFIX)
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
    assert kd.docker_names(tmp_path, SUFFIX).kernel in outcome.gone and outcome.failed == []
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
    fake.fail[("version",)] = (1, said)
    with pytest.raises(KernelRuntimeError) as exc:
        docker_runtime(tmp_path, fake).engine_version()
    assert str(exc.value) == "Docker is not running." and "DOCKER_CONTEXT" in exc.value.hint and "docker context use default" in exc.value.hint



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
# uvx hailer kernel stop
# --------------------------------------------------------------------------- #


def test_kernel_stop_removes_every_labelled_object_of_the_workspace(tmp_path):
    """Whichever session started it, running or not: a live session's kernel goes too (on request),
    but its owner lock stays with it."""
    fake = FakeDocker()
    _rt, running = started(tmp_path, fake)
    stray = fake.add_container("hailer-kernel-0123456789", workspace=str(tmp_path))
    other = fake.add_container("hailer-kernel-y-1", workspace=str(tmp_path / "another"))
    names = kd.docker_names(tmp_path, SUFFIX)
    report = kd.stop_workspace_kernels(make_config(tmp_path), runner=fake)
    assert report.done == [f"Removed {names.kernel}, {names.forwarder}, {stray.name}, {names.network}."]
    assert report.failed == [] and list(fake.containers.values()) == [other] and fake.networks == {}
    assert running.owner.path.exists(), "kernel stop never touches a live session's files in .hailer"
    assert kd.stop_workspace_kernels(make_config(tmp_path), runner=fake).done == [], "nothing to stop"
    running.stop()


def test_kernel_stop_reports_what_it_could_not_remove(tmp_path):
    fake = FakeDocker()
    _rt, running = started(tmp_path, fake)
    fake.fail[("network", "rm")] = (1, "Error response from daemon: network has active endpoints")
    report = kd.stop_workspace_kernels(make_config(tmp_path), runner=fake)
    names = kd.docker_names(tmp_path, SUFFIX)
    assert report.done == [f"Removed {names.kernel}, {names.forwarder}."]
    assert report.failed and all(line.startswith(f"Could not remove {names.network}: docker said:") for line in report.failed)


def test_kernel_stop_with_docker_down_or_missing(tmp_path):
    down = FakeDocker(engine=None)
    report = kd.stop_workspace_kernels(make_config(tmp_path), runner=down)
    assert report.done == [
        "Docker is not running, so containers a docker kernel may have left were not checked "
        "(start Docker Desktop and run uvx hailer kernel stop again to remove them)."
    ]
    assert report.failed == []
    assert kd.stop_workspace_kernels(make_config(tmp_path), runner=FakeDocker(installed=False)).done == [], (
        "no Docker at all: nothing to say"
    )


def test_status_lists_this_workspaces_running_docker_kernels(tmp_path):
    fake = FakeDocker()
    config = make_config(tmp_path)  # a local config: docker kernels are listed anyway
    assert kd.workspace_kernels(config, fake) == []
    _rt, running = started(tmp_path, fake)
    orphan = fake.add_container("hailer-kernel-z-1", workspace=str(tmp_path), **owned(GONE))
    fake.add_container("hailer-kernel-w-1", workspace=str(tmp_path), state="exited", **owned(GONE))
    assert kd.workspace_kernels(config, fake) == [
        f"docker {running.containers[0]} (its Hailer session is running)",
        f"docker {orphan.name} (its Hailer session has ended; the next start removes it)",
    ], "running kernels only; forwarders and stopped containers are not listed"
    assert kd.workspace_kernels(config, FakeDocker(engine=None)) == ["docker: not checked (Docker is not running.)"]
    assert kd.workspace_kernels(config, FakeDocker(installed=False)) == [], "no Docker: nothing to say"


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
