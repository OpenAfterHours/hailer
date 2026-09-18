"""Tests for hailer.kernel: the path map, .hailer/kernel.json, the kernel's environment, the
local runtime (its process layer faked: nothing is spawned or killed) and the docker runtime
(a scripted docker CLI, :class:`FakeDocker`: nothing is run). No Docker needed."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest

from hailer import __version__
from hailer import kernel as k
from hailer.errors import ConfigError, KernelRuntimeError
from hailer.kernel_image import VERSION_LABEL
from hailer.models import HailerConfig, KernelConfig, MarimoServer, ModelConfig, ProviderConfig

from test_marimo_client import FakeMarimo  # the offline marimo stand-in (health, token-checked /api/sessions)

TOKEN = "unit-test-token-0123456789"


def make_config(tmp_path: Path, **overrides) -> HailerConfig:
    base = dict(
        workspace=tmp_path,
        notebook=tmp_path / "notebooks" / "analysis.py",
        notebooks_dir=tmp_path / "notebooks",
        data_dir=tmp_path / "data",
        context_dir=tmp_path / ".config" / "hailer" / "context",
        skills_dir=tmp_path / ".config" / "hailer" / "skills",
        prompts_dir=tmp_path / ".config" / "hailer" / "prompts",
    )
    base.update(overrides)
    return HailerConfig(**base)


@pytest.fixture
def served():
    """A fake marimo server that requires TOKEN (what a server Hailer started looks like)."""
    srv = FakeMarimo()
    srv.token = TOKEN
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()


# --------------------------------------------------------------------------- #
# PathMap
# --------------------------------------------------------------------------- #


def docker_map(tmp_path: Path) -> k.PathMap:
    return k.docker_paths(make_config(tmp_path))


def test_identity_map_uses_the_notebook_file_key(tmp_path):
    from hailer.marimo_client import notebook_file_key

    identity = k.PathMap()
    nb = tmp_path / "notebooks" / ".." / "notebooks" / "a b.py"
    assert identity.identity
    assert identity.to_kernel(nb) == notebook_file_key(nb)
    assert identity.to_host("anything/at all.py") == "anything/at all.py"


def test_docker_map_translates_both_ways_including_subfolders(tmp_path):
    paths = docker_map(tmp_path)
    assert not paths.identity
    assert paths.to_kernel(tmp_path / "notebooks" / "analysis.py") == "/work/notebooks/analysis.py"
    assert paths.to_kernel(tmp_path / "notebooks" / "team" / "q2 churn.py") == "/work/notebooks/team/q2 churn.py"
    assert paths.to_kernel(tmp_path / "notebooks") == "/work/notebooks"
    assert paths.to_kernel(tmp_path / "data" / "25-01 sales.parquet") == "/work/data/25-01 sales.parquet"
    assert paths.to_kernel(tmp_path / "notebooks" / "sub" / ".." / "x.py") == "/work/notebooks/x.py", "normalised like the file key"
    assert Path(paths.to_host("/work/notebooks/team/q2 churn.py")) == tmp_path / "notebooks" / "team" / "q2 churn.py"
    assert Path(paths.to_host("/work/data/25-01 sales.parquet")) == tmp_path / "data" / "25-01 sales.parquet"
    assert Path(paths.to_host("/work/notebooks")) == tmp_path / "notebooks"


def test_paths_outside_every_mount(tmp_path):
    paths = docker_map(tmp_path)
    with pytest.raises(ValueError):
        paths.to_kernel(tmp_path / "elsewhere" / "x.py")
    with pytest.raises(ValueError):
        paths.to_kernel(tmp_path / "notebooks-old" / "x.py"), "a sibling with the same prefix is not inside"
    with pytest.raises(ValueError):
        paths.to_kernel(tmp_path)
    assert paths.to_host("/tmp/scratch.py") is None
    assert paths.to_host("/work/hailer.toml") is None
    assert paths.to_host("/work/notebooksX/a.py") is None
    assert paths.to_host("/work/notebooks/../../etc/passwd") is None, "normalised before matching"
    assert paths.to_host("notebooks/analysis.py") is None, "relative kernel paths have no host path"
    assert paths.to_host("") is None


def test_longest_mount_wins(tmp_path):
    paths = k.PathMap(
        (
            (tmp_path / "work", PurePosixPath("/work/notebooks")),
            (tmp_path / "work" / "data", PurePosixPath("/work/data")),
        )
    )
    assert paths.to_kernel(tmp_path / "work" / "data" / "a.csv") == "/work/data/a.csv"
    assert paths.to_kernel(tmp_path / "work" / "nb.py") == "/work/notebooks/nb.py"
    assert Path(paths.to_host("/work/data/a.csv")) == tmp_path / "work" / "data" / "a.csv"


@pytest.mark.skipif(os.name != "nt", reason="Windows paths compare case-insensitively")
def test_windows_host_paths_ignore_case_and_separators(tmp_path):
    paths = docker_map(tmp_path)
    shouted = str(tmp_path / "NOTEBOOKS" / "Team" / "Report.py").upper().replace("REPORT.PY", "Report.py")
    assert paths.to_kernel(shouted) == "/work/notebooks/TEAM/Report.py", "the case below the mount is kept"
    assert paths.to_kernel(str(tmp_path / "notebooks" / "a.py").replace("\\", "/")) == "/work/notebooks/a.py"
    mixed = k.PathMap(((Path(str(tmp_path).upper()) / "Notebooks", PurePosixPath("/work/notebooks")),))
    assert mixed.to_kernel(tmp_path / "notebooks" / "a.py") == "/work/notebooks/a.py"


@pytest.mark.skipif(os.name == "nt", reason="POSIX paths are case-sensitive")
def test_posix_host_paths_are_case_sensitive(tmp_path):
    paths = docker_map(tmp_path)
    with pytest.raises(ValueError):
        paths.to_kernel(tmp_path / "NOTEBOOKS" / "a.py")
    assert paths.to_kernel(tmp_path / "notebooks" / "a.py") == "/work/notebooks/a.py"


def test_path_map_does_not_resolve_symlinks(tmp_path, monkeypatch):
    """Like marimo's file keys: a notebooks folder reached through a junction keeps its own path."""
    monkeypatch.setattr(Path, "resolve", lambda self, strict=False: Path(str(self).replace("linked", "real")))
    paths = k.PathMap(((tmp_path / "linked", PurePosixPath("/work/notebooks")),))
    assert paths.to_kernel(tmp_path / "linked" / "a.py") == "/work/notebooks/a.py"
    assert "linked" in paths.to_host("/work/notebooks/a.py")


# --------------------------------------------------------------------------- #
# .hailer/kernel.json
# --------------------------------------------------------------------------- #


def test_local_state_round_trips_without_docker_fields(tmp_path):
    state = k.KernelState(runtime="local", url="http://127.0.0.1:2718", port=2718, token=TOKEN, pid=4242)
    path = k.write_kernel_state(tmp_path, state)
    assert path == tmp_path / ".hailer" / "kernel.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert set(raw) == {"runtime", "url", "port", "token", "hailer_version", "started", "pid"}
    assert raw["hailer_version"] == __version__ and raw["started"]
    assert k.read_kernel_state(tmp_path) == state
    assert TOKEN not in repr(state), "the token stays out of reprs"
    server = state.server()
    assert server == MarimoServer(
        url="http://127.0.0.1:2718", server_id="127.0.0.1:2718", pid=4242, source="kernel", token=TOKEN, runtime="local"
    )


def test_docker_state_round_trips_with_its_mounts(tmp_path):
    mounts = ((str(tmp_path / "notebooks"), "/work/notebooks"), (str(tmp_path / "data"), "/work/data"))
    state = k.KernelState(
        runtime="docker",
        url="http://127.0.0.1:2731",
        port=2731,
        token=TOKEN,
        image="ghcr.io/openafterhours/hailer-kernel:0.2.5",
        containers=("hailer-kernel-abc", "hailer-fwd-abc"),
        network="hailer-net-abc",
        network_access=False,
        mounts=mounts,
    )
    k.write_kernel_state(tmp_path, state)
    raw = json.loads(k.kernel_state_path(tmp_path).read_text(encoding="utf-8"))
    assert "pid" not in raw and raw["containers"] == ["hailer-kernel-abc", "hailer-fwd-abc"]
    assert raw["mounts"] == [list(pair) for pair in mounts] and raw["network_access"] is False
    loaded = k.read_kernel_state(tmp_path)
    assert loaded == state
    server = loaded.server()
    assert server.runtime == "docker" and server.paths is not None
    assert server.paths.to_kernel(tmp_path / "data" / "a.csv") == "/work/data/a.csv"


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        "[]",
        '{"runtime": "local", "url": "http://127.0.0.1:2718", "port": 2718}',  # no token
        '{"runtime": "local", "url": "http://127.0.0.1:2718", "port": "2718", "token": "t"}',
        '{"runtime": "local", "url": "ftp://x", "port": 1, "token": "t"}',
        '{"runtime": "docker", "url": "http://127.0.0.1:1", "port": 1, "token": "t", "mounts": [["only one"]]}',
    ],
)
def test_malformed_state_is_ignored(tmp_path, content):
    path = k.kernel_state_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")
    assert k.read_kernel_state(tmp_path) is None
    assert k.delete_kernel_state(tmp_path, token="whatever"), "a malformed file belongs to nobody"


def test_delete_state_only_removes_the_server_it_names(tmp_path):
    assert not k.delete_kernel_state(tmp_path), "nothing to delete"
    k.write_kernel_state(tmp_path, k.KernelState(runtime="local", url="http://127.0.0.1:2718", port=2718, token="newer"))
    assert not k.delete_kernel_state(tmp_path, token="older"), "a newer start replaced it"
    assert k.read_kernel_state(tmp_path) is not None
    assert k.delete_kernel_state(tmp_path, token="newer")
    assert k.read_kernel_state(tmp_path) is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_state_file_is_private(tmp_path):
    path = k.write_kernel_state(tmp_path, k.KernelState(runtime="local", url="http://127.0.0.1:2718", port=2718, token=TOKEN))
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


# --------------------------------------------------------------------------- #
# The kernel's environment, descriptions, attaching
# --------------------------------------------------------------------------- #


def test_kernel_environment_drops_every_secret(tmp_path):
    internal = ProviderConfig(
        id="internal",
        base_url="https://llm.example.internal/v1",
        env_key="CORP_LLM_CREDENTIAL",  # no secret-looking suffix: removed because a provider names it
        env_http_headers={"X-Client-Id": "CORP_CLIENT_ID"},
    )
    config = make_config(tmp_path, providers={"internal": internal}, model=ModelConfig(provider="internal"))
    environ = {
        "PATH": "/usr/bin",
        "SYSTEMROOT": "C:\\Windows",
        "HAILER_LOG_LEVEL": "DEBUG",
        "OPENAI_API_KEY": "sk-1",
        "CORP_LLM_CREDENTIAL": "c-1",
        "CORP_CLIENT_ID": "id-1",
        "HAILER_MARIMO_TOKEN": "m-1",
        "GITHUB_TOKEN": "g-1",
        "DB_PASSWORD": "p-1",
        "AWS_SECRET": "s-1",
        "lower_case_api_key": "l-1",
    }
    env = k.kernel_environment(config, environ)
    assert env == {"PATH": "/usr/bin", "SYSTEMROOT": "C:\\Windows", "HAILER_LOG_LEVEL": "DEBUG"}
    assert environ["OPENAI_API_KEY"] == "sk-1", "the caller's mapping is not changed"


@pytest.mark.skipif(os.name != "nt", reason="environment names are case-insensitive on Windows only")
def test_kernel_environment_names_ignore_case_on_windows(tmp_path):
    internal = ProviderConfig(id="internal", base_url="https://x/v1", env_key="Corp_Llm_Credential")
    config = make_config(tmp_path, providers={"internal": internal})
    assert k.kernel_environment(config, {"CORP_LLM_CREDENTIAL": "c", "Path": "p"}) == {"Path": "p"}


def test_describe_runtime():
    assert k.describe_runtime(KernelConfig()) == "local (runs as you; not isolated)"
    assert k.describe_runtime(KernelConfig(runtime="docker"), version="0.2.5") == "docker (hailer-kernel 0.2.5; no network; data read-only)"
    assert k.describe_runtime(KernelConfig(runtime="docker", network=True), version="0.2.5") == (
        "docker (hailer-kernel 0.2.5; network on; data read-only)"
    )
    assert f"hailer-kernel {__version__};" in k.describe_runtime(KernelConfig(runtime="docker"))


def test_effective_image():
    assert KernelConfig().effective_image == f"ghcr.io/openafterhours/hailer-kernel:{__version__}"
    assert KernelConfig(image="registry.example/hailer-kernel:dev").effective_image == "registry.example/hailer-kernel:dev"


def test_attach_runtime_turns_a_local_session_into_docker_only(tmp_path):
    config = make_config(tmp_path)
    local = MarimoServer(url="http://127.0.0.1:2718", runtime="local")
    docker = MarimoServer(url="http://127.0.0.1:2731", runtime="docker", source="kernel")
    assert k.attach_runtime(config, None) is config
    assert k.attach_runtime(config, local) is config
    attached = k.attach_runtime(config, docker)
    assert attached.kernel.runtime == "docker" and attached.kernel.network is False
    k.write_kernel_state(tmp_path, k.KernelState(runtime="docker", url=docker.url, port=2731, token=TOKEN, network_access=True))
    assert k.attach_runtime(config, docker).kernel.network is True, "the network setting it was started with"
    docker_config = replace(config, kernel=KernelConfig(runtime="docker"))
    assert k.attach_runtime(docker_config, docker) is docker_config
    assert k.attach_runtime(docker_config, local) is docker_config, "never downgraded to local"


def test_runtime_for_never_falls_back_to_local(tmp_path):
    assert isinstance(k.runtime_for(make_config(tmp_path)), k.LocalRuntime)
    fake = FakeDocker(installed=False)
    docker = k.runtime_for(make_config(tmp_path, kernel=KernelConfig(runtime="docker")), runner=fake)
    assert isinstance(docker, k.DockerRuntime) and docker.runner is fake
    with pytest.raises(KernelRuntimeError) as exc:
        docker.start(2731)
    assert str(exc.value) == "Docker is not installed." and 'runtime = "local"' in exc.value.hint
    with pytest.raises(ConfigError):
        k.runtime_for(make_config(tmp_path, kernel=KernelConfig(runtime="podman")))


# --------------------------------------------------------------------------- #
# LocalRuntime
# --------------------------------------------------------------------------- #


class FakeProc:
    def __init__(self, exit_code=None, pid=4242):
        self.pid = pid
        self.returncode = exit_code
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


class Procs:
    """Records what the runtime asks of the operating system."""

    def __init__(self, *, healthy=True, proc=None):
        self.healthy = healthy
        self.proc = proc or FakeProc()
        self.calls: list[tuple] = []
        self.killed: list[int] = []
        self.removed: list[str] = []

    def spawn(self, cmd, cwd, log_path, *, env=None, stdin_text=None):
        self.calls.append(("spawn", cmd, cwd, log_path, env, stdin_text))
        return self.proc

    def attach(self, cmd, cwd, *, env=None, stdin_text=None):
        self.calls.append(("attach", cmd, cwd, None, env, stdin_text))
        return self.proc

    def wait_for_health(self, url, timeout, should_stop=None):
        self.calls.append(("health", url, timeout))
        if callable(self.healthy):
            return self.healthy()
        return self.healthy

    def local_processes(self) -> k.LocalProcesses:
        return k.LocalProcesses(
            spawn=self.spawn,
            attach=self.attach,
            kill_tree=lambda pid: None,
            kill_pid=self.killed.append,
            wait_for_health=self.wait_for_health,
            remove_registry_entry=lambda url: (self.removed.append(url) or True),
        )


def runtime(tmp_path, procs: Procs, **kw) -> k.LocalRuntime:
    return k.LocalRuntime(
        make_config(tmp_path),
        procs=procs.local_processes(),
        environ=kw.pop("environ", {"PATH": "p", "OPENAI_API_KEY": "sk-secret"}),
        token_factory=lambda: TOKEN,
        **kw,
    )


def test_local_runtime_describes_itself(tmp_path):
    rt = runtime(tmp_path, Procs())
    assert rt.name == "local" and rt.paths.identity
    assert rt.describe() == "local (runs as you; not isolated)"
    assert rt.check() == [k.Check("kernel", True, "local (runs as you; not isolated)", fatal=False)]
    assert "ctx.packages.add()" in rt.prompt_notes() and rt.prompt_notes() == k.LOCAL_PROMPT_NOTES
    rt.prepare(say=pytest.fail)  # nothing to download


def test_start_passes_the_token_on_stdin_scrubs_the_environment_and_records_the_server(tmp_path):
    procs = Procs()
    running = runtime(tmp_path, procs).start(2718)
    kind, cmd, cwd, log_path, env, stdin_text = procs.calls[0]
    assert kind == "spawn" and cwd == tmp_path and log_path == tmp_path / ".hailer" / "marimo.log"
    assert cmd == [sys.executable, "-m", "marimo", "edit", "notebooks", "--token-password-file", "-", "--headless", "--port", "2718", "--skip-update-check"]
    assert stdin_text == TOKEN and TOKEN not in " ".join(cmd)
    assert env == {"PATH": "p"}, "no API key in the kernel's environment"
    assert procs.calls[1] == ("health", "http://127.0.0.1:2718", k.START_TIMEOUT_SEC)
    assert running.server == MarimoServer(
        url="http://127.0.0.1:2718", server_id="127.0.0.1:2718", pid=4242, source="kernel", token=TOKEN, runtime="local"
    )
    assert running.log_hint == str(tmp_path / ".hailer" / "marimo.log") and "4242" in running.stop_hint
    state = k.read_kernel_state(tmp_path)
    assert (state.runtime, state.url, state.port, state.token, state.pid) == ("local", "http://127.0.0.1:2718", 2718, TOKEN, 4242)

    running.stop()
    assert procs.proc.terminated and procs.removed == ["http://127.0.0.1:2718"]
    assert k.read_kernel_state(tmp_path) is None
    running.stop()  # idempotent


def test_foreground_start_attaches_to_this_terminal(tmp_path):
    procs = Procs()
    running = runtime(tmp_path, procs).start(2720, foreground=True)
    kind, cmd, _cwd, _log, env, stdin_text = procs.calls[0]
    assert kind == "attach" and "--headless" in cmd and stdin_text == TOKEN and "OPENAI_API_KEY" not in env
    assert running.log_hint == "this terminal" and running.log_tail() == []
    assert k.read_kernel_state(tmp_path).url == "http://127.0.0.1:2720"
    procs.proc.returncode = 0  # marimo exits (Ctrl+C in its terminal)
    assert running.wait() == 0
    running.stop()
    assert k.read_kernel_state(tmp_path) is None


def test_start_reports_an_early_exit_with_the_log_tail_and_cleans_up(tmp_path):
    procs = Procs(healthy=False, proc=FakeProc(exit_code=3))
    log = tmp_path / ".hailer" / "marimo.log"
    log.parent.mkdir(parents=True)
    log.write_text("\n".join(f"line {i}" for i in range(20)), encoding="utf-8")
    with pytest.raises(KernelRuntimeError) as exc:
        runtime(tmp_path, procs).start(2718)
    assert str(exc.value) == "Marimo exited early (code 3)."
    assert exc.value.hint.startswith(f"Log: {log}") and "    line 19" in exc.value.hint and "line 4\n" not in exc.value.hint
    assert k.read_kernel_state(tmp_path) is None, "nothing recorded for a server that never came up"


def test_start_that_times_out_stops_the_server(tmp_path):
    procs = Procs(healthy=False)
    with pytest.raises(KernelRuntimeError) as exc:
        runtime(tmp_path, procs, start_timeout=7).start(2718)
    assert str(exc.value) == "Marimo did not answer on http://127.0.0.1:2718 within 7 s."
    assert procs.proc.terminated


def test_ctrl_c_while_starting_stops_the_server(tmp_path):
    def interrupted():
        raise KeyboardInterrupt

    procs = Procs(healthy=interrupted)
    with pytest.raises(KeyboardInterrupt):
        runtime(tmp_path, procs).start(2718)
    assert procs.proc.terminated


def test_start_that_cannot_spawn_is_a_runtime_error(tmp_path):
    procs = Procs()

    def broken(*args, **kwargs):
        raise FileNotFoundError(2, "No such file", "python")

    procs.spawn = broken
    with pytest.raises(KernelRuntimeError) as exc:
        runtime(tmp_path, procs).start(2718)
    assert str(exc.value).startswith("Could not start marimo:") and exc.value.hint


def test_start_removes_a_stale_record(tmp_path):
    k.write_kernel_state(tmp_path, k.KernelState(runtime="local", url="http://127.0.0.1:9", port=9, token="old"))
    procs = Procs(healthy=False)
    with pytest.raises(KernelRuntimeError):
        runtime(tmp_path, procs).start(2718)
    assert k.read_kernel_state(tmp_path) is None


def test_find_running_needs_a_server_that_answers_with_its_token(tmp_path, served):
    procs = Procs()
    rt = runtime(tmp_path, procs)
    assert rt.find_running() is None, "no record"
    port = served.server_address[1]
    k.write_kernel_state(tmp_path, k.KernelState(runtime="local", url=served.url, port=port, token="not-its-token", pid=77))
    assert rt.find_running() is None
    k.write_kernel_state(tmp_path, k.KernelState(runtime="docker", url=served.url, port=port, token=TOKEN))
    assert rt.find_running() is None, "a docker kernel is not the local runtime's"
    k.write_kernel_state(tmp_path, k.KernelState(runtime="local", url=served.url, port=port, token=TOKEN, pid=77))
    running = rt.find_running()
    assert running is not None and running.server.token == TOKEN and running.server.pid == 77
    assert running.alive()
    running.stop()
    assert procs.killed == [77] and procs.removed == [served.url], "an earlier run's server is stopped by pid"
    assert k.read_kernel_state(tmp_path) is None


# --------------------------------------------------------------------------- #
# Process helpers
# --------------------------------------------------------------------------- #


def test_spawn_marimo_uses_background_flags_log_and_stdin(tmp_path, monkeypatch):
    calls = []

    class Stdin:
        def __init__(self):
            self.written = b""
            self.closed = False

        def write(self, data):
            self.written += data

        def close(self):
            self.closed = True

    class Popen:
        def __init__(self, cmd, **kwargs):
            calls.append((cmd, kwargs))
            self.pid = 1
            self.stdin = Stdin()

    monkeypatch.setattr(k.subprocess, "Popen", Popen)
    log = tmp_path / ".hailer" / "marimo.log"
    k.spawn_marimo(["python", "-m", "marimo"], tmp_path, log)
    cmd, kwargs = calls[0]
    assert cmd == ["python", "-m", "marimo"] and kwargs["cwd"] == str(tmp_path)
    assert log.parent.is_dir(), "log directory created"
    assert kwargs["stderr"] == subprocess.STDOUT and kwargs["stdin"] == subprocess.DEVNULL and kwargs["env"] is None
    expected = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    assert kwargs["creationflags"] == expected, "own process group on Windows so Ctrl+C in the chat is not delivered to marimo"

    proc = k.spawn_marimo(["python"], tmp_path, log, env={"PATH": "p"}, stdin_text=TOKEN)
    _cmd, kwargs = calls[1]
    assert kwargs["stdin"] == subprocess.PIPE and kwargs["env"] == {"PATH": "p"}
    assert proc.stdin.written == TOKEN.encode() and proc.stdin.closed, "the token, then EOF"


def test_stop_process_terminates_then_kills():
    killed_trees: list[int] = []
    proc = FakeProc()
    k.stop_process(proc, kill_tree=killed_trees.append)
    assert proc.terminated
    assert killed_trees == ([4242] if os.name == "nt" else []), "Windows: the kernel children go too"

    class Stubborn(FakeProc):
        killed = False

        def terminate(self):
            self.terminated = True  # stays alive

        def kill(self):
            self.killed = True
            self.returncode = -9

        def wait(self, timeout=None):
            if self.returncode is None:
                raise subprocess.TimeoutExpired("marimo", timeout)
            return self.returncode

    stubborn = Stubborn()
    k.stop_process(stubborn, timeout=0.01, kill_tree=lambda pid: None)
    assert stubborn.terminated and stubborn.killed
    k.stop_process(FakeProc(exit_code=0), kill_tree=killed_trees.append)  # already gone: nothing to do


# --------------------------------------------------------------------------- #
# DockerRuntime: a scripted docker CLI, nothing is run
# --------------------------------------------------------------------------- #

IMAGE = f"ghcr.io/openafterhours/hailer-kernel:{__version__}"


class FakeDocker:
    """A scripted ``docker`` CLI (the :class:`~hailer.kernel.DockerRunner` protocol). Records every
    argv and answers like Docker 29: ``version``, ``image inspect`` (the version label), ``ps`` and
    ``network ls`` (leftovers), ``inspect`` (running state, labels, exit code) and ``logs``; anything
    else succeeds silently. ``fail`` maps an argv prefix to (exit code, stderr); ``timeout`` holds
    prefixes that time out."""

    def __init__(self, *, installed=True, engine="29.4.3 linux", image_version=__version__, pulled_version=__version__, pull_code=0, build_code=0):
        self.installed = installed
        self.engine = engine  # None: the engine is down
        self.image_version = image_version  # None: the image is not on this machine
        self.pulled_version = pulled_version
        self.pull_code = pull_code
        self.build_code = build_code
        self.running: dict[str, bool] = {}
        self.exit_codes: dict[str, int] = {}
        self.labels: dict[str, dict] = {}
        self.leftovers: list[str] = []
        self.leftover_networks: list[str] = []
        self.logs = ["marimo is starting", f"URL: http://0.0.0.0:2718?access_token={TOKEN}"]
        self.fail: dict[tuple[str, ...], tuple[int, str]] = {}
        self.timeout: set[tuple[str, ...]] = set()
        self.stream_hook = None  # called with the argv inside stream(), e.g. to raise KeyboardInterrupt
        self.calls: list[list[str]] = []
        self.streams: list[list[str]] = []

    @staticmethod
    def _done(args, code=0, out="", err=""):
        return subprocess.CompletedProcess(["docker", *args], code, out, err)

    @staticmethod
    def _labels_of(args) -> dict:
        return dict(args[i + 1].split("=", 1) for i, a in enumerate(args) if a == "--label")

    def run(self, args, *, input=None, timeout=None, check=True, capture=True):
        args = list(args)
        self.calls.append(args)
        if not self.installed:
            raise k.docker_not_installed()
        for prefix in self.timeout:
            if tuple(args[: len(prefix)]) == prefix:
                raise subprocess.TimeoutExpired(["docker", *args], timeout)
        for prefix, (code, err) in self.fail.items():
            if tuple(args[: len(prefix)]) == prefix:
                return self._done(args, code, "", err)
        head = args[0]
        if head == "version":
            if self.engine is None:
                return self._done(args, 1, "", "error during connect: open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.")
            return self._done(args, 0, self.engine + "\n")
        if args[:2] == ["image", "inspect"]:
            if self.image_version is None:
                return self._done(args, 1, "", f"Error response from daemon: No such image: {args[-1]}")
            return self._done(args, 0, json.dumps({VERSION_LABEL: self.image_version}) + "\n")
        if head == "ps":
            return self._done(args, 0, "".join(f"{name}\n" for name in self.leftovers))
        if args[:2] == ["network", "ls"]:
            return self._done(args, 0, "".join(f"{name}\n" for name in self.leftover_networks))
        if head in ("run", "create"):
            name = args[args.index("--name") + 1]
            self.running[name] = head == "run"
            self.labels[name] = self._labels_of(args)
        elif head == "start":
            self.running[args[1]] = True
        elif head == "rm":
            for name in args[2:]:
                self.running.pop(name, None)
        elif head == "inspect":
            fmt, names = args[2], args[3:]
            if fmt == "{{json .Config.Labels}}":
                if names[0] not in self.labels:
                    return self._done(args, 1, "", f"error: no such object: {names[0]}")
                return self._done(args, 0, json.dumps(self.labels[names[0]]) + "\n")
            if fmt == "{{.State.ExitCode}}":
                return self._done(args, 0, f"{self.exit_codes.get(names[0], 0)}\n")
            known = [name for name in names if name in self.running]
            out = "".join(f"/{name} {'true' if self.running[name] else 'false'} {self.exit_codes.get(name, 0)}\n" for name in known)
            return self._done(args, 0 if len(known) == len(names) else 1, out)
        elif head == "logs":
            return self._done(args, 0, "\n".join(self.logs) + "\n")
        return self._done(args)

    def stream(self, args):
        args = list(args)
        self.streams.append(args)
        if not self.installed:
            raise k.docker_not_installed()
        if self.stream_hook is not None:
            self.stream_hook(args)
        if args[0] == "pull":
            if self.pull_code == 0:
                self.image_version = self.pulled_version
            return self.pull_code
        if args[0] == "build":
            return self.build_code
        return 0

    def commands(self, *head: str) -> list[list[str]]:
        return [c for c in self.calls if c[: len(head)] == list(head)]


def healthy(url, timeout, should_stop=None):
    return True


def docker_runtime(tmp_path: Path, fake: FakeDocker, *, kernel: dict | None = None, **kw) -> k.DockerRuntime:
    config = make_config(tmp_path, kernel=KernelConfig(runtime="docker", **(kernel or {})))
    kw.setdefault("health", healthy)
    kw.setdefault("user", lambda: None)
    return k.DockerRuntime(config, runner=fake, token_factory=lambda: TOKEN, **kw)


def labels(tmp_path: Path, role: str) -> list[str]:
    return [
        "--label", f"org.openafterhours.hailer.workspace={tmp_path}",
        "--label", f"org.openafterhours.hailer.version={__version__}",
        "--label", f"org.openafterhours.hailer.role={role}",
    ]  # fmt: skip


def follows(seq: list[str], part: list[str]) -> bool:
    return any(seq[i : i + len(part)] == part for i in range(len(seq) - len(part) + 1))


def test_workspace_id_and_names(tmp_path):
    ident = k.workspace_id(tmp_path)
    assert re.fullmatch(r"[0-9a-f]{10}", ident)
    assert ident == hashlib.sha256(os.path.normcase(str(tmp_path.resolve())).encode("utf-8")).hexdigest()[:10]
    assert k.docker_names(tmp_path) == k.DockerNames(f"hailer-kernel-{ident}", f"hailer-fwd-{ident}", f"hailer-net-{ident}")
    assert k.workspace_id(tmp_path / "other") != ident, "two workspaces run side by side"
    if os.name == "nt":
        assert k.workspace_id(Path(str(tmp_path).upper())) == ident, "Windows paths compare case-insensitively"


def test_mount_arg_quotes_like_csv():
    assert k.mount_arg("/srv/notebooks", k.KERNEL_NOTEBOOKS_DIR) == "type=bind,source=/srv/notebooks,target=/work/notebooks"
    assert k.mount_arg("/srv/data", k.KERNEL_DATA_DIR, readonly=True) == "type=bind,source=/srv/data,target=/work/data,readonly"
    assert k.mount_arg(r"C:\Users\Ann Lee\data", "/work/data") == r"type=bind,source=C:\Users\Ann Lee\data,target=/work/data"
    assert k.mount_arg("/srv/q1,q2", "/work/data") == 'type=bind,"source=/srv/q1,q2",target=/work/data'
    assert k.mount_arg('/srv/"best" data', "/work/data") == 'type=bind,"source=/srv/""best"" data",target=/work/data'


def test_linux_host_user(monkeypatch):
    monkeypatch.setattr(k.os, "getuid", lambda: 1234, raising=False)
    monkeypatch.setattr(k.os, "getgid", lambda: 5678, raising=False)
    monkeypatch.setattr(k.sys, "platform", "linux")
    assert k.linux_host_user() == (1234, 5678)
    monkeypatch.setattr(k.os, "getuid", lambda: 0, raising=False)
    assert k.linux_host_user() is None, "never run the kernel as root"
    for platform in ("win32", "darwin"):
        monkeypatch.setattr(k.sys, "platform", platform)
        assert k.linux_host_user() is None, "Docker Desktop maps file ownership itself"


def test_prompt_notes_per_runtime(tmp_path):
    assert k.runtime_prompt_notes(KernelConfig()) == k.LOCAL_PROMPT_NOTES
    offline = k.runtime_prompt_notes(KernelConfig(runtime="docker"))
    assert "No internet access" in offline and "ctx.packages.add is unavailable" in offline and "/tmp" in offline
    assert "ctx.packages.add()" not in offline, "never told to install packages"
    online = k.runtime_prompt_notes(KernelConfig(runtime="docker", network=True))
    assert "can reach the internet" in online and "No internet" not in online
    rt = k.DockerRuntime(make_config(tmp_path), runner=FakeDocker())  # built for `kernel stop` from a local config
    assert rt.name == "docker" and rt.prompt_notes() == offline
    assert rt.describe() == f"docker (hailer-kernel {__version__}; no network; data read-only)"
    assert rt.paths.to_kernel(tmp_path / "data" / "a.csv") == "/work/data/a.csv"


def test_check_rows_when_docker_is_ready(tmp_path):
    rows = docker_runtime(tmp_path, FakeDocker()).check()
    assert [(r.name, r.ok) for r in rows] == [("kernel", True), ("docker", True), ("image", True), ("data", True)]
    assert rows[1].summary == "Docker 29.4.3 (Linux engine)"
    assert rows[2].summary == f"{IMAGE} (Hailer {__version__})"


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
    fake = FakeDocker(image_version=None)
    rt = docker_runtime(tmp_path, fake)
    image_row = next(r for r in rt.check() if r.name == "image")
    assert not image_row.ok and not image_row.fatal and "uvx hailer kernel pull" in image_row.hint
    said: list[str] = []
    rt.prepare(say=said.append)
    assert fake.streams == [["pull", IMAGE]] and said == [f"Downloading the kernel image {IMAGE} (first use; this can take a few minutes) ..."]
    assert not fake.commands("run")
    rt.start(2731)
    assert fake.streams == [["pull", IMAGE]], "no second pull: the image is here now"


def test_a_failed_pull_names_the_build_command_and_the_image_setting(tmp_path):
    fake = FakeDocker(image_version=None, pull_code=1)
    with pytest.raises(KernelRuntimeError) as exc:
        docker_runtime(tmp_path, fake).start(2731)
    assert str(exc.value) == f"Could not download the kernel image {IMAGE}."
    assert "uvx hailer kernel build" in exc.value.hint and "[kernel].image" in exc.value.hint
    assert not fake.commands("run")


@pytest.mark.parametrize(("present", "pulled"), [("0.1.0", __version__), (None, "0.1.0"), ("", __version__)])
def test_an_image_for_another_hailer_version_is_refused(tmp_path, present, pulled):
    fake = FakeDocker(image_version=present, pulled_version=pulled)
    rt = docker_runtime(tmp_path, fake)
    with pytest.raises(KernelRuntimeError) as exc:
        rt.start(2731)
    shown = present if present is not None else pulled
    assert f"is for Hailer {shown or '(no version label)'}; this is Hailer {__version__}." in str(exc.value)
    assert "uvx hailer kernel pull" in exc.value.hint and "uvx hailer kernel build" in exc.value.hint
    assert not fake.commands("run")
    if present is not None:
        image_row = next(r for r in rt.check() if r.name == "image")
        assert not image_row.ok and image_row.fatal


def test_start_runs_the_hardened_kernel_offline_behind_a_forwarder(tmp_path):
    fake = FakeDocker()
    waits = []

    def health(url, timeout, should_stop=None):
        waits.append((url, timeout, should_stop()))
        return True

    names = k.docker_names(tmp_path)
    running = docker_runtime(tmp_path, fake, health=health).start(2731)

    assert [c[:2] for c in fake.calls if c[0] != "inspect"] == [
        ["version", "--format"], ["image", "inspect"], ["ps", "-a"], ["network", "ls"],
        ["network", "create"], ["run", "-d"], ["create", "--pull"], ["network", "connect"], ["start", names.forwarder],
    ]  # fmt: skip
    assert fake.commands("network", "create") == [["network", "create", "--internal", *labels(tmp_path, "network"), names.network]]
    assert fake.commands("run") == [[
        "run", "-d", "--pull", "never", "--name", names.kernel, "--network", names.network,
        "--init", "--read-only", "--tmpfs", "/tmp", "--tmpfs", "/home/analyst:uid=1000,gid=1000",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "256",
        "--memory", "4g", "--cpus", "2", "-w", "/work", *labels(tmp_path, "kernel"),
        "--mount", f"type=bind,source={tmp_path / 'notebooks'},target=/work/notebooks",
        "--mount", f"type=bind,source={tmp_path / 'data'},target=/work/data,readonly",
        IMAGE, "marimo", "edit", "notebooks", "--host", "0.0.0.0", "--port", "2718",
        "--headless", "--skip-update-check", "--token-password", TOKEN,
    ]]  # fmt: skip
    assert fake.commands("create") == [[
        "create", "--pull", "never", "--name", names.forwarder, "-p", "127.0.0.1:2731:2718",
        "--init", "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--pids-limit", "64", "--memory", "64m", *labels(tmp_path, "forwarder"),
        IMAGE, "python", "-m", "hailer._forward", names.kernel, "2718", "2718",
    ]]  # fmt: skip
    assert fake.commands("network", "connect") == [["network", "connect", names.network, names.forwarder]]

    run = fake.commands("run")[0]
    assert "-p" not in run and "--publish" not in run, "no port on the kernel: the forwarder publishes it"
    assert run.count("--mount") == 2 and not {"-v", "--volume", "--privileged", "--user"} & set(run), "nothing else is mounted"
    assert "docker.sock" not in " ".join(run) and f"source={tmp_path}," not in " ".join(run), "not the workspace, not the Docker socket"
    assert all(TOKEN not in " ".join(c) for c in fake.calls if c[0] != "run"), "the token only reaches marimo"
    assert waits == [("http://127.0.0.1:2731", k.START_TIMEOUT_SEC, False)], "should_stop: both containers run"
    assert (tmp_path / "data").is_dir() and (tmp_path / "notebooks").is_dir(), "created before Docker could create them"

    assert isinstance(running, k.DockerKernel)
    assert running.server == MarimoServer(
        url="http://127.0.0.1:2731", server_id="127.0.0.1:2731", source="kernel", token=TOKEN, runtime="docker",
        paths=k.docker_paths(make_config(tmp_path)),
    )  # fmt: skip
    assert running.log_hint == f"docker logs {names.kernel}" and "uvx hailer kernel stop" in running.stop_hint
    state = k.read_kernel_state(tmp_path)
    assert (state.runtime, state.url, state.port, state.token, state.image) == ("docker", "http://127.0.0.1:2731", 2731, TOKEN, IMAGE)
    assert state.containers == (names.kernel, names.forwarder) and state.network == names.network and state.network_access is False
    assert state.mounts == ((str(tmp_path / "notebooks"), "/work/notebooks"), (str(tmp_path / "data"), "/work/data"))

    running.stop()
    assert fake.calls[-2:] == [["rm", "-f", names.kernel, names.forwarder], ["network", "rm", names.network]]
    assert k.read_kernel_state(tmp_path) is None
    running.stop()  # idempotent


def test_start_on_a_linux_host_runs_the_kernel_as_the_user(tmp_path):
    fake = FakeDocker()
    docker_runtime(tmp_path, fake, user=lambda: (1234, 5678)).start(2731)
    run = fake.commands("run")[0]
    assert follows(run, ["--tmpfs", "/home/analyst:uid=1234,gid=5678", "--user", "1234:5678", "-e", "HOME=/home/analyst"])
    assert "--user" not in fake.commands("create")[0], "the forwarder touches no files"


def test_start_with_network_access_publishes_the_kernel_directly(tmp_path):
    fake = FakeDocker()
    names = k.docker_names(tmp_path)
    rt = docker_runtime(tmp_path, fake, kernel={"network": True, "memory": "8g", "cpus": 1.5})
    assert rt.describe() == f"docker (hailer-kernel {__version__}; network on; data read-only)"
    running = rt.start(2731)
    run = fake.commands("run")[0]
    assert follows(run, ["--name", names.kernel, "-p", "127.0.0.1:2731:2718", "--init"])
    assert "--network" not in run and follows(run, ["--memory", "8g", "--cpus", "1.5"])
    assert not fake.commands("create") and not fake.commands("network", "create") and not fake.commands("start")
    state = k.read_kernel_state(tmp_path)
    assert state.network_access is True and state.containers == (names.kernel,) and state.network is None
    running.stop()
    assert fake.calls[-1] == ["rm", "-f", names.kernel] and not fake.commands("network", "rm")


def test_start_refuses_to_orphan_a_live_local_server(tmp_path, served):
    k.write_kernel_state(tmp_path, k.KernelState(runtime="local", url=served.url, port=served.server_address[1], token=TOKEN, pid=77))
    fake = FakeDocker(image_version=None)
    with pytest.raises(KernelRuntimeError) as exc:
        docker_runtime(tmp_path, fake).start(2731)
    assert str(exc.value) == f"A local marimo server Hailer started for this workspace is still running at {served.url}."
    assert "uvx hailer kernel stop" in exc.value.hint
    assert not fake.streams, "refused before any download"
    assert not fake.commands("run") and not fake.commands("rm") and k.read_kernel_state(tmp_path).runtime == "local"


def test_start_refuses_to_start_over_a_live_docker_kernel(tmp_path, served):
    k.write_kernel_state(tmp_path, k.KernelState(runtime="docker", url=served.url, port=served.server_address[1], token=TOKEN))
    fake = FakeDocker()
    with pytest.raises(KernelRuntimeError) as exc:
        docker_runtime(tmp_path, fake).start(2731)
    assert "already running" in str(exc.value) and "uvx hailer kernel stop" in exc.value.hint
    assert not fake.commands("run") and not fake.commands("rm")


def test_start_drops_a_stale_record_and_removes_leftover_containers(tmp_path):
    names = k.docker_names(tmp_path)
    k.write_kernel_state(tmp_path, k.KernelState(runtime="docker", url="http://127.0.0.1:9", port=9, token="old"))
    fake = FakeDocker()
    fake.leftovers = [names.kernel, names.forwarder]
    fake.leftover_networks = [names.network]
    docker_runtime(tmp_path, fake).start(2731)
    label = f"label=org.openafterhours.hailer.workspace={tmp_path}"
    assert fake.commands("ps") == [["ps", "-a", "--filter", label, "--format", "{{.Names}}"]]
    assert fake.commands("network", "ls") == [["network", "ls", "--filter", label, "--format", "{{.Name}}"]]
    order = [c[:2] for c in fake.calls]
    assert order.index(["rm", "-f"]) < order.index(["network", "rm"]) < order.index(["network", "create"])
    assert fake.commands("rm") == [["rm", "-f", names.kernel, names.forwarder]]
    assert k.read_kernel_state(tmp_path).token == TOKEN


def test_a_kernel_that_exits_early_is_reported_with_its_log_and_removed(tmp_path):
    fake = FakeDocker()
    names = k.docker_names(tmp_path)

    def health(url, timeout, should_stop=None):
        fake.running[names.kernel] = False
        fake.exit_codes[names.kernel] = 3
        assert should_stop() is True
        return False

    with pytest.raises(KernelRuntimeError) as exc:
        docker_runtime(tmp_path, fake, health=health).start(2731)
    assert str(exc.value) == "The kernel container exited early (code 3)."
    hint = exc.value.hint
    assert hint.startswith(f"Last lines of docker logs {names.kernel}:") and "    marimo is starting" in hint
    assert TOKEN not in hint and "access_token=<token>" in hint, "the token is masked in the log tail"
    assert ["logs", "--tail", "15", names.kernel] in fake.calls
    assert fake.calls[-2:] == [["rm", "-f", names.kernel, names.forwarder], ["network", "rm", names.network]]
    assert k.read_kernel_state(tmp_path) is None


def test_a_forwarder_that_exits_is_reported_with_its_own_log(tmp_path):
    fake = FakeDocker()
    names = k.docker_names(tmp_path)

    def health(url, timeout, should_stop=None):
        fake.running[names.forwarder] = False
        fake.exit_codes[names.forwarder] = 1
        return False

    with pytest.raises(KernelRuntimeError) as exc:
        docker_runtime(tmp_path, fake, health=health).start(2731)
    assert str(exc.value) == "The forwarder container exited early (code 1)."
    assert ["logs", "--tail", "15", names.forwarder] in fake.calls


def test_a_kernel_that_never_answers_times_out_and_is_removed(tmp_path):
    fake = FakeDocker()
    with pytest.raises(KernelRuntimeError) as exc:
        docker_runtime(tmp_path, fake, health=lambda url, timeout, should_stop=None: False, start_timeout=7).start(2731)
    assert str(exc.value) == "Marimo did not answer on http://127.0.0.1:2731 within 7 s."
    assert fake.commands("rm") and fake.commands("network", "rm")


def test_a_docker_error_mid_start_removes_what_was_created(tmp_path):
    fake = FakeDocker()
    names = k.docker_names(tmp_path)
    fake.fail[("start",)] = (1, "Error response from daemon: ports are not available: exposing port TCP 127.0.0.1:2731")
    with pytest.raises(KernelRuntimeError) as exc:
        docker_runtime(tmp_path, fake).start(2731)
    assert str(exc.value) == "Docker could not start the forwarder container."
    assert exc.value.hint == "docker said: Error response from daemon: ports are not available: exposing port TCP 127.0.0.1:2731"
    assert fake.calls[-2:] == [["rm", "-f", names.kernel, names.forwarder], ["network", "rm", names.network]]
    assert k.read_kernel_state(tmp_path) is None


def test_a_docker_command_that_hangs_is_an_error_and_cleaned_up(tmp_path):
    fake = FakeDocker()
    fake.timeout.add(("run",))
    with pytest.raises(KernelRuntimeError) as exc:
        docker_runtime(tmp_path, fake).start(2731)
    assert str(exc.value) == "Docker did not start the kernel container within 60 s."
    assert fake.commands("rm") and fake.commands("network", "rm")


def test_ctrl_c_while_waiting_removes_the_containers(tmp_path):
    fake = FakeDocker()

    def interrupted(url, timeout, should_stop=None):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        docker_runtime(tmp_path, fake, health=interrupted).start(2731)
    assert fake.commands("rm") and fake.commands("network", "rm") and k.read_kernel_state(tmp_path) is None


def test_find_running_needs_the_token_and_this_workspaces_label(tmp_path, served):
    fake = FakeDocker()
    names = k.docker_names(tmp_path)
    rt = docker_runtime(tmp_path, fake)
    assert rt.find_running() is None, "no record"
    state = k.KernelState(
        runtime="docker", url=served.url, port=served.server_address[1], token=TOKEN,
        containers=(names.kernel, names.forwarder), network=names.network,
        mounts=((str(tmp_path / "notebooks"), "/work/notebooks"), (str(tmp_path / "data"), "/work/data")),
    )  # fmt: skip
    k.write_kernel_state(tmp_path, state)
    assert rt.find_running() is None, "its container is gone"
    fake.labels[names.kernel] = {k.LABEL_WORKSPACE: str(tmp_path / "elsewhere")}
    assert rt.find_running() is None, "another workspace's container"
    fake.labels[names.kernel] = {k.LABEL_WORKSPACE: str(tmp_path)}
    running = rt.find_running()
    assert isinstance(running, k.DockerKernel) and running.server.token == TOKEN and running.server.runtime == "docker"
    assert running.containers == (names.kernel, names.forwarder) and running.network == names.network
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
    kernel = k.DockerKernel(
        server=MarimoServer(url="http://127.0.0.1:2731", token=TOKEN, runtime="docker"),
        workspace=tmp_path, runner=Broken(), containers=("hailer-kernel-x", "hailer-fwd-x"), network="hailer-net-x",
    )  # fmt: skip
    kernel.stop()
    assert k.read_kernel_state(tmp_path) is None
    assert kernel.log_tail() == []


def test_foreground_follows_the_kernel_log_until_it_stops_or_ctrl_c(tmp_path):
    fake = FakeDocker()
    names = k.docker_names(tmp_path)
    running = docker_runtime(tmp_path, fake).start(2731, foreground=True)
    assert fake.commands("run")[0][:2] == ["run", "-d"], "detached either way"
    assert running.wait() == 0
    assert fake.streams[-1] == ["logs", "-f", "--tail", "0", names.kernel], "new lines only: marimo's banner shows the in-container URL"
    fake.exit_codes[names.kernel] = 137
    assert running.wait() == 137, "the kernel's exit code when it stops by itself"

    def ctrl_c(args):
        raise KeyboardInterrupt

    fake.stream_hook = ctrl_c
    assert running.wait() == 0, "Ctrl+C ends the wait; the caller stops and removes the containers"


def test_kernel_stop_removes_a_docker_kernel_and_strays(tmp_path):
    names = k.docker_names(tmp_path)
    k.write_kernel_state(
        tmp_path,
        k.KernelState(runtime="docker", url="http://127.0.0.1:2731", port=2731, token=TOKEN, containers=(names.kernel, names.forwarder), network=names.network),
    )
    fake = FakeDocker()
    fake.leftovers = ["hailer-kernel-0123456789"]
    done = k.stop_workspace_kernels(make_config(tmp_path), runner=fake)
    assert done == [
        f"Stopped the docker kernel at http://127.0.0.1:2731 (removed {names.kernel}, {names.forwarder}, {names.network}).",
        "Removed hailer-kernel-0123456789.",
    ]
    assert ["rm", "-f", names.kernel, names.forwarder] in fake.calls and ["network", "rm", names.network] in fake.calls
    assert ["rm", "-f", "hailer-kernel-0123456789"] in fake.calls
    assert k.read_kernel_state(tmp_path) is None
    assert k.stop_workspace_kernels(make_config(tmp_path), runner=FakeDocker()) == [], "nothing to stop"


def test_kernel_stop_stops_a_live_local_server_and_never_kills_a_stale_pid(tmp_path, served):
    procs = Procs()
    port = served.server_address[1]
    k.write_kernel_state(tmp_path, k.KernelState(runtime="local", url=served.url, port=port, token=TOKEN, pid=77))
    done = k.stop_workspace_kernels(make_config(tmp_path), runner=FakeDocker(installed=False), procs=procs.local_processes())
    assert done == [f"Stopped the local marimo server at {served.url} (process 77)."]
    assert procs.killed == [77] and k.read_kernel_state(tmp_path) is None
    k.write_kernel_state(tmp_path, k.KernelState(runtime="local", url="http://127.0.0.1:9", port=9, token=TOKEN, pid=78))
    done = k.stop_workspace_kernels(make_config(tmp_path), runner=FakeDocker(), procs=procs.local_processes())
    assert done == ["Removed the record of a marimo server that no longer answers (http://127.0.0.1:9)."]
    assert procs.killed == [77], "a pid from a record that does not answer may belong to anything now"


def test_kernel_stop_without_docker(tmp_path, served):
    k.write_kernel_state(tmp_path, k.KernelState(runtime="docker", url="http://127.0.0.1:9", port=9, token=TOKEN))
    done = k.stop_workspace_kernels(make_config(tmp_path), runner=FakeDocker(engine=None))
    assert done == ["Removed the record of a docker kernel that no longer answers (http://127.0.0.1:9)."]
    k.write_kernel_state(tmp_path, k.KernelState(runtime="docker", url=served.url, port=served.server_address[1], token=TOKEN))
    with pytest.raises(KernelRuntimeError) as exc:
        k.stop_workspace_kernels(make_config(tmp_path), runner=FakeDocker(engine=None))
    assert str(exc.value) == "Docker is not running."
    assert k.read_kernel_state(tmp_path) is not None, "kept: the containers still need removing"


# --------------------------------------------------------------------------- #
# The data folder, as the container sees it
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("path", [r"\\fileserver\team\sales", r"\\?\UNC\fileserver\team\sales", "//fileserver/team/sales"])
def test_unc_data_folders_are_flagged_on_windows(path, monkeypatch):
    scanned: list[Path] = []
    monkeypatch.setattr(k, "_links_outside", lambda folder, limit=0: (scanned.append(folder) or []))  # never touch a share
    problems = k.data_path_problems(Path(path), windows=True, drive_type=lambda root: 3)
    assert len(problems) == 1 and "network share (UNC path)" in problems[0] and "[hailer].data_dir" in problems[0]
    assert scanned == [], "a share is not walked for links"
    assert k.data_path_problems(Path(path), windows=False) == [], "Windows only"


def test_mapped_network_drives_are_flagged_and_local_drives_are_not(monkeypatch):
    monkeypatch.setattr(k, "_links_outside", lambda folder, limit=0: [])
    drives = {"Z:\\": k.DRIVE_REMOTE, "C:\\": 3}
    problems = k.data_path_problems(Path("Z:/sales"), windows=True, drive_type=lambda root: drives.get(root, 0))
    assert len(problems) == 1 and "a mapped network drive (Z:)" in problems[0]
    assert k.data_path_problems(Path("C:/sales"), windows=True, drive_type=lambda root: drives.get(root, 0)) == []
    assert k.data_path_problems(Path(r"\\?\C:\sales"), windows=True, drive_type=lambda root: drives.get(root, 0)) == []


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
    problems = k.data_path_problems(data, windows=False)
    assert len(problems) == 1 and problems[0].startswith("1 link(s) in the data folder point outside it (shared)")
    assert "recent" not in problems[0], "a link inside the folder resolves in the container too"
    assert k.data_path_problems(tmp_path / "missing", windows=False) == []
    rt = k.DockerRuntime(make_config(tmp_path, kernel=KernelConfig(runtime="docker")), runner=FakeDocker())
    row = next(r for r in rt.check() if r.name == "data")
    assert not row.ok and not row.fatal and "(shared)" in row.hint


# --------------------------------------------------------------------------- #
# SubprocessDockerRunner
# --------------------------------------------------------------------------- #


def test_subprocess_runner_without_docker_says_so(monkeypatch):
    monkeypatch.setattr(k.shutil, "which", lambda name: None)
    runner = k.SubprocessDockerRunner()
    for call in (lambda: runner.run(["version"]), lambda: runner.stream(["pull", IMAGE])):
        with pytest.raises(KernelRuntimeError) as exc:
            call()
        assert str(exc.value) == "Docker is not installed."


def test_subprocess_runner_runs_docker_in_text_mode_with_empty_stdin(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "29.4.3 linux\n", "")

    monkeypatch.setattr(k.subprocess, "run", fake_run)
    runner = k.SubprocessDockerRunner("/usr/bin/docker")
    assert runner.run(["version"], timeout=5, check=False).stdout == "29.4.3 linux\n"
    argv, kwargs = calls[0]
    assert argv == ["/usr/bin/docker", "version"]
    assert kwargs["text"] and kwargs["encoding"] == "utf-8" and kwargs["capture_output"] and kwargs["timeout"] == 5
    assert kwargs["check"] is False and kwargs["stdin"] == subprocess.DEVNULL, "docker never reads the chat's terminal"
    runner.run(["login", "--password-stdin"], input="secret")
    assert calls[1][1]["input"] == "secret" and calls[1][1]["stdin"] is None


def test_subprocess_runner_stream_waits_in_short_steps_and_stops_docker_on_ctrl_c(monkeypatch):
    class Proc:
        interrupt = False
        last = None

        def __init__(self, argv, **kwargs):
            self.argv, self.kwargs = argv, kwargs
            self.waits = 0
            self.terminated = False
            Proc.last = self

        def wait(self, timeout=None):
            self.waits += 1
            if Proc.interrupt and self.waits == 2:
                raise KeyboardInterrupt
            if self.waits < 3:
                raise subprocess.TimeoutExpired("docker", timeout)
            return 0

        def terminate(self):
            self.terminated = True

    monkeypatch.setattr(k.subprocess, "Popen", Proc)
    runner = k.SubprocessDockerRunner("docker")
    assert runner.stream(["pull", IMAGE]) == 0
    assert Proc.last.argv == ["docker", "pull", IMAGE] and Proc.last.kwargs["stdin"] == subprocess.DEVNULL and Proc.last.waits == 3
    Proc.interrupt = True
    with pytest.raises(KeyboardInterrupt):
        runner.stream(["logs", "-f", "hailer-kernel-x"])
    assert Proc.last.terminated
