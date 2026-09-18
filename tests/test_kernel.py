"""Tests for hailer.kernel: the path map, .hailer/kernel.json, the kernel's environment and the
local runtime (its process layer faked: nothing is spawned or killed). No Docker needed."""

from __future__ import annotations

import json
import os
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
    with pytest.raises(KernelRuntimeError) as exc:
        k.runtime_for(make_config(tmp_path, kernel=KernelConfig(runtime="docker")))
    assert "not implemented" in str(exc.value) and 'runtime = "local"' in exc.value.hint
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
    assert "ctx.packages.add()" in rt.prompt_notes()


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
