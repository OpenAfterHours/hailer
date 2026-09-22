"""Tests for hailer.kernel: the kernel's environment and prompt notes, and the local
runtime (its process layer faked: nothing is spawned or killed). The docker runtime has
tests/test_kernel_docker.py. No Docker needed."""

from __future__ import annotations

import os
import stat
import subprocess
import sys

import pytest

from fake_docker import FakeDocker
from fake_kernel import TOKEN, FakeProc, Procs, make_config
from hailer import kernel as k
from hailer.errors import ConfigError, KernelRuntimeError
from hailer.kernel_image import contract_tag
from hailer.models import KernelConfig, MarimoServer, ModelConfig, ProviderConfig

# --------------------------------------------------------------------------- #
# The kernel's environment and descriptions
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
        "PWD": "/home/ann",
        "OLDPWD": "/home",
        "HTTPS_PROXY": "http://proxy.example:3128",
        "DATABASE_URL": "postgres://ann:pw@db/sales",  # URL values are passed: proxies must keep working
        "OPENAI_API_KEY": "sk-1",
        "CORP_LLM_CREDENTIAL": "c-1",
        "CORP_CLIENT_ID": "id-1",
        "HAILER_MARIMO_TOKEN": "m-1",
        "GITHUB_TOKEN": "g-1",
        "DB_PASSWORD": "p-1",
        "AWS_SECRET": "s-1",
        "lower_case_api_key": "l-1",
        "SMTP_PASSWD": "x-1",
        "FTP_PWD": "x-2",
        "GOOGLE_APPLICATION_CREDENTIALS": "x-3",
        "SALES_DB_CONNECTION_STRING": "x-4",
        "MAPBOX_APIKEY": "x-5",
        "PGPASSWORD": "x-6",
        "MYSQL_PWD": "x-7",
        "TOKEN": "x-8",
        "SECRET": "x-9",
        "PASSWORD": "x-10",
    }
    env = k.kernel_environment(config, environ)
    assert env == {
        "PATH": "/usr/bin",
        "SYSTEMROOT": "C:\\Windows",
        "HAILER_LOG_LEVEL": "DEBUG",
        "PWD": "/home/ann",
        "OLDPWD": "/home",
        "HTTPS_PROXY": "http://proxy.example:3128",
        "DATABASE_URL": "postgres://ann:pw@db/sales",
    }
    assert environ["OPENAI_API_KEY"] == "sk-1", "the caller's mapping is not changed"
    assert len(k.withheld_variables(config, environ)) == len(environ) - len(env)


def test_pass_env_lets_named_variables_through(tmp_path):
    config = make_config(tmp_path, kernel=KernelConfig(pass_env=("DB_PASSWORD", "OPENAI_API_KEY")))
    env = k.kernel_environment(config, {"DB_PASSWORD": "p", "OPENAI_API_KEY": "sk", "GITHUB_TOKEN": "g", "PATH": "x"})
    assert env == {"DB_PASSWORD": "p", "OPENAI_API_KEY": "sk", "PATH": "x"}, "unchanged, even a provider key the user names"


@pytest.mark.skipif(os.name != "nt", reason="environment names are case-insensitive on Windows only")
def test_kernel_environment_names_ignore_case_on_windows(tmp_path):
    internal = ProviderConfig(id="internal", base_url="https://x/v1", env_key="Corp_Llm_Credential")
    config = make_config(tmp_path, providers={"internal": internal}, kernel=KernelConfig(pass_env=("db_password",)))
    assert k.kernel_environment(config, {"CORP_LLM_CREDENTIAL": "c", "Path": "p", "DB_PASSWORD": "d"}) == {"Path": "p", "DB_PASSWORD": "d"}


def test_describe_runtime():
    assert k.describe_runtime(KernelConfig()) == "local (runs as you; not isolated)"
    assert k.describe_runtime(KernelConfig(runtime="docker")) == f"docker (hailer-kernel {contract_tag()}; no network; data read-only)"
    assert k.describe_runtime(KernelConfig(runtime="docker", network=True)) == (
        f"docker (hailer-kernel {contract_tag()}; network on: the internet and this machine; data read-only)"
    )


def test_effective_image():
    assert KernelConfig().effective_image == f"ghcr.io/openafterhours/hailer-kernel:{contract_tag()}"
    assert KernelConfig(image="registry.example/hailer-kernel:dev").effective_image == "registry.example/hailer-kernel:dev"


def test_prompt_notes_per_runtime():
    assert k.runtime_prompt_notes(KernelConfig()) == k.LOCAL_PROMPT_NOTES
    offline = k.runtime_prompt_notes(KernelConfig(runtime="docker"))
    assert "No internet access" in offline and "ctx.packages.add is unavailable" in offline
    assert "/tmp is scratch space in memory, shared by every notebook in the container and wiped when the kernel stops" in offline
    assert "ctx.packages.add()" not in offline, "never told to install packages"
    online = k.runtime_prompt_notes(KernelConfig(runtime="docker", network=True))
    assert "the internet, and also services on the user's machine and in other containers" in online
    assert "No internet" not in online


def test_runtime_for_never_falls_back_to_local(tmp_path):
    from hailer.kernel_docker import DockerRuntime

    assert isinstance(k.runtime_for(make_config(tmp_path)), k.LocalRuntime)
    fake = FakeDocker(installed=False)
    docker = k.runtime_for(make_config(tmp_path, kernel=KernelConfig(runtime="docker")), runner=fake)
    assert isinstance(docker, DockerRuntime) and docker.runner is fake
    with pytest.raises(KernelRuntimeError) as exc:
        docker.start(2731)
    assert str(exc.value) == "Docker is not installed." and 'runtime = "local"' in exc.value.hint
    with pytest.raises(ConfigError):
        k.runtime_for(make_config(tmp_path, kernel=KernelConfig(runtime="podman")))


def test_the_public_api_is_small():
    assert "LocalProcesses" not in k.__all__ and "DockerRuntime" not in k.__all__
    assert all(hasattr(k, name) for name in k.__all__)
    assert not hasattr(k, "DockerRuntime"), "the docker runtime lives in hailer.kernel_docker"


# --------------------------------------------------------------------------- #
# LocalRuntime
# --------------------------------------------------------------------------- #


def runtime(tmp_path, procs: Procs, **kw) -> k.LocalRuntime:
    """The local runtime with its process layer faked."""
    return k.LocalRuntime(
        kw.pop("config", None) or make_config(tmp_path),
        procs=procs.local_processes(),
        environ=kw.pop("environ", {"PATH": "p", "OPENAI_API_KEY": "sk-secret"}),
        token_factory=lambda: TOKEN,
        **kw,
    )


def test_local_runtime_describes_itself(tmp_path):
    rt = runtime(tmp_path, Procs())
    assert rt.name == "local"
    assert rt.describe() == "local (runs as you; not isolated)"
    [row] = rt.check()
    assert row.ok and not row.fatal
    assert row.summary == (
        "local (runs as you; not isolated); 1 secret-looking environment variable(s) withheld from notebook code "
        "([kernel].pass_env lets named ones through)"
    )
    passing = runtime(tmp_path, Procs(), config=make_config(tmp_path, kernel=KernelConfig(pass_env=("DB_PASSWORD",))))
    assert passing.check()[0].summary.endswith("; passed through by [kernel].pass_env: DB_PASSWORD")
    rt.prepare(say=pytest.fail)  # nothing to download, nothing to warn about
    assert not hasattr(rt, "prompt_notes"), "one prompt path: runtime_prompt_notes"


def test_start_passes_the_token_on_stdin_and_scrubs_the_environment(tmp_path):
    procs = Procs()
    running = runtime(tmp_path, procs).start(2718)
    kind, cmd, cwd, log_path, env, stdin_text = procs.calls[0]
    assert kind == "spawn" and cwd == tmp_path and log_path == k.local_log_path(tmp_path)
    assert log_path.name == f"marimo-{os.getpid()}.log", "one log per Hailer process"
    assert cmd == [sys.executable, "-m", "marimo", "edit", "notebooks", "--token-password-file", "-", "--headless", "--port", "2718", "--skip-update-check"]
    assert stdin_text == TOKEN and TOKEN not in " ".join(cmd)
    assert env == {"PATH": "p"}, "no API key in the kernel's environment"
    assert procs.calls[1] == ("health", "http://127.0.0.1:2718", k.START_TIMEOUT_SEC, TOKEN), "waits for its own token"
    assert running.server == MarimoServer(url="http://127.0.0.1:2718", pid=4242, token=TOKEN, runtime="local")
    assert running.log_hint == str(log_path)
    assert [p.name for p in (tmp_path / ".hailer").glob("*.json")] == ["last-kernel.json"], "no kernel record"
    # the sandbox: the kernel knows the host's own folders
    from hailer.marimo_client import notebook_file_key

    assert running.native_paths and running.notebooks_path == notebook_file_key(tmp_path / "notebooks")
    assert running.data_path == str(tmp_path / "data") and running.data_dir == tmp_path / "data"
    assert running.describe() == "local (runs as you; not isolated)"
    assert running.notebook_url("q3/r.py") == f"http://127.0.0.1:2718/?file={notebook_file_key(tmp_path / 'notebooks')}/q3/r.py&view-as=present"
    assert running.home_url(with_token=True) == f"http://127.0.0.1:2718/?access_token={TOKEN}" and TOKEN not in repr(running)

    log_path.write_text(f"URL: http://localhost:2718?access_token={TOKEN}\n", encoding="utf-8")
    running.stop()
    assert procs.proc.terminated
    assert not log_path.exists(), "the log (it holds the signed-in URL) goes with the server"
    running.stop()  # idempotent


def test_foreground_start_attaches_to_this_terminal(tmp_path):
    procs = Procs()
    running = runtime(tmp_path, procs).start(2720, foreground=True)
    kind, cmd, _cwd, _log, env, stdin_text = procs.calls[0]
    assert kind == "attach" and "--headless" in cmd and stdin_text == TOKEN and "OPENAI_API_KEY" not in env
    assert running.log_hint == "this terminal" and running.log_tail() == []
    procs.proc.returncode = 0  # marimo exits by itself (the shutdown button)
    assert running.wait() == 0 and running.ended == "marimo shut itself down."
    procs.proc.returncode = 1  # ended from outside: kernel stop in another terminal, Task Manager
    assert running.wait() == 1
    assert running.ended.startswith("marimo stopped (exit code 1): it was ended from outside this terminal (uvx hailer kernel stop")
    running.stop()


def test_start_reports_an_early_exit_with_the_log_tail_and_cleans_up(tmp_path):
    procs = Procs(healthy=False, proc=FakeProc(exit_code=3))
    log = k.local_log_path(tmp_path)
    log.parent.mkdir(parents=True)
    log.write_text("\n".join([*(f"line {i}" for i in range(19)), f"URL: http://localhost:2718?access_token={TOKEN}"]), encoding="utf-8")
    with pytest.raises(KernelRuntimeError) as exc:
        runtime(tmp_path, procs).start(2718)
    assert str(exc.value) == "Marimo exited early (code 3)."
    assert exc.value.hint.startswith("Last lines of its log:") and "    line 18" in exc.value.hint and "line 3\n" not in exc.value.hint
    assert TOKEN not in exc.value.hint and "access_token=<token>" in exc.value.hint, "the token is masked in the tail"
    assert not log.exists(), "the tail is in the hint; the log never outlives the start"


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


@pytest.mark.parametrize("bound", [False, True], ids=["exited", "still-starting"])
def test_a_start_never_takes_another_sessions_server_on_the_same_port_for_its_own(tmp_path, bound):
    """Two sessions picked the same free port: the other session's server answers /health there,
    but only a server that holds this start's token counts. Otherwise every call would get 401."""
    from fake_marimo import serving
    from hailer.marimo_client import wait_for_health

    with serving(token="the-other-sessions-token") as other:
        port = other.server_address[1]
        procs = Procs(proc=FakeProc(exit_code=None if bound else 1))  # 1: this marimo could not bind the port
        layer = procs.local_processes()
        layer.wait_for_health = wait_for_health  # the real wait, against the real (other) server
        rt = k.LocalRuntime(make_config(tmp_path), procs=layer, environ={}, token_factory=lambda: TOKEN, start_timeout=0.3)
        with pytest.raises(KernelRuntimeError) as exc:
            rt.start(port)
    expected = f"Marimo did not answer on http://127.0.0.1:{port} within 0 s." if bound else "Marimo exited early (code 1)."
    assert str(exc.value) == expected
    assert procs.proc.terminated or not bound


# --------------------------------------------------------------------------- #
# A docker kernel's notebooks, later run locally
# --------------------------------------------------------------------------- #


def test_a_local_start_warns_when_a_docker_kernel_last_ran_the_notebooks(tmp_path):
    config = make_config(tmp_path)
    said: list[str] = []
    k.note_kernel_start(tmp_path, "docker", config.notebooks_root)
    rt = runtime(tmp_path, Procs())
    rt.prepare(say=said.append)
    assert len(said) == 2 and said[0].startswith(f"Warning: the notebooks in {config.notebooks_root} were last run by the isolated docker kernel")
    assert "runs on this machine as you" in said[0] and "uvx hailer notebook --kernel docker" in said[1]
    rt.start(2718).stop()
    said.clear()
    rt.prepare(say=said.append)
    assert said == [], "once per switch: the local start is recorded now"
    k.note_kernel_start(tmp_path, "docker", tmp_path / "elsewhere")
    rt.prepare(say=said.append)
    assert said == [], "another notebooks folder"
    assert not (tmp_path / "notebooks" / ".hailer").exists(), "kept in .hailer/, which is never mounted"


# --------------------------------------------------------------------------- #
# Process helpers
# --------------------------------------------------------------------------- #


def test_spawn_marimo_uses_background_flags_a_fresh_private_log_and_stdin(tmp_path, monkeypatch):
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
    log.parent.mkdir(parents=True)
    log.write_text(f"an old server's banner: access_token={TOKEN}\n", encoding="utf-8")
    k.spawn_marimo(["python", "-m", "marimo"], tmp_path, log)
    cmd, kwargs = calls[0]
    assert cmd == ["python", "-m", "marimo"] and kwargs["cwd"] == str(tmp_path)
    assert log.read_text(encoding="utf-8") == "", "a fresh log per start: old tokens do not pile up"
    assert kwargs["stderr"] == subprocess.STDOUT and kwargs["stdin"] == subprocess.DEVNULL and kwargs["env"] is None
    expected = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    assert kwargs["creationflags"] == expected, "own process group on Windows so Ctrl+C in the chat is not delivered to marimo"
    if os.name != "nt":
        assert stat.S_IMODE(log.stat().st_mode) == 0o600, "the log holds the signed-in URL"

    proc = k.spawn_marimo(["python"], tmp_path, log, env={"PATH": "p"}, stdin_text=TOKEN)
    _cmd, kwargs = calls[1]
    assert kwargs["stdin"] == subprocess.PIPE and kwargs["env"] == {"PATH": "p"}
    assert proc.stdin.written == TOKEN.encode() and proc.stdin.closed, "the token, then EOF"


def test_log_tail_masks_the_token(tmp_path):
    log = tmp_path / "marimo.log"
    log.write_text(f"a\nURL: http://localhost:2718?access_token={TOKEN}\n", encoding="utf-8")
    assert k.log_tail(log, token=TOKEN) == ["a", "URL: http://localhost:2718?access_token=<token>"]
    assert k.log_tail(tmp_path / "missing.log") == []


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
