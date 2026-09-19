"""Regressions from the final adversarial review; no Docker or host processes are started."""

from dataclasses import replace
import subprocess

import pytest

from fake_docker import FakeDocker
from fake_kernel import Procs, answers, make_config
from hailer import config as cfg, kernel as k, kernel_docker as kd
from hailer.errors import KernelRuntimeError
from hailer.models import KernelConfig


def runtime(tmp_path, fake, **options):
    config = make_config(tmp_path, kernel=KernelConfig(runtime="docker", **options))
    return kd.DockerRuntime(config, runner=fake, probe=answers(), health=lambda *a, **kw: True, user=lambda: None)


@pytest.mark.parametrize("marker", ["project/.git", "project/hailer.toml", ".vscode/nested/.git", "hailer.toml"])
def test_nested_repository_or_workspace_is_refused_before_docker(tmp_path, marker):
    rt = runtime(tmp_path, FakeDocker())
    path = rt.config.notebooks_root / marker
    path.parent.mkdir(parents=True)
    path.write_text("untrusted settings", encoding="utf-8")
    assert marker in " ".join(cfg.docker_mount_problems(rt.config))
    with pytest.raises(KernelRuntimeError, match="contains repository or workspace controls"):
        rt.start(2718, foreground=True)
    assert rt.runner.calls == [], "the mount check precedes even the engine probe"


def test_control_scan_does_not_follow_symlinks(tmp_path):
    root = tmp_path / "notebooks"
    root.mkdir()
    external = tmp_path / "elsewhere"
    external.mkdir()
    (external / ".git").mkdir()
    try:
        (root / "link").symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation unavailable")
    assert cfg.notebook_control_files(root) == []
    assert (external / ".git").exists()


def test_unreadable_notebooks_tree_is_not_approved(tmp_path, monkeypatch):
    rt = runtime(tmp_path, FakeDocker())
    def unreadable(folder):
        raise PermissionError("cannot read child folder")
    monkeypatch.setattr(cfg, "notebook_control_files", unreadable)
    with pytest.raises(KernelRuntimeError, match="Cannot inspect"):
        rt.start(2718)
    assert rt.runner.calls == []


def test_temp_directory_does_not_exempt_ssh_credentials(tmp_path, monkeypatch):
    secret = tmp_path / "home" / ".ssh"
    monkeypatch.setattr(cfg, "_credential_folders", lambda windows: [secret])
    monkeypatch.setattr(cfg, "_in_temp", lambda folder: True)
    config = make_config(tmp_path / "workspace", data_dir=secret / "nested", kernel=KernelConfig(runtime="docker"))
    assert "credentials" in " ".join(cfg.docker_mount_problems(config))


def test_nested_editor_settings_are_reported_after_stop(tmp_path):
    fake = FakeDocker()
    rt = runtime(tmp_path, fake)
    running = rt.start(2718)
    (rt.config.notebooks_root / "project" / ".idea").mkdir(parents=True)
    report = kd.stop_workspace_kernels(rt.config, runner=fake, probe=answers())
    assert report.warnings == [kd.planted_warning(rt.config.notebooks_root, ["project/.idea"])]
    assert running.planted() == ["project/.idea"]
    assert (rt.config.notebooks_root / "project" / ".idea").is_dir(), "warnings never remove user files"


@pytest.mark.parametrize("docker", [False, True])
def test_state_write_failure_cleans_up_the_kernel(tmp_path, monkeypatch, docker):
    fake = FakeDocker()
    procs = Procs()
    rt = runtime(tmp_path, fake) if docker else k.LocalRuntime(make_config(tmp_path), procs=procs.local_processes(), probe=answers())
    def full_disk(*args):
        raise OSError("disk full")
    monkeypatch.setattr(kd if docker else k, "write_kernel_state", full_disk)
    with pytest.raises(KernelRuntimeError, match="Could not record"):
        rt.start(2718)
    assert not fake.containers and not fake.networks if docker else procs.proc.terminated
    assert k.read_kernel_state(tmp_path) is None


def test_failed_automatic_cleanup_keeps_record_and_can_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(kd, "_sleep", lambda seconds: None)
    fake = FakeDocker()
    running = runtime(tmp_path, fake).start(2718)
    record = k.read_kernel_state(tmp_path)
    fake.fail[("rm",)] = (1, "engine unavailable")
    fake.fail[("network", "rm")] = (1, "engine unavailable")
    running.stop()
    assert k.read_kernel_state(tmp_path) == record
    assert "Could not finish stopping" in running.stop_error
    fake.fail.clear()
    running.stop()
    assert running.stop_error == "" and k.read_kernel_state(tmp_path) is None
    assert not fake.containers and not fake.networks


def test_empty_inspect_response_is_not_proof_of_death(tmp_path):
    class Unknown:
        def run(self, *args, **kwargs):
            return subprocess.CompletedProcess(args, 0, "", "")
    state = k.KernelState(runtime="docker", url="http://127.0.0.1:2718", port=2718, token="test", container_ids=("id",))
    assert not kd.containers_gone(state, Unknown())


def test_diagnostics_keep_busy_kernel_runtime_with_real_discovery(tmp_path, monkeypatch):
    from hailer import cli, marimo_client as mc
    config = make_config(tmp_path)
    state = k.KernelState(runtime="docker", url="http://127.0.0.1:2718", port=2718, token="test", container_ids=("id",))
    k.write_kernel_state(tmp_path, state)
    monkeypatch.setattr(k, "answers_with_token", answers())
    monkeypatch.setattr(k, "record_is_gone", lambda state: False)
    monkeypatch.setattr(mc, "discover_servers", lambda *args: [])
    discovered = cli._discovered_server(config)
    assert discovered.runtime == "docker"
    assert cli._attach_runtime(config, discovered).kernel.runtime == "docker"
    # An explicit pin still takes precedence over a record from another server.
    pinned = replace(config, marimo_url="http://127.0.0.1:9999")
    assert cli._discovered_server(pinned).url == pinned.marimo_url
