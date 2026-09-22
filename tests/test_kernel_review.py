"""Regressions from the final adversarial review; no Docker or host processes are started."""

from fake_docker import FakeDocker
from fake_kernel import make_config
from hailer import config as cfg, kernel_docker as kd
from hailer.models import KernelConfig


def runtime(tmp_path, fake, **options):
    config = make_config(tmp_path, kernel=KernelConfig(runtime="docker", **options))
    return kd.DockerRuntime(config, runner=fake, health=lambda *a, **kw: True, user=lambda: None)


def test_temp_directory_does_not_exempt_ssh_credentials(tmp_path, monkeypatch):
    secret = tmp_path / "home" / ".ssh"
    monkeypatch.setattr(cfg, "_credential_folders", lambda windows: [secret])
    monkeypatch.setattr(cfg, "_in_temp", lambda folder: True)
    config = make_config(tmp_path / "workspace", data_dir=secret / "nested", kernel=KernelConfig(runtime="docker"))
    assert "credentials" in " ".join(cfg.docker_mount_problems(config))


def test_failed_automatic_cleanup_says_so_and_can_retry(tmp_path, monkeypatch):
    from test_kernel_docker import NoSync

    monkeypatch.setattr(kd, "_sleep", lambda seconds: None)
    monkeypatch.setattr(kd, "NotebookSync", NoSync)
    fake = FakeDocker()
    running = runtime(tmp_path, fake).start(2718)
    fake.fail[("rm",)] = (1, "engine unavailable")
    fake.fail[("network", "rm")] = (1, "engine unavailable")
    running.stop()
    assert "Could not finish stopping" in running.stop_error and "uvx hailer kernel stop" in running.stop_error
    assert len(fake.containers) == 2, "still labelled: kernel stop or the next start removes them"
    assert not running.owner.path.exists(), "ownerless now: the next start may remove them"
    fake.fail.clear()
    running.stop()
    assert running.stop_error == ""
    assert not fake.containers and not fake.networks
