"""Shared pieces for the kernel runtime tests (test helper): a workspace configuration, a liveness
probe stub (no connects to closed ports) and a fake process layer for the local runtime."""

from __future__ import annotations

from pathlib import Path

from hailer import kernel as k
from hailer.models import HailerConfig

TOKEN = "unit-test-token-0123456789"
LIVE = "http://127.0.0.1:2718"


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


def answers(*live: tuple[str, str]):
    """A liveness probe that says yes only for the (url, token) pairs given."""
    alive = set(live)
    return lambda url, token: (url, token) in alive


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
