"""Dependency preparation stays responsive and never owns runtime resources."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
import time

import pytest

from hailer import agent as agent_mod
from hailer.errors import AgentError


@pytest.fixture
def fresh_warmup(monkeypatch):
    previous = agent_mod._dependency_warmup
    if previous is not None:
        # Other agent tests may already have completed the process-wide preparation.
        previous.result(timeout=15)
    monkeypatch.setattr(agent_mod, "_dependency_warmup", None)


async def _wait_for_thread(event: threading.Event) -> None:
    async with asyncio.timeout(5):
        while not event.is_set():
            await asyncio.sleep(0.001)


def test_warmup_imports_once_on_a_daemon_after_disabling_tracing(fresh_warmup, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls: list[threading.Thread] = []
    monkeypatch.delenv("HAILER_TRACING", raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    def prepare() -> None:
        import os

        assert os.environ["LANGSMITH_TRACING"] == "false"
        calls.append(threading.current_thread())
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(agent_mod, "_load_agent_dependencies", prepare)
    completion = agent_mod.start_dependency_warmup()
    try:
        assert entered.wait(5)
        assert agent_mod.start_dependency_warmup() is completion
        assert not completion.cancel()
        assert len(calls) == 1 and calls[0].daemon
        assert calls[0] is not threading.current_thread()
    finally:
        release.set()
        completion.result(timeout=5)


def test_cancelled_wait_closes_its_loop_while_imports_continue(fresh_warmup, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def prepare() -> None:
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(agent_mod, "_load_agent_dependencies", prepare)

    async def cancel_first_wait() -> None:
        waiter = asyncio.create_task(agent_mod.await_dependency_warmup())
        await _wait_for_thread(entered)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(waiter, timeout=1)

    started = time.monotonic()
    try:
        asyncio.run(cancel_first_wait())
        # Runner shutdown must not join the blocked worker via the default executor.
        assert time.monotonic() - started < 2
        assert agent_mod._dependency_warmup.running()
    finally:
        release.set()
        agent_mod._dependency_warmup.result(timeout=5)
    # Completion is reusable from a new loop; closing the first loop lost no result.
    asyncio.run(agent_mod.await_dependency_warmup())


def test_failure_reaches_remaining_waiters_without_an_abandoned_exception(fresh_warmup, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls: list[bool] = []

    def prepare() -> None:
        calls.append(True)
        entered.set()
        assert release.wait(5)
        raise ModuleNotFoundError("missing package; sk-private-value-must-not-be-shown")

    monkeypatch.setattr(agent_mod, "_load_agent_dependencies", prepare)

    async def exercise() -> None:
        errors = []
        asyncio.get_running_loop().set_exception_handler(lambda _loop, context: errors.append(context))
        first = asyncio.create_task(agent_mod.await_dependency_warmup())
        second = asyncio.create_task(agent_mod.await_dependency_warmup())
        try:
            await _wait_for_thread(entered)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            release.set()
            with pytest.raises(AgentError, match="Could not prepare chat dependencies") as err:
                await asyncio.wait_for(second, timeout=5)
            assert "ModuleNotFoundError" in err.value.hint
            assert "sk-private" not in err.value.hint
            with pytest.raises(AgentError):
                await agent_mod.await_dependency_warmup()
            await asyncio.sleep(0)
            assert not errors
        finally:
            release.set()
            await asyncio.gather(first, second, return_exceptions=True)

    asyncio.run(exercise())
    assert calls == [True]


def test_real_cold_warmup_only_imports_without_clients_network_or_sqlite():
    # A separate interpreter prevents earlier tests from making imports appear cheap or hiding
    # resources created on first import. Audit hooks also reject accidental outbound requests.
    script = r'''
import asyncio
import sys
import threading
from hailer.agent import await_dependency_warmup

assert not any(name in sys.modules for name in ("langchain", "langchain_openai", "openai"))

def audit(event, args):
    if event in ("socket.connect", "sqlite3.connect"):
        raise AssertionError("warmup attempted " + event)

def profile(frame, event, arg):
    if event != "call" or frame.f_code.co_name != "__init__":
        return
    module = frame.f_globals.get("__name__", "")
    instance = frame.f_locals.get("self")
    if module.startswith(("openai.", "httpx.", "httpx2.")) and instance is not None:
        if any(base.__name__ in ("BaseClient", "Client", "AsyncClient") for base in type(instance).__mro__):
            raise AssertionError("warmup constructed " + type(instance).__name__)

async def exercise():
    # Windows creates the loop's self-pipe with a loopback socket; install the audit only
    # after the event loop exists, so it checks imports rather than asyncio itself.
    sys.addaudithook(audit)
    threading.setprofile(profile)
    await await_dependency_warmup()

asyncio.run(exercise())
assert all(name in sys.modules for name in ("langchain.agents", "langchain_openai", "openai.resources"))
print("imports only")
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "imports only"
