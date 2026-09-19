"""Offline tests for hailer.marimo_client against a local fake marimo server."""

from __future__ import annotations

import json
import os
import socket
import threading
from dataclasses import replace
from pathlib import Path
from urllib.parse import quote

import pytest
from fake_marimo import running

from fake_marimo import FakeMarimo
from hailer import marimo_client as mc
from hailer.errors import MarimoExecutionError, MarimoUnavailableError, NoSessionError, NotebookPathError
from hailer.models import HailerConfig, KernelConfig, MarimoServer, MarimoSession

# --------------------------------------------------------------------------- #
# Fake server (tests/fake_marimo.py)
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake():
    srv = FakeMarimo().start()
    try:
        yield srv
    finally:
        srv.stop()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _config(tmp_path: Path, **overrides) -> HailerConfig:
    base = dict(
        workspace=tmp_path,
        notebook=tmp_path / "notebooks" / "analysis.py",
        data_dir=tmp_path / "data",
        context_dir=tmp_path / ".config" / "hailer" / "context",
        skills_dir=tmp_path / ".config" / "hailer" / "skills",
        prompts_dir=tmp_path / ".config" / "hailer" / "prompts",
    )
    base.update(overrides)
    return HailerConfig(**base)


# --------------------------------------------------------------------------- #
# health / sessions
# --------------------------------------------------------------------------- #


def test_health_true_and_false(fake):
    assert mc.MarimoClient(fake.url).health() is True
    assert mc.MarimoClient(f"http://127.0.0.1:{_free_port()}").health() is False


def test_sessions_parsed(fake, tmp_path):
    nb = tmp_path / "notebooks" / "analysis.py"
    fake.sessions = {"s1": {"filename": "notebooks/analysis.py", "path": str(nb)}, "s2": {"filename": None, "path": None}}
    sessions = mc.MarimoClient(fake.url).sessions()
    assert sessions == [
        MarimoSession("s1", "notebooks/analysis.py", str(nb)),
        MarimoSession("s2", None, None),
    ]


# --------------------------------------------------------------------------- #
# execute
# --------------------------------------------------------------------------- #


def test_execute_success_collects_stdout_and_output(fake):
    seen: list[str] = []
    result = mc.MarimoClient(fake.url).execute("print('x')", session_id="s1", on_stdout=seen.append)
    assert result.success is True
    assert result.stdout == "hello world\n"
    assert result.output == "42"
    assert result.mimetype == "text/plain"
    assert seen == ["hello ", "world\n"]
    req = fake.requests[-1]
    assert req["headers"]["Marimo-Session-Id"] == "s1"
    assert req["headers"]["Content-Type"] == "application/json"
    assert req["body"] == {"code": "print('x')"}
    assert "hello world" in result.as_text() and "42" in result.as_text()


def test_execute_crlf_stream(fake):
    fake.mode = "crlf"
    result = mc.MarimoClient(fake.url).execute("1", session_id="s1")
    assert result.success and result.stdout == "hello world\n" and result.output == "42"


def test_execute_stderr_and_failure(fake):
    fake.mode = "stderr"
    errs: list[str] = []
    result = mc.MarimoClient(fake.url).execute("boom", session_id="s1", on_stderr=errs.append)
    assert result.success is False
    assert "boom" in result.stderr and errs == ["Traceback: boom\n"]
    assert "[stderr]" in result.as_text()


def test_execute_json_error_body(fake):
    fake.mode = "json_error"
    with pytest.raises(MarimoExecutionError) as exc:
        mc.MarimoClient(fake.url).execute("1", session_id="stale")
    assert "Session not found: stale" in str(exc.value)
    assert exc.value.hint


def test_execute_non_stream_200(fake):
    fake.mode = "plain_json_200"
    with pytest.raises(MarimoExecutionError) as exc:
        mc.MarimoClient(fake.url).execute("1", session_id="s1")
    assert "not a stream" in str(exc.value)


def test_execute_stream_without_done(fake):
    fake.mode = "nodone"
    with pytest.raises(MarimoExecutionError) as exc:
        mc.MarimoClient(fake.url).execute("1", session_id="s1")
    assert "without a result" in str(exc.value)


def test_execute_resolves_session_when_not_given(fake, tmp_path):
    nb = tmp_path / "notebooks" / "analysis.py"
    fake.sessions = {"abc": {"filename": "notebooks/analysis.py", "path": str(nb)}}
    result = mc.MarimoClient(fake.url, notebook=nb, workspace=tmp_path).execute("1")
    assert result.success
    assert fake.requests[-1]["headers"]["Marimo-Session-Id"] == "abc"


def test_bearer_token_sent_and_401_mapped(fake):
    fake.token = "secret-token"
    fake.sessions = {"s1": {"filename": "a.py", "path": "a.py"}}
    ok = mc.MarimoClient(fake.url, token="secret-token")
    assert ok.sessions()[0].session_id == "s1"
    ok.execute("1", session_id="s1")
    assert fake.requests[-1]["headers"]["Authorization"] == "Bearer secret-token"
    with pytest.raises(MarimoUnavailableError) as exc:
        mc.MarimoClient(fake.url, token="wrong").sessions()
    assert "HAILER_MARIMO_TOKEN" in exc.value.hint


def test_connection_refused_hint_has_launch_command(tmp_path):
    nb = tmp_path / "notebooks" / "analysis.py"
    client = mc.MarimoClient(f"http://127.0.0.1:{_free_port()}", notebook=nb, workspace=tmp_path)
    with pytest.raises(MarimoUnavailableError) as exc:
        client.sessions()
    assert "uvx hailer notebook --foreground" in exc.value.hint
    assert "uv run" not in exc.value.hint, "no project .venv is needed"


# --------------------------------------------------------------------------- #
# server token / shutdown_session
# --------------------------------------------------------------------------- #


def test_server_token_is_read_from_the_page_and_cached(fake):
    client = mc.MarimoClient(fake.url)
    assert client.server_token() == "skew-token-123"
    assert client.server_token() == "skew-token-123"
    assert fake.page_hits == 1, "cached on the instance"


def test_server_token_missing_is_actionable(fake):
    fake.mode = "no_token_tag"
    with pytest.raises(MarimoUnavailableError) as exc:
        mc.MarimoClient(fake.url).server_token()
    assert "server token" in str(exc.value) and "marimo-server-token" in exc.value.hint


def test_shutdown_session_sends_token_and_closes(fake):
    fake.sessions = {"s1": {"filename": "a.py", "path": "a.py"}, "s2": {"filename": "b.py", "path": "b.py"}}
    client = mc.MarimoClient(fake.url)
    client.shutdown_session("s1")
    req = fake.requests[-1]
    assert req["path"] == "/api/home/shutdown_session"
    assert req["headers"]["Marimo-Server-Token"] == "skew-token-123"
    assert req["body"] == {"sessionId": "s1"}
    assert [s.session_id for s in client.sessions()] == ["s2"]
    with pytest.raises(MarimoUnavailableError) as exc:
        client.shutdown_session("nope")
    assert "HTTP 500" in str(exc.value) and "Session not found" in str(exc.value)


def test_shutdown_session_stale_token_is_actionable_and_refetched(fake):
    fake.sessions = {"s1": {"filename": "a.py", "path": "a.py"}}
    client = mc.MarimoClient(fake.url)
    assert client.server_token() == "skew-token-123"
    fake.mode = "stale_token"
    with pytest.raises(MarimoUnavailableError) as exc:
        client.shutdown_session("s1")
    assert "HTTP 401" in str(exc.value) and "restarted" in exc.value.hint
    fake.mode = "success"
    client.shutdown_session("s1")  # the token was dropped and fetched again
    assert fake.page_hits == 2
    assert client.sessions() == []


# --------------------------------------------------------------------------- #
# resolve_session
# --------------------------------------------------------------------------- #


def test_resolve_no_sessions(fake, tmp_path):
    nb = tmp_path / "notebooks" / "analysis.py"
    with pytest.raises(NoSessionError) as exc:
        mc.MarimoClient(fake.url, notebook=nb, workspace=tmp_path).resolve_session(nb)
    assert "not open in a browser" in str(exc.value)
    assert exc.value.hint.startswith("Open ") and f"?file={mc.notebook_file_key(nb)}" in exc.value.hint


def test_resolve_by_absolute_path_and_case(fake, tmp_path):
    nb = tmp_path / "notebooks" / "analysis.py"
    stored = str(nb).upper() if os.name == "nt" else str(nb)
    fake.sessions = {"other": {"filename": "x.py", "path": str(tmp_path / "x.py")}, "mine": {"filename": "notebooks/analysis.py", "path": stored}}
    assert mc.MarimoClient(fake.url).resolve_session(nb).session_id == "mine"


def test_resolve_relative_session_path_against_the_workspace(fake, tmp_path):
    nb = tmp_path / "notebooks" / "analysis.py"
    fake.sessions = {"mine": {"filename": "notebooks/analysis.py", "path": None}}
    assert mc.MarimoClient(fake.url, workspace=tmp_path).resolve_session(nb).session_id == "mine"
    # a bare filename is not the same notebook: no more matching by name alone
    fake.sessions = {"other": {"filename": "analysis.py", "path": None}}
    with pytest.raises(NoSessionError):
        mc.MarimoClient(fake.url, workspace=tmp_path).resolve_session(nb)


def test_resolve_same_filename_in_two_subfolders(fake, tmp_path):
    a = tmp_path / "notebooks" / "a" / "report.py"
    b = tmp_path / "notebooks" / "b" / "report.py"
    fake.sessions = {"sA": {"filename": "a/report.py", "path": str(a)}}
    with pytest.raises(NoSessionError):
        mc.MarimoClient(fake.url, workspace=tmp_path).resolve_session(b)
    fake.sessions = {"sA": {"filename": "a/report.py", "path": str(a)}, "sB": {"filename": "b/report.py", "path": str(b)}}
    client = mc.MarimoClient(fake.url, workspace=tmp_path)
    assert client.resolve_session(a).session_id == "sA"
    assert client.resolve_session(b).session_id == "sB"


def test_match_session_ignores_untitled_and_prefers_the_latest(tmp_path):
    nb = tmp_path / "notebooks" / "analysis.py"
    sessions = [
        MarimoSession("untitled", None, None),
        MarimoSession("old", "notebooks/analysis.py", str(nb)),
        MarimoSession("other", "x.py", str(tmp_path / "x.py")),
        MarimoSession("new", "notebooks/analysis.py", str(nb).upper() if os.name == "nt" else str(nb)),
    ]
    assert mc.match_session(sessions, nb, tmp_path).session_id == "new"
    assert mc.match_session(sessions[:1], nb, tmp_path) is None
    assert mc.match_session([], nb) is None
    assert mc.match_session([MarimoSession("rel", "notebooks/analysis.py", "notebooks/analysis.py")], nb, tmp_path).session_id == "rel"
    assert mc.match_session([MarimoSession("rel", "notebooks/analysis.py", "notebooks/analysis.py")], nb, tmp_path / "elsewhere") is None


def test_resolve_multiple_no_match_lists_sessions(fake, tmp_path):
    nb = tmp_path / "notebooks" / "analysis.py"
    fake.sessions = {"a": {"filename": "one.py", "path": str(tmp_path / "one.py")}, "b": {"filename": "two.py", "path": str(tmp_path / "two.py")}}
    with pytest.raises(NoSessionError) as exc:
        mc.MarimoClient(fake.url).resolve_session(nb)
    assert "one.py" in exc.value.hint and "two.py" in exc.value.hint


def test_resolve_single_session_without_notebook(fake):
    fake.sessions = {"only": {"filename": "n.py", "path": "n.py"}}
    assert mc.MarimoClient(fake.url).resolve_session(None).session_id == "only"


# --------------------------------------------------------------------------- #
# registry discovery / find_server
# --------------------------------------------------------------------------- #


def _write_entry(directory: Path, name: str, **entry) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(entry), encoding="utf-8")


def test_discover_servers_filters_dead_and_garbage(fake, tmp_path):
    reg = tmp_path / "servers"
    live_port = fake.server_address[1]
    _write_entry(reg, "127.0.0.1_live.json", server_id=f"127.0.0.1:{live_port}", pid=1234, host="127.0.0.1", port=live_port, base_url="", started_at="x", version="0.24.2")
    _write_entry(reg, "0.0.0.0_dead.json", server_id="0.0.0.0:1", pid=1, host="0.0.0.0", port=_free_port(), base_url="", started_at="x", version="0.23.9")
    (reg / "garbage.json").write_text("{not json", encoding="utf-8")
    found = mc.discover_servers(reg)
    assert [s.url for s in found] == [fake.url]
    assert found[0].pid == 1234 and found[0].version == "0.24.2" and found[0].source == "registry"


def test_discover_servers_missing_dir(tmp_path):
    assert mc.discover_servers(tmp_path / "nope") == []


def test_url_from_entry_bind_all_becomes_loopback():
    assert mc._url_from_entry({"host": "0.0.0.0", "port": 2718, "base_url": "/nb/"}) == "http://127.0.0.1:2718/nb"
    assert mc._url_from_entry({"host": "::", "port": 2718}) == "http://[::1]:2718"
    assert mc._url_from_entry({"host": "localhost", "port": "bad"}) is None


def test_find_server_prefers_config(fake, tmp_path):
    cfg = _config(tmp_path, marimo_url="http://localhost:9999/")
    server = mc.find_server(cfg, registry=tmp_path / "servers")
    assert server == MarimoServer(url="http://localhost:9999", source="config")


def _register(reg: Path, *servers) -> None:
    """Registry entries for ``servers`` the way marimo writes them (``<host>_<port>.json``)."""
    for srv in servers:
        port = srv.server_address[1]
        _write_entry(reg, f"127.0.0.1_{port}.json", server_id=f"127.0.0.1:{port}", pid=1, host="127.0.0.1", port=port, base_url="", started_at="", version="0.24.2")


def test_find_server_pinned_url_is_returned_even_when_down(tmp_path):
    cfg = _config(tmp_path, marimo_url=f"http://127.0.0.1:{_free_port()}")
    assert mc.find_server(cfg, registry=tmp_path / "servers").url == cfg.marimo_url


def test_find_server_ignores_a_lone_server_for_another_workspace(fake, tmp_path):
    """One live server, started on another worktree's notebooks folder: not ours, even though it is the only one."""
    reg = tmp_path / "servers"
    ws = tmp_path / "wt-a"
    other = tmp_path / "wt-b"
    fake.root = str(other / "notebooks")
    fake.sessions = {"s1": {"filename": "analysis.py", "path": str(other / "notebooks" / "analysis.py")}}
    _register(reg, fake)
    assert mc.find_server(_config(ws, notebook=ws / "notebooks" / "analysis.py"), registry=reg) is None


def test_find_server_picks_the_server_started_on_this_workspace(fake, tmp_path):
    """Two live servers, no tab open on either: the one whose root is our notebooks folder wins."""
    reg = tmp_path / "servers"
    ws = tmp_path / "wt-a"
    fake.root = str(tmp_path / "wt-b" / "notebooks")
    with running() as ours:
        ours.root = str(ws / "notebooks")
        _register(reg, fake, ours)
        found = mc.find_server(_config(ws, notebook=ws / "notebooks" / "analysis.py"), registry=reg)
        assert found is not None and found.url == ours.url and found.source == "registry"


def test_find_server_picks_the_server_hosting_a_notebook_of_this_workspace(fake, tmp_path):
    """A server started elsewhere (single-file mode: no root) that hosts one of our notebooks is ours."""
    reg = tmp_path / "servers"
    ws = tmp_path / "wt-a"
    fake.root = str(tmp_path / "wt-b" / "notebooks")
    with running() as ours:
        ours.sessions = {"s2": {"filename": "other.py", "path": str(ws / "notebooks" / "sub" / "other.py")}}
        _register(reg, fake, ours)
        found = mc.find_server(_config(ws, notebook=ws / "notebooks" / "analysis.py"), registry=reg)
        assert found is not None and found.url == ours.url
        assert ours.page_hits == 0, "a session inside the notebooks folder is proof enough; no root lookup"


def test_find_server_prefers_the_server_hosting_the_active_notebook(fake, tmp_path):
    reg = tmp_path / "servers"
    ws = tmp_path
    fake.root = str(ws / "notebooks")  # ours, but the active notebook is open on the other one
    with running() as active:
        active.sessions = {"s1": {"filename": "analysis.py", "path": str(ws / "notebooks" / "analysis.py")}}
        _register(reg, fake, active)
        assert mc.find_server(_config(ws), registry=reg).url == active.url


def test_find_server_does_not_claim_a_server_on_a_parent_folder(fake, tmp_path):
    """A server on the main checkout does not belong to a worktree nested inside it."""
    reg = tmp_path / "servers"
    main = tmp_path / "repo"
    worktree = main / ".claude" / "worktrees" / "wt"
    fake.root = str(main)
    _register(reg, fake)
    assert mc.find_server(_config(worktree, notebook=worktree / "notebooks" / "analysis.py"), registry=reg) is None
    fake.root = str(main / "notebooks")
    assert mc.find_server(_config(worktree, notebook=worktree / "notebooks" / "analysis.py"), registry=reg) is None


def test_root_reads_workspace_files_with_the_server_token(fake, tmp_path):
    fake.root = str(tmp_path / "notebooks")
    assert mc.MarimoClient(fake.url).root() == str(tmp_path / "notebooks")
    request = fake.requests[-1]
    assert request["path"] == "/api/home/workspace_files"
    assert request["headers"].get("Marimo-Server-Token") == fake.server_token
    fake.root = None
    assert mc.MarimoClient(fake.url).root() is None, "a single-file server has no folder"


def test_workspace_affinity_is_none_for_something_that_is_not_marimo(fake, tmp_path):
    fake.mode = "no_token_tag"  # answers /health and /api/sessions, but no server token and no workspace_files
    assert mc.workspace_affinity(mc.MarimoClient(fake.url), _config(tmp_path)) == mc.AFFINITY_NONE
    fake.mode = "sessions_500"
    assert mc.workspace_affinity(mc.MarimoClient(fake.url), _config(tmp_path)) == mc.AFFINITY_NONE


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def test_notebook_file_key_is_absolute_posix(tmp_path):
    nb = tmp_path / "notebooks" / ".." / "notebooks" / "my analysis.py"
    key = mc.notebook_file_key(nb)
    assert key == Path(os.path.normpath(str((tmp_path / "notebooks" / "my analysis.py").absolute()))).as_posix()
    assert "\\" not in key and ".." not in key
    assert Path(key).is_absolute()


def test_notebook_file_key_does_not_resolve_symlinks(tmp_path, monkeypatch):
    """marimo normalises without resolving links; a key through a junction must stay as given."""
    nb = tmp_path / "linked" / "nb.py"
    monkeypatch.setattr(Path, "resolve", lambda self, strict=False: Path(str(self).replace("linked", "real")))
    assert "linked" in mc.notebook_file_key(nb) and "real" not in mc.notebook_file_key(nb)
    relative = Path("notebooks") / "x.py"
    assert mc.notebook_file_key(relative) == (Path.cwd() / relative).as_posix()


def test_open_notebook_url_defaults_to_app_view(tmp_path):
    nb = tmp_path / "notebooks" / "my analysis.py"
    url = mc.open_notebook_url(MarimoServer(url="http://127.0.0.1:2718"), nb, tmp_path)
    key = mc.notebook_file_key(nb)
    assert url == f"http://127.0.0.1:2718/?file={quote(key, safe='/:')}&view-as=present"
    assert "my%20analysis.py" in url and "%5C" not in url, "absolute posix key, no backslashes"
    # the workspace no longer influences the key (absolute keys work on folder and single-file servers)
    assert mc.open_notebook_url(MarimoServer(url="http://127.0.0.1:2718"), nb, tmp_path / "elsewhere") == url
    assert mc.open_notebook_url(MarimoServer(url="http://127.0.0.1:2718"), nb) == url


def test_open_notebook_url_edit_view(tmp_path):
    nb = tmp_path / "notebooks" / "analysis.py"
    server = MarimoServer(url="http://127.0.0.1:2718/")
    expected = f"http://127.0.0.1:2718/?file={quote(mc.notebook_file_key(nb), safe='/:')}"
    assert mc.open_notebook_url(server, nb, tmp_path, view="edit") == expected
    with pytest.raises(ValueError):
        mc.open_notebook_url(server, nb, tmp_path, view="kiosk")


def test_client_notebook_url_uses_app_view(fake, tmp_path):
    nb = tmp_path / "notebooks" / "analysis.py"
    client = mc.MarimoClient(fake.url, notebook=nb, workspace=tmp_path)
    assert client.notebook_url().endswith(f"?file={quote(mc.notebook_file_key(nb), safe='/:')}&view-as=present")


def test_launch_command_is_hailers_foreground_mode():
    assert mc.launch_command() == ["uvx", "hailer", "notebook", "--foreground"]
    assert mc.launch_command(port=2718)[-2:] == ["--port", "2718"]


def test_snippets_are_valid_python():
    import ast

    for snippet in (mc.CM_HELP_CODE, mc.LIST_CELLS_CODE, mc.NOTEBOOK_GLOBALS_CODE):
        ast.parse(snippet)
    code = mc.build_create_cell_code("df = pl.DataFrame({'a': [1]})\ndf", name="demo")
    # top-level `async with` is only valid in the scratchpad; wrap to check syntax
    ast.parse("async def _():\n" + "\n".join("    " + line for line in code.splitlines()))
    assert "hide_code=False" in code and "name='demo'" in code and "ctx.run_cell(cid)" in code


def test_sessions_server_error_is_actionable(fake, tmp_path):
    fake.mode = "sessions_500"
    client = mc.MarimoClient(fake.url, notebook=tmp_path / "nb.py", workspace=tmp_path)
    with pytest.raises(MarimoUnavailableError) as info:
        client.sessions()
    assert "HTTP 500" in str(info.value)
    assert "kernel manager exploded" in str(info.value)
    assert "uvx hailer notebook --foreground" in info.value.hint
    # resolve_session (used by preflight) must surface the same actionable error, never a raw HTTPError
    with pytest.raises(MarimoUnavailableError):
        client.resolve_session(tmp_path / "nb.py")


def test_sse_parser_handles_comments_and_multiline_data():
    lines = [b": comment\r\n", b"event: stdout\r\n", b"data: {\"data\":\r\n", b"data: \"x\"}\r\n", b"\r\n", b"event: done\n", b"data: {\"success\": true, \"output\": {\"data\": \"\", \"mimetype\": \"text/plain\"}}\n", b"\n"]
    events = list(mc._iter_sse(iter(lines)))
    assert events[0][0] == "stdout" and json.loads(events[0][1]) == {"data": "x"}
    assert events[1][0] == "done"


# --------------------------------------------------------------------------- #
# Helpers for `hailer notebook`: ports, readiness waits, registry cleanup, hint
# --------------------------------------------------------------------------- #


def test_find_free_port_prefers_the_requested_port():
    port = _free_port()
    assert mc.find_free_port(port) == port


def test_find_free_port_falls_back_when_busy(fake):
    busy = fake.server_address[1]
    assert mc.port_in_use(busy)
    chosen = mc.find_free_port(busy)
    assert chosen != busy and not mc.port_in_use(chosen)


def test_wait_for_health_true_for_live_server(fake):
    assert mc.wait_for_health(fake.url, timeout=2.0, interval=0.05)


def test_wait_for_health_false_when_nothing_listens():
    url = f"http://127.0.0.1:{_free_port()}"
    assert not mc.wait_for_health(url, timeout=0.3, interval=0.05)


def test_wait_for_health_stops_early_when_process_died():
    url = f"http://127.0.0.1:{_free_port()}"
    calls = []

    def died():
        calls.append(1)
        return True

    assert not mc.wait_for_health(url, timeout=30.0, interval=0.05, should_stop=died)
    assert len(calls) == 1, "returns as soon as should_stop() is True instead of waiting for the timeout"


def test_wait_for_session_returns_when_notebook_opens(fake, tmp_path):
    nb = tmp_path / "notebooks" / "analysis.py"
    client = mc.MarimoClient(fake.url, notebook=nb, workspace=tmp_path)

    def open_tab():
        fake.sessions["s_new"] = {"filename": str(nb), "path": str(nb)}

    threading.Timer(0.3, open_tab).start()
    session = mc.wait_for_session(client, nb, timeout=5.0, interval=0.05)
    assert session is not None and session.session_id == "s_new"


def test_wait_for_session_none_on_timeout(fake, tmp_path):
    nb = tmp_path / "notebooks" / "analysis.py"
    client = mc.MarimoClient(fake.url, notebook=nb, workspace=tmp_path)
    assert mc.wait_for_session(client, nb, timeout=0.3, interval=0.05) is None


def test_wait_for_session_cancelled_before_request():
    class Client:
        def resolve_session(self, notebook):
            pytest.fail("a cancelled wait must not start a request")

    assert mc.wait_for_session(Client(), None, should_stop=lambda: True) is None


def test_wait_for_session_cancelled_during_long_poll_interval(monkeypatch):
    stopped = threading.Event()
    attempts = []

    class Clock:
        now = 0.0
        sleeps = []

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            self.sleeps.append(seconds)
            self.now += seconds
            stopped.set()

    class Client:
        def resolve_session(self, notebook):
            attempts.append(notebook)
            raise NoSessionError("not open")

    clock = Clock()
    monkeypatch.setattr(mc, "time", clock)
    assert mc.wait_for_session(Client(), None, timeout=60, interval=30, should_stop=stopped.is_set) is None
    assert len(attempts) == 1
    assert clock.sleeps == [0.1], "cancellation must not wait for the full polling interval"


@pytest.mark.parametrize("session_ready", [False, True])
def test_wait_for_session_cancelled_during_request(monkeypatch, session_ready):
    stopped = threading.Event()

    class Client:
        def resolve_session(self, notebook):
            stopped.set()
            if session_ready:
                return MarimoSession("s1", None, None)
            raise NoSessionError("not open")

    monkeypatch.setattr(mc.time, "sleep", lambda seconds: pytest.fail("no sleep after cancellation"))
    assert mc.wait_for_session(Client(), None, should_stop=stopped.is_set) is None


def test_registry_entry_path_and_removal(tmp_path):
    registry = tmp_path / "servers"
    registry.mkdir()
    path = mc.registry_entry_path("http://127.0.0.1:2718", registry)
    assert path == registry / "127.0.0.1_2718.json"
    path.write_text("{}", encoding="utf-8")
    assert mc.remove_registry_entry("http://127.0.0.1:2718", registry)
    assert not path.exists()
    assert not mc.remove_registry_entry("http://127.0.0.1:2718", registry), "already gone"


def test_marimo_server_command_flags(tmp_path):
    cmd = mc.marimo_server_command(tmp_path / "notebooks", tmp_path, 2731)
    assert cmd[0] == mc.sys.executable and cmd[1:4] == ["-m", "marimo", "edit"]
    assert cmd[4] == "notebooks", "marimo is started on the folder"
    for flag in ("--headless", "--skip-update-check"):
        assert flag in cmd
    assert cmd[cmd.index("--token-password-file") + 1] == "-", "the token is read from stdin, never the command line"
    assert "--no-token" not in cmd and "--token-password" not in cmd
    assert cmd[cmd.index("--port") + 1] == "2731"


def test_marimo_server_command_without_headless_lets_marimo_open_the_browser(tmp_path):
    cmd = mc.marimo_server_command(tmp_path / "nbs", tmp_path, 2718, headless=False)
    assert cmd == [mc.sys.executable, "-m", "marimo", "edit", "nbs", "--token-password-file", "-", "--port", "2718", "--skip-update-check"]


# --------------------------------------------------------------------------- #
# Tokens: signed-in URLs, identifying the server Hailer started
# --------------------------------------------------------------------------- #


def test_open_notebook_url_signs_in_only_when_asked(tmp_path):
    """Fail safe: the token is only added on request (a URL the user opens), never by default."""
    nb = tmp_path / "notebooks" / "analysis.py"
    plain = mc.open_notebook_url(MarimoServer(url="http://127.0.0.1:2718"), nb)
    server = MarimoServer(url="http://127.0.0.1:2718", token="t0k/en+=")
    assert mc.open_notebook_url(server, nb) == plain, "no token unless asked"
    assert mc.open_notebook_url(server, nb, with_token=True) == f"{plain}&access_token=t0k%2Fen%2B%3D"
    assert mc.open_notebook_url(server, nb, view="edit", with_token=True).endswith("&access_token=t0k%2Fen%2B%3D")
    assert mc.home_url(server) == mc.home_url(MarimoServer(url="http://127.0.0.1:2718/")) == "http://127.0.0.1:2718/"
    assert mc.home_url(server, with_token=True) == "http://127.0.0.1:2718/?access_token=t0k%2Fen%2B%3D"


def test_server_token_is_not_in_its_repr():
    server = MarimoServer(url="http://127.0.0.1:2718", token="very-secret-token")
    assert "very-secret-token" not in repr(server)


def test_answers_with_token_identifies_the_server_that_holds_it(fake):
    fake.token = "right-token"
    assert mc.answers_with_token(fake.url, "right-token")
    assert not mc.answers_with_token(fake.url, "wrong-token")
    assert mc.answers_with_token(fake.url, None), "without a token only /health is checked"
    fake.token = None  # a --no-token server accepts any bearer token, so it is never taken for Hailer's
    assert not mc.answers_with_token(fake.url, "right-token")
    assert not mc.answers_with_token(f"http://127.0.0.1:{_free_port()}", "right-token")


def test_client_links_leave_the_token_out_unless_asked(fake, tmp_path):
    nb = tmp_path / "notebooks" / "analysis.py"
    fake.token = "right-token"
    for_model = mc.MarimoClient(fake.url, "right-token", notebook=nb, workspace=tmp_path)
    with pytest.raises(NoSessionError) as exc:
        for_model.resolve_session(nb)
    assert "right-token" not in exc.value.hint and "access_token" not in exc.value.hint
    for_user = mc.MarimoClient(fake.url, "right-token", notebook=nb, workspace=tmp_path, token_in_links=True)
    with pytest.raises(NoSessionError) as exc:
        for_user.resolve_session(nb)
    assert "&access_token=right-token" in exc.value.hint


# --------------------------------------------------------------------------- #
# Path map at the HTTP boundary (a kernel in a container)
# --------------------------------------------------------------------------- #


def _docker_client(fake, tmp_path, **kw):
    from hailer.kernel import docker_paths

    config = _config(tmp_path, notebooks_dir=tmp_path / "notebooks")
    return mc.MarimoClient(fake.url, notebook=config.notebook, workspace=tmp_path, paths=docker_paths(config), **kw)


def test_client_maps_kernel_session_paths_back_to_host_paths(fake, tmp_path):
    nb = tmp_path / "notebooks" / "analysis.py"
    nested = tmp_path / "notebooks" / "team" / "q2 churn.py"
    fake.sessions = {
        "s1": {"filename": "/work/notebooks/analysis.py", "path": "/work/notebooks/analysis.py"},
        "s2": {"filename": "team/q2 churn.py", "path": "/work/notebooks/team/q2 churn.py"},
        "s3": {"filename": "/tmp/scratch.py", "path": "/tmp/scratch.py"},
        "s4": {"filename": None, "path": None},
    }
    client = _docker_client(fake, tmp_path)
    by_id = {s.session_id: s for s in client.sessions()}
    assert Path(by_id["s1"].path) == nb and Path(by_id["s1"].filename) == nb
    assert Path(by_id["s2"].path) == nested and by_id["s2"].filename == "team/q2 churn.py", "relative names are left as they are"
    assert by_id["s3"].path == "/tmp/scratch.py", "outside every mount: kept, so it never matches a host notebook"
    assert by_id["s4"].path is None
    assert client.resolve_session(nb).session_id == "s1"
    assert client.resolve_session(nested).session_id == "s2"
    assert client.execute("1").success  # the configured notebook's session, found through the map
    assert fake.requests[-1]["headers"]["Marimo-Session-Id"] == "s1"


def test_client_sends_kernel_paths_in_the_file_key(fake, tmp_path):
    nb = tmp_path / "notebooks" / "team" / "q2 churn.py"
    client = _docker_client(fake, tmp_path)
    assert client.notebook_url(nb) == f"{fake.url}/?file=/work/notebooks/team/q2%20churn.py&view-as=present"
    with pytest.raises(NoSessionError) as exc:
        client.resolve_session(nb)
    assert "?file=/work/notebooks/team/q2%20churn.py" in exc.value.hint
    assert str(tmp_path) not in exc.value.hint.split("Open sessions")[0], "no host path in the URL"
    # a notebook the container cannot see has no URL of its own; the home page is offered instead
    assert client.notebook_url(tmp_path / "elsewhere" / "x.py") == f"{fake.url}/"
    server = MarimoServer(url=fake.url, paths=client.paths)
    with pytest.raises(NotebookPathError):
        mc.open_notebook_url(server, tmp_path / "elsewhere" / "x.py")


# --------------------------------------------------------------------------- #
# find_server: marimo_url, then .hailer/kernel.json, then the registry (local only)
# --------------------------------------------------------------------------- #


def _record(tmp_path: Path, url: str, *, runtime: str = "local", token: str = "right-token", **extra):
    from hailer.kernel import KernelState, write_kernel_state

    port = int(url.rsplit(":", 1)[1])
    write_kernel_state(tmp_path, KernelState(runtime=runtime, url=url, port=port, token=token, **extra))


def _kernel_cfg(tmp_path: Path, runtime: str = "local", **overrides) -> HailerConfig:
    return _config(tmp_path, notebooks_dir=tmp_path / "notebooks", kernel=KernelConfig(runtime=runtime), **overrides)


def _registry_with(tmp_path: Path, fake) -> Path:
    reg = tmp_path / "servers"
    _write_entry(reg, "a.json", server_id="a", pid=1, host="127.0.0.1", port=fake.server_address[1], base_url="", started_at="", version="")
    return reg


def test_find_server_prefers_the_server_hailer_started_over_the_registry(fake, tmp_path):
    fake.token = "right-token"
    _record(tmp_path, fake.url, pid=4242)
    other = FakeMarimo().start()  # a --no-token server of the user's own, in the registry
    try:
        server = mc.find_server(_kernel_cfg(tmp_path), registry=_registry_with(tmp_path, other))
    finally:
        other.stop()
    assert server is not None and server.url == fake.url
    assert (server.source, server.token, server.runtime, server.pid, server.paths) == ("kernel", "right-token", "local", 4242, None)


def test_find_server_ignores_a_record_whose_server_is_gone_or_not_ours(fake, tmp_path):
    from hailer.kernel import read_kernel_state

    reg = _registry_with(tmp_path, fake)
    fake.root = str(_kernel_cfg(tmp_path).notebooks_root)
    _record(tmp_path, f"http://127.0.0.1:{_free_port()}")  # stale: nothing listens there
    assert mc.find_server(_kernel_cfg(tmp_path), registry=reg).source == "registry"
    assert read_kernel_state(tmp_path) is not None, "no pid: it cannot be proven dead, so it is kept"
    _record(tmp_path, fake.url)  # the port now belongs to a --no-token server: not the one recorded
    server = mc.find_server(_kernel_cfg(tmp_path), registry=reg)
    assert server.source == "registry" and server.token is None


def test_find_server_drops_a_record_whose_process_is_gone(tmp_path, monkeypatch):
    """A stale record would cost a health check (about a second on Windows) on every tool call."""
    from hailer import kernel
    from hailer.kernel import read_kernel_state

    probes: list[str] = []
    monkeypatch.setattr(kernel, "answers_with_token", lambda url, token, timeout=1.0: probes.append(url) or False)
    monkeypatch.setattr(kernel, "process_running", lambda pid: {77: False, 78: None}.get(pid, True))
    _record(tmp_path, "http://127.0.0.1:9", pid=78)  # the OS cannot say: kept
    assert mc.find_server(_kernel_cfg(tmp_path), registry=tmp_path / "none") is None
    assert read_kernel_state(tmp_path) is not None
    _record(tmp_path, "http://127.0.0.1:9", pid=77)
    assert mc.find_server(_kernel_cfg(tmp_path), registry=tmp_path / "none") is None
    assert read_kernel_state(tmp_path) is None, "its process is gone: deleted"
    assert mc.find_server(_kernel_cfg(tmp_path), registry=tmp_path / "none") is None
    assert len(probes) == 2, "the next call does not probe again"


def test_find_server_marimo_url_takes_the_records_token_and_paths(tmp_path):
    url = f"http://127.0.0.1:{_free_port()}"
    _record(tmp_path, url, runtime="docker", mounts=((str(tmp_path / "notebooks"), "/work/notebooks"),))
    server = mc.find_server(_kernel_cfg(tmp_path, marimo_url=url + "/"), registry=tmp_path / "none")
    assert server.source == "kernel" and server.token == "right-token" and server.runtime == "docker"
    assert server.paths is not None and server.paths.to_kernel(tmp_path / "notebooks" / "a.py") == "/work/notebooks/a.py"
    other = mc.find_server(_kernel_cfg(tmp_path, marimo_url="http://127.0.0.1:9"), registry=tmp_path / "none")
    assert other == MarimoServer(url="http://127.0.0.1:9", source="config"), "any other URL: as configured, no token"


def test_find_server_for_docker_fails_closed(fake, tmp_path):
    reg = _registry_with(tmp_path, fake)
    docker = _kernel_cfg(tmp_path, "docker")
    assert mc.find_server(docker, registry=reg) is None, "never the registry"
    assert mc.find_server(replace(docker, marimo_url=fake.url), registry=reg) is None, "never a configured local URL"
    fake.token = "right-token"
    _record(tmp_path, fake.url, runtime="local")
    assert mc.find_server(docker, registry=reg) is None, "never a local server Hailer started"
    assert mc.find_server(replace(docker, marimo_url=fake.url), registry=reg) is None
    _record(tmp_path, fake.url, runtime="docker", mounts=((str(tmp_path / "notebooks"), "/work/notebooks"),))
    server = mc.find_server(docker, registry=reg)
    assert server is not None and server.runtime == "docker" and server.paths is not None


def test_find_server_local_request_may_attach_to_a_docker_kernel(fake, tmp_path):
    fake.token = "right-token"
    _record(tmp_path, fake.url, runtime="docker", mounts=((str(tmp_path / "notebooks"), "/work/notebooks"),))
    server = mc.find_server(_kernel_cfg(tmp_path), registry=tmp_path / "none")
    assert server is not None and server.runtime == "docker" and server.token == "right-token"


def test_launch_hint_offers_one_command_route_first():
    hint = mc.launch_hint()
    assert hint.index("uvx hailer        (or: uvx hailer notebook)") < hint.index("uvx hailer notebook --foreground")
    assert "starts marimo for this workspace" in hint, "bare hailer starts its own server"
    assert "Or run marimo on its own in another terminal:" in hint
    assert "uv run" not in hint


def test_unavailable_error_uses_one_command_hint(tmp_path):
    client = mc.MarimoClient(f"http://127.0.0.1:{_free_port()}", timeout=0.5, notebook=tmp_path / "nbs" / "nb.py", workspace=tmp_path)
    with pytest.raises(MarimoUnavailableError) as info:
        client.sessions()
    assert info.value.hint == mc.launch_hint()
