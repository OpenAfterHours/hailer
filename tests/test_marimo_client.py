"""Offline tests for hailer.marimo_client against a local fake marimo server."""

from __future__ import annotations

import json
import os
import socket
import threading
from pathlib import Path
from urllib.parse import quote

import pytest

from fake_marimo import FakeMarimo
from hailer import marimo_client as mc
from hailer.errors import MarimoExecutionError, MarimoUnavailableError, NoSessionError
from hailer.models import MarimoServer, MarimoSession

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


def _local_client(fake, tmp_path: Path, **kw) -> mc.MarimoClient:
    """A client for a local kernel: it knows the notebooks folder by its host path."""
    return mc.MarimoClient(fake.url, notebooks_path=mc.notebook_file_key(tmp_path / "notebooks"), native_paths=True, **kw)


def _docker_client(fake, **kw) -> mc.MarimoClient:
    """A client for a kernel in a container: it knows the notebooks folder as /work/notebooks."""
    return mc.MarimoClient(fake.url, notebooks_path="/work/notebooks", **kw)


# --------------------------------------------------------------------------- #
# health / sessions
# --------------------------------------------------------------------------- #


def test_health_true_and_false(fake):
    assert mc.MarimoClient(fake.url).health() is True
    assert mc.MarimoClient(f"http://127.0.0.1:{_free_port()}").health() is False


def test_sessions_parsed(fake, tmp_path):
    nb = tmp_path / "notebooks" / "analysis.py"
    fake.sessions = {"s1": {"filename": "notebooks/analysis.py", "path": str(nb)}, "s2": {"filename": None, "path": None}}
    assert _local_client(fake, tmp_path).sessions() == [
        MarimoSession("s1", "notebooks/analysis.py", str(nb), name="analysis.py"),
        MarimoSession("s2", None, None),
    ]
    assert mc.MarimoClient(fake.url).sessions()[0].name is None, "no notebooks folder: no names"


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
    result = _local_client(fake, tmp_path, notebook="analysis.py").execute("1")
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
    assert "HAILER_MARIMO_TOKEN" not in exc.value.hint and "uvx hailer again" in exc.value.hint


def test_connection_refused_hint_says_how_to_get_a_new_kernel(tmp_path):
    client = mc.MarimoClient(f"http://127.0.0.1:{_free_port()}", notebooks_path="/work/notebooks", notebook="analysis.py")
    with pytest.raises(MarimoUnavailableError) as exc:
        client.sessions()
    assert exc.value.hint == mc.launch_hint() and "uvx hailer" in exc.value.hint
    assert "uv run" not in exc.value.hint and "--foreground" not in exc.value.hint


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
    with pytest.raises(NoSessionError) as exc:
        _local_client(fake, tmp_path).resolve_session("analysis.py")
    assert "not open in a browser" in str(exc.value)
    key = mc.notebook_file_key(tmp_path / "notebooks" / "analysis.py")
    assert exc.value.hint.startswith("Open ") and f"?file={quote(key, safe='/:')}" in exc.value.hint


def test_resolve_by_name_whatever_the_case_of_the_host_path(fake, tmp_path):
    nb = tmp_path / "notebooks" / "analysis.py"
    stored = str(nb).upper().replace("ANALYSIS.PY", "analysis.py") if os.name == "nt" else str(nb)
    fake.sessions = {"other": {"filename": "x.py", "path": str(tmp_path / "x.py")}, "mine": {"filename": "notebooks/analysis.py", "path": stored}}
    client = _local_client(fake, tmp_path)
    assert client.resolve_session("analysis.py").session_id == "mine"
    assert {s.session_id: s.name for s in client.sessions()} == {"other": None, "mine": "analysis.py"}, "x.py is outside the folder"


def test_relative_or_bare_session_paths_never_match(fake, tmp_path):
    """marimo reports absolute paths for notebooks opened by an absolute key; a bare or relative
    filename is not the same notebook (no matching by file name alone)."""
    fake.sessions = {"other": {"filename": "analysis.py", "path": None}, "rel": {"filename": "notebooks/analysis.py", "path": None}}
    with pytest.raises(NoSessionError):
        _local_client(fake, tmp_path).resolve_session("analysis.py")


def test_resolve_same_filename_in_two_subfolders(fake, tmp_path):
    a = tmp_path / "notebooks" / "a" / "report.py"
    b = tmp_path / "notebooks" / "b" / "report.py"
    fake.sessions = {"sA": {"filename": "a/report.py", "path": str(a)}}
    with pytest.raises(NoSessionError):
        _local_client(fake, tmp_path).resolve_session("b/report.py")
    fake.sessions = {"sA": {"filename": "a/report.py", "path": str(a)}, "sB": {"filename": "b/report.py", "path": str(b)}}
    client = _local_client(fake, tmp_path)
    assert client.resolve_session("a/report.py").session_id == "sA"
    assert client.resolve_session("b/report.py").session_id == "sB"


def test_match_session_ignores_untitled_and_prefers_the_latest():
    sessions = [
        MarimoSession("untitled", None, None),
        MarimoSession("old", "analysis.py", "/work/notebooks/analysis.py", name="analysis.py"),
        MarimoSession("outside", "x.py", "/tmp/analysis.py"),
        MarimoSession("new", "analysis.py", "/work/notebooks/analysis.py", name="analysis.py"),
    ]
    assert mc.match_session(sessions, "analysis.py").session_id == "new"
    assert mc.match_session(sessions[:1], "analysis.py") is None
    assert mc.match_session([], "analysis.py") is None
    assert mc.match_session(sessions, "sub/analysis.py") is None
    assert mc.match_session(sessions, "ANALYSIS.py") is None, "case matters unless the kernel's paths fold it"
    assert mc.match_session(sessions, "ANALYSIS.py", fold=True).session_id == "new"


def test_resolve_multiple_no_match_lists_sessions(fake, tmp_path):
    fake.sessions = {"a": {"filename": "one.py", "path": str(tmp_path / "one.py")}, "b": {"filename": "two.py", "path": str(tmp_path / "two.py")}}
    with pytest.raises(NoSessionError) as exc:
        _local_client(fake, tmp_path).resolve_session("analysis.py")
    assert "one.py" in exc.value.hint and "two.py" in exc.value.hint


def test_resolve_single_session_without_notebook(fake):
    fake.sessions = {"only": {"filename": "n.py", "path": "n.py"}}
    assert mc.MarimoClient(fake.url).resolve_session(None).session_id == "only"


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


def test_open_notebook_url_defaults_to_app_view():
    url = mc.open_notebook_url(MarimoServer(url="http://127.0.0.1:2718"), "/work/notebooks/my analysis.py")
    assert url == "http://127.0.0.1:2718/?file=/work/notebooks/my%20analysis.py&view-as=present"
    key = "C:/Users/ann/notebooks/a.py"
    assert mc.open_notebook_url(MarimoServer(url="http://127.0.0.1:2718"), key).endswith(f"?file={key}&view-as=present")


def test_open_notebook_url_edit_view():
    server = MarimoServer(url="http://127.0.0.1:2718/")
    assert mc.open_notebook_url(server, "/work/notebooks/a.py", view="edit") == "http://127.0.0.1:2718/?file=/work/notebooks/a.py"
    with pytest.raises(ValueError):
        mc.open_notebook_url(server, "/work/notebooks/a.py", view="kiosk")


def test_client_notebook_url_uses_app_view(fake, tmp_path):
    client = _local_client(fake, tmp_path, notebook="q3/analysis.py")
    key = mc.notebook_file_key(tmp_path / "notebooks" / "q3" / "analysis.py")
    assert client.notebook_url().endswith(f"?file={quote(key, safe='/:')}&view-as=present")
    assert mc.MarimoClient(fake.url, notebook="x.py").notebook_url() == f"{fake.url}/", "no notebooks folder: the home page"


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
    client = _local_client(fake, tmp_path, notebook="nb.py")
    with pytest.raises(MarimoUnavailableError) as info:
        client.sessions()
    assert "HTTP 500" in str(info.value)
    assert "kernel manager exploded" in str(info.value)
    assert "Check the marimo server log" in info.value.hint and "uvx hailer again" in info.value.hint
    # resolve_session (used by preflight) must surface the same actionable error, never a raw HTTPError
    with pytest.raises(MarimoUnavailableError):
        client.resolve_session("nb.py")


def test_sse_parser_handles_comments_and_multiline_data():
    lines = [b": comment\r\n", b"event: stdout\r\n", b"data: {\"data\":\r\n", b"data: \"x\"}\r\n", b"\r\n", b"event: done\n", b"data: {\"success\": true, \"output\": {\"data\": \"\", \"mimetype\": \"text/plain\"}}\n", b"\n"]
    events = list(mc._iter_sse(iter(lines)))
    assert events[0][0] == "stdout" and json.loads(events[0][1]) == {"data": "x"}
    assert events[1][0] == "done"


# --------------------------------------------------------------------------- #
# Helpers for `hailer notebook`: ports, readiness waits, hint
# --------------------------------------------------------------------------- #


def test_find_free_port_prefers_the_requested_port():
    port = _free_port()
    assert mc.find_free_port(port) == port


def test_find_free_port_falls_back_when_busy(fake):
    busy = fake.server_address[1]
    chosen = mc.find_free_port(busy)
    assert chosen != busy
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", chosen))  # free when chosen


def test_wait_for_health_true_for_live_server(fake):
    assert mc.wait_for_health(fake.url, timeout=2.0, interval=0.05)


def test_wait_for_health_with_a_token_accepts_only_the_server_that_holds_it(fake):
    """Two starts can pick the same port: another session's server answers /health there, but not
    with this start's token."""
    fake.token = "this-starts-token"
    assert mc.wait_for_health(fake.url, timeout=2.0, token="this-starts-token", interval=0.05)
    assert not mc.wait_for_health(fake.url, timeout=0.2, token="another-token", interval=0.05)


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


def test_wait_for_session_returns_when_notebook_opens(fake):
    client = _docker_client(fake)

    def open_tab():
        fake.sessions["s_new"] = {"filename": "/work/notebooks/analysis.py", "path": "/work/notebooks/analysis.py"}

    threading.Timer(0.3, open_tab).start()
    session = mc.wait_for_session(client, "analysis.py", timeout=5.0, interval=0.05)
    assert session is not None and session.session_id == "s_new"


def test_wait_for_session_none_on_timeout(fake):
    assert mc.wait_for_session(_docker_client(fake), "analysis.py", timeout=0.3, interval=0.05) is None


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


def test_marimo_server_command_flags(tmp_path):
    cmd = mc.marimo_server_command(tmp_path / "notebooks", tmp_path, 2731)
    assert cmd[0] == mc.sys.executable and cmd[1:4] == ["-m", "marimo", "edit"]
    assert cmd[4] == "notebooks", "marimo is started on the folder"
    for flag in ("--headless", "--skip-update-check"):
        assert flag in cmd
    assert cmd[cmd.index("--token-password-file") + 1] == "-", "the token is read from stdin, never the command line"
    assert "--no-token" not in cmd and "--token-password" not in cmd
    assert cmd[cmd.index("--port") + 1] == "2731"


# --------------------------------------------------------------------------- #
# Tokens: signed-in URLs, identifying the server Hailer started
# --------------------------------------------------------------------------- #


def test_open_notebook_url_signs_in_only_when_asked():
    """Fail safe: the token is only added on request (a URL the user opens), never by default."""
    nb = "/work/notebooks/analysis.py"
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


def test_client_links_leave_the_token_out_unless_asked(fake):
    fake.token = "right-token"
    for_model = _docker_client(fake, token="right-token", notebook="analysis.py")
    with pytest.raises(NoSessionError) as exc:
        for_model.resolve_session("analysis.py")
    assert "right-token" not in exc.value.hint and "access_token" not in exc.value.hint
    for_user = _docker_client(fake, token="right-token", notebook="analysis.py", token_in_links=True)
    with pytest.raises(NoSessionError) as exc:
        for_user.resolve_session("analysis.py")
    assert "&access_token=right-token" in exc.value.hint


# --------------------------------------------------------------------------- #
# Names <-> kernel paths (the runtime layer's only path mapping)
# --------------------------------------------------------------------------- #


def test_name_in_folder_for_a_container():
    assert mc.name_in_folder("/work/notebooks", "/work/notebooks/team/q2 churn.py", native=False) == "team/q2 churn.py"
    assert mc.name_in_folder("/work/notebooks/", "/work/notebooks/a.py", native=False) == "a.py"
    for outside in ("/tmp/scratch.py", "/work/hailer.toml", "/work/notebooksX/a.py", "/work/notebooks", "/work/notebooks/../../etc/passwd", "notebooks/a.py", "", None, 3):
        assert mc.name_in_folder("/work/notebooks", outside, native=False) is None, outside
    assert mc.name_in_folder("/work/notebooks", "/work/notebooks/sub/../a.py", native=False) == "a.py", "normalised first"
    assert mc.name_in_folder("/work/notebooks", "/WORK/notebooks/a.py", native=False) is None, "POSIX paths are case-sensitive"
    assert mc.name_in_folder(None, "/work/notebooks/a.py", native=False) is None


def test_name_in_folder_for_the_local_kernel(tmp_path):
    root = mc.notebook_file_key(tmp_path / "notebooks")
    assert mc.name_in_folder(root, str(tmp_path / "notebooks" / "team" / "r.py"), native=True) == "team/r.py"
    assert mc.name_in_folder(root, f"{root}/a.py", native=True) == "a.py", "either separator"
    assert mc.name_in_folder(root, str(tmp_path / "notebooks-old" / "a.py"), native=True) is None, "a sibling is not inside"
    assert mc.name_in_folder(root, str(tmp_path / "elsewhere.py"), native=True) is None
    assert mc.name_in_folder(root, "notebooks/a.py", native=True) is None, "relative paths never match"
    if os.name == "nt":
        assert mc.name_in_folder(root, str(tmp_path / "NOTEBOOKS" / "A.py").upper(), native=True) == "A.PY", "Windows ignores case"


def test_name_in_folder_follows_a_link_to_the_notebooks_folder(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    try:
        (tmp_path / "linked").symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("creating symlinks needs a privilege here")
    root = mc.notebook_file_key(tmp_path / "linked")
    assert mc.name_in_folder(root, str(real / "a.py"), native=True) == "a.py"


def test_the_client_speaks_names_to_a_kernel_in_a_container(fake):
    fake.sessions = {
        "s1": {"filename": "/work/notebooks/analysis.py", "path": "/work/notebooks/analysis.py"},
        "s2": {"filename": "team/q2 churn.py", "path": "/work/notebooks/team/q2 churn.py"},
        "s3": {"filename": "/tmp/scratch.py", "path": "/tmp/scratch.py"},
        "s4": {"filename": None, "path": None},
    }
    client = _docker_client(fake, notebook="analysis.py")
    assert {s.session_id: s.name for s in client.sessions()} == {"s1": "analysis.py", "s2": "team/q2 churn.py", "s3": None, "s4": None}
    assert client.resolve_session("team/q2 churn.py").session_id == "s2"
    assert client.execute("1").success  # the default notebook's session
    assert fake.requests[-1]["headers"]["Marimo-Session-Id"] == "s1"
    assert client.notebook_url("team/q2 churn.py") == f"{fake.url}/?file=/work/notebooks/team/q2%20churn.py&view-as=present"
    with pytest.raises(NoSessionError) as exc:
        client.resolve_session("other.py")
    assert "?file=/work/notebooks/other.py" in exc.value.hint


# --------------------------------------------------------------------------- #
# marimo's file explorer
# --------------------------------------------------------------------------- #


def test_file_endpoints_send_the_bearer_and_the_server_token(fake, tmp_path):
    """Like marimo 0.24.2: every file endpoint is @requires("edit") (the bearer token) and needs the
    skew-protection header; no browser session is involved."""
    fake.token = "right-token"
    fake.serve_files(tmp_path)
    client = _docker_client(fake, token="right-token")
    created = client.create_file("/work/notebooks/q3", "a.py", b"import marimo\napp = marimo.App()\n")
    assert created["success"] and created["info"]["path"] == "/work/notebooks/q3/a.py" and created["info"]["isMarimoFile"]
    assert [e["name"] for e in client.list_files("/work/notebooks")] == ["q3"]
    assert client.file_details("/work/notebooks/q3/a.py")["contents"].startswith("import marimo")
    assert client.update_file("/work/notebooks/q3/a.py", "x = 1\n")["success"]
    assert client.delete_file("/work/notebooks/q3/a.py")["success"]
    posts = [r for r in fake.requests if r["path"].startswith("/api/files/")]
    assert len(posts) == 5
    assert all(r["headers"]["Authorization"] == "Bearer right-token" and r["headers"]["Marimo-Server-Token"] == "skew-token-123" for r in posts)
    assert posts[0]["headers"]["Content-Type"].startswith("multipart/form-data; boundary=")
    assert posts[0]["body"] == {"path": "/work/notebooks/q3", "type": "file", "name": "a.py", "file": ("a.py", b"import marimo\napp = marimo.App()\n")}


def test_file_endpoints_without_the_tokens_are_refused(fake, tmp_path):
    fake.token = "right-token"
    fake.serve_files(tmp_path)
    with pytest.raises(MarimoUnavailableError) as exc:
        _docker_client(fake, token="wrong").list_files("/work/notebooks")
    assert "HTTP 401" in str(exc.value)
    client = _docker_client(fake, token="right-token")
    client.server_token()
    fake.mode = "stale_token"
    with pytest.raises(MarimoUnavailableError) as exc:
        client.update_file("/work/notebooks/a.py", "")
    assert "server token" in str(exc.value) and "restarted" in exc.value.hint


def test_file_details_of_a_missing_file_is_an_http_error(fake, tmp_path):
    import urllib.error

    fake.serve_files(tmp_path)
    with pytest.raises(urllib.error.HTTPError) as exc:
        _docker_client(fake).file_details("/work/notebooks/missing.py")
    assert exc.value.code == 500
    assert _docker_client(fake).list_files("/work/notebooks/missing") == [], "a missing folder lists as empty"


def test_unavailable_error_uses_one_command_hint(tmp_path):
    client = mc.MarimoClient(f"http://127.0.0.1:{_free_port()}", timeout=0.5, notebooks_path="/work/notebooks", notebook="nb.py")
    with pytest.raises(MarimoUnavailableError) as info:
        client.sessions()
    assert info.value.hint == mc.launch_hint()
